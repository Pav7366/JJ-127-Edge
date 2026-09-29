# JJ-127-Edge

One container = one simulated traffic camera. It reads a video file, finds the
vehicles, reads their number plates, and publishes a JSON detection over MQTT -
then does it all again on the next frame.

```
video file ──► frame sampler ──► YOLO26s (vehicles) ──► plate detector + OCR ──► JSON ──► MQTT
   (mount)      stride/loop       (ONNX, 36.5 MB)        (ONNX, 7.5 + 5.1 MB)   6 keys   QoS 1
```

* **Offline by design** - every model is in the image before it starts; the
  container never calls out to the internet while streaming.
* **No PyTorch anywhere** - the vehicle model is an ONNX graph you commit, so the
  image ships onnxruntime + headless OpenCV only (836 MB, down from 1.84 GB with
  the old `yolov8n.pt` setup) and the build needs no 2 GB of torch wheels.
* **One thread per camera** - `yolo26s` + two ONNX models, CPU only, ~1.5 cores
  and ~450 MB RAM per camera.
* **Resilient** - a dead broker only causes buffering (bounded, oldest dropped),
  a bad frame only fails that frame, a missing file fails the container loudly.

---

## 1. Quick start

The vehicle model is **yours to supply**: put your YOLO26s ONNX graph at
`models/yolo26s.onnx` and the build bakes it into the image. Nothing is trained,
downloaded or converted at build time, and the build fails with instructions if
the file is missing.

```bash
# 1. config
cp .env.example .env

# 2. your vehicle model (this is the only model you have to provide)
#    - must be an Ultralytics-style YOLO26 export
#    - 640x640 static input, output (1, 4 + num_classes, num_anchors)
cp /path/to/your/yolo26s.onnx models/yolo26s.onnx
python scripts/download_models.py --models-dir models   # verifies the graph

# 3. some footage (or drop your own .mp4/.avi into ./data)
#    The bundled assets/sample_car.jpg makes the demo detect real vehicles;
#    pass --source-image to pan/zoom over your own photo instead.
python scripts/make_test_video.py --output data/cam01.mp4 --seconds 12
cp data/cam01.mp4 data/cam02.mp4 && cp data/cam01.mp4 data/cam03.mp4

# 4. build (installs runtime wheels, bakes in the models)
docker build -t anpr-edge .

# 5. run three cameras + a broker + a live monitor
docker compose up --build
docker compose logs -f monitor
```

No footage? Dry run the whole pipeline on the host, no broker needed:

```bash
CAMERA_ID=cam01 LAT=12.9716 LON=77.5946 \
MQTT_BROKER=localhost:1883 \
VIDEO_PATH=data/cam01.mp4 \
python main.py --dry-run
```

You should see one JSON line per detection:

```json
{"plate_string":"KA01AB1234","confidence":0.97,"camera_id":"cam01","lat":12.9716,"lon":77.5946,"timestamp":"2026-09-26T10:48:41.851Z"}
```

Single container, no compose:

```bash
docker run --rm \
  -e CAMERA_ID=cam01 -e LAT=12.9716 -e LON=77.5946 \
  -e VIDEO_PATH=/data/cam01.mp4 -e MQTT_BROKER=mosquitto:1883 \
  -v "$PWD/data:/data" anpr-edge
```

### CLI flags

| Flag | Purpose |
| --- | --- |
| `--dry-run` | run the CV pipeline, print payloads to stdout, never touch MQTT |
| `--print-config` | dump the resolved configuration (passwords masked) and exit |
| `--check-models` | load and warm up every model, then exit (post-deploy sanity check) |
| `--video PATH` | override `VIDEO_PATH` without editing env |
| `--log-level`, `--log-format` | override `LOG_LEVEL` / `LOG_FORMAT` |

---

## 2. The MQTT contract

The topic and the payload below are **not ours to choose**. They are dictated by
the backend's `contract/event_contract.json`, which is the single source of truth
for the whole platform. `ingest` validates every message against that schema and
drops anything that fails, so a key spelled `latitude` here instead of `lon` costs
us every sighting in the city and shows up only as a `Dropping invalid sighting`
line in someone else's logs.

**Topic**

```
anpr/{camera_id}/sightings
```

e.g. `anpr/cam-pune-hinjewadi-01/sightings` (template via `MQTT_TOPIC_TEMPLATE`).
`{camera_id}` must equal the `camera_id` in the payload; `ingest` drops any
mismatch rather than guess which of the two is right.

**Payload** - the six required keys, plus whichever optional ones this detection
actually has:

```json
{
  "plate_string":  "KA01AB1234",
  "confidence":    0.97,
  "camera_id":     "cam-01",
  "lat":           12.9716,
  "lon":           77.5946,
  "timestamp":     "2026-09-26T10:48:41.851Z",
  "image_ref":     "plates/cam-01_000000044_....jpg",
  "direction":     "E",
  "lane":          2,
  "vehicle_type":  "car"
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `plate_string` | yes | upper-cased, spaces/dashes stripped (`ka 01 ab 1234` → `KA01AB1234`) |
| `confidence` | yes | mean per-character OCR confidence, `0.0`-`1.0`, not rescaled |
| `camera_id` | yes | from `CAMERA_ID`; must match the topic |
| `lat` / `lon` | yes | surveyed camera position from env |
| `timestamp` | yes | ISO 8601 UTC, milliseconds, `Z` suffix |
| `image_ref` | no | a file path or a base64 JPEG per `IMAGE_REF_MODE`; key omitted when `none` |
| `direction` | no | from `CAMERA_DIRECTION`, one of `N S E W NE NW SE SW` |
| `lane` | no | from `CAMERA_LANE` |
| `vehicle_type` | no | `car` \| `bike` \| `truck` \| `bus`, mapped from the detector's class |

Optional keys are **omitted when there is nothing to say** rather than sent as
`null`. The schema marks them nullable but not required, so a key carrying only
`null` tells a consumer nothing that absence does not.

`vehicle_type` is translated from the detector's COCO label to the contract's
four-value enum (`motorcycle` → `bike`, `bicycle` → `bike`). A class outside the
enum is **dropped, not forwarded**: it would fail validation and take the whole
sighting down with it. Absent is always safer than wrong, because downstream the
class selects a map pin and a filter, and a wrong pin is believed. Under
`PLATE_SCOPE=frame` there is no vehicle box to inherit a class from, so the key is
absent — which the contract treats as the honest "could not classify" answer.

Optional MQTT 5 **user properties** carry the extra operational detail (plate
box, region, detection confidence) *without* touching the payload - enable with
`MQTT_PUBLISH_DIAGNOSTICS=true`. They are also emitted as JSON log fields:

```json
{"event":"detection","plate":"KA01AB1234","confidence":0.97,"region":"IN-KA",
 "vehicle_class":"car","plate_box":[310,190,394,214],"source":"crop",
 "mqtt_delivery":"sent","frame_index":44,"video_time_seconds":7.3}
```

Subscribe:

```bash
mosquitto_sub -h localhost -t 'anpr/#' -v
```

### Running against the platform

This container is the real producer for
[`JJ-127-DEVOPS`](../JJ-127-DEVOPS), which brings up the broker, `ingest`, PostGIS,
the GIS API and the dashboard, and runs one of these per camera behind its `edge`
profile. From the DEVOPS checkout:

```bash
make passwd          # generate the broker password file from .env
make up              # broker, db, ingest, gis, frontend
make anpr-up         # + the three real CV cameras
make anpr-logs
```

`CAMERA_ID` must equal the camera's MQTT username, and `mosquitto/config/acl.txt`
must grant it write access to `anpr/{CAMERA_ID}/sightings`, or every message is
refused at the broker. DEVOPS's `.env` already carries the three Pune cameras
with matching ACL entries, so the defaults line up.

---

## 3. Configuration

Everything is environment driven; `.env.example` is the annotated full list.
Required: `CAMERA_ID`, `LAT`, `LON`, `VIDEO_PATH`, `MQTT_BROKER`.
Invalid values fail immediately with a `config_invalid` log line and exit code 2.

### Video

| Variable | Default | Meaning |
| --- | --- | --- |
| `VIDEO_PATH` | – | clip inside the container (mount it, do not bake it) |
| `FRAME_STRIDE` | `1` | process every Nth frame; `3` ≈ 3× faster, fewer misses |
| `LOOP_VIDEO` | `true` | restart the clip at EOF (keeps a demo running forever) |
| `REALTIME_FPS` | `0` | throttle to a simulated frame rate; `0` = as fast as possible |
| `MAX_FRAMES` | `0` | stop after N sampled frames; `0` = unlimited |
| `FRAME_MAX_WIDTH` | `0` | downscale wider frames before inference (big speedup, some accuracy cost) |
| `TIMESTAMP_SOURCE` | `wall` | `wall` = processing time, `video` = `VIDEO_START_TIME` + clip time |
| `VIDEO_START_TIME` | now | ISO 8601 epoch used by `TIMESTAMP_SOURCE=video` |

### Models

| Variable | Default | Meaning |
| --- | --- | --- |
| `VEHICLE_WEIGHTS` | `/models/yolo26s.onnx` | baked ONNX graph (see [Re-exporting the vehicle model](#re-exporting-the-vehicle-model)) |
| `VEHICLE_CLASSES` | `car,truck,bus,motorcycle` | classes to publish (plate detection only runs on these) |
| `VEHICLE_CONF_THRESHOLD` | `0.30` | YOLO confidence gate |
| `VEHICLE_IOU` | `0.50` | NMS IoU threshold |
| `VEHICLE_MAX_DET` | `20` | cap on vehicles kept per frame |
| `VEHICLE_MIN_BOX_PX` | `32` | drop boxes smaller than this in the source frame |
| `VEHICLE_IMGSZ` | `640` | YOLO input size; the exported graph is **static**, so this only takes effect if the graph was exported dynamically (see below) |
| `PLATE_MODEL` | `yolo-v9-t-416-license-plate-end2end` | plate detector |
| `PLATE_SCOPE` | `crop` | `crop` per vehicle, `frame` whole frame, `both` crop then frame |
| `PLATE_CONF_THRESHOLD` | `0.30` | plate detector gate |
| `OCR_MODEL` | `cct-s-v2-global-model` | plate OCR |
| `OCR_MIN_CONFIDENCE` | `0.60` | readings below this are dropped |
| `PLATE_MIN_LENGTH` | `5` | reject obvious fragments |
| `PLATE_PATTERN` | – | regex allow-list, e.g. `^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$` |
| `INFERENCE_THREADS` | `1` | intra-op threads; `1` is best when running many containers |

### Output and MQTT

| Variable | Default | Meaning |
| --- | --- | --- |
| `IMAGE_REF_MODE` | `none` | `none` \| `path` \| `base64` |
| `PLATE_IMAGE_DIR` | `/data/plates` | where `path` mode writes JPEGs |
| `PLATE_IMAGE_MAX_WIDTH` | `320` | downscale stored plate images |
| `PLATE_IMAGE_JPEG_QUALITY` | `85` | JPEG quality |
| `PLATE_DEDUP_WINDOW_SECONDS` | `10` | suppress a repeat of the same plate inside the window |
| `MQTT_BROKER` | `localhost:1883` | `host:port`, `mqtts://…`, or host + `MQTT_BROKER_PORT` |
| `MQTT_QOS` | `1` | at-least-once |
| `MQTT_PROTOCOL` | `v5` | `v5` or `v311` |
| `MQTT_USERNAME` / `MQTT_PASSWORD` | – | optional auth |
| `MQTT_TLS` / `MQTT_CA_CERTS` | `false` / – | TLS |
| `MQTT_MAX_QUEUE` | `1000` | offline buffer size; oldest dropped when full |
| `MQTT_RECONNECT_MIN_DELAY` / `_MAX_DELAY` | `1` / `60` | backoff bounds |
| `MQTT_PUBLISH_DIAGNOSTICS` | `false` | add box/region as user properties |

`LAT`/`LON` default to `0.0`; set them to the real camera position so detections
are not all pinned to Null Island.

---

## 4. Model choices and why

| Stage | Model | Size | Why |
| --- | --- | --- | --- |
| Vehicle | **your** `models/yolo26s.onnx` | ~36 MB | Supplied by you and baked in unchanged; the build only verifies it, so swapping in a retrained or differently sized graph is a file replacement, not a rebuild of anything else |
| Plate detection | `yolo-v9-t-416-license-plate-end2end` (ONNX) | 7.5 MB | Purpose-trained for plates, T=smallest tier; no PyTorch needed at inference, so it runs inside the already-loaded ONNX stack |
| Plate OCR | `cct-s-v2-global-model` (ONNX) | 5.1 MB | `cct-s-v2` is the best accuracy/speed trade-off of the `fast-plate-ocr` family and handles Latin + Indian plates without per-country models |

The vehicle graph keeps the standard YOLO layout (`(1, 4 + 80, 8400)` of
`cx, cy, w, h` plus per-class scores), so `vehicle_detector.py` owns the letterbox
transform, the class/confidence gate and NMS. Keeping that logic as plain
functions means the decode maths is unit tested without loading a model. The
graph output is *not* NMS-free at inference time despite YOLO26's architecture,
hence the explicit NMS step.

### The vehicle model contract

`models/yolo26s.onnx` is copied into the image verbatim, so anything the runtime
detector assumes has to be true of your graph. `scripts/download_models.py`
checks all of it at build time and prints the verdict:

| Property | Expected | Why |
| --- | --- | --- |
| input | `images`, `[1, 3, H, W]` float32 | the detector feeds one letterboxed RGB blob |
| H, W | equal (a static square) | `letterbox()` pads to a square; a non-square input is refused |
| output | `(1, 4 + num_classes, num_anchors)` | `cx, cy, w, h` down axis 0, then one score per class |
| metadata | `names` key present | the COCO label map; without it class ids cannot be mapped to names and the container refuses to start |
| size | 640 for the default | a static graph, so the input size is fixed at export time |

Anything Ultralytics exports satisfies this. If you re-export at another size,
rebuild - `VEHICLE_IMGSZ` cannot resize a graph that is already built. To
regenerate the graph from a `.pt` on a workstation:

```bash
pip install -r requirements-build.txt
python scripts/download_models.py --export-vehicle yolo26s --imgsz 512
```

### Cost of the graph size

The vehicle stage is the largest single line item in a frame, so the export size
matters more than any other knob. Measured on the bundled 518x330 demo clip, one
thread, 30 frames:

| Graph | mean | median | p95 | boxes/frame |
| --- | --- | --- | --- | --- |
| `imgsz=640` (default) | 297 ms | 296 ms | 333 ms | 3.0 |
| `imgsz=512` | 179 ms | 177 ms | 194 ms | 3.0 |
| `imgsz=416` | 119 ms | 114 ms | 151 ms | 3.0 |
| *(old)* `yolov8n.pt` | 102 ms | 89 ms | 165 ms | 4.8 |

640 is kept as the default because it is what the model was trained at and it is
what catches small, distant vehicles in real 1080p+ footage. If your streams are
already low resolution, or you would rather spend the CPU budget elsewhere,
supply a 512 or 416 graph - on this clip the detection count is identical. The old
`yolov8n` was ~3x faster but produced *more* boxes per frame (4.8), i.e. it was
double-detecting the same vehicles and sending redundant crops to the plate stage.

Plate detection is applied to **vehicle crops** by default. Running it on a
300x200 crop instead of a 1920x1080 frame is ~10x less pixel work, and small or
distant plates get upscaled into the detector's fixed 416px input. Switch to
`PLATE_SCOPE=frame` if vehicle detection is unreliable (e.g. heavy occlusion), or
`both` to catch plates the crops missed.

**Accuracy.** The pipeline is tuned to keep end-to-end plate accuracy above 90%
on clear, well-lit traffic footage (small vehicle crops, `0.30` detector
threshold, `0.60` OCR gate, `PLATE_PATTERN` as a cheap second filter). No number
can be guaranteed without your own footage: measure it locally with the script
below and tune `FRAME_STRIDE`, `VEHICLE_CONF_THRESHOLD` and `OCR_MIN_CONFIDENCE`.

```bash
# rough precision check against known plates in one clip
docker compose run --rm edge sh -c "\
  sed -i 's/^LOOP_VIDEO=.*/LOOP_VIDEO=false/;s/^MAX_FRAMES=.*/MAX_FRAMES=0/' .env 2>/dev/null; \
  python /app/main.py --dry-run" | grep -o '"plate_string":"[^"]*"' | sort | uniq -c
```

---

## 5. Project layout

```
.
├── main.py                    # entrypoint: config, logging, publisher lifecycle
├── Dockerfile                 # 2-stage build; stage 1 bakes in the weights
├── docker-compose.yml         # 3 cameras + Mosquitto + live monitor
├── deploy/mosquitto.conf      # broker config (anonymous, persistence, stdout log)
├── requirements.txt           # runtime: onnxruntime, headless OpenCV, paho-mqtt
├── requirements-build.txt     # optional, host-only: export a graph from a .pt
├── pyproject.toml             # packaging, pytest and ruff config
├── .env.example               # every supported variable, annotated
├── models/
│   └── yolo26s.onnx           # YOUR vehicle model - a build input, committed
├── assets/sample_car.jpg      # photo the demo video pans across
├── data/                      # generated demo footage (gitignored)
├── scripts/
│   ├── download_models.py     # verify the vehicle graph, bake the plate/OCR caches
│   └── make_test_video.py     # generate demo footage
├── src/anpr_edge/
│   ├── config.py              # env -> validated dataclasses
│   ├── logging_setup.py       # JSON logs, UTC timestamps, stdout isolation
│   ├── payload.py             # the JSON contract with the backend
│   ├── video_source.py        # sampling, looping, throttling, timestamps
│   ├── models/
│   │   ├── vehicle_detector.py  # ONNX Runtime session + letterbox/decode/NMS
│   │   └── plate_reader.py      # plate detection + OCR wrapper
│   ├── mqtt_publisher.py      # paho, backoff, bounded offline buffer
│   └── pipeline.py            # stages, gating, dedup, stats, shutdown
└── tests/                     # 130 unit tests, no model downloads
```

---

## 6. Manual setup (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# verifies models/yolo26s.onnx and caches the plate/OCR models (needs network
# once; after this the whole pipeline runs offline)
python scripts/download_models.py --models-dir models

export CAMERA_ID=cam01 LAT=12.9716 LON=77.5946
export VIDEO_PATH=data/cam01.mp4 MQTT_BROKER=localhost:1883
export VEHICLE_WEIGHTS=models/yolo26s.onnx
python main.py --dry-run
```

Run a local broker if you do not have one:

```bash
docker run --rm -it -p 1883:1883 eclipse-mosquitto:2 \
  mosquitto -c /mosquitto-no-auth.conf
```

### Tests

```bash
pip install pytest ruff
pytest            # 130 tests, ~2.5 s, no model downloads, no network
ruff check .
```

The tests inject fake models, so they never load YOLO or ONNX - that keeps the
suite fast and makes CI (and the offline promise) easy to honour.

---

## 7. Performance

Measured in the container on one laptop, single container, defaults
(`INFERENCE_THREADS=1`, 518x330 frames, the bundled sample photo which holds
3 vehicles per frame):

| Stage | Latency |
| --- | --- |
| Vehicle detection (YOLO26s ONNX, 640) | ~300 ms |
| Plate detection + OCR (per vehicle crop) | ~200-400 ms |
| Whole frame, 1 vehicle | ~0.5 s |
| Whole frame, 3 vehicles (`PLATE_SCOPE=crop`) | ~0.7 s (measured 678 ms) |

The vehicle stage is now the single biggest line item, so re-exporting the graph
at 512 or 416 is the first tuning knob - see the measured table in
[Model choices](#4-model-choices-and-why). Expect roughly **0.3-2 frames/s per
container** depending on how many vehicles are in frame: the per-vehicle crop is
what costs, which is the price of accuracy. `FRAME_STRIDE` is the other knob that
matters for throughput - `FRAME_STRIDE=3` roughly triples the frame rate at the
cost of temporal resolution. Set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`MKL_NUM_THREADS` to `1` (the image does) - leaving them unset makes ONNX Runtime
oversubscribe the CPU and can slow a frame down by an order of magnitude. Running N
cameras means N containers, one thread each.

A real `pipeline_finished` rollup from the container:

```json
{"event":"pipeline_finished","uptime_seconds":40.0,"frames_read":34,"frames_processed":12,
 "frames_failed":0,"vehicles_detected":36,"plates_read":12,"plates_rejected":0,
 "plates_deduped":11,"published":0,"avg_frame_latency_ms":677.7,"sampled_fps":0.3}
```

---

## 8. Operations

**Offline buffering.** While the broker is down, payloads queue in memory
(`MQTT_MAX_QUEUE`, oldest dropped, `mqtt_queue_overflow` logged) and flush on the
next connect. Nothing is written to disk, so a crash loses the buffer - a
deliberate trade: it keeps the container stateless and small.

**Shutdown.** `SIGTERM`/`SIGINT` finish the current frame, then disconnect
cleanly (`docker stop` sends `SIGTERM`; the default 10 s grace period is plenty).

**Health.** The worker exposes no HTTP endpoint; supervise it with
`restart: unless-stopped` and watch the `event":"stats"` lines. `docker compose
ps` plus `logs` is the intended health check.

**Disk.** `IMAGE_REF_MODE=path` writes one small JPEG per detection into
`PLATE_IMAGE_DIR`; it is not rotated, so point it at a volume you are happy to
clear, or keep the default `none`.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `config_invalid` on start | Read the JSON line: it names the variable and the offending value. `docker compose logs edge` |
| `video_open_failed` | Wrong path/mount, or an exotic codec. Check `-v "$PWD/data:/data"` and try re-encoding (`ffmpeg -i in.avi -c:v libx264 out.mp4`) |
| `no plates detected` / `frame_no_plates` | Normal for most frames. Loosen with `PLATE_SCOPE=both`, `PLATE_CONF_THRESHOLD=0.2`, `OCR_MIN_CONFIDENCE=0.4`, or `FRAME_STRIDE=1` |
| `plate_rejected reason=low_confidence` | Working as intended; lower `OCR_MIN_CONFIDENCE` if you need more recall |
| `plate_rejected reason=pattern_mismatch` | `PLATE_PATTERN` is filtering - check it matches your country's format |
| `mqtt_initial_connect_timeout` | Broker not up yet. Not fatal: the node buffers and connects when the broker appears |
| `mqtt_queue_overflow` | Broker down longer than the buffer allows; raise `MQTT_MAX_QUEUE` or fix the broker |
| `mqtt_connect_refused` | Wrong host/port/credentials, or the broker rejects anonymous clients |
| `FileNotFoundError` for weights | The graph is baked at build time; `python scripts/download_models.py --models-dir models` (or rebuild the image) |
| ONNX `InvalidArgument ... Expected: 640` | The baked graph is a fixed size. Re-export at the size you need (`--imgsz`) or set `VEHICLE_IMGSZ` to match the graph |
| Container exits immediately with code 2 | Config validation failed - read the first log line |
| One frame throws | Logged as `frame_failed` and skipped; the stream continues. Repeated failures mean bad footage or a missing codec |
| `WARNING 'half' is deprecated` | ONNX Runtime prints this from C++ on **stdout**, straight into the JSON stream, and it ignores both `sys.stdout` and the session log severity. Muted by giving the logger a private copy of fd 1 and pointing fd 1 at `/dev/null`; set `QUIET_ONNX=0` to keep native output on stdout (`container_start` reports `native_stdout_detached`) |

## 10. Notes and limits

* Simulated cameras: a **file**, not a live stream. For a real camera, replace
  `VideoSource` with a capture source (RTSP/GStreamer) and keep everything else.
* Ultralytics is not involved in the build or the image at all. If you do use
  the optional `--export-vehicle` path, its first-run telemetry is disabled
  there (`sync: false`, `YOLO_CONFIG_DIR` pointed at a temp dir).
* Plates are read, not stored. Treat `plate_string` as personal data: define
  retention and access control where the subscriber writes it to disk.
* The image has no `HEALTHCHECK` and no root privileges; it runs as uid 10001
  and needs write access only to `PLATE_IMAGE_DIR`.
* The final image is **836 MB**, down from 1.84 GB. There is no build toolchain
  and no heavyweight build dependency left: the vehicle model is an ONNX graph
  you commit, so the whole build is `pip install -r requirements.txt` plus baking
  in models. The image is onnxruntime (67 MB), headless OpenCV (153 MB) and
  numpy, on a slim Debian base. Two earlier savings are still baked in:
  `PIP_NO_CACHE_DIR=1` (a 103 MB pip cache used to land in `$HOME`), and no GUI
  OpenCV wheel (that came from Ultralytics, which is no longer involved). The
  remaining bulk is the Python base image plus OpenCV, so going smaller would mean
  an Alpine/musl rebuild of the whole wheel stack.

---

## Appendix: bundled demo asset

`assets/sample_car.jpg` is a single still used to synthesise the demo footage -
`scripts/make_test_video.py` pans and zooms across it so the vehicle detector has
something real to find (a synthetic rectangle is not detected by a COCO
detector). It was
taken from the `fast-alpr` project's public test images (MIT licensed,
<https://github.com/robinfrere/fast-alpr>); the visible plate belongs to nobody
and is not a real registration. Replace it with your own photo via
`--source-image` before showing the demo to anyone.
