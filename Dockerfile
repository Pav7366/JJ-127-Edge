# syntax=docker/dockerfile:1.7
###############################################################################
# ANPR edge container - one simulated traffic camera per container instance.
#
# Build:
#   docker build -t anpr-edge .
#
# Run:
#   docker run --rm \
#     -e CAMERA_ID=cam01 -e LAT=12.9716 -e LON=77.5946 \
#     -e VIDEO_PATH=/data/cam01.mp4 -e MQTT_BROKER=mosquitto:1883 \
#     -v "$PWD/data:/data:ro" anpr-edge
#
# There is nothing to compile and no heavyweight build toolchain: the vehicle
# model is an ONNX graph committed to the repo (models/yolo26s.onnx), so this is
# a single stage that installs the runtime requirements, bakes in the models and
# ships. The container has no network access once started.
###############################################################################

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="anpr-edge" \
      org.opencontainers.image.description="Edge node of a city-wide ANPR traffic monitoring demo" \
      org.opencontainers.image.licenses="MIT"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOME=/opt/anpr \
    VEHICLE_WEIGHTS=/models/yolo26s.onnx \
    PLATE_IMAGE_DIR=/data/plates \
    MPLBACKEND=Agg \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    FRAME_STRIDE=1 \
    LOOP_VIDEO=true

# Small models gain nothing from extra threads, and 1 thread per container keeps
# N simulated cameras on one laptop from fighting over the same cores.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    INFERENCE_THREADS=1

# onnxruntime + headless OpenCV + the plate/OCR stack. No torch, no ultralytics.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# The vehicle model is a build input, not a build artefact: it is committed at
# models/yolo26s.onnx and baked into /models unchanged. The COPY copies the
# directory rather than the file so that a missing model produces the actionable
# error from scripts/download_models.py instead of a bare "not found" from COPY.
# That script also verifies the graph's shape/metadata and warms the plate + OCR
# caches so the runtime never reaches the network.
COPY models /models
COPY scripts /app/scripts
RUN mkdir -p /opt/anpr/.cache \
    && python /app/scripts/download_models.py --models-dir /models \
    && rm -rf /opt/anpr/.cache/pip \
    && du -sh /models /opt/anpr/.cache

WORKDIR /app
COPY main.py /app/main.py
COPY src /app/src

# libgomp1 is required by onnxruntime; OpenCV's bundled FFmpeg handles mp4/avi
# decoding, so no system ffmpeg package is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data/plates \
    && useradd --create-home --uid 10001 anpr \
    && chown -R anpr:anpr /app /data /models /opt/anpr

USER anpr
VOLUME ["/data"]

ENTRYPOINT ["python", "/app/main.py"]
