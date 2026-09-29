"""The per-camera processing loop: video -> vehicles -> plates -> MQTT.

Every stage is individually guarded so a bad frame, a corrupt crop or a dead
broker degrades the stream instead of terminating the container.
"""

from __future__ import annotations

import base64
import signal
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2

from .config import Config
from .logging_setup import get_logger, log_event
from .models.plate_reader import PlateReader, PlateReading, normalise_plate
from .models.vehicle_detector import VehicleDetector
from .mqtt_publisher import MQTTPublisher
from .payload import DetectionPayload, build_payload
from .video_source import FrameInfo, VideoSource

__all__ = ["Pipeline", "PipelineStats", "PublisherProtocol"]

_LOG = get_logger("anpr_edge.pipeline")


class PublisherProtocol(Protocol):
    """Minimal publisher surface used by the pipeline (easy to fake in tests)."""

    def publish(
        self, payload: DetectionPayload, diagnostics: dict[str, Any] | None = ...
    ) -> bool: ...

    def flush(self) -> int: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass
class PipelineStats:
    """Counters emitted with the periodic ``stats`` log line."""

    frames_read: int = 0
    frames_processed: int = 0
    frames_failed: int = 0
    vehicles_detected: int = 0
    plates_read: int = 0
    plates_rejected: int = 0
    plates_deduped: int = 0
    published: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _latency_total_ms: float = 0.0

    def record_latency(self, milliseconds: float) -> None:
        """Add one frame's end-to-end processing time to the rolling average."""
        self._latency_total_ms += milliseconds

    def as_dict(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        processed = max(self.frames_processed, 1)
        data = {
            "uptime_seconds": round(elapsed, 1),
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_failed": self.frames_failed,
            "vehicles_detected": self.vehicles_detected,
            "plates_read": self.plates_read,
            "plates_rejected": self.plates_rejected,
            "plates_deduped": self.plates_deduped,
            "published": self.published,
            "avg_frame_latency_ms": round(self._latency_total_ms / processed, 1),
            "sampled_fps": round(self.frames_processed / elapsed, 2),
        }
        if extra:
            data.update(extra)
        return data


class _DedupCache:
    """Suppresses repeated reads of the same plate within a time window."""

    def __init__(self, window_seconds: float, max_entries: int = 256) -> None:
        self._window = window_seconds
        self._max_entries = max_entries
        self._seen: OrderedDict[str, float] = OrderedDict()

    def seen_recently(self, key: str, now: float) -> bool:
        if self._window <= 0:
            return False
        previous = self._seen.get(key)
        if previous is not None and now - previous < self._window:
            self._seen.move_to_end(key)
            return True
        self._seen[key] = now
        self._seen.move_to_end(key)
        while len(self._seen) > self._max_entries:
            self._seen.popitem(last=False)
        return False


class Pipeline:
    """Runs the whole edge pipeline for a single simulated camera."""

    def __init__(
        self,
        config: Config,
        *,
        publisher: PublisherProtocol | None = None,
        video: VideoSource | None = None,
        detector: VehicleDetector | None = None,
        reader: PlateReader | None = None,
    ) -> None:
        self.config = config
        self.stats = PipelineStats()
        self.publisher = publisher if publisher is not None else MQTTPublisher(config.mqtt)
        self.video = video if video is not None else VideoSource(config.frames)
        self.detector = detector if detector is not None else VehicleDetector(config.models)
        self.reader = reader if reader is not None else PlateReader(config.models)
        self._dedup = _DedupCache(config.output.dedup_window_seconds)
        self._stop = threading.Event()

    # ------------------------------------------------------------- life cycle
    def request_stop(self, *_args: object) -> None:
        """Signal a graceful shutdown (wired to SIGTERM/SIGINT in ``main``)."""
        if not self._stop.is_set():
            log_event(_LOG, 20, "shutdown_requested")
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def start_models(self) -> None:
        self.detector.load()
        self.reader.load()
        self.detector.warmup()
        self.reader.warmup()

    def run(self) -> int:
        """Process frames until the video ends or a stop is requested."""
        self.start_models()
        log_event(
            _LOG,
            20,
            "pipeline_started",
            video_path=self.config.frames.path,
            stride=self.config.frames.stride,
            loop=self.config.frames.loop,
            plate_scope=self.config.models.plate_scope,
            plate_detector=self.config.models.plate_model,
            ocr_model=self.config.models.ocr_model,
            ocr_min_confidence=self.config.models.ocr_min_confidence,
            dedup_window_seconds=self.config.output.dedup_window_seconds,
            topic=self.config.mqtt.topic,
        )

        try:
            self.video.open()
        except (OSError, RuntimeError) as exc:
            log_event(
                _LOG,
                40,
                "video_open_failed",
                video_path=self.config.frames.path,
                error=str(exc),
                hint="check VIDEO_PATH, the read-only bind mount and the file's codec "
                "(OpenCV needs a decodable mp4/avi; install ffmpeg if the clip is exotic)",
            )
            raise

        last_stats = time.monotonic()
        try:
            for info in self.video.frames():
                if self._stop.is_set():
                    log_event(_LOG, 20, "stopping_mid_stream", frames_processed=self.stats.frames_processed)
                    break
                self.stats.frames_read = info.index + 1
                try:
                    self._process_frame(info)
                except Exception as exc:  # never let one frame kill the container
                    self.stats.frames_failed += 1
                    log_event(
                        _LOG,
                        40,
                        "frame_failed",
                        frame_index=info.index,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                self.stats.frames_processed += 1

                now = time.monotonic()
                if self.config.runtime.stats_interval_seconds and (
                    now - last_stats >= self.config.runtime.stats_interval_seconds
                ):
                    last_stats = now
                    self._log_stats()
        finally:
            self.video.close()

        self._log_stats(final=True)
        return 0

    # ------------------------------------------------------------------ stages
    def _process_frame(self, info: FrameInfo) -> None:
        started = time.perf_counter()
        frame = info.image

        vehicles = self.detector.detect(frame)
        self.stats.vehicles_detected += len(vehicles)
        vehicle_boxes = [
            (box.x1, box.y1, box.x2, box.y2, box.class_name)
            for box in vehicles
        ]
        plate_started = time.perf_counter()
        readings = self.reader.read(frame, vehicle_boxes)
        plate_ms = (time.perf_counter() - plate_started) * 1000
        self.stats.plates_read += len(readings)

        if not readings:
            self.stats.record_latency((time.perf_counter() - started) * 1000)
            log_event(
                _LOG,
                20,
                "frame_no_plates",
                frame_index=info.index,
                vehicles=len(vehicles),
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return

        timestamp = self.video.timestamp_for(info)
        for reading in readings:
            self._handle_reading(reading, info, timestamp)

        self.stats.record_latency((time.perf_counter() - started) * 1000)
        log_event(
            _LOG,
            20,
            "frame_processed",
            frame_index=info.index,
            vehicles=len(vehicles),
            plates=len(readings),
            plate_stage_ms=round(plate_ms, 1),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _handle_reading(self, reading: PlateReading, info: FrameInfo, timestamp: str) -> None:
        text = normalise_plate(reading.text)
        base_fields = {
            "vehicle_class": reading.vehicle_class or "unknown",
            "detection_confidence": round(reading.detection_confidence, 3),
            "source": reading.source,
            "plate_box": list(reading.box),
        }

        reject = self.reader.reject_reason(text, reading.confidence)
        if reject:
            self.stats.plates_rejected += 1
            log_event(
                _LOG,
                20,
                "plate_rejected",
                frame_index=info.index,
                plate=text,
                confidence=round(reading.confidence, 4),
                reason=reject,
                min_confidence=self.config.models.ocr_min_confidence,
                **base_fields,
            )
            return

        if self._dedup.seen_recently(text, time.monotonic()):
            self.stats.plates_deduped += 1
            log_event(
                _LOG,
                20,
                "plate_suppressed_duplicate",
                frame_index=info.index,
                plate=text,
                confidence=round(reading.confidence, 4),
                window_seconds=self.config.output.dedup_window_seconds,
                **base_fields,
            )
            return

        image_ref = self._store_plate_image(reading, info)
        payload = build_payload(
            camera_id=self.config.camera_id,
            plate_string=text,
            confidence=reading.confidence,
            lat=self.config.latitude,
            lon=self.config.longitude,
            timestamp=timestamp,
            image_ref=image_ref,
            # Surveyed per camera, not per frame - the pipeline cannot know which
            # way traffic flows past the lens from a single still image.
            direction=self.config.direction,
            lane=self.config.lane,
            # The raw COCO label; build_payload maps it onto the contract's enum.
            # Absent under PLATE_SCOPE=frame, which has no vehicle box to inherit it
            # from, and an absent class is the honest answer the contract asks for.
            vehicle_class=reading.vehicle_class,
        )
        diagnostics = {
            **base_fields,
            "region": reading.region,
            "min_char_confidence": round(min(reading.char_confidences), 4) if reading.char_confidences else None,
        }
        sent = self.publisher.publish(payload, diagnostics)
        if sent:
            self.stats.published += 1

        log_event(
            _LOG,
            20,
            "detection",
            frame_index=info.index,
            plate=text,
            confidence=round(reading.confidence, 4),
            region=reading.region,
            video_time_seconds=round(info.video_time_seconds, 3),
            timestamp=timestamp,
            mqtt_delivery="sent" if sent else "buffered",
            image_ref=image_ref,
            **base_fields,
        )

    # ------------------------------------------------------------------ output
    def _store_plate_image(self, reading: PlateReading, info: FrameInfo) -> str | None:
        mode = self.config.output.image_ref_mode
        if mode == "none" or reading.plate_image is None:
            return None

        image = reading.plate_image
        max_width = self.config.output.image_max_width
        if max_width and image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(
                image,
                (max_width, max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        if mode == "base64":
            params = (
                [int(cv2.IMWRITE_JPEG_QUALITY), self.config.output.jpeg_quality]
                if self.config.output.image_format == "jpg"
                else []
            )
            ok, buffer = cv2.imencode(f".{self.config.output.image_format}", image, params)
            if not ok:
                log_event(_LOG, 30, "plate_image_encode_failed", frame_index=info.index)
                return None
            return base64.b64encode(buffer.tobytes()).decode("ascii")

        directory = Path(self.config.output.plate_image_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
            name = f"{self.config.camera_id}_{info.index:09d}_{stamp}_{reading.text}.{self.config.output.image_format}"
            path = directory / name
            params = (
                [int(cv2.IMWRITE_JPEG_QUALITY), self.config.output.jpeg_quality]
                if self.config.output.image_format == "jpg"
                else []
            )
            if not cv2.imwrite(str(path), image, params):
                log_event(_LOG, 30, "plate_image_write_failed", path=str(path))
                return None
            return str(path)
        except OSError as exc:
            log_event(_LOG, 30, "plate_image_write_error", error=str(exc), path=str(directory))
            return None

    def _log_stats(self, final: bool = False) -> None:
        publisher_stats = {}
        try:
            publisher_stats = self.publisher.stats()
        except Exception as exc:  # pragma: no cover - defensive
            publisher_stats = {"error": str(exc)}
        log_event(
            _LOG,
            20,
            "pipeline_stats" if not final else "pipeline_finished",
            **self.stats.as_dict(publisher_stats),
        )


def install_signal_handlers(pipeline: Pipeline) -> None:
    """Stop the loop cleanly on SIGTERM/SIGINT (``docker stop`` sends SIGTERM)."""

    def _handler(signum: int, _frame: object) -> None:
        log_event(_LOG, 20, "signal_received", signal=signal.Signals(signum).name)
        pipeline.request_stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            log_event(_LOG, 30, "signal_handler_unavailable", signal=sig.name)
