"""Pipeline orchestration with fake models: dedup, gating, stats."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pytest

from anpr_edge.config import load_config
from anpr_edge.models.plate_reader import PlateReading
from anpr_edge.models.vehicle_detector import VehicleBox
from anpr_edge.pipeline import Pipeline
from anpr_edge.video_source import FrameInfo


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, payload, diagnostics=None) -> bool:
        self.published.append((payload, diagnostics))
        return True

    def start(self) -> None: ...

    def close(self) -> None: ...

    def flush(self) -> int:
        return 0

    def stats(self) -> dict[str, Any]:
        return {"connected": True}


class FakeDetector:
    def __init__(self, boxes: list[VehicleBox] | None = None) -> None:
        self.boxes = boxes if boxes is not None else [VehicleBox(10, 10, 60, 40, 0.9, 2, "car")]
        self.calls = 0

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def detect(self, frame: np.ndarray) -> list[VehicleBox]:
        self.calls += 1
        return self.boxes


class FakeReader:
    def __init__(self, readings: list[PlateReading] | None = None) -> None:
        self.readings = readings if readings is not None else [
            PlateReading("KA01AB1234", 0.95, 0.9, (12, 12, 40, 26))
        ]
        self.calls = 0

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def read(self, frame, vehicles) -> list[PlateReading]:
        self.calls += 1
        return self.readings

    def reject_reason(self, text: str, confidence: float) -> str | None:
        if confidence < 0.5:
            return "low_confidence"
        return None


class FakeVideo:
    def __init__(self, frames: int = 3) -> None:
        self._frames = frames
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        self.opened += 1

    def close(self) -> None:
        self.closed += 1

    def frames(self):
        for index in range(self._frames):
            yield FrameInfo(
                image=np.zeros((48, 64, 3), dtype=np.uint8),
                index=index,
                sampled_index=index,
                video_time_seconds=float(index),
                wall_time=1_700_000_000.0 + index,
            )

    def timestamp_for(self, info: FrameInfo) -> str:
        return "2026-01-02T03:04:05.000Z"


def build_pipeline(env, sample_video, reader=None, publisher=None, **overrides) -> Pipeline:
    settings = {
        "REALTIME_FPS": "0",
        "LOOP_VIDEO": "false",
        "MAX_FRAMES": "0",
        # Deduplication is on in production; most tests want every frame.
        "PLATE_DEDUP_WINDOW_SECONDS": "0",
    }
    settings.update(overrides)
    env(VIDEO_PATH=str(sample_video), **settings)
    config = load_config()
    return Pipeline(
        config,
        publisher=publisher or FakePublisher(),
        video=FakeVideo(),
        detector=FakeDetector(),
        reader=reader or FakeReader(),
    )


def test_a_detection_reaches_the_publisher(env, sample_video):
    publisher = FakePublisher()
    pipeline = build_pipeline(env, sample_video, publisher=publisher)
    assert pipeline.run() == 0
    assert len(publisher.published) == 3
    payload, _ = publisher.published[0]
    assert payload.plate_string == "KA01AB1234"
    assert payload.camera_id == "cam-test"


def test_payload_carries_configured_camera_and_location(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(env, sample_video, publisher=publisher).run()
    payload = publisher.published[0][0]
    assert payload.lat == pytest.approx(12.9716)
    assert payload.lon == pytest.approx(77.5946)


def test_vehicle_class_reaches_the_payload_as_a_contract_type(env, sample_video):
    # The detector reports COCO names; the platform stores its own four-value enum.
    reader = FakeReader([
        PlateReading("KA01AB1234", 0.95, 0.9, (12, 12, 40, 26), vehicle_class="motorcycle")
    ])
    publisher = FakePublisher()
    build_pipeline(env, sample_video, reader=reader, publisher=publisher).run()
    assert publisher.published[0][0].to_dict()["vehicle_type"] == "bike"


def test_unclassified_vehicle_omits_the_type_entirely(env, sample_video):
    # No vehicle box to inherit a class from (PLATE_SCOPE=frame). An honest gap is
    # the contract's answer; a guessed class would be believed downstream.
    publisher = FakePublisher()
    build_pipeline(env, sample_video, publisher=publisher).run()
    assert "vehicle_type" not in publisher.published[0][0].to_dict()


def test_direction_and_lane_come_from_surveyed_config(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(
        env, sample_video, publisher=publisher, CAMERA_DIRECTION="E", CAMERA_LANE="2",
    ).run()
    event = publisher.published[0][0].to_dict()
    assert event["direction"] == "E"
    assert event["lane"] == 2


def test_direction_and_lane_are_absent_when_not_configured(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(env, sample_video, publisher=publisher).run()
    event = publisher.published[0][0].to_dict()
    assert "direction" not in event
    assert "lane" not in event


def test_published_event_carries_only_the_required_contract_keys(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(env, sample_video, publisher=publisher).run()
    assert set(publisher.published[0][0].to_dict()) == {
        "plate_string",
        "confidence",
        "camera_id",
        "lat",
        "lon",
        "timestamp",
    }


def test_deduplication_suppresses_repeats_of_the_same_plate(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(env, sample_video, publisher=publisher, PLATE_DEDUP_WINDOW_SECONDS="60").run()
    assert len(publisher.published) == 1


def test_dedup_can_be_disabled(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(env, sample_video, publisher=publisher, PLATE_DEDUP_WINDOW_SECONDS="0").run()
    assert len(publisher.published) == 3


def test_different_plates_are_not_deduplicated(env, sample_video):
    class AlternatingReader(FakeReader):
        def read(self, frame, vehicles):
            self.calls += 1
            text = "KA01AB1234" if self.calls % 2 else "KA02CD5678"
            return [PlateReading(text, 0.95, 0.9, (12, 12, 40, 26))]

    publisher = FakePublisher()
    build_pipeline(
        env, sample_video, reader=AlternatingReader(), publisher=publisher,
        PLATE_DEDUP_WINDOW_SECONDS="60",
    ).run()
    assert {payload.plate_string for payload, _ in publisher.published} == {
        "KA01AB1234",
        "KA02CD5678",
    }


def test_low_confidence_readings_are_not_published(env, sample_video):
    reader = FakeReader([PlateReading("KA01AB1234", 0.1, 0.9, (12, 12, 40, 26))])
    publisher = FakePublisher()
    pipeline = build_pipeline(env, sample_video, reader=reader, publisher=publisher)
    pipeline.run()
    assert publisher.published == []
    assert pipeline.stats.plates_rejected == 3


def test_vehicles_with_no_plates_publish_nothing(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(env, sample_video, reader=FakeReader([]), publisher=publisher).run()
    assert publisher.published == []
    assert publisher is not None


def test_stats_track_frames_and_detections(env, sample_video):
    pipeline = build_pipeline(env, sample_video, PLATE_DEDUP_WINDOW_SECONDS="0")
    pipeline.run()
    stats = pipeline.stats.as_dict()
    assert stats["frames_processed"] == 3
    assert stats["frames_failed"] == 0
    assert stats["vehicles_detected"] == 3
    assert stats["plates_read"] == 3
    assert stats["published"] == 3
    assert isinstance(stats["avg_frame_latency_ms"], float)


def test_latency_average_is_recorded(env, sample_video):
    from anpr_edge.pipeline import PipelineStats

    stats = PipelineStats(frames_processed=2)
    stats.record_latency(100.0)
    stats.record_latency(300.0)
    assert stats.as_dict()["avg_frame_latency_ms"] == 200.0


def test_one_failing_frame_does_not_stop_the_stream(env, sample_video):
    class ExplodingReader(FakeReader):
        def read(self, frame, vehicles):
            self.calls += 1
            raise RuntimeError("boom")

    publisher = FakePublisher()
    pipeline = build_pipeline(env, sample_video, reader=ExplodingReader(), publisher=publisher)
    pipeline.run()
    assert pipeline.stats.frames_failed == 3
    assert pipeline.stats.frames_processed == 3
    assert publisher.published == []


def test_image_reference_is_attached_when_configured(env, sample_video, tmp_path):
    target = tmp_path / "plates"
    publisher = FakePublisher()
    build_pipeline(
        env,
        sample_video,
        reader=FakeReader([_reading_with_image()]),
        publisher=publisher,
        IMAGE_REF_MODE="path",
        PLATE_IMAGE_DIR=str(target),
    ).run()
    payload = publisher.published[0][0]
    assert payload.image_ref is not None
    assert str(target) in payload.image_ref
    assert (target / payload.image_ref.split("/")[-1]).is_file()


def test_base64_image_reference_is_inline_jpeg(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(
        env,
        sample_video,
        reader=FakeReader([_reading_with_image()]),
        publisher=publisher,
        IMAGE_REF_MODE="base64",
    ).run()
    image_ref = publisher.published[0][0].image_ref
    assert image_ref is not None
    assert base64.b64decode(image_ref)[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_image_reference_is_none_by_default(env, sample_video):
    publisher = FakePublisher()
    build_pipeline(
        env, sample_video, reader=FakeReader([_reading_with_image()]), publisher=publisher
    ).run()
    assert publisher.published[0][0].image_ref is None


def _reading_with_image() -> PlateReading:
    return PlateReading(
        "KA01AB1234",
        0.95,
        0.9,
        (12, 12, 40, 26),
        plate_image=np.full((16, 40, 3), 200, dtype=np.uint8),
    )


def test_stop_request_ends_the_stream_early(env, sample_video):
    class StoppingReader(FakeReader):
        def read(self, frame, vehicles):
            self.calls += 1
            if self.calls == 1:
                self.pipeline.request_stop()
            return super().read(frame, vehicles)

    pipeline = build_pipeline(env, sample_video, reader=StoppingReader())
    pipeline.reader.pipeline = pipeline  # type: ignore[attr-defined]
    assert pipeline.run() == 0
    assert pipeline.stats.frames_processed == 1
    assert pipeline.stats.published == 1


def test_pipeline_opens_and_closes_the_video(env, sample_video):
    pipeline = build_pipeline(env, sample_video)
    pipeline.run()
    assert pipeline.video.opened == 1
    assert pipeline.video.closed == 1
