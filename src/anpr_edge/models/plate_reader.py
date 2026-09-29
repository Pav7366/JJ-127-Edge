"""License plate detection + OCR stage (fast-alpr / fast-plate-ocr).

``PLATE_SCOPE`` controls where the plate detector looks:

``crop``  (default)
    Run the plate detector on each vehicle box. Small or distant plates get
    upscaled into the detector's fixed input, which is the best recall/speed
    trade-off for traffic camera footage.
``frame``
    Run the plate detector once on the whole frame. Cheapest option, but plates
    smaller than a few dozen pixels after the 384/416px downscale are missed.
``both``
    ``crop`` first, then ``frame`` for anything the vehicle boxes missed.
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import ModelConfig
from ..logging_setup import get_logger, log_event

__all__ = ["PlateReader", "PlateReading"]

_LOG = get_logger("anpr_edge.models.plate")


@dataclass(frozen=True, slots=True)
class PlateReading:
    """A plate read, with the plate box mapped back to full-frame coordinates."""

    text: str
    confidence: float
    detection_confidence: float
    box: tuple[int, int, int, int]
    char_confidences: tuple[float, ...] = ()
    region: str | None = None
    vehicle_class: str | None = None
    vehicle_index: int | None = None
    source: str = "crop"
    plate_image: np.ndarray | None = field(default=None, repr=False, compare=False)


class PlateReader:
    """Plate detection + OCR wrapper around :class:`fast_alpr.ALPR`."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._alpr = None
        self._pattern = re.compile(config.plate_pattern) if config.plate_pattern else None
        self._scope = config.plate_scope
        # Vehicle boxes frequently overlap; only run the plate detector once per
        # plate-sized region.
        self._iou_threshold = 0.45

    # ----------------------------------------------------------------- gating
    def reject_reason(self, text: str, confidence: float) -> str | None:
        """Return why a reading must not be published, or ``None`` to accept it.

        Keeping this next to the model config means the thresholds live in one
        place: ``OCR_MIN_CONFIDENCE``, ``PLATE_MIN_LENGTH`` and the optional
        ``PLATE_PATTERN`` regex all apply to every reading.
        """
        if not text:
            return "empty_text"
        if len(text) < self._config.plate_min_length:
            return "too_short"
        if confidence < self._config.ocr_min_confidence:
            return "low_confidence"
        if self._pattern is not None and not self._pattern.match(text):
            return "pattern_mismatch"
        return None

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        """Instantiate the ALPR pipeline (downloads weights on first use)."""
        from fast_alpr import ALPR

        started = time.perf_counter()
        self._alpr = ALPR(
            detector_model=self._config.plate_model,
            detector_conf_thresh=self._config.plate_conf_threshold,
            ocr_model=self._config.ocr_model,
            ocr_device=self._config.ocr_device,
        )
        log_event(
            _LOG,
            20,
            "plate_models_loaded",
            plate_model=self._config.plate_model,
            plate_conf_threshold=self._config.plate_conf_threshold,
            ocr_model=self._config.ocr_model,
            ocr_device=self._config.ocr_device,
            scope=self._scope,
            load_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def warmup(self, rounds: int = 2) -> None:
        """Force ONNX Runtime to initialise its arenas on a dummy crop."""
        if self._alpr is None:
            self.load()
        assert self._alpr is not None
        dummy = np.full((240, 640, 3), 128, dtype=np.uint8)
        for _ in range(max(0, rounds)):
            self._alpr.predict(dummy)
        log_event(_LOG, 20, "plate_models_warm", rounds=rounds)

    # --------------------------------------------------------------- inference
    def read(
        self,
        frame: np.ndarray,
        vehicles: list[tuple[int, int, int, int, str]] | None = None,
    ) -> list[PlateReading]:
        """Detect and read every plate in ``frame``.

        ``vehicles`` is a list of ``(x1, y1, x2, y2, class_name)`` tuples in
        full-frame coordinates; the returned readings use full-frame boxes too.
        """
        if self._alpr is None:
            self.load()
        assert self._alpr is not None

        # Detectors hand out floats; crop slicing needs ints, so normalise once.
        regions = self._unique_regions(_as_pixel_regions(vehicles))
        readings: list[PlateReading] = []
        covered: list[tuple[int, int, int, int]] = []

        if self._scope in {"crop", "both"} and regions:
            for index, (x1, y1, x2, y2, class_name) in enumerate(regions):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                found = self._run(
                    crop,
                    offset=(x1, y1),
                    source="crop",
                    vehicle_index=index,
                    vehicle_class=class_name,
                )
                readings.extend(found)
                covered.extend(reading.box for reading in found)

        if self._scope == "frame" or (self._scope == "both" and not readings):
            frame_reads = self._run(frame, offset=(0, 0), source="frame")
            if self._scope == "both":
                frame_reads = [r for r in frame_reads if not self._covered(r.box, covered)]
            readings.extend(frame_reads)

        return [reading for reading in readings if reading.text]

    def _unique_regions(
        self, vehicles: list[tuple[int, int, int, int, str]]
    ) -> list[tuple[int, int, int, int, str]]:
        """Drop vehicle boxes that mostly overlap an already selected one."""
        kept: list[tuple[int, int, int, int, str]] = []
        for box in vehicles:
            if any(self._iou(box[:4], other[:4]) >= self._iou_threshold for other in kept):
                continue
            kept.append(box)
        return kept

    def _run(
        self,
        image: np.ndarray,
        *,
        offset: tuple[int, int],
        source: str,
        vehicle_index: int | None = None,
        vehicle_class: str | None = None,
    ) -> list[PlateReading]:
        assert self._alpr is not None
        try:
            results = self._alpr.predict(image)
        except Exception as exc:  # a single bad crop must not kill the stream
            log_event(
                _LOG,
                40,
                "plate_stage_error",
                stage="detect+ocr",
                source=source,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return []

        height, width = image.shape[:2]
        readings: list[PlateReading] = []
        for item in results:
            detection = item.detection
            box = detection.bounding_box
            x1 = max(0, min(box.x1, width)) + offset[0]
            y1 = max(0, min(box.y1, height)) + offset[1]
            x2 = max(0, min(box.x2, width)) + offset[0]
            y2 = max(0, min(box.y2, height)) + offset[1]
            if x2 - x1 < self._config.plate_min_px or y2 - y1 < self._config.plate_min_px:
                continue

            ocr = item.ocr
            if ocr is None:
                continue
            text = normalise_plate(ocr.text)
            if not text:
                continue
            confidences = self._char_confidences(ocr.confidence, len(text))
            confidence = statistics.fmean(confidences) if confidences else 0.0

            frame_crop = self._crop_plate(image, (x1 - offset[0], y1 - offset[1], x2 - offset[0], y2 - offset[1]))
            readings.append(
                PlateReading(
                    text=text,
                    confidence=confidence,
                    detection_confidence=float(detection.confidence),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    char_confidences=confidences,
                    region=ocr.region,
                    source=source,
                    vehicle_index=vehicle_index,
                    vehicle_class=vehicle_class,
                    plate_image=frame_crop,
                )
            )
        return readings

    @staticmethod
    def _crop_plate(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _char_confidences(raw: float | list[float], text_length: int) -> tuple[float, ...]:
        if isinstance(raw, list):
            values = [float(value) for value in raw[:text_length] if value is not None]
        else:
            values = [float(raw)]
        return tuple(value for value in values if 0.0 <= value <= 1.0)

    @staticmethod
    def _iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if not inter:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union else 0.0

    def _covered(self, box: tuple[int, ...], others: list[tuple[int, ...]]) -> bool:
        return any(self._iou(box, other) >= self._iou_threshold for other in others)


def normalise_plate(text: str) -> str:
    """Uppercase and drop everything that cannot appear on a plate."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _as_pixel_regions(
    vehicles: list[tuple[int, int, int, int, str]] | None,
) -> list[tuple[int, int, int, int, str]]:
    """Coerce vehicle boxes to clipped integer pixel coordinates."""
    regions: list[tuple[int, int, int, int, str]] = []
    for vehicle in vehicles or []:
        x1, y1, x2, y2, class_name = vehicle
        regions.append((int(x1), int(y1), int(x2), int(y2), str(class_name)))
    return regions
