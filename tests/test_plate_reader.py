"""Plate text normalisation and per-reading gates."""

from __future__ import annotations

import pytest

from anpr_edge.config import load_config
from anpr_edge.models.plate_reader import PlateReader, normalise_plate


def reader(env, **overrides) -> PlateReader:
    """Build a reader whose thresholds come from the environment."""
    env(**overrides)
    return PlateReader(load_config().models)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ka01ab1234", "KA01AB1234"),
        ("KA 01 AB 1234", "KA01AB1234"),
        ("KA-01-AB-1234", "KA01AB1234"),
        (" 5AU5341 ", "5AU5341"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalise_plate_strips_noise(raw, expected):
    assert normalise_plate(raw) == expected


def test_empty_text_is_rejected(env):
    assert reader(env).reject_reason("", 0.99) == "empty_text"


def test_short_text_is_rejected(env):
    assert reader(env, PLATE_MIN_LENGTH="6").reject_reason("AB12", 0.99) == "too_short"


def test_low_confidence_is_rejected(env):
    assert reader(env, OCR_MIN_CONFIDENCE="0.8").reject_reason("KA01AB1234", 0.5) == "low_confidence"


def test_confident_reading_is_accepted(env):
    assert reader(env, OCR_MIN_CONFIDENCE="0.8").reject_reason("KA01AB1234", 0.9) is None


def test_pattern_filter_rejects_foreign_plates(env):
    subject = reader(env, PLATE_PATTERN=r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$")
    assert subject.reject_reason("KA01AB1234", 0.99) is None
    assert subject.reject_reason("123-ABC", 0.99) == "pattern_mismatch"


def test_float_vehicle_boxes_are_coerced_to_int_pixels(env):
    """Detectors emit floats; crop slicing needs ints."""
    import numpy as np

    subject = reader(env)
    subject._alpr = _FakeALPR()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    # Must not raise TypeError even though the boxes are floats.
    assert subject.read(frame, [(1.7, 2.2, 40.9, 30.4, "car")]) == []
    # crop = frame[int(2.2):int(30.4), int(1.7):int(40.9)]
    assert subject._alpr.seen[0].shape == (28, 39, 3)


class _FakeALPR:
    """Stand-in for :class:`fast_alpr.ALPR` that records the crops it received."""

    def __init__(self) -> None:
        self.seen: list[object] = []

    def predict(self, image):
        self.seen.append(image)
        return []
