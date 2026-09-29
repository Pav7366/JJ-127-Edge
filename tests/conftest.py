"""Shared pytest fixtures.

The unit tests never load a neural network: models are injected as fakes so the
suite runs in well under a second on any laptop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for entry in (str(SRC), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A tiny synthetic clip (solid frames) that OpenCV can decode."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    if not writer.isOpened():  # pragma: no cover - environment dependent
        pytest.skip("OpenCV cannot write mp4 in this environment")
    for index in range(10):
        frame = (index * 25) % 255
        writer.write(numpy.full((48, 64, 3), frame, dtype="uint8"))
    writer.release()
    return path


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, sample_video: Path):
    """Populate the environment for ``load_config`` (with a real video file)."""

    def _apply(**overrides: str) -> None:
        values = {
            "CAMERA_ID": "cam-test",
            "LAT": "12.9716",
            "LON": "77.5946",
            "VIDEO_PATH": str(sample_video),
            "MQTT_BROKER": "localhost:1883",
            "MQTT_USERNAME": "",
            "MQTT_PASSWORD": "",
        }
        values.update(overrides)
        for key, value in values.items():
            if value == "":
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

    return _apply
