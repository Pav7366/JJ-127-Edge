"""The vehicle model is a build input, so its absence must be actionable.

``scripts/download_models.py`` is what the Docker build runs; the tests here pin
the two contracts the build depends on: the repo ships an ONNX graph (never a
``.pt``), and a missing graph produces instructions rather than a stack trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

download_models = pytest.importorskip("scripts.download_models")


def test_the_committed_vehicle_model_is_an_onnx_graph():
    from anpr_edge.config import DEFAULT_VEHICLE_MODEL

    assert DEFAULT_VEHICLE_MODEL.endswith(".onnx")
    assert DEFAULT_VEHICLE_MODEL == "yolo26s.onnx"


def test_the_repo_ships_the_vehicle_model():
    # The Dockerfile does COPY models /models, so a missing file here is a build
    # failure. This is the test that tells you to restore the binary.
    model = ROOT / "models" / "yolo26s.onnx"
    assert model.is_file(), f"missing vehicle model: {model}"
    assert model.stat().st_size > 1_000_000, "vehicle model looks truncated"


def test_ignores_do_not_exclude_the_vehicle_model():
    # Both ignore files end with an explicit re-include, because the "last
    # matching rule wins" order is easy to break by accident. Note the two tools
    # differ: git's "*" crosses "/" (so "*.onnx" does match models/yolo26s.onnx
    # and really does need the later negation), while Docker's does not. Making
    # the negation last in both files means the test does not have to care.
    for name in (".gitignore", ".dockerignore"):
        rule = _matching_ignore_rule(ROOT / name, "models/yolo26s.onnx")
        assert rule == "!models/yolo26s.onnx", f"{name} would exclude the vehicle model ({rule})"


def _matching_ignore_rule(ignore_file: Path, relative: str) -> str | None:
    """Return the last pattern in ``ignore_file`` that matches ``relative``."""
    import fnmatch

    winner: str | None = None
    for raw in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern = line[1:] if line.startswith("!") else line
        if fnmatch.fnmatch(relative, pattern.rstrip("/")):
            winner = line
    return winner


def test_missing_model_explains_how_to_supply_one(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        download_models.require_vehicle_onnx("yolo26s", tmp_path)

    message = str(excinfo.value)
    assert "no vehicle model" in message
    assert str(tmp_path / "yolo26s.onnx") in message
    # It must point at both routes: supply a graph, or export one.
    assert "--export-vehicle" in message
    assert "requirements-build.txt" in message


def test_supplied_model_is_accepted(tmp_path: Path):
    model = tmp_path / "yolo26s.onnx"
    model.write_bytes(b"not a real graph, but it is there")

    assert download_models.require_vehicle_onnx("yolo26s", tmp_path) == model


def test_model_stem_is_configurable(tmp_path: Path):
    model = tmp_path / "my-detector.onnx"
    model.write_bytes(b"x")

    assert download_models.require_vehicle_onnx("my-detector", tmp_path) == model
    with pytest.raises(SystemExit):
        download_models.require_vehicle_onnx("yolo26s", tmp_path)
