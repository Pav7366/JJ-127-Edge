"""Tests for the structured logging helpers."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import pytest

from anpr_edge.logging_setup import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    get_logger,
    isolate_native_stdout,
    log_event,
)


def test_json_formatter_is_one_compact_object_per_record() -> None:
    formatter = JsonFormatter({"camera_id": "cam-01"})
    record = logging.LogRecord("anpr_edge.demo", logging.INFO, __file__, 1, "started", (), None)

    line = formatter.format(record)

    assert "\n" not in line
    assert line.startswith("{") and line.endswith("}")
    assert '"camera_id":"cam-01"' in line
    assert '"event":"started"' in line
    assert '"level":"INFO"' in line
    assert '"logger":"anpr_edge.demo"' in line


def test_json_formatter_keeps_structured_extras() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("anpr_edge.demo", logging.INFO, __file__, 1, "e", (), None)
    record.frame_index = 7  # type: ignore[attr-defined]

    assert '"frame_index":7' in formatter.format(record)


def test_json_formatter_renders_unserialisable_extras() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("anpr_edge.demo", logging.INFO, __file__, 1, "e", (), None)
    record.path = object()  # type: ignore[attr-defined]

    assert '"path":"' in formatter.format(record)


def test_json_timestamps_are_utc_with_millis() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("anpr_edge.demo", logging.INFO, __file__, 1, "e", (), None)

    match = re.search(
        r'"ts":"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d{3}Z"',
        formatter.format(record),
    )

    assert match, formatter.format(record)


def test_log_event_drops_none_fields(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        log_event(get_logger("anpr_edge.demo"), logging.INFO, "thing", kept=1, dropped=None)

    record = caplog.records[-1]
    assert record.kept == 1  # type: ignore[attr-defined]
    assert not hasattr(record, "dropped")


def test_configure_logging_writes_json_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json", camera_id="cam-07")

    log_event(get_logger("anpr_edge.demo"), logging.INFO, "started")

    out = capsys.readouterr().out
    assert '"event":"started"' in out
    assert '"camera_id":"cam-07"' in out


def test_configure_logging_text_format_is_plain(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="text", camera_id="cam-07")

    log_event(get_logger("anpr_edge.demo"), logging.INFO, "started")

    out = capsys.readouterr().out
    assert "started" in out
    assert "camera_id=cam-07" in out
    assert not out.lstrip().startswith("{")


def test_configure_logging_replaces_existing_handlers() -> None:
    root = logging.getLogger()

    configure_logging(level="INFO", fmt="json")
    configure_logging(level="INFO", fmt="json")

    assert len(root.handlers) == 1


def test_text_formatter_renders_extras_as_json() -> None:
    formatter = TextFormatter({"camera_id": "cam-01"})
    record = logging.LogRecord("anpr_edge.demo", logging.WARNING, __file__, 1, "slow", (), None)
    record.latency_ms = 12.5  # type: ignore[attr-defined]

    line = formatter.format(record)

    assert line.startswith("camera_id=cam-01 WARNING anpr_edge.demo slow")
    assert "latency_ms=12.5" in line


def test_isolate_native_stdout_splits_the_streams(capfd: pytest.CaptureFixture[str]) -> None:
    stream = isolate_native_stdout()
    try:
        os.write(1, b"native chatter\n")  # what a C++ library would do
        stream.write("our log line\n")
        stream.flush()
    finally:
        os.dup2(stream.fileno(), 1)
        stream.close()

    captured = capfd.readouterr()
    assert "native chatter" not in captured.out
    assert "our log line" in captured.out


def test_logging_can_target_the_isolated_stream(capfd: pytest.CaptureFixture[str]) -> None:
    stream = isolate_native_stdout()
    try:
        configure_logging(level="INFO", fmt="json", stream=stream, camera_id="cam-09")
        log_event(get_logger("anpr_edge.demo"), logging.INFO, "kept")
        os.write(1, b"dropped\n")
        stream.flush()
    finally:
        logging.getLogger().handlers.clear()
        os.dup2(stream.fileno(), 1)
        stream.close()

    captured = capfd.readouterr()
    assert '"event":"kept"' in captured.out
    assert '"camera_id":"cam-09"' in captured.out
    assert "dropped" not in captured.out


def test_missing_onnxruntime_does_not_break_logging_setup() -> None:
    """The log helpers must not depend on onnxruntime being importable."""
    import anpr_edge.logging_setup as module

    assert not hasattr(module, "quiet_onnx_runtime")
    assert callable(isolate_native_stdout)
    assert isinstance(configure_logging.__defaults__[0], str)


def test_third_party_loggers_are_quietened() -> None:
    configure_logging(level="DEBUG", fmt="json")

    for name in ("ultralytics", "onnxruntime", "paho"):
        assert logging.getLogger(name).level == logging.WARNING


def test_thread_aware_env_sets_single_thread_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from anpr_edge.logging_setup import thread_aware_env

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)

    thread_aware_env()

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_annotations_are_exported() -> None:
    from anpr_edge import logging_setup

    assert set(logging_setup.__all__) == {
        "configure_logging",
        "get_logger",
        "isolate_native_stdout",
        "log_event",
    }
    unused: Any = None
    assert unused is None
