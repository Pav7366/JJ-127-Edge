"""Structured (JSON) logging to stdout with a per-camera context field."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, TextIO

__all__ = [
    "configure_logging",
    "get_logger",
    "isolate_native_stdout",
    "log_event",
]

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def __init__(self, static_fields: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = dict(self.static_fields)
        payload["ts"] = _utc_iso(record)
        payload["level"] = record.levelname
        payload["logger"] = record.name
        payload["event"] = record.getMessage()
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str, separators=(",", ":"))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return json.dumps({"level": record.levelname, "event": "log_serialisation_failed"})


def _utc_iso(record: logging.LogRecord) -> str:
    """UTC ISO-8601 timestamp with millisecond precision.

    ``Formatter.formatTime`` renders *local* time, which would make the trailing
    ``Z`` a lie whenever the container does not run in UTC.
    """
    seconds = time.gmtime(record.created)
    return (
        f"{seconds.tm_year:04d}-{seconds.tm_mon:02d}-{seconds.tm_mday:02d}"
        f"T{seconds.tm_hour:02d}:{seconds.tm_min:02d}:{seconds.tm_sec:02d}"
        f".{int(record.msecs):03d}Z"
    )


class TextFormatter(logging.Formatter):
    """Human friendly single line formatter for local debugging."""

    def __init__(self, static_fields: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        rendered = " ".join(f"{key}={json.dumps(value, default=str)}" for key, value in extras.items())
        prefix = " ".join(f"{key}={value}" for key, value in self.static_fields.items())
        head = " ".join(part for part in (prefix, record.levelname, record.name) if part)
        return f"{head} {record.getMessage()} {rendered}".rstrip()


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    *,
    stream: TextIO | None = None,
    **static_fields: Any,
) -> None:
    """Install a single stdout handler and quiet down chatty third-party loggers.

    ``stream`` defaults to stdout. Pass the stream returned by
    :func:`isolate_native_stdout` to keep logs flowing after fd 1 has been
    detached from the process.
    """
    static = {key: value for key, value in static_fields.items() if value is not None}
    formatter: logging.Formatter = (
        JsonFormatter(static) if fmt == "json" else TextFormatter(static)
    )

    handler = logging.StreamHandler(stream=stream or sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # These libraries are extremely chatty at INFO level and add nothing to an
    # operational log stream.
    for noisy in ("ultralytics", "onnxruntime", "onnxruntime.capi._pybind_state", "paho", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (kept as a function so callers never import logging)."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, event: str, /, **fields: Any) -> None:
    """Log a single structured event.

    ``event`` becomes the human readable message while every other keyword is
    emitted as a structured field, which keeps ``jq``-style filtering easy.
    """
    safe = {key: value for key, value in fields.items() if value is not None}
    logger.log(level, event, extra=safe, stacklevel=3)


def thread_aware_env() -> None:  # pragma: no cover - convenience for entrypoints
    """Default BLAS/OMP thread counts to 1 when running many containers."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def isolate_native_stdout() -> TextIO:
    """Detach fd 1 from the process and return a private stream for our logs.

    Some native libraries print to *stdout* from C++ - ONNX Runtime emits a
    ``'half' is deprecated`` notice for every session - and such writes ignore
    both Python's ``sys.stdout`` and the session log severity, so they land in
    the middle of the JSON log stream. The only reliable separation is to give
    the logger its own descriptor: fd 1 is duplicated, the copy becomes the log
    stream, and fd 1 itself is pointed at /dev/null so native chatter is dropped.

    stderr is deliberately left alone, so tracebacks and crash output still reach
    ``docker logs``. Set ``QUIET_ONNX=0`` to keep native output on stdout.
    """
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
    finally:
        os.close(devnull)
    return os.fdopen(saved, "w", buffering=1, encoding="utf-8", errors="replace")


