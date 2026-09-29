#!/usr/bin/env python3
"""Entrypoint for a single ANPR edge container (one simulated camera).

    python main.py --dry-run          # run the CV pipeline, print payloads
    python main.py --print-config     # dump the resolved configuration and exit
    python main.py --check-models     # download/verify weights, then exit
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anpr_edge.config import Config, ConfigError, load_config
from anpr_edge.logging_setup import configure_logging, get_logger, isolate_native_stdout, log_event
from anpr_edge.payload import DetectionPayload
from anpr_edge.pipeline import Pipeline, install_signal_handlers

_LOG = get_logger("anpr_edge.main")


class ConsolePublisher:
    """Drop-in replacement for :class:`MQTTPublisher` used by ``--dry-run``."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.messages: list[DetectionPayload] = []
        self.diagnostics: list[dict[str, Any]] = []

    def publish(self, payload: DetectionPayload, diagnostics: dict[str, Any] | None = None) -> bool:
        self.messages.append(payload)
        if diagnostics:
            self.diagnostics.append(diagnostics)
        if self.verbose:
            print(f"--> {json.dumps(payload.to_dict(), separators=(',', ':'))}", flush=True)
        return True

    def flush(self) -> int:
        return 0

    def stats(self) -> dict[str, Any]:
        return {"connected": False, "published": len(self.messages), "mode": "dry-run"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anpr-edge",
        description="Edge node of a city-wide ANPR network: reads a video file, "
        "detects vehicles and plates, and publishes JSON detections over MQTT.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not connect to MQTT; print payloads to stdout instead",
    )
    parser.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="load (downloading if needed) and warm up all models, then exit",
    )
    parser.add_argument("--log-level", help="override LOG_LEVEL (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument(
        "--log-format", choices=("json", "text"), help="override LOG_FORMAT (default: json)"
    )
    parser.add_argument(
        "--video", help="override VIDEO_PATH (handy for local testing without touching env)"
    )
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides (env stays the source of truth for everything else)."""
    if args.video:
        config = dataclasses.replace(config, frames=dataclasses.replace(config.frames, path=args.video))
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config()
    except ConfigError as exc:
        # Logging is not configured yet, so report on plain stderr.
        print(json.dumps({"level": "ERROR", "event": "config_invalid", "error": str(exc)}), file=sys.stderr)
        return 2

    if args.video:
        try:
            config = apply_overrides(config, args)
            config.validate()
        except ConfigError as exc:
            print(json.dumps({"level": "ERROR", "event": "config_invalid", "error": str(exc)}), file=sys.stderr)
            return 2

    # ONNX Runtime prints from C++ on stdout, straight into the JSON stream, so
    # the logger gets a private descriptor and fd 1 is detached (QUIET_ONNX=0 to
    # keep native output). Everything the app prints must then use ``log_stream``.
    log_stream = isolate_native_stdout() if config.runtime.quiet_onnx else None
    out = log_stream or sys.stdout

    configure_logging(
        level=args.log_level or config.runtime.log_level,
        fmt=args.log_format or config.runtime.log_format,
        stream=log_stream,
        camera_id=config.camera_id,
    )
    log_event(
        _LOG,
        20,
        "container_start",
        version=_version(),
        python=sys.version.split()[0],
        pid=os.getpid(),
        video_path=config.frames.path,
        broker=f"{config.mqtt.host}:{config.mqtt.port}",
        topic=config.mqtt.topic,
        dry_run=args.dry_run,
        native_stdout_detached=config.runtime.quiet_onnx,
    )

    if args.print_config:
        print(json.dumps(config.redacted(), indent=2, default=str), file=out)
        return 0

    if args.check_models:
        pipeline = Pipeline(config, publisher=ConsolePublisher(verbose=False))
        pipeline.start_models()
        log_event(_LOG, 20, "models_ready", **{"ok": True})
        return 0

    publisher = ConsolePublisher() if args.dry_run else None
    pipeline = Pipeline(config, publisher=publisher)
    install_signal_handlers(pipeline)

    exit_code = 0
    try:
        if publisher is None:
            pipeline.publisher.start()
        exit_code = pipeline.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        log_event(_LOG, 20, "interrupted")
    except ConfigError as exc:
        log_event(_LOG, 40, "config_invalid", error=str(exc))
        exit_code = 2
    except Exception as exc:  # pragma: no cover - last resort
        log_event(_LOG, 40, "fatal_error", error=str(exc), error_type=type(exc).__name__)
        exit_code = 1
    finally:
        if publisher is None:
            pipeline.publisher.close()  # type: ignore[union-attr]
        log_event(_LOG, 20, "container_exit", exit_code=exit_code)
    return exit_code


def _version() -> str:
    try:
        from anpr_edge import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
