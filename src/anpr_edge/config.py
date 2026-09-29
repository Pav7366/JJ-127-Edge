"""Environment driven configuration for the ANPR edge container.

Every per-camera knob is read from the process environment so that a single
image can be started N times with different ``docker run -e`` flags or
docker-compose service definitions.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_VEHICLE_MODEL",
    "Config",
    "ConfigError",
    "FrameConfig",
    "ModelConfig",
    "MqttConfig",
    "OutputConfig",
    "RuntimeConfig",
    "load_config",
]

# The topic the platform's ingest and debug subscriber are subscribed to. Owned by
# the backend's contract/event_contract.json, not by this repo; the two must agree
# or every sighting is published to a topic nobody reads.
DEFAULT_TOPIC_TEMPLATE = "anpr/{camera_id}/sightings"
DEFAULT_MODELS_DIR = "/models"
# Vehicle stage: an Ultralytics YOLO26 checkpoint exported to ONNX at build time
# and executed with onnxruntime, so the runtime image needs no torch.
DEFAULT_VEHICLE_MODEL = "yolo26s.onnx"
# Placeholder written over the password by ``Config.redacted``.
REDACTED = "***"
_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}
_PLATE_SCOPES = {"crop", "frame", "both"}
# event.json_schema -> properties.direction.enum in the platform contract.
_DIRECTIONS = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}
_TIMESTAMP_SOURCES = {"wall", "video"}
_IMAGE_REF_MODES = {"none", "path", "base64"}
_MQTT_PROTOCOLS = {"v5", "v311"}


class ConfigError(RuntimeError):
    """Raised when the environment contains missing or invalid configuration."""


def _raw(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _str(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = _raw(name, default)
    if value is None:
        if required:
            raise ConfigError(f"Required environment variable {name} is not set")
        raise ConfigError(f"Environment variable {name} resolved to an empty value")
    return value


def _int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = _raw(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = _raw(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value != value:  # NaN
        raise ConfigError(f"{name} must be a number, got {raw!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")


def _csv(name: str, default: str) -> tuple[str, ...]:
    raw = _raw(name, default) or ""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _choice(name: str, default: str, allowed: set[str]) -> str:
    raw = (_raw(name, default) or default).lower()
    if raw not in allowed:
        raise ConfigError(f"{name} must be one of {sorted(allowed)}, got {raw!r}")
    return raw


def _optional_int(name: str, *, minimum: int, maximum: int) -> int | None:
    """Read an optional int, treating unset and empty alike as 'not supplied'."""
    raw = _raw(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be in {minimum}..{maximum}, got {value}")
    return value


def _optional_direction(name: str) -> str | None:
    """Read an optional compass direction, upper-cased and enum-checked."""
    raw = _raw(name)
    if raw is None:
        return None
    value = raw.upper()
    if value not in _DIRECTIONS:
        raise ConfigError(f"{name} must be one of {sorted(_DIRECTIONS)}, got {raw!r}")
    return value


def _parse_broker() -> tuple[str, int, bool]:
    """Resolve broker host/port/TLS from MQTT_BROKER(_HOST/_PORT)/MQTT_TLS.

    ``MQTT_BROKER`` accepts ``host``, ``host:port``, ``mqtt://host:port`` and
    ``mqtts://host:port``. Explicit ``MQTT_BROKER_HOST``/``MQTT_BROKER_PORT``
    take precedence over the host part of the URL.
    """
    default_host = _str("MQTT_BROKER_HOST", "localhost")
    default_port = _int("MQTT_BROKER_PORT", 1883, minimum=1, maximum=65535)
    url = _raw("MQTT_BROKER")

    host, port, tls = default_host, default_port, _bool("MQTT_TLS", False)
    if url:
        candidate = url
        if "://" in candidate:
            scheme, _, candidate = candidate.partition("://")
            tls = tls or scheme.lower() in {"mqtts", "ssl", "tls"}
        if candidate.startswith("["):  # IPv6 literal, e.g. [::1]:1883
            host_part, _, port_part = candidate.partition("]")
            host = host_part.lstrip("[")
            port = int(port_part.lstrip(":")) if port_part.lstrip(":") else port
        elif ":" in candidate:
            host_part, _, port_part = candidate.rpartition(":")
            host = host_part or host
            if port_part:
                try:
                    port = int(port_part)
                except ValueError as exc:
                    raise ConfigError(f"MQTT_BROKER has an invalid port: {url!r}") from exc
        else:
            host = candidate
    if not host:
        raise ConfigError("MQTT broker host resolved to an empty value")
    if not 1 <= port <= 65535:
        raise ConfigError(f"MQTT broker port must be in 1..65535, got {port}")
    return host, port, tls


@dataclass(frozen=True)
class MqttConfig:
    """Connection and publishing behaviour for the central broker."""

    host: str
    port: int
    tls: bool
    ca_certs: str | None
    topic: str
    qos: int
    retain: bool
    keepalive: int
    protocol: str
    client_id: str
    username: str | None
    password: str | None
    connect_wait_seconds: float
    reconnect_min_delay: int
    reconnect_max_delay: int
    max_queue: int
    publish_diagnostics: bool

    @classmethod
    def from_env(cls, camera_id: str) -> MqttConfig:
        host, port, tls = _parse_broker()
        template = _str("MQTT_TOPIC_TEMPLATE", DEFAULT_TOPIC_TEMPLATE)
        if "{camera_id}" not in template:
            raise ConfigError("MQTT_TOPIC_TEMPLATE must contain the '{camera_id}' placeholder")
        return cls(
            host=host,
            port=port,
            tls=tls,
            ca_certs=_raw("MQTT_CA_CERTS"),
            topic=template.format(camera_id=camera_id),
            qos=_int("MQTT_QOS", 1, minimum=0, maximum=2),
            retain=_bool("MQTT_RETAIN", False),
            keepalive=_int("MQTT_KEEPALIVE", 60, minimum=5, maximum=3600),
            protocol=_choice("MQTT_PROTOCOL", "v5", _MQTT_PROTOCOLS),
            client_id=_str("MQTT_CLIENT_ID", f"anpr-edge-{camera_id}"),
            username=_raw("MQTT_USERNAME"),
            password=_raw("MQTT_PASSWORD"),
            connect_wait_seconds=_float("MQTT_CONNECT_WAIT_SECONDS", 30.0, minimum=0.0),
            reconnect_min_delay=_int("MQTT_RECONNECT_MIN_DELAY", 1, minimum=1, maximum=3600),
            reconnect_max_delay=_int("MQTT_RECONNECT_MAX_DELAY", 60, minimum=1, maximum=3600),
            max_queue=_int("MQTT_MAX_QUEUE", 1000, minimum=1, maximum=1_000_000),
            publish_diagnostics=_bool("MQTT_PUBLISH_DIAGNOSTICS", False),
        )

    @property
    def broker(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and per-stage thresholds."""

    vehicle_weights: str
    vehicle_classes: tuple[str, ...]
    vehicle_conf_threshold: float
    vehicle_iou: float
    vehicle_imgsz: int
    vehicle_max_det: int
    vehicle_min_box_px: int
    plate_model: str
    plate_conf_threshold: float
    plate_scope: str
    plate_min_px: int
    ocr_model: str
    ocr_device: str
    ocr_min_confidence: float
    plate_min_length: int
    plate_pattern: str | None
    inference_threads: int
    vehicle_threads: int

    @classmethod
    def from_env(cls) -> ModelConfig:
        # The vehicle stage runs from an ONNX graph baked into the image, so the
        # default is the exported model rather than a .pt checkpoint.
        weights = _str("VEHICLE_WEIGHTS", "")
        if not weights:
            default_path = Path(DEFAULT_MODELS_DIR) / DEFAULT_VEHICLE_MODEL
            weights = str(default_path) if default_path.is_file() else DEFAULT_VEHICLE_MODEL
        device = (_raw("OCR_DEVICE", "cpu") or "cpu").lower()
        if device not in {"cpu", "auto", "cuda"}:
            raise ConfigError(f"OCR_DEVICE must be one of ['auto', 'cpu', 'cuda'], got {device!r}")
        pattern = _raw("PLATE_PATTERN")
        if pattern:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ConfigError(f"PLATE_PATTERN is not a valid regex: {exc}") from exc
        return cls(
            vehicle_weights=weights,
            vehicle_classes=_csv("VEHICLE_CLASSES", "car,motorcycle,bus,truck"),
            vehicle_conf_threshold=_float("VEHICLE_CONF_THRESHOLD", 0.30, minimum=0.0, maximum=1.0),
            vehicle_iou=_float("VEHICLE_IOU", 0.50, minimum=0.0, maximum=1.0),
            vehicle_imgsz=_int("VEHICLE_IMGSZ", 640, minimum=64, maximum=4096),
            vehicle_max_det=_int("VEHICLE_MAX_DET", 20, minimum=1, maximum=1000),
            vehicle_min_box_px=_int("VEHICLE_MIN_BOX_PX", 32, minimum=1, maximum=4096),
            plate_model=_str("PLATE_MODEL", "yolo-v9-t-416-license-plate-end2end"),
            plate_conf_threshold=_float("PLATE_CONF_THRESHOLD", 0.30, minimum=0.0, maximum=1.0),
            plate_scope=_choice("PLATE_SCOPE", "crop", _PLATE_SCOPES),
            plate_min_px=_int("PLATE_MIN_PX", 12, minimum=1, maximum=4096),
            ocr_model=_str("OCR_MODEL", "cct-s-v2-global-model"),
            ocr_device=device,
            ocr_min_confidence=_float("OCR_MIN_CONFIDENCE", 0.50, minimum=0.0, maximum=1.0),
            plate_min_length=_int("PLATE_MIN_LENGTH", 2, minimum=1, maximum=16),
            plate_pattern=pattern,
            inference_threads=_int("INFERENCE_THREADS", 1, minimum=1, maximum=64),
            vehicle_threads=_int("VEHICLE_THREADS", 0, minimum=0, maximum=64),
        )


@dataclass(frozen=True)
class FrameConfig:
    """Video input sampling behaviour."""

    path: str
    stride: int
    loop: bool
    realtime_fps: float
    max_frames: int
    max_width: int
    timestamp_source: str
    video_start_time: str

    @classmethod
    def from_env(cls) -> FrameConfig:
        source = _choice("TIMESTAMP_SOURCE", "wall", _TIMESTAMP_SOURCES)
        return cls(
            path=_str("VIDEO_PATH", required=True),
            stride=_int("FRAME_STRIDE", 1, minimum=1, maximum=10000),
            loop=_bool("LOOP_VIDEO", True),
            realtime_fps=_float("REALTIME_FPS", 0.0, minimum=0.0, maximum=1000.0),
            max_frames=_int("MAX_FRAMES", 0, minimum=0, maximum=10_000_000),
            max_width=_int("FRAME_MAX_WIDTH", 0, minimum=0, maximum=16384),
            timestamp_source=source,
            video_start_time=_raw("VIDEO_START_TIME", "") or "",
        )


@dataclass(frozen=True)
class OutputConfig:
    """Plate image persistence and MQTT payload enrichment."""

    image_ref_mode: str
    plate_image_dir: str
    image_format: str
    image_max_width: int
    jpeg_quality: int
    dedup_window_seconds: float

    @classmethod
    def from_env(cls) -> OutputConfig:
        image_format = (_raw("PLATE_IMAGE_FORMAT", "jpg") or "jpg").lower()
        if image_format not in {"jpg", "png"}:
            raise ConfigError(f"PLATE_IMAGE_FORMAT must be 'jpg' or 'png', got {image_format!r}")
        mode = _choice("IMAGE_REF_MODE", "none", _IMAGE_REF_MODES)
        return cls(
            image_ref_mode=mode,
            plate_image_dir=_str("PLATE_IMAGE_DIR", "/data/plates"),
            image_format=image_format,
            image_max_width=_int("PLATE_IMAGE_MAX_WIDTH", 320, minimum=0, maximum=16384),
            jpeg_quality=_int("PLATE_IMAGE_JPEG_QUALITY", 85, minimum=1, maximum=100),
            dedup_window_seconds=_float("PLATE_DEDUP_WINDOW_SECONDS", 10.0, minimum=0.0, maximum=86400.0),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Process level knobs (logging, stats, shutdown)."""

    log_level: str
    log_format: str
    stats_interval_seconds: float
    quiet_onnx: bool

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        log_level = (_raw("LOG_LEVEL", "INFO") or "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"LOG_LEVEL must be a standard level name, got {log_level!r}")
        return cls(
            log_level=log_level,
            log_format=_choice("LOG_FORMAT", "json", {"json", "text"}),
            stats_interval_seconds=_float("STATS_INTERVAL_SECONDS", 30.0, minimum=0.0, maximum=86400.0),
            quiet_onnx=_bool("QUIET_ONNX", True),
        )


@dataclass(frozen=True)
class Config:
    """Fully resolved configuration for one simulated camera node."""

    camera_id: str
    latitude: float
    longitude: float
    direction: str | None
    lane: int | None
    frames: FrameConfig
    models: ModelConfig
    mqtt: MqttConfig
    output: OutputConfig
    runtime: RuntimeConfig
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Config:
        camera_id = _str("CAMERA_ID", required=True)
        # No ':' or '/': the id is interpolated into an MQTT topic and matched
        # against a per-camera ACL entry, and either character would add a topic
        # level (or silently miss the grant) instead of naming a camera.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", camera_id):
            raise ConfigError(
                "CAMERA_ID may only contain letters, digits, '_', '.' and '-' (1-64 chars); "
                "':' and '/' are rejected because they would break the MQTT topic"
            )
        config = cls(
            camera_id=camera_id,
            latitude=_float("LAT", 0.0, minimum=-90.0, maximum=90.0),
            longitude=_float("LON", 0.0, minimum=-180.0, maximum=180.0),
            # A camera bolted to a pole knows its own direction of travel and lane
            # numbering; neither is inferable from a single frame, so both are
            # surveyed configuration and are left absent when not supplied.
            direction=_optional_direction("CAMERA_DIRECTION"),
            lane=_optional_int("CAMERA_LANE", minimum=0, maximum=99),
            frames=FrameConfig.from_env(),
            models=ModelConfig.from_env(),
            mqtt=MqttConfig.from_env(camera_id),
            output=OutputConfig.from_env(),
            runtime=RuntimeConfig.from_env(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail fast on values that would only blow up mid-stream."""
        if not Path(self.frames.path).is_file():
            raise ConfigError(f"VIDEO_PATH does not exist or is not a file: {self.frames.path}")
        if self.mqtt.reconnect_max_delay < self.mqtt.reconnect_min_delay:
            raise ConfigError("MQTT_RECONNECT_MAX_DELAY must be >= MQTT_RECONNECT_MIN_DELAY")
        if self.output.image_ref_mode == "path":
            Path(self.output.plate_image_dir).mkdir(parents=True, exist_ok=True)
        if self.models.plate_scope in {"frame", "both"} and not self.models.vehicle_classes:
            raise ConfigError("VEHICLE_CLASSES must not be empty when PLATE_SCOPE is 'frame' or 'both'")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the configuration."""
        return asdict(self)  # type: ignore[arg-type]

    def redacted(self) -> dict[str, Any]:
        """Same as :meth:`to_dict` but with the MQTT password masked."""
        data = self.to_dict()
        mqtt = data.get("mqtt")
        if isinstance(mqtt, dict) and mqtt.get("password"):
            mqtt["password"] = REDACTED
        return data


def load_config() -> Config:
    """Build a :class:`Config` from the current environment."""
    return Config.from_env()
