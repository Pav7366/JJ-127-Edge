"""Configuration loading and validation."""

from __future__ import annotations

import pytest

from anpr_edge.config import ConfigError, load_config


def test_minimal_environment_builds_expected_defaults(env):
    env()
    config = load_config()

    assert config.camera_id == "cam-test"
    assert config.latitude == pytest.approx(12.9716)
    assert config.longitude == pytest.approx(77.5946)
    assert config.mqtt.host == "localhost"
    assert config.mqtt.port == 1883
    assert config.mqtt.topic == "anpr/cam-test/sightings"
    assert config.mqtt.qos == 1
    assert config.frames.stride == 1
    assert config.frames.loop is True
    assert config.models.plate_scope == "crop"
    assert config.output.dedup_window_seconds == pytest.approx(10.0)
    assert config.runtime.log_format == "json"
    assert config.runtime.quiet_onnx is True
    assert config.direction is None
    assert config.lane is None


def test_onnx_quieting_can_be_switched_off(env):
    env(QUIET_ONNX="0")
    config = load_config()

    assert config.runtime.quiet_onnx is False


def test_broker_accepts_host_and_port_separately(env):
    env(MQTT_BROKER="", MQTT_BROKER_HOST="mqtt.internal", MQTT_BROKER_PORT="8883")
    config = load_config()
    assert (config.mqtt.host, config.mqtt.port) == ("mqtt.internal", 8883)


def test_broker_accepts_ipv6_literal(env):
    env(MQTT_BROKER="[::1]:1884")
    config = load_config()
    assert (config.mqtt.host, config.mqtt.port) == ("::1", 1884)


def test_explicit_topic_template_overrides_the_default(env):
    env(MQTT_TOPIC_TEMPLATE="city/{camera_id}/plates")
    assert load_config().mqtt.topic == "city/cam-test/plates"


def test_direction_and_lane_are_read_when_supplied(env):
    env(CAMERA_DIRECTION="e", CAMERA_LANE="2")
    config = load_config()
    assert config.direction == "E"
    assert config.lane == 2


def test_unknown_direction_is_rejected(env):
    env(CAMERA_DIRECTION="sideways")
    with pytest.raises(ConfigError, match="CAMERA_DIRECTION"):
        load_config()


def test_lane_must_be_an_integer(env):
    env(CAMERA_LANE="middle")
    with pytest.raises(ConfigError, match="CAMERA_LANE"):
        load_config()


def test_camera_id_rejects_characters_that_would_break_the_topic(env):
    # The id is interpolated into anpr/{camera_id}/sightings and matched against a
    # per-camera ACL grant, so a ':' or '/' would add a topic level instead of
    # naming a camera - and every publish would be refused at the broker.
    for bad in ("cam:pune", "cam/pune", "cam 01"):
        env(CAMERA_ID=bad)
        with pytest.raises(ConfigError, match="CAMERA_ID"):
            load_config()


def test_camera_id_accepts_the_platform_naming_convention(env):
    env(CAMERA_ID="cam-pune-hinjewadi-01")
    assert load_config().mqtt.topic == "anpr/cam-pune-hinjewadi-01/sightings"


def test_topic_template_must_contain_the_placeholder(env):
    env(MQTT_TOPIC_TEMPLATE="city/fixed/topic")
    with pytest.raises(ConfigError, match="MQTT_TOPIC_TEMPLATE"):
        load_config()


def test_invalid_float_is_reported_with_the_variable_name(env):
    env(OCR_MIN_CONFIDENCE="high")
    with pytest.raises(ConfigError, match="OCR_MIN_CONFIDENCE"):
        load_config()


def test_stride_must_be_positive(env):
    env(FRAME_STRIDE="0")
    with pytest.raises(ConfigError, match="FRAME_STRIDE"):
        load_config()


def test_unknown_plate_scope_is_rejected(env):
    env(PLATE_SCOPE="sideways")
    with pytest.raises(ConfigError, match="PLATE_SCOPE"):
        load_config()


def test_missing_required_variable_is_reported(env):
    env(CAMERA_ID="")
    with pytest.raises(ConfigError, match="CAMERA_ID"):
        load_config()


def test_missing_video_file_is_reported(env, tmp_path):
    env(VIDEO_PATH=str(tmp_path / "nope.mp4"))
    with pytest.raises(ConfigError, match="VIDEO_PATH"):
        load_config()


def test_video_file_must_exist(env, tmp_path):
    env(VIDEO_PATH=str(tmp_path / "nope.mp4"))
    with pytest.raises(ConfigError, match="VIDEO_PATH"):
        load_config()


def test_existing_video_file_validates(env, sample_video):
    env(VIDEO_PATH=str(sample_video))
    config = load_config()
    assert config.frames.path == str(sample_video)
    config.validate()


def test_credentials_are_redacted(env):
    env(MQTT_USERNAME="admin", MQTT_PASSWORD="s3cret")
    redacted = load_config().redacted()
    assert redacted["mqtt"]["password"] == "***"
    assert "s3cret" not in str(redacted)
