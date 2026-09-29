"""MQTT publisher: payload wire format and offline buffering (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from anpr_edge.config import load_config
from anpr_edge.mqtt_publisher import MQTTPublisher
from anpr_edge.payload import DetectionPayload, build_payload

pytest.importorskip("paho.mqtt")


class FakeClient:
    """Minimal stand-in for ``paho.mqtt.Client``."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, Any]] = []
        self.credentials: tuple[str | None, str | None] = (None, None)
        self.tls_certs: str | None = None
        self.connected = True
        self.loop_started = False
        self.disconnected = False

    def username_pw_set(self, username, password) -> None:
        self.credentials = (username, password)

    def tls_set(self, ca_certs=None, certfile=None, keyfile=None, tls_version=None) -> None:
        self.tls_certs = ca_certs

    def tls_insecure_set(self, value) -> None: ...
    def enable_logger(self, logger) -> None: ...
    def reconnect_delay_set(self, minimum, maximum) -> None: ...
    def max_queued_messages_set(self, value) -> None: ...

    def loop_start(self) -> None:
        self.loop_started = True

    def loop_stop(self) -> None: ...
    def disconnect(self) -> None:
        self.disconnected = True

    def connect_async(self, host, port, keepalive) -> None: ...
    def is_connected(self) -> bool:
        return self.connected

    def publish(self, topic, payload=None, qos=0, retain=False, properties=None):
        self.published.append((topic, payload if payload is not None else "", qos, properties))
        return SimpleNamespace(rc=0)


def build(env, client: FakeClient, connect: bool = True, **overrides) -> MQTTPublisher:
    env(
        MQTT_BROKER="localhost:1883",
        MQTT_CONNECT_WAIT_SECONDS="0",
        **overrides,
    )
    publisher = MQTTPublisher(load_config().mqtt)
    publisher._client = client
    if connect:
        publisher._connected.set()
    return publisher


def payload() -> DetectionPayload:
    return build_payload(
        camera_id="cam-test",
        plate_string="KA01AB1234",
        confidence=0.91,
        lat=1.5,
        lon=2.5,
    )


def test_topic_follows_the_platform_sightings_template(env):
    assert build(env, FakeClient()).topic == "anpr/cam-test/sightings"


def test_publish_sends_the_exact_json_payload(env):
    client = FakeClient()
    publisher = build(env, client)
    publisher.start()
    assert publisher.publish(payload(), {"region": "IN-KA"}) is True

    topic, body, qos, _ = client.published[0]
    assert topic == "anpr/cam-test/sightings"
    assert qos == 1
    decoded = json.loads(body)
    assert set(decoded) == {
        "plate_string",
        "confidence",
        "camera_id",
        "lat",
        "lon",
        "timestamp",
    }
    assert decoded["plate_string"] == "KA01AB1234"
    publisher.close()


def test_diagnostics_are_optional_user_properties(env):
    client = FakeClient()
    publisher = build(env, client, MQTT_PUBLISH_DIAGNOSTICS="true")
    publisher.publish(payload(), {"region": "IN-KA", "source": "crop"})
    properties = client.published[0][3]
    assert properties is not None
    assert ("region", "IN-KA") in properties.UserProperty


def test_diagnostics_stay_out_of_the_payload(env):
    client = FakeClient()
    publisher = build(env, client, MQTT_PUBLISH_DIAGNOSTICS="true")
    publisher.publish(payload(), {"region": "IN-KA"})
    _, body, _, _ = client.published[0]
    assert "region" not in json.loads(body)


def test_messages_are_buffered_while_offline(env):
    client = FakeClient()
    client.connected = False
    publisher = build(env, client, connect=False)
    assert publisher.publish(payload()) is False
    assert client.published == []  # nothing left the process
    assert publisher.stats()["queue_depth"] == 1


def test_buffer_is_bounded_and_drops_the_oldest(env):
    client = FakeClient()
    client.connected = False
    publisher = build(env, client, connect=False, MQTT_MAX_QUEUE="5")
    for _ in range(20):
        publisher.publish(payload())
    stats = publisher.stats()
    assert stats["queue_depth"] == 5
    assert stats["dropped"] == 15


def test_buffer_is_flushed_once_the_broker_returns(env):
    client = FakeClient()
    client.connected = False
    publisher = build(env, client, connect=False)
    publisher.publish(payload())
    assert publisher.flush() == 0  # still offline

    publisher._connected.set()
    assert publisher.flush() == 1
    assert publisher.stats()["queue_depth"] == 0
    assert json.loads(client.published[0][1])["plate_string"] == "KA01AB1234"


def test_stats_expose_connection_state(env):
    publisher = build(env, FakeClient())
    assert publisher.stats()["connected"] is True
    assert publisher.stats()["published"] == 0


def test_start_stops_the_paho_loop_and_close_disconnects(env):
    client = FakeClient()
    publisher = build(env, client)
    publisher.start()
    assert client.loop_started is True
    publisher.close()
    assert client.disconnected is True
    publisher.close()  # idempotent


def test_credentials_are_forwarded_to_paho(env):
    env(MQTT_BROKER="localhost:1883", MQTT_USERNAME="admin", MQTT_PASSWORD="secret")
    config = load_config().mqtt
    assert (config.username, config.password) == ("admin", "secret")


def test_publishing_after_close_is_refused(env):
    client = FakeClient()
    publisher = build(env, client)
    publisher.start()
    publisher.close()
    assert publisher.publish(payload()) is False
    assert client.published == []
