"""Resilient MQTT publisher built on paho-mqtt.

Design notes
------------
* ``connect_async`` + ``loop_start`` gives us automatic reconnection with
  exponential backoff, so a broker that is down (or restarted) never crashes the
  container and never blocks the inference loop.
* While disconnected, payloads are buffered in a bounded in-memory deque and
  flushed on the next successful connect. Overflow drops the oldest entries and
  logs a warning rather than growing without bound.
* Publishing never blocks: ``client.publish`` is non-blocking and we simply
  count the messages that the broker has not acknowledged yet.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

from .config import MqttConfig
from .logging_setup import get_logger, log_event
from .payload import DetectionPayload

__all__ = ["MQTTPublisher"]

_LOG = get_logger("anpr_edge.mqtt")


class MQTTPublisher:
    """Publishes detection payloads to ``anpr/{camera_id}/sightings``."""

    def __init__(self, config: MqttConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._pending: deque[str] = deque()
        self._connected = threading.Event()
        self._stopped = False
        self.published = 0
        self.dropped = 0
        self.queued = 0

        protocol = (
            mqtt.MQTTv5 if config.protocol == "v5" else mqtt.MQTTv311
        )
        self._client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=config.client_id,
            protocol=protocol,
        )
        if config.username:
            self._client.username_pw_set(config.username, config.password)
        if config.tls:
            self._client.tls_set(ca_certs=config.ca_certs)
        self._client.reconnect_delay_set(
            min_delay=config.reconnect_min_delay, max_delay=config.reconnect_max_delay
        )
        self._client.max_queued_messages_set(config.max_queue)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_connect_fail = self._on_connect_fail
        self._client.on_publish = self._on_publish

    # ------------------------------------------------------------------ setup
    @property
    def topic(self) -> str:
        """Topic every detection is published to."""
        return self._config.topic

    def start(self) -> None:
        """Begin connecting in the background (never raises on broker outage)."""
        log_event(
            _LOG,
            20,
            "mqtt_connecting",
            broker=self._config.broker,
            topic=self._config.topic,
            client_id=self._config.client_id,
            qos=self._config.qos,
            tls=self._config.tls,
            protocol=self._config.protocol,
        )
        # retry_first_connection keeps the background loop retrying the very first
        # connection attempt as well (otherwise a broker that is still booting is
        # only retried on the second loop iteration).
        self._client.connect_async(
            self._config.host,
            self._config.port,
            keepalive=self._config.keepalive,
        )
        self._client.loop_start()
        if self._config.connect_wait_seconds > 0:
            if self._connected.wait(self._config.connect_wait_seconds):
                log_event(_LOG, 20, "mqtt_connected", broker=self._config.broker)
            else:
                # Not fatal: detections are buffered until the broker shows up.
                log_event(
                    _LOG,
                    30,
                    "mqtt_initial_connect_timeout",
                    broker=self._config.broker,
                    waited_seconds=self._config.connect_wait_seconds,
                    action="buffering_until_connected",
                )

    def close(self) -> None:
        """Flush what we can and disconnect cleanly."""
        if self._stopped:
            return
        self._stopped = True
        try:
            if self._client.is_connected():
                self._client.disconnect()
        except Exception as exc:  # pragma: no cover - defensive
            log_event(_LOG, 30, "mqtt_disconnect_error", error=str(exc))
        try:
            self._client.loop_stop()
        except Exception as exc:  # pragma: no cover - defensive
            log_event(_LOG, 30, "mqtt_loop_stop_error", error=str(exc))
        log_event(
            _LOG,
            20,
            "mqtt_stopped",
            published=self.published,
            dropped=self.dropped,
            still_queued=len(self._pending),
        )

    def __enter__(self) -> MQTTPublisher:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- publishing
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    def publish(self, payload: DetectionPayload, diagnostics: dict[str, Any] | None = None) -> bool:
        """Publish one payload. Returns ``True`` if it went to the broker now.

        Returns ``False`` when the message was buffered (or dropped) instead.
        """
        body = payload.to_json()
        properties = None
        if self._config.publish_diagnostics and diagnostics:
            properties = Properties(PacketTypes.PUBLISH)
            for key, value in diagnostics.items():
                # paho appends to UserProperty on every assignment.
                properties.UserProperty = (str(key), str(value))

        if self._connected.is_set() and not self._stopped:
            info = self._client.publish(
                self._config.topic,
                payload=body,
                qos=self._config.qos,
                retain=self._config.retain,
                properties=properties,
            )
            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                self.published += 1
                return True
            # Queue full or socket race: fall through to buffering.
            log_event(
                _LOG,
                30,
                "mqtt_publish_deferred",
                reason=mqtt.error_string(info.rc),
                plate=payload.plate_string,
            )

        with self._lock:
            if len(self._pending) >= self._config.max_queue:
                self._pending.popleft()
                self.dropped += 1
                log_event(
                    _LOG,
                    30,
                    "mqtt_queue_overflow",
                    max_queue=self._config.max_queue,
                    action="dropped_oldest",
                )
            self._pending.append(body)
            self.queued += 1
        return False

    def flush(self) -> int:
        """Try to drain the offline buffer. Returns the number of messages sent."""
        if not self._connected.is_set() or self._stopped:
            return 0
        sent = 0
        while True:
            with self._lock:
                if not self._pending:
                    break
                body = self._pending.popleft()
            info = self._client.publish(
                self._config.topic, payload=body, qos=self._config.qos, retain=self._config.retain
            )
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                with self._lock:
                    self._pending.appendleft(body)
                log_event(_LOG, 30, "mqtt_flush_paused", reason=mqtt.error_string(info.rc))
                break
            sent += 1
        if sent:
            self.published += sent
            log_event(_LOG, 20, "mqtt_queue_flushed", messages=sent, remaining=len(self._pending))
        return sent

    def stats(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "published": self.published,
            "queued_total": self.queued,
            "queue_depth": len(self._pending),
            "dropped": self.dropped,
        }

    # --------------------------------------------------------------- paho API
    def _on_connect(self, _client, _userdata, _flags, reason_code, _properties=None) -> None:
        if getattr(reason_code, "is_failure", False):
            log_event(_LOG, 30, "mqtt_connect_refused", reason_code=str(reason_code))
            return
        first = not self._connected.is_set()
        self._connected.set()
        log_event(_LOG, 20, "mqtt_connected", broker=self._config.broker, reconnected=not first)
        if first or self._pending:
            self.flush()

    def _on_disconnect(self, _client, _userdata, _flags=None, reason_code=None, _properties=None) -> None:
        was_connected = self._connected.is_set()
        self._connected.clear()
        if was_connected and not self._stopped:
            log_event(
                _LOG,
                30,
                "mqtt_disconnected",
                reason_code=str(reason_code),
                action="will_retry_with_backoff",
                retry_min_seconds=self._config.reconnect_min_delay,
                retry_max_seconds=self._config.reconnect_max_delay,
            )

    def _on_connect_fail(self, _client, _userdata) -> None:
        log_event(
            _LOG,
            30,
            "mqtt_connect_failed",
            broker=self._config.broker,
            action="will_retry_with_backoff",
        )

    def _on_publish(self, _client, _userdata, _mid, _reason_code=None, _properties=None) -> None:  # pragma: no cover
        return
