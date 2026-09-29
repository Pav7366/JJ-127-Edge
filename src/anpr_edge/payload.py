"""Detection payload construction.

The published JSON document is the contract with the downstream backend, so the
schema lives in exactly one place: :meth:`DetectionPayload.to_dict`.

The field names and the topic are dictated by the platform's
``contract/event_contract.json``, not by this repo. Ingest validates every payload
against that schema and drops anything that fails, so a key spelled ``latitude``
here instead of ``lon`` costs us every sighting in the city with nothing but a
``Dropping invalid sighting`` line in someone else's logs. The six required keys
are therefore fixed by the contract; only the optional ones are ours to fill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "CONTRACT_VEHICLE_TYPES",
    "DetectionPayload",
    "build_payload",
    "to_contract_vehicle_type",
    "utc_now_iso",
]

# event.json_schema -> properties.vehicle_type.enum in the platform contract.
CONTRACT_VEHICLE_TYPES = frozenset({"car", "bike", "truck", "bus"})

# The vehicle detector is a COCO YOLO graph, so it names classes in COCO's
# vocabulary; the contract uses its own four-value enum. Only the classes with no
# contract equivalent are renamed here - anything absent is left absent, because a
# class the contract does not define would fail validation and take the whole
# sighting (plate, position and all) down with it. An honest gap beats a wrong pin:
# downstream the class selects a map pin and a filter, and a wrong pin is believed.
_COCO_TO_CONTRACT = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "bike",
    "bicycle": "bike",
}


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def to_contract_vehicle_type(class_name: str | None) -> str | None:
    """Map a detector class name onto the contract's ``vehicle_type`` enum.

    Returns ``None`` for anything the contract does not define, which the payload
    renders by omitting the key rather than sending an explicit null.
    """
    if not class_name:
        return None
    mapped = _COCO_TO_CONTRACT.get(str(class_name).strip().lower())
    return mapped if mapped in CONTRACT_VEHICLE_TYPES else None


@dataclass(frozen=True, slots=True)
class DetectionPayload:
    """One ANPR reading, serialised straight onto the MQTT topic."""

    camera_id: str
    plate_string: str
    confidence: float
    timestamp: str
    lat: float
    lon: float
    image_ref: str | None = None
    direction: str | None = None
    lane: int | None = None
    vehicle_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the wire format. Key order matches the agreed schema.

        The six required keys always appear, in contract order. The optional ones
        are omitted when there is nothing to say: the schema marks them nullable
        but not required, and a key carrying only ``null`` is indistinguishable
        from an absent one to every consumer, so sending it is pure noise.
        """
        event: dict[str, Any] = {
            "plate_string": self.plate_string,
            "confidence": round(float(self.confidence), 4),
            "camera_id": self.camera_id,
            "lat": float(self.lat),
            "lon": float(self.lon),
            "timestamp": self.timestamp,
        }
        if self.image_ref is not None:
            event["image_ref"] = self.image_ref
        if self.direction is not None:
            event["direction"] = self.direction
        if self.lane is not None:
            event["lane"] = int(self.lane)
        if self.vehicle_type is not None:
            event["vehicle_type"] = self.vehicle_type
        return event

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


def build_payload(
    *,
    camera_id: str,
    plate_string: str,
    confidence: float,
    lat: float,
    lon: float,
    timestamp: str | None = None,
    image_ref: str | None = None,
    direction: str | None = None,
    lane: int | None = None,
    vehicle_class: str | None = None,
) -> DetectionPayload:
    """Build a payload, stamping the detection time when not supplied.

    ``vehicle_class`` is the raw detector label (e.g. ``motorcycle``); it is
    translated to the contract's enum here so no caller has to know the mapping.
    """
    return DetectionPayload(
        camera_id=camera_id,
        plate_string=plate_string,
        confidence=float(confidence),
        lat=float(lat),
        lon=float(lon),
        timestamp=timestamp or utc_now_iso(),
        image_ref=image_ref,
        direction=direction,
        lane=None if lane is None else int(lane),
        vehicle_type=to_contract_vehicle_type(vehicle_class),
    )
