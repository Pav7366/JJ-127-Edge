"""The published payload must match the platform contract exactly.

The key names and their optionality are not a style choice: ingest validates every
payload against contract/event_contract.json and drops anything that fails, so a
spelling drift here is a silent city-wide data loss rather than a failing test.
These tests therefore pin the wire format, not just the plumbing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from anpr_edge.payload import (
    CONTRACT_VEHICLE_TYPES,
    DetectionPayload,
    build_payload,
    to_contract_vehicle_type,
    utc_now_iso,
)

# The contract's event.required, in the order the platform's example lists them.
REQUIRED_KEYS = [
    "plate_string",
    "confidence",
    "camera_id",
    "lat",
    "lon",
    "timestamp",
]


def make_payload(**overrides) -> DetectionPayload:
    fields = {
        "camera_id": "cam-01",
        "plate_string": "KA01AB1234",
        "confidence": 0.93,
        "lat": 12.9716,
        "lon": 77.5946,
    }
    fields.update(overrides)
    return build_payload(**fields)


def test_payload_carries_exactly_the_required_keys_by_default():
    assert list(make_payload().to_dict()) == REQUIRED_KEYS


def test_payload_is_valid_json_with_primitive_values():
    decoded = json.loads(make_payload().to_json())
    assert decoded["camera_id"] == "cam-01"
    assert decoded["plate_string"] == "KA01AB1234"
    assert decoded["confidence"] == 0.93
    for key in ("lat", "lon", "confidence"):
        assert isinstance(decoded[key], float)


def test_timestamp_defaults_to_now_in_utc():
    decoded = json.loads(make_payload().to_json())
    parsed = datetime.strptime(decoded["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 30


def test_timestamp_matches_the_contract_pattern():
    # The contract checks the timestamp with a regex rather than format:date-time,
    # because jsonschema only enforces the latter when an optional library is
    # installed - and a silently unenforced format is worse than an explicit one.
    import re

    pattern = re.compile(
        r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$"
    )
    assert pattern.match(make_payload().to_dict()["timestamp"])


def test_explicit_timestamp_is_preserved():
    decoded = json.loads(make_payload(timestamp="2026-01-02T03:04:05.678Z").to_json())
    assert decoded["timestamp"] == "2026-01-02T03:04:05.678Z"


def test_optional_keys_are_omitted_rather_than_sent_as_null():
    # The schema marks these nullable but not required, and a key holding only null
    # tells a consumer nothing that absence does not.
    assert set(make_payload().to_dict()) == set(REQUIRED_KEYS)


def test_optional_keys_are_included_when_supplied():
    decoded = make_payload(
        image_ref="plates/abc.jpg",
        direction="E",
        lane=2,
        vehicle_class="car",
    ).to_dict()
    assert decoded["image_ref"] == "plates/abc.jpg"
    assert decoded["direction"] == "E"
    assert decoded["lane"] == 2
    assert decoded["vehicle_type"] == "car"


def test_optional_key_order_follows_the_contract():
    decoded = make_payload(
        image_ref="plates/abc.jpg",
        direction="E",
        lane=2,
        vehicle_class="truck",
    ).to_dict()
    assert list(decoded) == [*REQUIRED_KEYS, "image_ref", "direction", "lane", "vehicle_type"]


def test_lane_zero_is_published_rather_than_treated_as_absent():
    # `if lane is not None` rather than a truthiness check: lane 0 is a real lane.
    assert make_payload(lane=0).to_dict()["lane"] == 0


def test_vehicle_class_is_mapped_onto_the_contract_enum():
    # The detector speaks COCO, the contract does not.
    assert make_payload(vehicle_class="motorcycle").to_dict()["vehicle_type"] == "bike"
    assert make_payload(vehicle_class="bus").to_dict()["vehicle_type"] == "bus"
    assert make_payload(vehicle_class="truck").to_dict()["vehicle_type"] == "truck"


def test_vehicle_class_is_case_insensitive():
    assert make_payload(vehicle_class="Car").to_dict()["vehicle_type"] == "car"


def test_unknown_vehicle_class_is_dropped_not_forwarded():
    # A class outside the enum would fail validation and take the whole sighting
    # with it, so an unrecognised label is dropped instead.
    assert "vehicle_type" not in make_payload(vehicle_class="rickshaw").to_dict()
    assert "vehicle_type" not in make_payload(vehicle_class=None).to_dict()


def test_to_contract_vehicle_type_covers_only_the_contract_enum():
    assert to_contract_vehicle_type("motorcycle") == "bike"
    assert to_contract_vehicle_type("bicycle") == "bike"
    assert to_contract_vehicle_type("car") == "car"
    assert to_contract_vehicle_type("  CAR  ") == "car"
    assert to_contract_vehicle_type("") is None
    assert to_contract_vehicle_type(None) is None
    assert to_contract_vehicle_type("aeroplane") is None
    # Anything we do map must be a value the contract actually accepts.
    for name in ("car", "motorcycle", "truck", "bus", "bicycle"):
        assert to_contract_vehicle_type(name) in CONTRACT_VEHICLE_TYPES


def test_confidence_is_rounded_to_four_places():
    assert make_payload(confidence=0.123456789).to_dict()["confidence"] == 0.1235


def test_image_ref_round_trips():
    decoded = json.loads(make_payload(image_ref="plates/abc.jpg").to_json())
    assert decoded["image_ref"] == "plates/abc.jpg"


def test_utc_now_iso_is_parseable_and_utc():
    assert utc_now_iso().endswith("Z")
