"""Tests for the ONNX vehicle stage: letterbox, decode, gating and NMS.

No model file is loaded here. The exported graph keeps the standard
``(1, 4 + num_classes, num_anchors)`` YOLO layout, so these tests build small
synthetic tensors with the same shape contract and check the decode maths.
"""

from __future__ import annotations

import numpy as np
import pytest

from anpr_edge.models.vehicle_detector import decode_detections, letterbox

cv2 = pytest.importorskip("cv2")

NUM_CLASSES = 80
CAR, MOTORCYCLE, BUS, TRUCK, PERSON = 2, 3, 5, 7, 0
COCO = {CAR: "car", MOTORCYCLE: "motorcycle", BUS: "bus", TRUCK: "truck", PERSON: "person"}


def make_output(rows: list[dict[int, float]], anchors: int = 8) -> np.ndarray:
    """Build a (1, 4 + NUM_CLASSES, anchors) tensor from cxcywh + class scores."""
    width = 4 + NUM_CLASSES
    out = np.zeros((1, width, anchors), dtype=np.float32)
    for index, row in enumerate(rows):
        out[0, 0, index] = row["cx"]
        out[0, 1, index] = row["cy"]
        out[0, 2, index] = row["w"]
        out[0, 3, index] = row["h"]
        for class_id, score in row.get("scores", {}).items():
            out[0, 4 + class_id, index] = score
    return out


def decode(
    output,
    *,
    frame_shape=(480, 640),
    classes=None,
    class_names=COCO,
    conf=0.3,
    iou=0.5,
    max_det=20,
    min_box=8,
    scale=1.0,
    pad=0.0,
    pad_x=None,
    pad_y=None,
):
    return decode_detections(
        output,
        scale=scale,
        pad_x=pad if pad_x is None else pad_x,
        pad_y=pad if pad_y is None else pad_y,
        frame_shape=frame_shape,
        class_ids=classes if classes is not None else [CAR, MOTORCYCLE, BUS, TRUCK],
        class_names=class_names,
        num_classes=NUM_CLASSES,
        conf_threshold=conf,
        iou_threshold=iou,
        max_det=max_det,
        min_box_px=min_box,
    )


# ------------------------------------------------------------------ letterbox ---


def test_letterbox_produces_chw_float_blob_scaled_to_unit_range():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    blob, scale, pad_x, pad_y = letterbox(frame, 640)

    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    # A 640x480 frame scales to fit the width exactly, so the 114-grey bars
    # needed to square it off are top and bottom only.
    assert scale == pytest.approx(1.0)
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx(80.0)
    assert blob.max() == pytest.approx(1.0, abs=1e-3)
    assert blob[0, 0, 0, 0] == pytest.approx(114 / 255.0, abs=1e-3)


def test_letterbox_pads_tall_frames_vertically_and_keeps_aspect_ratio():
    frame = np.zeros((800, 400, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox(frame, 640)

    # 400x800 -> scale 0.8 -> 320x640 centred, so 160px of grey on each side.
    assert scale == pytest.approx(0.8)
    assert pad_x == pytest.approx(160.0)
    assert pad_y == pytest.approx(0.0)


def test_letterbox_pads_wide_frames_horizontally():
    frame = np.zeros((200, 800, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox(frame, 640)

    # 800x200 -> scale 0.8 -> 640x160 centred, so (640 - 160) / 2 of grey above
    # and below.
    assert scale == pytest.approx(0.8)
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx(240.0)


def test_letterbox_pads_with_grey_114():
    frame = np.zeros((200, 800, 3), dtype=np.uint8)
    blob, _, _, _ = letterbox(frame, 640)

    padding = blob[0, :, :10, :]
    assert np.allclose(padding, 114 / 255.0, atol=1e-3)


# --------------------------------------------------------------------- decode ---


def test_decode_returns_box_in_full_frame_coordinates():
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 200.0, "h": 100.0, "scores": {CAR: 0.9}}])
    boxes = decode(out)

    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (220, 190, 420, 290)
    assert box.class_id == CAR
    assert box.class_name == "car"
    assert box.confidence == pytest.approx(0.9)
    assert box.as_tuple() == (220, 190, 420, 290)
    assert box.area == 200 * 100


def test_decode_undoes_letterbox_scale_and_padding():
    # Letterbox of a 640x480 frame into 640x640: scale 1.0, 80px of vertical pad.
    # Without the 80px subtracted the box would be reported 80px too high.
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 100.0, "h": 60.0, "scores": {BUS: 0.8}}])
    assert decode(out)[0].as_tuple() == (270, 210, 370, 270)
    assert decode(out, pad_y=80.0)[0].as_tuple() == (270, 130, 370, 190)

    # A 400x800 frame is letterboxed at scale 0.8 with 160px of horizontal pad
    # only. A box on the centre of the frame (200, 400) lands at (320, 320) in
    # the padded square, and a 200x100 frame box is 160x80 there.
    out = make_output([{"cx": 320.0, "cy": 320.0, "w": 160.0, "h": 80.0, "scores": {CAR: 0.7}}])
    boxes = decode(out, frame_shape=(800, 400), scale=0.8, pad_x=160.0)
    assert boxes[0].as_tuple() == (100, 350, 300, 450)

    # Dropping the pad shifts the box right by exactly pad / scale = 200px, and
    # it gets clipped at the frame edge - proof the padding is really applied.
    boxes = decode(out, frame_shape=(800, 400), scale=0.8)
    assert boxes[0].as_tuple() == (300, 350, 400, 450)


def test_decode_drops_detections_below_the_confidence_threshold():
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 100.0, "h": 100.0, "scores": {CAR: 0.29}}])
    assert decode(out, conf=0.3) == []


def test_decode_ignores_classes_outside_the_configured_set():
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 100.0, "h": 100.0, "scores": {PERSON: 0.99}}])
    assert decode(out) == []
    # ...but an allowed class on the same anchor is still picked up.
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 100.0, "h": 100.0, "scores": {CAR: 0.4}}])
    assert len(decode(out, classes=[CAR])) == 1


def test_decode_prefers_the_best_vehicle_class_over_a_higher_ignored_score():
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 100.0, "h": 100.0, "scores": {CAR: 0.6, PERSON: 0.99}}])
    boxes = decode(out, classes=[CAR])
    assert len(boxes) == 1
    assert boxes[0].class_id == CAR
    assert boxes[0].confidence == pytest.approx(0.6)


def test_decode_clamps_boxes_to_the_frame():
    out = make_output([{"cx": 10.0, "cy": 10.0, "w": 100.0, "h": 100.0, "scores": {CAR: 0.9}}])
    box = decode(out)[0]
    assert box.x1 == 0 and box.y1 == 0
    assert box.x2 == 60 and box.y2 == 60


def test_decode_drops_boxes_below_the_minimum_size():
    rows = [
        {"cx": 100.0, "cy": 100.0, "w": 20.0, "h": 20.0, "scores": {CAR: 0.9}},
        {"cx": 400.0, "cy": 300.0, "w": 100.0, "h": 100.0, "scores": {CAR: 0.9}},
    ]
    boxes = decode(make_output(rows), min_box=32)
    assert len(boxes) == 1
    assert boxes[0].as_tuple() == (350, 250, 450, 350)


def test_decode_suppresses_overlapping_duplicates():
    rows = [
        {"cx": 300.0, "cy": 240.0, "w": 200.0, "h": 120.0, "scores": {CAR: 0.9}},
        {"cx": 312.0, "cy": 244.0, "w": 200.0, "h": 120.0, "scores": {CAR: 0.85}},
    ]
    assert len(decode(make_output(rows))) == 1


def test_decode_keeps_distant_objects_and_respects_max_det():
    rows = [
        {"cx": 100.0, "cy": 100.0, "w": 40.0, "h": 40.0, "scores": {CAR: 0.9}},
        {"cx": 300.0, "cy": 200.0, "w": 40.0, "h": 40.0, "scores": {CAR: 0.85}},
        {"cx": 500.0, "cy": 350.0, "w": 40.0, "h": 40.0, "scores": {CAR: 0.8}},
    ]
    assert len(decode(make_output(rows))) == 3
    assert len(decode(make_output(rows), max_det=2)) == 2


def test_decode_keeps_distinct_vehicle_classes_on_the_same_object():
    rows = [
        {"cx": 300.0, "cy": 240.0, "w": 200.0, "h": 120.0, "scores": {CAR: 0.9, BUS: 0.6}},
    ]
    boxes = decode(make_output(rows), iou=0.5)
    # class-agnostic NMS collapses the two labels onto the strongest one.
    assert len(boxes) == 1
    assert boxes[0].class_name == "car"


def test_decode_accepts_the_transposed_layout():
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 200.0, "h": 100.0, "scores": {CAR: 0.9}}])
    boxes = decode(np.ascontiguousarray(out[0].T))
    assert len(boxes) == 1
    assert boxes[0].as_tuple() == (220, 190, 420, 290)


def test_decode_handles_empty_and_odd_outputs():
    assert decode(np.zeros((1, 4 + NUM_CLASSES, 0), dtype=np.float32)) == []
    assert decode(np.zeros((1, 2, 5), dtype=np.float32)) == []
    assert decode(make_output([{"cx": 1.0, "cy": 1.0, "w": 1.0, "h": 1.0, "scores": {CAR: 0.9}}]), classes=[]) == []


def test_decode_labels_unknown_ids_defensively():
    out = make_output([{"cx": 320.0, "cy": 240.0, "w": 100.0, "h": 100.0, "scores": {CAR: 0.9}}])
    boxes = decode(out, class_names={})
    assert boxes[0].class_name == str(CAR)


# ------------------------------------------------------- graph input shape ---


class _FakeInput:
    def __init__(self, shape):
        self.name = "images"
        self.shape = shape


class _FakeSession:
    def __init__(self, shape):
        self._input = _FakeInput(shape)

    def get_inputs(self):
        return [self._input]


def _detector(env, imgsz: int = 640):
    from anpr_edge.config import load_config
    from anpr_edge.models.vehicle_detector import VehicleDetector

    env(VEHICLE_IMGSZ=str(imgsz))
    return VehicleDetector(load_config().models)


def test_static_graph_size_wins_over_the_configured_size(env):
    # Ultralytics exports a fixed square input, so a stale VEHICLE_IMGSZ must
    # not feed the graph a tensor of the wrong shape.
    detector = _detector(env, imgsz=640)
    detector._session = _FakeSession([1, 3, 416, 416])
    assert detector._resolve_imgsz() == 416


def test_matching_graph_size_is_used_as_is(env):
    detector = _detector(env, imgsz=512)
    detector._session = _FakeSession([1, 3, 512, 512])
    assert detector._resolve_imgsz() == 512


def test_dynamic_graph_falls_back_to_the_configured_size(env):
    detector = _detector(env, imgsz=640)
    detector._session = _FakeSession([1, 3, "height", "width"])
    assert detector._resolve_imgsz() == 640

    detector = _detector(env, imgsz=512)
    detector._session = _FakeSession([1, 3, None, None])
    assert detector._resolve_imgsz() == 512


def test_non_square_graph_falls_back_to_the_configured_size(env):
    # A rectangle is not something letterbox() can feed, so the setting is used
    # and the graph error surfaces instead of being silently mis-sized.
    detector = _detector(env, imgsz=640)
    detector._session = _FakeSession([1, 3, 640, 480])
    assert detector._resolve_imgsz() == 640
