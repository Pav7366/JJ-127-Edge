"""Vehicle detection stage (YOLO26 exported to ONNX, CPU only via ONNX Runtime).

The exported graph keeps the standard YOLO layout - ``(1, 4 + num_classes,
num_anchors)`` of ``cx, cy, w, h`` plus per-class scores in letterboxed pixel
space - so the letterbox transform, the class/confidence gating and the NMS all
live here rather than inside the model. Keeping them as module level functions
makes the decoding testable without downloading a model.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import ModelConfig
from ..logging_setup import get_logger, log_event

__all__ = ["VehicleBox", "VehicleDetector", "decode_detections", "letterbox"]

_LOG = get_logger("anpr_edge.models.vehicle")

_DEFAULT_CLASS_NAMES = {"bicycle", "bus", "car", "motorcycle", "truck"}


@dataclass(frozen=True, slots=True)
class VehicleBox:
    """One detected vehicle in full-frame pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize ``frame`` into a square ``size`` blob, preserving aspect ratio.

    Returns the CHW float32 blob plus the ``(scale, pad_x, pad_y)`` needed to map
    predictions back to full-frame pixels. Padding uses the same grey value
    (114) as the training pipeline.
    """
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, min(size, round(width * scale)))
    new_height = max(1, min(size, round(height * scale)))
    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas[top : top + new_height, left : left + new_width] = resized

    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return blob, scale, pad_x, pad_y


def decode_detections(
    output: np.ndarray,
    *,
    scale: float,
    pad_x: float,
    pad_y: float,
    frame_shape: tuple[int, int],
    class_ids: list[int],
    class_names: dict[int, str],
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    min_box_px: int,
    num_classes: int | None = None,
) -> list[VehicleBox]:
    """Turn one raw model output into full-frame :class:`VehicleBox` values.

    ``output`` is the ``(1, 4 + num_classes, num_anchors)`` tensor the
    Ultralytics exporter produces, i.e. box terms down axis 0 followed by one
    score column per class. ``num_classes`` (known from the graph metadata)
    pins the layout so the ``(num_anchors, 4 + num_classes)`` transpose of a
    re-exported graph is still read correctly. Scores are taken only over
    ``class_ids`` so a high score on an ignored class cannot mask a vehicle.
    """
    height, width = frame_shape
    predictions = np.squeeze(output, axis=0) if output.ndim == 3 else output
    if predictions.ndim != 2:
        return []

    if num_classes is None:
        features_on_rows = True
    else:
        expected = 4 + num_classes
        if predictions.shape[0] == expected:
            features_on_rows = True
        elif predictions.shape[1] == expected:
            features_on_rows = False
        else:
            return []

    boxes = predictions[:4, :].T if features_on_rows else predictions[:, :4]
    columns = predictions[4:, :].T if features_on_rows else predictions[:, 4:]
    if columns.shape[1] == 0:
        return []

    if not class_ids:
        return []
    wanted = [class_id for class_id in class_ids if 0 <= class_id < columns.shape[1]]
    if not wanted:
        return []
    scores = columns[:, wanted]
    best = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), best]
    keep = confidences >= conf_threshold
    if not keep.any():
        return []

    candidates = boxes[keep]
    confidences = confidences[keep]
    class_columns = [wanted[index] for index in best[keep]]

    # cx, cy, w, h in letterboxed pixels -> full-frame xyxy.
    centres_x = (candidates[:, 0] - pad_x) / scale
    centres_y = (candidates[:, 1] - pad_y) / scale
    box_width = candidates[:, 2] / scale
    box_height = candidates[:, 3] / scale
    xyxy = np.column_stack(
        [
            centres_x - box_width / 2,
            centres_y - box_height / 2,
            centres_x + box_width / 2,
            centres_y + box_height / 2,
        ]
    )
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)

    # NMSBoxes wants (x, y, w, h); zero-sized boxes are dropped first so they
    # cannot distort the IoU maths of the real candidates.
    for_boxes = np.column_stack(
        [xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]]
    )
    positive = (for_boxes[:, 2] > 0) & (for_boxes[:, 3] > 0)
    for_boxes = for_boxes[positive]
    confidences = confidences[positive]
    class_columns = [class_column for class_column, ok in zip(class_columns, positive, strict=True) if ok]
    if not len(for_boxes):
        return []
    selected = cv2.dnn.NMSBoxes(
        for_boxes.tolist(),
        confidences.tolist(),
        float(conf_threshold),
        float(iou_threshold),
        top_k=max(1, int(max_det)),
    )
    indices = (
        np.array(selected, dtype=np.int64).reshape(-1)
        if len(selected)
        else np.empty(0, dtype=np.int64)
    )

    boxes: list[VehicleBox] = []
    for index in indices:
        x1, y1, x2, y2 = (round(value) for value in xyxy[index])
        if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
            continue
        class_id = int(class_columns[index])
        boxes.append(
            VehicleBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(confidences[index]),
                class_id=class_id,
                class_name=class_names.get(class_id, str(class_id)),
            )
        )
    return boxes


def _class_names_from_metadata(session: object) -> dict[int, str]:
    """Read the COCO label map Ultralytics embeds in the ONNX graph."""
    meta = session.get_modelmeta()  # type: ignore[attr-defined]
    raw = meta.custom_metadata_map.get("names", "")
    if not raw:
        raise ValueError(
            "the vehicle ONNX has no 'names' metadata; re-export it with "
            "ultralytics (scripts/download_models.py) so class ids can be mapped"
        )
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:  # pragma: no cover - malformed graph
        raise ValueError(f"unreadable 'names' metadata in the vehicle ONNX: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"unexpected 'names' metadata type: {type(parsed).__name__}")
    return {int(key): str(value) for key, value in parsed.items()}


class VehicleDetector:
    """Run the exported YOLO26 graph through ONNX Runtime."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._session = None
        self._input_name = ""
        self._imgsz = config.vehicle_imgsz
        self._class_names: dict[int, str] = {}
        self._class_ids: list[int] = []

    def _resolve_imgsz(self) -> int:
        """Prefer the size baked into the graph over the configured one.

        Ultralytics exports a static square input by default, so VEHICLE_IMGSZ
        alone cannot change it - the graph has to be re-exported. Reading the
        shape back means a mismatched setting degrades to a warning instead of an
        ONNX Runtime shape error on the first frame.
        """
        shape = self._session.get_inputs()[0].shape  # type: ignore[union-attr]
        height, width = (shape[-2], shape[-1]) if len(shape) >= 2 else (None, None)
        if isinstance(height, int) and isinstance(width, int) and height == width:
            if height != self._config.vehicle_imgsz:
                log_event(
                    _LOG,
                    30,
                    "vehicle_imgsz_from_graph",
                    graph_imgsz=height,
                    configured_imgsz=self._config.vehicle_imgsz,
                    hint="re-export the model to change the inference size",
                )
            return height
        return self._config.vehicle_imgsz

    def load(self) -> None:
        """Create the ONNX Runtime session (no download happens here)."""
        import onnxruntime as ort

        weights = self._config.vehicle_weights
        started = time.perf_counter()
        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"vehicle weights not found: {weights} "
                "(bake them with scripts/download_models.py or rebuild the image)"
            )

        options = ort.SessionOptions()
        options.log_severity_level = 3  # keep ORT's own warnings off the JSON stream
        options.intra_op_num_threads = max(1, int(self._config.vehicle_threads))
        self._session = ort.InferenceSession(
            weights, options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._imgsz = self._resolve_imgsz()
        self._class_names = _class_names_from_metadata(self._session)

        wanted = {name.lower() for name in self._config.vehicle_classes}
        if wanted and wanted != _DEFAULT_CLASS_NAMES:
            self._class_ids = [
                class_id for class_id, name in self._class_names.items() if name.lower() in wanted
            ]
            if not self._class_ids:
                raise ValueError(
                    f"None of VEHICLE_CLASSES={sorted(wanted)} exist in the model "
                    f"(available: {sorted(self._class_names.values())})"
                )
        else:
            self._class_ids = [
                class_id
                for class_id, name in self._class_names.items()
                if name.lower() in _DEFAULT_CLASS_NAMES
            ]

        log_event(
            _LOG,
            20,
            "vehicle_model_loaded",
            weights=weights,
            weights_present=True,
            format="onnx",
            classes={self._class_names[class_id]: class_id for class_id in self._class_ids},
            imgsz=self._imgsz,
            conf_threshold=self._config.vehicle_conf_threshold,
            iou_threshold=self._config.vehicle_iou,
            providers=self._session.get_providers(),
            load_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def detect(self, frame: np.ndarray) -> list[VehicleBox]:
        """Return vehicle boxes for ``frame``, in full-frame coordinates."""
        if self._session is None:
            self.load()

        blob, scale, pad_x, pad_y = letterbox(frame, self._imgsz)
        outputs = self._session.run(None, {self._input_name: blob})
        return decode_detections(
            outputs[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            frame_shape=frame.shape[:2],
            class_ids=self._class_ids,
            class_names=self._class_names,
            conf_threshold=self._config.vehicle_conf_threshold,
            iou_threshold=self._config.vehicle_iou,
            max_det=self._config.vehicle_max_det,
            min_box_px=self._config.vehicle_min_box_px,
            num_classes=len(self._class_names) or None,
        )

    def warmup(self, size: int = 320, rounds: int = 2) -> None:
        """Run a couple of dummy inferences so the first real frame is not slow."""
        if self._session is None:
            self.load()
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        for _ in range(max(0, rounds)):
            self.detect(dummy)
        log_event(
            _LOG,
            20,
            "vehicle_model_warm",
            rounds=rounds,
            size=size,
            classes={self._class_names[class_id]: class_id for class_id in self._class_ids},
        )
