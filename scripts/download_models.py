#!/usr/bin/env python3
"""Bake every model the edge container needs.

The vehicle model is a **build input**: this script does not fetch or train it.
Drop your own YOLO26s export at ``models/yolo26s.onnx`` (committed to the repo)
and the Dockerfile copies it into the image. The script's job is to

* fail loudly, with instructions, if that file is missing or unusable;
* verify the graph really is what the runtime expects (input shape, output
  layout, COCO label metadata) and run one dummy inference;
* warm the plate detector and plate OCR caches so the container is fully offline
  once started.

    python scripts/download_models.py --models-dir models

Fetching the two plate models needs ``fast-alpr[onnx]`` (i.e.
``requirements.txt``) and network access. Nothing here needs torch.

If you would rather export an Ultralytics checkpoint than upload a graph, that is
an explicit opt-in and needs the build extras:

    pip install -r requirements-build.txt
    python scripts/download_models.py --export-vehicle yolo26s
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_VEHICLE = "yolo26s"
DEFAULT_PLATE = "yolo-v9-t-416-license-plate-end2end"
DEFAULT_OCR = "cct-s-v2-global-model"
DEFAULT_IMGSZ = 640
DEFAULT_OPSET = 17

_MISSING_VEHICLE = """\
no vehicle model at {path}

The vehicle stage is built from an ONNX graph you supply, not from a checkpoint
this script downloads. Put your YOLO26s export there, then re-run:

    models/yolo26s.onnx

Any file name works as long as VEHICLE_WEIGHTS points at it. To generate one from
an Ultralytics checkpoint instead (needs torch + ultralytics + onnx):

    pip install -r requirements-build.txt
    python scripts/download_models.py --export-vehicle yolo26s
"""


def require_vehicle_onnx(name: str, models_dir: Path) -> Path:
    """Return the supplied vehicle graph, or explain how to supply one."""
    target = models_dir / f"{name}.onnx"
    if not target.is_file():
        raise SystemExit(_MISSING_VEHICLE.format(path=target))
    return target


def verify_vehicle_onnx(path: Path) -> dict:
    """Check the graph matches what the runtime detector assumes.

    Reports the input geometry, the output layout and whether the COCO label map
    is embedded. A mismatch here is a build failure, not a surprise at 3am on a
    camera node.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model_in = session.get_inputs()[0]
    model_out = session.get_outputs()[0]
    shape = list(model_in.shape)
    meta = session.get_modelmeta().custom_metadata_map
    names = meta.get("names", "")

    height, width = (shape[-2], shape[-1]) if len(shape) >= 2 else (None, None)
    static_square = isinstance(height, int) and height == width
    if not static_square:
        print(
            f"[warn] {path.name}: input shape {shape} is not a fixed square, so the "
            "detector will letterbox to VEHICLE_IMGSZ at runtime instead"
        )

    output = list(model_out.shape)
    if len(output) == 3 and output[-1] is not None:
        print(f"[ok] {path.name}: input {model_in.name} {shape} -> output {output}")
    else:
        print(
            f"[warn] {path.name}: output {output} is not the expected "
            "(1, 4 + num_classes, num_anchors) layout; see the vehicle_detector docs"
        )

    if names:
        count = len([pair for pair in names.replace("{", "").replace("}", "").split(",") if ":" in pair])
        print(f"[ok] {path.name}: {count} class names embedded in the graph metadata")
    else:
        print(
            f"[warn] {path.name}: no 'names' metadata, so the detector cannot map class ids to "
            "names and will refuse to start - re-export the graph with ultralytics, which "
            "embeds the label map"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    blob = np.zeros((1, 3, height or 640, width or 640), dtype=np.float32)
    session.run(None, {model_in.name: blob})
    print(f"[ok] {path.name}: {size_mb:.1f} MB, dummy inference passed")
    return {
        "file": path.name,
        "size_mb": round(size_mb, 1),
        "input_name": model_in.name,
        "input_shape": shape,
        "input_type": model_in.type,
        "output_name": model_out.name,
        "output_shape": output,
        "has_names": bool(names),
        "static_square": static_square,
    }


def export_vehicle_onnx(name: str, models_dir: Path, imgsz: int, opset: int) -> Path:
    """Opt-in: export an Ultralytics checkpoint to ONNX, keeping no ``.pt``."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the host
        raise SystemExit(
            "exporting needs the build extras: pip install -r requirements-build.txt"
        ) from exc

    target = models_dir / f"{name}.onnx"
    scratch = models_dir / ".export"
    scratch.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(scratch)
        produced = YOLO(f"{name}.pt", task="detect").export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=False,
            opset=opset,
            device="cpu",
            verbose=False,
        )
        # Ultralytics exports next to the process CWD, which is the scratch dir.
        exported = Path(produced)
        if not exported.is_absolute():
            exported = (scratch / exported).resolve()
    finally:
        os.chdir(previous_cwd)

    if not exported.is_file():
        found = sorted(scratch.glob("*.onnx"))
        if len(found) == 1:
            exported = found[0]
        else:
            raise SystemExit(f"ultralytics did not produce an ONNX file for {name!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), target)
    for leftover in scratch.glob("*"):
        leftover.unlink() if leftover.is_file() else shutil.rmtree(leftover, ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[ok] exported {target} (imgsz={imgsz}, opset={opset})")
    return target


def fetch_plate_models(plate_model: str, ocr_model: str) -> None:
    """Warm both ONNX model caches (``~/.cache/open-image-models``, ``~/.cache/fast-plate-ocr``)."""
    from fast_alpr import ALPR

    alpr = ALPR(
        detector_model=plate_model,
        detector_conf_thresh=0.3,
        ocr_model=ocr_model,
        ocr_device="cpu",
    )
    print("[ok] plate detector + OCR ready")
    return alpr


def disable_ultralytics_telemetry() -> None:
    """Stop Ultralytics from syncing settings/analytics on first use (export only)."""
    try:
        from ultralytics import SETTINGS

        SETTINGS.update({"sync": False})
        print("[ok] ultralytics telemetry disabled")
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[warn] could not disable ultralytics telemetry: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--vehicle", default=DEFAULT_VEHICLE, help="vehicle graph stem (default: yolo26s)")
    parser.add_argument(
        "--export-vehicle",
        metavar="CHECKPOINT",
        default=None,
        help="instead of using models/<name>.onnx, export this Ultralytics checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="export size (--export-vehicle only)")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="export opset (--export-vehicle only)")
    parser.add_argument("--plate", default=DEFAULT_PLATE)
    parser.add_argument("--ocr", default=DEFAULT_OCR)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"models dir : {models_dir}")
    print(f"cache dir  : {Path.home() / '.cache'}")
    print(json.dumps({"vehicle": args.vehicle, "plate": args.plate, "ocr": args.ocr}))

    if args.export_vehicle:
        disable_ultralytics_telemetry()
        vehicle = export_vehicle_onnx(args.export_vehicle, models_dir, args.imgsz, args.opset)
    else:
        vehicle = require_vehicle_onnx(args.vehicle, models_dir)
        print(f"[ok] using supplied vehicle model: {vehicle}")

    info = verify_vehicle_onnx(vehicle)
    print(json.dumps(info))
    fetch_plate_models(args.plate, args.ocr)
    print("all models baked - the container can now run fully offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
