#!/usr/bin/env python3
"""Generate a demo/testing clip so the container can be run without real footage.

Two modes:

1. ``--source-image photo.jpg`` (default when the file exists)
   Slides a slow zoom/pan across a real photo of a car. The vehicles and the
   plate stay realistic, so OCR actually has something to read.

2. no source image
   Draws a synthetic road scene with a moving vehicle and a rendered plate.

Both modes write a small H.264-friendly ``mp4v`` file that OpenCV can decode:

    python scripts/make_test_video.py --output data/cam01.mp4 --seconds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "cam01.mp4"
FALLBACK_IMAGE = ROOT / "assets" / "sample_car.jpg"


def _writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open a writer for {path}")
    return writer


def from_image(image_path: Path, output: Path, seconds: float, fps: float) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"could not read image: {image_path}")
    height, width = image.shape[:2]
    writer = _writer(output, (width, height), fps)
    frames = int(seconds * fps)

    for index in range(frames):
        phase = index / max(frames - 1, 1)
        # gentle zoom (1.0 -> 1.15) plus a small pan keeps every frame sharp
        # enough for detection while making the clip look like motion.
        scale = 1.0 + 0.15 * phase
        new_w, new_h = int(width * scale), int(height * scale)
        offset_x = int((new_w - width) * (0.25 + 0.5 * phase))
        offset_y = int((new_h - height) * 0.4)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = resized[offset_y : offset_y + height, offset_x : offset_x + width]
        writer.write(canvas if canvas.shape[:2] == (height, width) else image)
    writer.release()
    print(f"wrote {output} ({frames} frames, {width}x{height} @ {fps:.0f}fps) from {image_path}")


def synthetic(output: Path, seconds: float, fps: float, plate: str) -> None:
    width, height = 1280, 720
    writer = _writer(output, (width, height), fps)
    frames = int(seconds * fps)
    rng = np.random.default_rng(7)

    for index in range(frames):
        image = np.full((height, width, 3), 60, dtype=np.uint8)
        cv2.rectangle(image, (0, 430), (width, height), (90, 90, 95), -1)
        for x in range(0, width, 90):  # lane markings
            shift = int((index * 4) % 90)
            cv2.line(image, (x + shift, 560), (x + shift + 45, 560), (200, 200, 200), 4)

        car_w, car_h = 300, 170
        left = int((index / fps) * 120) % (width + car_w) - car_w
        top = 400 - car_h
        cv2.rectangle(image, (left, top), (left + car_w, top + car_h), (40, 40, 160), -1)
        cv2.rectangle(image, (left + 40, top + 90), (left + car_w - 40, top + car_h - 15), (25, 25, 25), -1)

        plate_w = 210
        plate_left = left + (car_w - plate_w) // 2
        plate_top = top + car_h - 70
        cv2.rectangle(
            image, (plate_left, plate_top), (plate_left + plate_w, plate_top + 60), (235, 235, 235), -1
        )
        cv2.rectangle(
            image, (plate_left, plate_top), (plate_left + plate_w, plate_top + 60), (20, 20, 20), 3
        )
        cv2.putText(
            image,
            plate,
            (plate_left + 22, plate_top + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        noise = rng.integers(0, 12, size=image.shape, dtype=np.uint8)
        writer.write(cv2.add(image, noise))
    writer.release()
    print(f"wrote {output} ({frames} frames, {width}x{height} @ {fps:.0f}fps, plate {plate})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-image", help="photo of a car to pan/zoom across")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--plate", default="KA01AB1234", help="synthetic-scene plate text")
    args = parser.parse_args()

    output = Path(args.output)
    source = Path(args.source_image) if args.source_image else (FALLBACK_IMAGE if FALLBACK_IMAGE.is_file() else None)
    if source:
        from_image(source, output, args.seconds, args.fps)
    else:
        synthetic(output, args.seconds, args.fps, args.plate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
