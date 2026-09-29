"""Model wrappers for the ANPR edge pipeline."""

from __future__ import annotations

from .plate_reader import PlateReader, PlateReading, normalise_plate
from .vehicle_detector import VehicleBox, VehicleDetector

__all__ = [
    "PlateReader",
    "PlateReading",
    "VehicleBox",
    "VehicleDetector",
    "normalise_plate",
]
