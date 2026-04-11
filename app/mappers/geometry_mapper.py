"""Pure mapping layer: canvas geometry -> backend Feature objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from core.assembly import Feature
from core.tolerance.linear import LinearTolerance
from core.tolerance.position import CircularTolerance

POINT_RADIUS_THRESHOLD = 4.5


GeometryType = Literal["point", "circle", "line", "unknown"]


@dataclass(frozen=True)
class GeometryPrimitive:
    """Normalized geometry primitive independent of UI frameworks."""

    type: GeometryType
    x: float
    y: float
    radius: float | None = None


def parse_geometry_primitives(canvas_objects: Sequence[Mapping[str, Any]]) -> list[GeometryPrimitive]:
    """Normalize raw canvas object dictionaries into geometry primitives.

    Supported inputs:
    - point (represented by small circles)
    - circle
    - line
    """
    primitives: list[GeometryPrimitive] = []

    for obj in canvas_objects:
        raw_type = str(obj.get("type", "unknown"))
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))

        if raw_type == "circle":
            radius = float(obj.get("radius", 0.0))
            geometry_type: GeometryType = "point" if radius <= POINT_RADIUS_THRESHOLD else "circle"
            primitives.append(
                GeometryPrimitive(
                    type=geometry_type,
                    x=left + radius,
                    y=top + radius,
                    radius=radius if geometry_type == "circle" else None,
                )
            )
            continue

        if raw_type == "line":
            x1 = float(obj.get("x1", 0.0))
            y1 = float(obj.get("y1", 0.0))
            x2 = float(obj.get("x2", 0.0))
            y2 = float(obj.get("y2", 0.0))
            primitives.append(
                GeometryPrimitive(
                    type="line",
                    x=left + ((x1 + x2) / 2.0),
                    y=top + ((y1 + y2) / 2.0),
                    radius=None,
                )
            )
            continue

        primitives.append(GeometryPrimitive(type="unknown", x=left, y=top, radius=None))

    return primitives


def map_geometry_to_features(canvas_objects: Sequence[Mapping[str, Any]]) -> list[Feature]:
    """Map canvas objects to backend Feature objects.

    Rules:
    - circle -> position tolerance feature (CircularTolerance using radius)
    - point  -> nominal feature (zero linear tolerance)
    - line   -> alignment reference basic (midpoint + zero linear tolerance)
    """
    features: list[Feature] = []

    for primitive in parse_geometry_primitives(canvas_objects):
        nominal = np.array([primitive.x, primitive.y], dtype=np.float64)

        if primitive.type == "circle":
            tolerance = CircularTolerance(radius=float(primitive.radius or 0.0))
            features.append(Feature(nominal=nominal, tolerance=tolerance))
            continue

        if primitive.type in {"point", "line"}:
            tolerance = LinearTolerance(sigma_x=0.0, sigma_y=0.0)
            features.append(Feature(nominal=nominal, tolerance=tolerance))
            continue

    return features
