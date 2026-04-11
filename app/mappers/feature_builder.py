"""Feature construction logic from normalized geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.mappers.geometry_mapper import GeometryPrimitive
from core.assembly import Feature
from core.tolerance.linear import LinearTolerance
from core.tolerance.position import CircularTolerance


@dataclass(frozen=True)
class FeatureEdit:
    """Editable feature values from UI controls."""

    feature_type: str
    x: float
    y: float
    tolerance: float


def primitive_to_feature(primitive: GeometryPrimitive, *, tolerance: float) -> Feature:
    """Convert one primitive into a backend feature with tolerance semantics."""
    nominal = np.array([primitive.x, primitive.y], dtype=np.float64)
    tol = max(0.0, float(tolerance))

    if primitive.type == "circle":
        return Feature(nominal=nominal, tolerance=CircularTolerance(radius=tol))
    return Feature(nominal=nominal, tolerance=LinearTolerance(sigma_x=tol, sigma_y=tol))


def build_features_from_primitives(
    primitives: list[GeometryPrimitive],
    *,
    selected_idx: int | None,
    selected_edit: FeatureEdit | None,
    tolerance_factor: float,
) -> tuple[list[Feature], list[tuple[float, float]]]:
    """Build features and overlay points, applying selected-feature edits when present."""
    features: list[Feature] = []
    overlay: list[tuple[float, float]] = []

    for idx, primitive in enumerate(primitives):
        current = primitive
        tolerance = primitive.radius if primitive.type == "circle" and primitive.radius is not None else 0.0

        if selected_idx is not None and selected_edit is not None and idx == selected_idx:
            current = GeometryPrimitive(type=selected_edit.feature_type, x=selected_edit.x, y=selected_edit.y, radius=primitive.radius)
            tolerance = selected_edit.tolerance

        tolerance *= float(tolerance_factor)
        features.append(primitive_to_feature(current, tolerance=tolerance))
        overlay.append((current.x, current.y))

    return features, overlay
