"""Failure probability estimators for radial tolerance checks."""

from __future__ import annotations

import numpy as np

from analysis._validation import PointArray, validate_points_2d


def failure_probability(points: PointArray, radius: float) -> float:
    """Return probability that sampled points exceed the given radial limit."""
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")

    validate_points_2d(points)
    radial_distance = np.linalg.norm(points, axis=1)
    return float(np.mean(radial_distance > radius))
