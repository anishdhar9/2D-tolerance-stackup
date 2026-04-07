"""Failure probability estimators for radial tolerance checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def failure_probability(points: NDArray[np.float64], radius: float) -> float:
    """Return probability that sampled points exceed the given radial limit."""
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2).")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one sample.")

    radial_distance = np.linalg.norm(points, axis=1)
    return float(np.mean(radial_distance > radius))
