"""Validation helpers shared by analysis utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


PointArray = NDArray[np.float64]


def validate_points_2d(points: PointArray) -> None:
    """Validate points array has shape ``(n_samples, 2)`` with at least one sample."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2).")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one sample.")
