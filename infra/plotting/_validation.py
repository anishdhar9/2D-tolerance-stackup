"""Validation helpers for plotting adapters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


PointArray = NDArray[np.float64]


def validate_points_2d(points: PointArray) -> None:
    """Validate points shape is ``(n_samples, 2)`` for plotting."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2).")
