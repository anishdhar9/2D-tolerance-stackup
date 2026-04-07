"""Statistical utilities for 2D Monte Carlo point clouds."""

from __future__ import annotations

import numpy as np

from analysis._validation import PointArray, validate_points_2d


def mean_2d(points: PointArray) -> PointArray:
    """Return the 2D mean vector for points of shape ``(n_samples, 2)``."""
    validate_points_2d(points)
    return np.mean(points, axis=0)


def covariance_2d(points: PointArray) -> PointArray:
    """Return the 2x2 sample covariance matrix for points of shape ``(n_samples, 2)``."""
    validate_points_2d(points)
    if points.shape[0] < 2:
        raise ValueError("At least two points are required to compute covariance.")
    return np.cov(points, rowvar=False)
from numpy.typing import NDArray


def mean_2d(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 2D mean vector for points of shape ``(n_samples, 2)``."""
    _validate_points(points)
    return np.mean(points, axis=0)


def covariance_2d(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 2x2 sample covariance matrix for points of shape ``(n_samples, 2)``."""
    _validate_points(points)
    if points.shape[0] < 2:
        raise ValueError("At least two points are required to compute covariance.")
    return np.cov(points, rowvar=False)


def _validate_points(points: NDArray[np.float64]) -> None:
    """Validate that points are a 2D array with shape ``(n_samples, 2)``."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2).")
