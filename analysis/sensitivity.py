"""Sensitivity analysis utilities for feature-importance estimation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from analysis._validation import PointArray, validate_points_2d


def feature_importance_by_variance(
    feature_points: Sequence[PointArray],
    total_points: PointArray,
) -> list[tuple[int, float]]:
    """Return feature importance ranking based on contribution to output variance.

    Importance for each feature is computed as::

        trace(cov(feature_i)) / trace(cov(total_output))

    The function returns a list of ``(feature_index, importance)`` sorted
    from highest to lowest contribution.
    """
    validate_points_2d(total_points)
    if total_points.shape[0] < 2:
        raise ValueError("At least two samples are required for variance-based importance.")
    if not feature_points:
        raise ValueError("feature_points must contain at least one feature sample array.")

    total_variance = float(np.trace(np.cov(total_points, rowvar=False)))
    if total_variance <= 0.0:
        return [(index, 0.0) for index in range(len(feature_points))]

    ranking: list[tuple[int, float]] = []
    for index, points in enumerate(feature_points):
        validate_points_2d(points)
        if points.shape[0] != total_points.shape[0]:
            raise ValueError("Each feature sample array must match total_points sample count.")

        feature_variance = float(np.trace(np.cov(points, rowvar=False)))
        ranking.append((index, feature_variance / total_variance))

    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking
