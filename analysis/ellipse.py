"""Confidence ellipse derivation utilities for 2D distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2


@dataclass(frozen=True)
class EllipseParameters:
    """Geometric parameters of a 2D confidence ellipse."""

    center: NDArray[np.float64]
    major_axis: float
    minor_axis: float
    angle: float


def confidence_ellipse_from_covariance(
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
    confidence: float = 0.95,
) -> EllipseParameters:
    """Compute confidence ellipse parameters from 2D mean and covariance."""
    if mean.shape != (2,):
        raise ValueError("mean must have shape (2,).")
    if covariance.shape != (2, 2):
        raise ValueError("covariance must have shape (2, 2).")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1.")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scale = float(np.sqrt(chi2.ppf(confidence, df=2)))
    major_axis = float(scale * np.sqrt(max(eigenvalues[0], 0.0)))
    minor_axis = float(scale * np.sqrt(max(eigenvalues[1], 0.0)))

    direction = eigenvectors[:, 0]
    angle = float(np.arctan2(direction[1], direction[0]))

    return EllipseParameters(
        center=mean.astype(np.float64),
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle=angle,
    )
