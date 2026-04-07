"""Angular tolerance model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.tolerance.base import Tolerance


@dataclass(frozen=True)
class AngularTolerance(Tolerance):
    """Angular variation converted into linear displacement at a lever arm."""

    sigma_theta: float
    lever_arm: float
    mean_theta: float = 0.0
    axis_angle: float = 0.0

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """Sample angular error and convert to tangential linear displacement."""
        self._validate_n_samples(n_samples)
        if self.lever_arm < 0.0:
            raise ValueError("lever_arm must be non-negative.")

        theta_error = np.random.normal(
            loc=self.mean_theta,
            scale=self.sigma_theta,
            size=n_samples,
        )
        linear_magnitude = self.lever_arm * theta_error

        x = linear_magnitude * np.cos(self.axis_angle)
        y = linear_magnitude * np.sin(self.axis_angle)
        return np.column_stack((x, y)).astype(np.float64)
