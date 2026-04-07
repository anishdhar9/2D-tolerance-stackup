"""Linear tolerance model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.tolerance.base import Tolerance


@dataclass(frozen=True)
class LinearTolerance(Tolerance):
    """Gaussian linear tolerance in x and y dimensions."""

    sigma_x: float
    sigma_y: float
    mean_x: float = 0.0
    mean_y: float = 0.0

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """Sample x/y displacements from independent Gaussian distributions."""
        self._validate_n_samples(n_samples)
        x = np.random.normal(loc=self.mean_x, scale=self.sigma_x, size=n_samples)
        y = np.random.normal(loc=self.mean_y, scale=self.sigma_y, size=n_samples)
        return np.column_stack((x, y)).astype(np.float64)
