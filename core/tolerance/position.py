"""Position tolerance model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.tolerance.base import Tolerance


@dataclass(frozen=True)
class CircularTolerance(Tolerance):
    """Uniformly distributed displacement inside a circular tolerance zone."""

    radius: float
    center_x: float = 0.0
    center_y: float = 0.0

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """Sample points uniformly within a circle and return ``(n_samples, 2)``."""
        self._validate_n_samples(n_samples)
        if self.radius < 0.0:
            raise ValueError("radius must be non-negative.")

        angles = np.random.uniform(0.0, 2.0 * np.pi, size=n_samples)
        radii = self.radius * np.sqrt(np.random.uniform(0.0, 1.0, size=n_samples))

        x = self.center_x + radii * np.cos(angles)
        y = self.center_y + radii * np.sin(angles)
        return np.column_stack((x, y)).astype(np.float64)
