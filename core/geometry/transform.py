"""2D transformation utilities for geometry operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.geometry.vector import Vector2D


@dataclass(frozen=True)
class Transform2D:
    """Rigid 2D transform with rotation and translation."""

    theta: float
    tx: float
    ty: float

    def rotation_matrix(self) -> NDArray[np.float64]:
        """Return the 2x2 rotation matrix for ``theta`` radians."""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        return np.array(
            [[cos_t, -sin_t], [sin_t, cos_t]],
            dtype=np.float64,
        )

    def translation_vector(self) -> NDArray[np.float64]:
        """Return the translation vector as a NumPy array."""
        return np.array([self.tx, self.ty], dtype=np.float64)

    def apply(self, vector: Vector2D) -> Vector2D:
        """Apply rotation and translation to a ``Vector2D``."""
        rotated = self.rotation_matrix() @ vector.to_numpy()
        transformed = rotated + self.translation_vector()
        return Vector2D(float(transformed[0]), float(transformed[1]))
