"""Vector primitives for 2D geometric calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Vector2D:
    """Immutable 2D vector backed by a NumPy array."""

    _components: NDArray[np.float64]

    def __init__(self, x: float, y: float) -> None:
        """Create a 2D vector from ``x`` and ``y`` coordinates."""
        components = np.array([x, y], dtype=np.float64)
        object.__setattr__(self, "_components", components)

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Vector2D":
        """Build a vector from an iterable with exactly two numeric values."""
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError("Vector2D requires exactly two values.")
        return cls(float(arr[0]), float(arr[1]))

    @property
    def x(self) -> float:
        """Return the x-component."""
        return float(self._components[0])

    @property
    def y(self) -> float:
        """Return the y-component."""
        return float(self._components[1])

    def to_numpy(self) -> NDArray[np.float64]:
        """Return a copy of the vector as a NumPy array."""
        return self._components.copy()

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Return vector addition with another ``Vector2D``."""
        summed = self._components + other._components
        return Vector2D(float(summed[0]), float(summed[1]))

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Return vector subtraction with another ``Vector2D``."""
        diff = self._components - other._components
        return Vector2D(float(diff[0]), float(diff[1]))

    def scale(self, factor: float) -> "Vector2D":
        """Return a new vector scaled by ``factor``."""
        scaled = self._components * factor
        return Vector2D(float(scaled[0]), float(scaled[1]))

    def dot(self, other: "Vector2D") -> float:
        """Return the dot product with another ``Vector2D``."""
        return float(np.dot(self._components, other._components))

    def magnitude(self) -> float:
        """Return the Euclidean norm of the vector."""
        return float(np.linalg.norm(self._components))
