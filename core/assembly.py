"""Assembly and feature domain models for 2D tolerance simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.tolerance.base import Tolerance


@dataclass(frozen=True)
class Feature:
    """A geometric feature with nominal 2D position and tolerance model."""

    nominal: NDArray[np.float64]
    tolerance: Tolerance

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """Sample feature positions by adding tolerance offsets to nominal position."""
        offsets = self.tolerance.sample(n_samples)
        return offsets + self.nominal


@dataclass(frozen=True)
class Assembly:
    """Collection of features whose summed positions define the assembly response."""

    features: tuple[Feature, ...]

    def __post_init__(self) -> None:
        """Validate that at least one feature exists."""
        if not self.features:
            raise ValueError("Assembly requires at least one feature.")

    def simulate(self) -> NDArray[np.float64]:
        """Return one sampled 2D position from all features in the assembly."""
        sampled = [feature.sample(1)[0] for feature in self.features]
        return np.sum(np.vstack(sampled), axis=0).astype(np.float64)
