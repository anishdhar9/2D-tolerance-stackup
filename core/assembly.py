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

    def sample_features(self, n_samples: int) -> list[NDArray[np.float64]]:
        """Return sampled point clouds for each feature."""
        return [feature.sample(n_samples) for feature in self.features]

    def sample_total(self, n_samples: int) -> NDArray[np.float64]:
        """Return sampled total assembly positions with shape ``(n_samples, 2)``."""
        feature_samples = self.sample_features(n_samples)
        return np.sum(np.stack(feature_samples, axis=0), axis=0).astype(np.float64)

    def simulate(self) -> NDArray[np.float64]:
        """Return one sampled 2D position from all features in the assembly."""
        return self.sample_total(1)[0]
