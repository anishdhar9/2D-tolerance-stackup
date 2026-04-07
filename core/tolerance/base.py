"""Base abstractions for tolerance models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from core.validation import validate_positive_int


class Tolerance(ABC):
    """Abstract base class for 2D tolerance models."""

    @abstractmethod
    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """Generate sampled 2D displacements with shape ``(n_samples, 2)``."""

    @staticmethod
    def _validate_n_samples(n_samples: int) -> None:
        """Validate that ``n_samples`` is a positive integer."""
        validate_positive_int(n_samples, name="n_samples")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
