"""Monte Carlo simulation orchestration for assembly sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Assembly(Protocol):
    """Protocol for assemblies that can produce one simulated 2D point."""

    def simulate(self) -> NDArray[np.float64]:
        """Return a single simulated 2D point as an array-like shape ``(2,)``."""


@dataclass(frozen=True)
class MonteCarloSimulator:
    """Simple Monte Carlo simulator that samples points from an assembly."""

    assembly: Assembly

    def run(self, n_samples: int) -> NDArray[np.float64]:
        """Run Monte Carlo sampling and return points with shape ``(n_samples, 2)``."""
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        samples = [np.asarray(self.assembly.simulate(), dtype=np.float64) for _ in range(n_samples)]
        points = np.vstack(samples)

        if points.shape != (n_samples, 2):
            raise ValueError(
                "assembly.simulate() must return a 2D point with shape (2,) for each sample."
            )

        return points
