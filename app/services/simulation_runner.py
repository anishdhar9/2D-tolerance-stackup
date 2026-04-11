"""Simulation orchestration service (non-UI)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.failure import failure_probability
from analysis.statistics import mean_2d
from core.assembly import Assembly, Feature
from core.simulation import MonteCarloSimulator


@dataclass(frozen=True)
class SimulationResult:
    """Structured simulation outputs for UI consumption."""

    points: NDArray[np.float64]
    mean: NDArray[np.float64]
    failure_probability: float
    failure_radius: float
    covariance_magnitude: float
    samples: int


def covariance_magnitude(points: NDArray[np.float64]) -> float:
    """Compute Frobenius norm of covariance matrix as scalar spread indicator."""
    covariance = np.cov(points, rowvar=False)
    return float(np.linalg.norm(covariance, ord="fro"))


def run_simulation(features: list[Feature], *, n_samples: int, failure_radius: float) -> SimulationResult:
    """Run Monte Carlo tolerance simulation for provided features."""
    assembly = Assembly(features=tuple(features))
    simulator = MonteCarloSimulator(assembly=assembly)
    points = simulator.run(int(n_samples))

    return SimulationResult(
        points=points,
        mean=mean_2d(points),
        failure_probability=failure_probability(points, radius=float(failure_radius)),
        failure_radius=float(failure_radius),
        covariance_magnitude=covariance_magnitude(points),
        samples=int(n_samples),
    )
