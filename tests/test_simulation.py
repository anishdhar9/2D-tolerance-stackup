"""Basic tests for tolerance sampling, assembly simulation, and failure metrics."""

from __future__ import annotations

import numpy as np

from analysis.failure import failure_probability
from core.assembly import Assembly, Feature
from core.simulation import MonteCarloSimulator
from core.tolerance.linear import LinearTolerance
from core.tolerance.position import CircularTolerance


def test_tolerance_sampling_shape() -> None:
    """Tolerance models should return arrays with shape (n_samples, 2)."""
    n_samples = 128

    linear = LinearTolerance(sigma_x=0.1, sigma_y=0.2)
    circular = CircularTolerance(radius=0.5)

    linear_samples = linear.sample(n_samples)
    circular_samples = circular.sample(n_samples)

    assert linear_samples.shape == (n_samples, 2)
    assert circular_samples.shape == (n_samples, 2)


def test_assembly_simulation_output_shape() -> None:
    """Monte Carlo simulation should return shape (n_samples, 2)."""
    feature_a = Feature(
        nominal=np.array([1.0, 2.0], dtype=np.float64),
        tolerance=LinearTolerance(sigma_x=0.0, sigma_y=0.0),
    )
    feature_b = Feature(
        nominal=np.array([3.0, -1.0], dtype=np.float64),
        tolerance=LinearTolerance(sigma_x=0.0, sigma_y=0.0),
    )

    assembly = Assembly(features=(feature_a, feature_b))
    simulator = MonteCarloSimulator(assembly=assembly)

    points = simulator.run(10)

    assert points.shape == (10, 2)
    assert np.allclose(points, np.array([[4.0, 1.0]] * 10, dtype=np.float64))


def test_failure_probability_correctness() -> None:
    """Failure probability should match expected exceedance ratio."""
    points = np.array(
        [
            [0.0, 0.0],  # inside r=1
            [1.0, 0.0],  # on boundary (not failure)
            [2.0, 0.0],  # outside
            [0.0, -2.0],  # outside
        ],
        dtype=np.float64,
    )

    probability = failure_probability(points, radius=1.0)

    assert probability == 0.5
