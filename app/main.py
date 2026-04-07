"""Streamlit UI orchestration for the 2D Tolerance Stack-Up Simulator."""

from __future__ import annotations

from typing import List

import numpy as np
import streamlit as st

from analysis.failure import failure_probability
from analysis.sensitivity import feature_importance_by_variance
from analysis.statistics import mean_2d
from core.assembly import Assembly, Feature
from core.simulation import MonteCarloSimulator
from core.tolerance.linear import LinearTolerance
from core.tolerance.position import CircularTolerance
from infra.plotting.scatter import add_target_circle, scatter_points


def _build_feature(index: int) -> Feature:
    """Build one feature from Streamlit inputs."""
    st.subheader(f"Feature {index + 1}")
    col1, col2 = st.columns(2)
    nominal_x = col1.number_input(f"Nominal X #{index + 1}", value=0.0, key=f"nx_{index}")
    nominal_y = col2.number_input(f"Nominal Y #{index + 1}", value=0.0, key=f"ny_{index}")

    tol_type = st.selectbox(
        f"Tolerance Type #{index + 1}",
        options=["linear", "circular"],
        key=f"tol_type_{index}",
    )

    if tol_type == "linear":
        col3, col4 = st.columns(2)
        sigma_x = col3.number_input(f"Sigma X #{index + 1}", min_value=0.0, value=0.1, key=f"sx_{index}")
        sigma_y = col4.number_input(f"Sigma Y #{index + 1}", min_value=0.0, value=0.1, key=f"sy_{index}")
        tolerance = LinearTolerance(sigma_x=sigma_x, sigma_y=sigma_y)
    else:
        radius = st.number_input(f"Radius #{index + 1}", min_value=0.0, value=0.1, key=f"r_{index}")
        tolerance = CircularTolerance(radius=radius)

    nominal = np.array([nominal_x, nominal_y], dtype=np.float64)
    return Feature(nominal=nominal, tolerance=tolerance)


def main() -> None:
    """Render UI and orchestrate simulation workflow."""
    st.title("2D Tolerance Stack-Up Simulator")
    st.caption("UI orchestrates core domain + analysis services only.")

    n_features = st.number_input("Number of Features", min_value=1, max_value=20, value=2, step=1)
    n_samples = st.number_input("Monte Carlo Samples", min_value=100, max_value=500000, value=5000, step=100)
    failure_radius = st.number_input("Failure Radius", min_value=0.0, value=1.0, step=0.1)

    features: List[Feature] = []
    for idx in range(int(n_features)):
        features.append(_build_feature(idx))

    if st.button("Run Simulation", type="primary"):
        assembly = Assembly(features=tuple(features))
        simulator = MonteCarloSimulator(assembly=assembly)
        points = simulator.run(int(n_samples))
        feature_samples = assembly.sample_features(int(n_samples))

        mean_pos = mean_2d(points)
        fail_prob = failure_probability(points, radius=float(failure_radius))
        importance = feature_importance_by_variance(feature_samples, points)

        fig = scatter_points(points, title="Monte Carlo Point Cloud")
        fig = add_target_circle(fig, radius=float(failure_radius))

        st.plotly_chart(fig, use_container_width=True)
        st.metric("Failure Probability", f"{fail_prob:.4f}")
        st.metric("Mean Position", f"({mean_pos[0]:.4f}, {mean_pos[1]:.4f})")

        st.subheader("Feature Importance (Variance Contribution)")
        ranking_rows = [
            {"Feature": f"Feature {index + 1}", "Contribution": contribution}
            for index, contribution in importance
        ]
        st.table(ranking_rows)


if __name__ == "__main__":
    main()
