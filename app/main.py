"""Interactive geometry drawing app (UI-only, no simulation orchestration)."""

from __future__ import annotations

import streamlit as st

from analysis.failure import failure_probability
from analysis.statistics import mean_2d
from app.ui.feature_mapper import build_ui_feature_specs, to_domain_features
from app.ui.geometry_canvas import render_geometry_canvas
from core.assembly import Assembly
from core.simulation import MonteCarloSimulator
from infra.plotting.scatter import add_target_circle, scatter_points


def main() -> None:
    """Render interactive UI and orchestrate simulation workflow."""
    st.set_page_config(page_title="2D Tolerance Stack-Up", layout="wide")
    st.title("2D Tolerance Stack-Up Simulator")
    st.caption("Interactive geometry UI maps user-drawn features into the unchanged simulation backend.")

    with st.sidebar:
        st.header("Simulation Settings")
        n_samples = st.number_input("Monte Carlo Samples", min_value=100, max_value=500000, value=5000, step=100)
        failure_radius = st.number_input("Failure Radius", min_value=0.0, value=1.0, step=0.1)
        units_per_px = st.number_input("Units per Pixel", min_value=0.001, value=0.01, step=0.001)
        canvas_width = st.slider("Canvas Width", min_value=400, max_value=1200, value=900, step=50)
        canvas_height = st.slider("Canvas Height", min_value=300, max_value=800, value=500, step=50)

    anchors = render_geometry_canvas(width=int(canvas_width), height=int(canvas_height))

    if not anchors:
        st.stop()

    ui_specs = build_ui_feature_specs(anchors, units_per_px=float(units_per_px))
    features = to_domain_features(ui_specs)

    if st.button("Run Simulation", type="primary", use_container_width=True):
        assembly = Assembly(features=tuple(features))
        simulator = MonteCarloSimulator(assembly=assembly)
        points = simulator.run(int(n_samples))

        mean_pos = mean_2d(points)
        fail_prob = failure_probability(points, radius=float(failure_radius))

        fig = scatter_points(points, title="Monte Carlo Point Cloud")
        fig = add_target_circle(fig, radius=float(failure_radius))

        st.plotly_chart(fig, use_container_width=True)
        st.metric("Failure Probability", f"{fail_prob:.4f}")
        st.metric("Mean Position", f"({mean_pos[0]:.4f}, {mean_pos[1]:.4f})")


if __name__ == "__main__":
    main()
