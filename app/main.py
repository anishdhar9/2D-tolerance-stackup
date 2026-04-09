"""Interactive Streamlit app for geometry-driven tolerance simulation."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from analysis.failure import failure_probability
from analysis.statistics import mean_2d
from app.mappers.geometry_mapper import map_geometry_to_features, parse_geometry_primitives
from core.assembly import Assembly
from core.simulation import MonteCarloSimulator
from infra.plotting.scatter import add_target_circle, scatter_points

POINT_DISPLAY_RADIUS = 4


def _signature_for_objects(objects: list[dict[str, Any]]) -> str:
    """Create stable hash for current canvas objects."""
    payload = json.dumps(objects, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    """Render geometry canvas, map to features, and run Monte Carlo simulation."""
    st.set_page_config(page_title="2D Tolerance Stack-Up", layout="wide")
    st.title("2D Tolerance Stack-Up Simulator")
    st.caption("Draw geometry, map to features, and run Monte Carlo simulation.")

    with st.sidebar:
        st.header("Canvas")
        canvas_width = st.slider("Canvas Width", min_value=400, max_value=1400, value=900, step=50)
        canvas_height = st.slider("Canvas Height", min_value=300, max_value=900, value=550, step=50)
        drawing_mode = st.radio("Mode", options=["point", "circle", "line"], index=0)

        st.header("Simulation")
        n_samples = st.number_input("Monte Carlo Samples", min_value=100, max_value=500000, value=5000, step=100)
        failure_radius = st.number_input("Failure Radius", min_value=0.0, value=1.0, step=0.1)
        debounce_seconds = st.slider("Debounce Seconds", min_value=0.2, max_value=3.0, value=0.9, step=0.1)

    canvas_result = st_canvas(
        fill_color="rgba(56, 189, 248, 0.2)",
        stroke_color="#0284C7",
        stroke_width=2,
        point_display_radius=POINT_DISPLAY_RADIUS,
        background_color="#FFFFFF",
        width=canvas_width,
        height=canvas_height,
        drawing_mode=drawing_mode,
        update_streamlit=True,
        key="geometry_capture_canvas",
    )

    raw_objects = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
    primitives = parse_geometry_primitives(raw_objects)
    features = map_geometry_to_features(raw_objects)

    st.subheader("Captured Geometry")
    if primitives:
        st.dataframe(
            [
                {
                    "type": primitive.type,
                    "x": round(primitive.x, 3),
                    "y": round(primitive.y, 3),
                    "radius": None if primitive.radius is None else round(primitive.radius, 3),
                }
                for primitive in primitives
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No objects drawn yet.")

    now = time.time()
    signature = _signature_for_objects(raw_objects)

    if "last_signature" not in st.session_state:
        st.session_state.last_signature = ""
    if "last_change_ts" not in st.session_state:
        st.session_state.last_change_ts = now
    if "last_simulated_signature" not in st.session_state:
        st.session_state.last_simulated_signature = ""
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None

    if signature != st.session_state.last_signature:
        st.session_state.last_signature = signature
        st.session_state.last_change_ts = now

    stopped_drawing = (now - st.session_state.last_change_ts) >= float(debounce_seconds)
    simulate_clicked = st.button("Simulate", type="primary", use_container_width=True)

    should_auto_simulate = (
        bool(raw_objects)
        and stopped_drawing
        and signature != st.session_state.last_simulated_signature
    )
    should_simulate = simulate_clicked or should_auto_simulate

    if simulate_clicked and not raw_objects:
        st.warning("Draw at least one object before simulating.")

    if should_simulate and raw_objects and features:
        assembly = Assembly(features=tuple(features))
        simulator = MonteCarloSimulator(assembly=assembly)
        points = simulator.run(int(n_samples))

        st.session_state.simulation_result = {
            "points": points,
            "mean": mean_2d(points),
            "failure_probability": failure_probability(points, radius=float(failure_radius)),
            "failure_radius": float(failure_radius),
            "samples": int(n_samples),
        }
        st.session_state.last_simulated_signature = signature

    result = st.session_state.simulation_result
    if result is not None:
        st.subheader("Simulation Results")
        fig = scatter_points(result["points"], title="Monte Carlo Point Cloud")
        fig = add_target_circle(fig, radius=float(result["failure_radius"]))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Samples", f"{result['samples']}")
        m2.metric("Failure Probability", f"{result['failure_probability']:.4f}")
        m3.metric("Mean Position", f"({result['mean'][0]:.4f}, {result['mean'][1]:.4f})")


if __name__ == "__main__":
    main()
