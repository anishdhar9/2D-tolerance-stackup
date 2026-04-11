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
from infra.plotting.interactive_plot import build_interactive_plot

POINT_DISPLAY_RADIUS = 4


def _signature_for_objects(objects: list[dict[str, Any]]) -> str:
    """Create stable hash for current canvas objects."""
    payload = json.dumps(objects, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_mode_header(mode: str) -> None:
    if mode == "Draw":
        st.info("Draw mode: create new points/circles/lines.")
    elif mode == "Edit":
        st.info("Edit mode: drag, resize, or reposition existing shapes.")
    else:
        st.info("Simulation mode: review geometry and run simulation.")


def main() -> None:
    """Render geometry canvas, map to features, and run Monte Carlo simulation."""
    st.set_page_config(page_title="2D Tolerance Stack-Up", layout="wide")
    st.title("2D Tolerance Stack-Up Simulator")
    st.caption("Switch modes to draw/edit geometry, then simulate stack-up behavior.")

    with st.sidebar:
        st.header("Interaction Mode")
        mode = st.radio("Mode", options=["Draw", "Edit", "Simulation"], index=0, label_visibility="collapsed")

        st.header("Canvas")
        canvas_width = st.slider("Canvas Width", min_value=400, max_value=1400, value=900, step=50)
        canvas_height = st.slider("Canvas Height", min_value=300, max_value=900, value=550, step=50)

        draw_tool = st.radio("Draw Tool", options=["point", "circle", "line"], index=0, disabled=mode != "Draw")

        st.header("Simulation")
        n_samples = st.number_input("Monte Carlo Samples", min_value=100, max_value=500000, value=5000, step=100)
        failure_radius = st.number_input("Failure Radius", min_value=0.0, value=1.0, step=0.1)
        debounce_seconds = st.slider("Debounce Seconds", min_value=0.2, max_value=3.0, value=0.9, step=0.1)

    _render_mode_header(mode)

    if mode == "Draw":
        drawing_mode = draw_tool
    else:
        drawing_mode = "transform"

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
        selected_idx = st.selectbox(
            "Selected Object",
            options=list(range(len(primitives))),
            format_func=lambda i: f"#{i + 1} - {primitives[i].type}",
            help="Select an object to inspect/highlight.",
        )

        rows = []
        for idx, primitive in enumerate(primitives):
            is_selected = idx == selected_idx
            rows.append(
                {
                    "selected": "👉" if is_selected else "",
                    "type": primitive.type,
                    "x": round(primitive.x, 3),
                    "y": round(primitive.y, 3),
                    "radius": None if primitive.radius is None else round(primitive.radius, 3),
                }
            )

        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.success(
            f"Selected object #{selected_idx + 1}: {primitives[selected_idx].type} at "
            f"({primitives[selected_idx].x:.2f}, {primitives[selected_idx].y:.2f})."
        )
    else:
        selected_idx = None
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
        mode == "Simulation"
        and bool(raw_objects)
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

        overlay = [(p.x, p.y) for p in primitives]
        figure = build_interactive_plot(
            result["points"],
            failure_radius=float(result["failure_radius"]),
            geometry_overlay=overlay,
            title="Monte Carlo Results + Geometry Overlay",
        )

        if selected_idx is not None:
            selected = primitives[selected_idx]
            figure.add_scatter(
                x=[selected.x],
                y=[selected.y],
                mode="markers",
                name="Selected Geometry",
                marker={"size": 14, "color": "#ffbf00", "symbol": "star"},
            )

        st.plotly_chart(figure, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Samples", f"{result['samples']}")
        m2.metric("Failure Probability", f"{result['failure_probability']:.4f}")
        m3.metric("Mean Position", f"({result['mean'][0]:.4f}, {result['mean'][1]:.4f})")


if __name__ == "__main__":
    main()
