"""Interactive Streamlit app for geometry-driven tolerance simulation."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from analysis.failure import failure_probability
from analysis.statistics import mean_2d
from app.mappers.geometry_mapper import GeometryPrimitive, parse_geometry_primitives
from core.assembly import Assembly, Feature
from core.simulation import MonteCarloSimulator
from core.tolerance.linear import LinearTolerance
from core.tolerance.position import CircularTolerance
from infra.plotting.interactive_plot import build_interactive_plot

POINT_DISPLAY_RADIUS = 4
MAX_AUTO_UPDATES_PER_SEC = 5
AUTO_UPDATE_INTERVAL_SEC = 1.0 / MAX_AUTO_UPDATES_PER_SEC


def _signature_for_payload(payload: list[dict[str, Any]]) -> str:
    """Create stable hash for current UI payload."""
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _render_mode_header(mode: str) -> None:
    if mode == "Draw":
        st.info("Draw mode: create new points/circles/lines.")
    elif mode == "Edit":
        st.info("Edit mode: drag/resize/reposition existing shapes.")
    else:
        st.info("Simulation mode: tune feature parameters and run analysis.")


def _covariance_magnitude(points: np.ndarray) -> float:
    covariance = np.cov(points, rowvar=False)
    return float(np.linalg.norm(covariance, ord="fro"))


def _primitive_to_feature(primitive: GeometryPrimitive, tolerance: float) -> Feature:
    nominal = np.array([primitive.x, primitive.y], dtype=np.float64)
    tol = max(0.0, float(tolerance))

    if primitive.type == "circle":
        return Feature(nominal=nominal, tolerance=CircularTolerance(radius=tol))

    return Feature(nominal=nominal, tolerance=LinearTolerance(sigma_x=tol, sigma_y=tol))


def _build_features_with_sidebar_edits(
    primitives: list[GeometryPrimitive],
    *,
    selected_idx: int | None,
    edited_type: str,
    edited_x: float,
    edited_y: float,
    edited_tolerance: float,
    tolerance_factor: float,
) -> tuple[list[Feature], list[tuple[float, float]]]:
    """Build backend features using sidebar-edited selected primitive values."""
    features: list[Feature] = []
    overlay_points: list[tuple[float, float]] = []

    for idx, primitive in enumerate(primitives):
        current = primitive
        tolerance = primitive.radius if primitive.type == "circle" else 0.0

        if selected_idx is not None and idx == selected_idx:
            current = GeometryPrimitive(type=edited_type, x=float(edited_x), y=float(edited_y), radius=primitive.radius)
            tolerance = float(edited_tolerance)

        tolerance *= float(tolerance_factor)
        features.append(_primitive_to_feature(current, tolerance=tolerance))
        overlay_points.append((current.x, current.y))

    return features, overlay_points


def main() -> None:
    st.set_page_config(page_title="2D Tolerance Stack-Up", layout="wide")
    st.title("2D Tolerance Stack-Up Simulator")
    st.caption("Draw/edit geometry with throttled near real-time simulation feedback (max 5 updates/sec).")

    with st.sidebar:
        st.header("Interaction")
        mode = st.radio("Mode", options=["Draw", "Edit", "Simulation"], index=0, label_visibility="collapsed")

        st.header("Canvas")
        canvas_width = st.slider("Canvas Width", min_value=400, max_value=1400, value=900, step=50)
        canvas_height = st.slider("Canvas Height", min_value=300, max_value=900, value=550, step=50)
        draw_tool = st.radio("Draw Tool", options=["point", "circle", "line"], index=0, disabled=mode != "Draw")

        st.header("Simulation")
        n_samples = st.slider("Monte Carlo Samples", min_value=500, max_value=30000, value=5000, step=500)
        failure_radius = st.slider("Failure Radius", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        tolerance_factor = st.slider("Tolerance Scale (Live)", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

    _render_mode_header(mode)
    drawing_mode = draw_tool if mode == "Draw" else "transform"

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

    selected_idx: int | None = None
    edited_type = "point"
    edited_x = 0.0
    edited_y = 0.0
    edited_tolerance = 0.0

    if primitives:
        with st.sidebar:
            st.header("Feature Editor")
            selected_idx = st.select_slider(
                "Selected Feature",
                options=list(range(len(primitives))),
                value=0,
                format_func=lambda i: f"Feature {i + 1} ({primitives[i].type})",
            )

            selected = primitives[selected_idx]
            default_tol = selected.radius if selected.type == "circle" and selected.radius is not None else 0.0

            edited_type = st.select_slider("Type", options=["point", "circle", "line"], value=selected.type)
            edited_x = st.slider("Position X", min_value=-200.0, max_value=2000.0, value=float(selected.x), step=0.5)
            edited_y = st.slider("Position Y", min_value=-200.0, max_value=2000.0, value=float(selected.y), step=0.5)
            edited_tolerance = st.slider("Tolerance", min_value=0.0, max_value=50.0, value=float(default_tol), step=0.1)

    features, overlay = _build_features_with_sidebar_edits(
        primitives,
        selected_idx=selected_idx,
        edited_type=edited_type,
        edited_x=edited_x,
        edited_y=edited_y,
        edited_tolerance=edited_tolerance,
        tolerance_factor=float(tolerance_factor),
    )

    st.subheader("Captured Geometry")
    if primitives:
        rows = []
        for idx, primitive in enumerate(primitives):
            x, y, t = primitive.x, primitive.y, primitive.type
            tolerance = primitive.radius if primitive.type == "circle" else 0.0
            if selected_idx is not None and idx == selected_idx:
                x, y, t, tolerance = edited_x, edited_y, edited_type, edited_tolerance

            rows.append(
                {
                    "selected": "👉" if idx == selected_idx else "",
                    "type": t,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "tolerance": round(tolerance, 3),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No objects drawn yet.")

    now = time.time()
    signature_payload = raw_objects + [
        {"selected_idx": selected_idx},
        {"edited_type": edited_type, "edited_x": edited_x, "edited_y": edited_y, "edited_tolerance": edited_tolerance},
        {"tolerance_factor": tolerance_factor},
    ]
    signature = _signature_for_payload(signature_payload)

    if "last_signature" not in st.session_state:
        st.session_state.last_signature = ""
    if "last_change_ts" not in st.session_state:
        st.session_state.last_change_ts = now
    if "last_simulated_signature" not in st.session_state:
        st.session_state.last_simulated_signature = ""
    if "last_simulation_ts" not in st.session_state:
        st.session_state.last_simulation_ts = 0.0
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None

    if signature != st.session_state.last_signature:
        st.session_state.last_signature = signature
        st.session_state.last_change_ts = now

    simulate_clicked = st.button("Simulate", type="primary", use_container_width=True)

    time_since_last_sim = now - float(st.session_state.last_simulation_ts)
    is_throttle_window_open = time_since_last_sim >= AUTO_UPDATE_INTERVAL_SEC
    should_auto_simulate = (
        mode != "Draw"
        and bool(features)
        and signature != st.session_state.last_simulated_signature
        and is_throttle_window_open
    )
    should_simulate = simulate_clicked or should_auto_simulate

    if simulate_clicked and not features:
        st.warning("Draw at least one feature before simulating.")

    if should_simulate and features:
        assembly = Assembly(features=tuple(features))
        simulator = MonteCarloSimulator(assembly=assembly)
        points = simulator.run(int(n_samples))

        st.session_state.simulation_result = {
            "points": points,
            "mean": mean_2d(points),
            "failure_probability": failure_probability(points, radius=float(failure_radius)),
            "failure_radius": float(failure_radius),
            "covariance_magnitude": _covariance_magnitude(points),
            "samples": int(n_samples),
        }
        st.session_state.last_simulated_signature = signature
        st.session_state.last_simulation_ts = time.time()

    result = st.session_state.simulation_result
    if result is not None:
        st.subheader("Real-Time Feedback")
        m1, m2, m3 = st.columns(3)
        m1.metric("Failure Probability", f"{result['failure_probability']:.4f}")
        m2.metric("Mean Position", f"({result['mean'][0]:.4f}, {result['mean'][1]:.4f})")
        m3.metric("Covariance Magnitude", f"{result['covariance_magnitude']:.6f}")

        figure = build_interactive_plot(
            result["points"],
            failure_radius=float(result["failure_radius"]),
            geometry_overlay=overlay,
            title="Simulation Results (Blue=Valid, Red=Failure)",
        )
        if selected_idx is not None and selected_idx < len(overlay):
            figure.add_scatter(
                x=[overlay[selected_idx][0]],
                y=[overlay[selected_idx][1]],
                mode="markers",
                name="Selected Feature",
                marker={"size": 14, "color": "#ffbf00", "symbol": "star"},
            )
        st.plotly_chart(figure, use_container_width=True)


if __name__ == "__main__":
    main()
