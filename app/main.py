"""Streamlit entrypoint for modular geometry-driven tolerance simulation."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import streamlit as st

from app.mappers import FeatureEdit, build_features_from_primitives
from app.services.simulation_runner import SimulationResult, run_simulation
from app.ui import render_canvas, render_feature_editor, render_geometry_table, resolve_canvas_mode
from infra.plotting.interactive_plot import build_interactive_plot

MAX_AUTO_UPDATES_PER_SEC = 5
AUTO_UPDATE_INTERVAL_SEC = 1.0 / MAX_AUTO_UPDATES_PER_SEC


def _signature_for_payload(payload: list[dict[str, Any]]) -> str:
    """Create stable hash for UI state tracking."""
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _render_layout_controls() -> dict[str, float | int | str]:
    """Render top-level sidebar controls and return current UI settings."""
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

    return {
        "mode": mode,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "draw_tool": draw_tool,
        "n_samples": n_samples,
        "failure_radius": failure_radius,
        "tolerance_factor": tolerance_factor,
    }


def _init_session_state(now: float) -> None:
    """Ensure required session state keys exist."""
    st.session_state.setdefault("last_signature", "")
    st.session_state.setdefault("last_change_ts", now)
    st.session_state.setdefault("last_simulated_signature", "")
    st.session_state.setdefault("last_simulation_ts", 0.0)
    st.session_state.setdefault("simulation_result", None)


def _should_auto_simulate(*, mode: str, has_features: bool, signature: str, now: float) -> bool:
    """Apply throttled auto-simulation policy for near real-time interaction."""
    time_since_last_sim = now - float(st.session_state.last_simulation_ts)
    is_throttle_window_open = time_since_last_sim >= AUTO_UPDATE_INTERVAL_SEC

    return (
        mode != "Draw"
        and has_features
        and signature != st.session_state.last_simulated_signature
        and is_throttle_window_open
    )


def _render_feedback(result: SimulationResult, *, overlay: list[tuple[float, float]], selected_idx: int | None) -> None:
    """Render metrics and interactive simulation plot."""
    st.subheader("Real-Time Feedback")
    m1, m2, m3 = st.columns(3)
    m1.metric("Failure Probability", f"{result.failure_probability:.4f}")
    m2.metric("Mean Position", f"({result.mean[0]:.4f}, {result.mean[1]:.4f})")
    m3.metric("Covariance Magnitude", f"{result.covariance_magnitude:.6f}")

    figure = build_interactive_plot(
        result.points,
        failure_radius=float(result.failure_radius),
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


def main() -> None:
    """Run the modular Streamlit UI workflow."""
    st.set_page_config(page_title="2D Tolerance Stack-Up", layout="wide")
    st.title("2D Tolerance Stack-Up Simulator")
    st.caption("Modular UI: drawing, mapping, simulation, and plotting are separated.")

    controls = _render_layout_controls()

    drawing_mode = resolve_canvas_mode(mode=str(controls["mode"]), draw_tool=str(controls["draw_tool"]))
    primitives = render_canvas(
        width=int(controls["canvas_width"]),
        height=int(controls["canvas_height"]),
        drawing_mode=drawing_mode,
    )

    editor_state = render_feature_editor(primitives)
    edit_payload = {
        "type": editor_state.feature_type,
        "x": editor_state.x,
        "y": editor_state.y,
        "tolerance": editor_state.tolerance,
    }
    render_geometry_table(primitives, selected_idx=editor_state.selected_idx, edited=edit_payload)

    features, overlay = build_features_from_primitives(
        primitives,
        selected_idx=editor_state.selected_idx,
        selected_edit=FeatureEdit(
            feature_type=editor_state.feature_type,
            x=editor_state.x,
            y=editor_state.y,
            tolerance=editor_state.tolerance,
        )
        if editor_state.selected_idx is not None
        else None,
        tolerance_factor=float(controls["tolerance_factor"]),
    )

    now = time.time()
    _init_session_state(now)

    payload_signature = _signature_for_payload(
        [
            *[{"type": p.type, "x": p.x, "y": p.y, "radius": p.radius} for p in primitives],
            {
                "selected_idx": editor_state.selected_idx,
                "edit": edit_payload,
                "tolerance_factor": controls["tolerance_factor"],
            },
        ]
    )

    if payload_signature != st.session_state.last_signature:
        st.session_state.last_signature = payload_signature
        st.session_state.last_change_ts = now

    simulate_clicked = st.button("Simulate", type="primary", use_container_width=True)
    should_simulate = simulate_clicked or _should_auto_simulate(
        mode=str(controls["mode"]),
        has_features=bool(features),
        signature=payload_signature,
        now=now,
    )

    if simulate_clicked and not features:
        st.warning("Draw at least one feature before simulating.")

    if should_simulate and features:
        result = run_simulation(
            features,
            n_samples=int(controls["n_samples"]),
            failure_radius=float(controls["failure_radius"]),
        )
        st.session_state.simulation_result = result
        st.session_state.last_simulated_signature = payload_signature
        st.session_state.last_simulation_ts = time.time()

    if st.session_state.simulation_result is not None:
        _render_feedback(
            st.session_state.simulation_result,
            overlay=overlay,
            selected_idx=editor_state.selected_idx,
        )


if __name__ == "__main__":
    main()
