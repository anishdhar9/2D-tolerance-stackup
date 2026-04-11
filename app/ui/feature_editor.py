"""Sidebar feature editor controls."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.mappers.geometry_mapper import GeometryPrimitive


@dataclass(frozen=True)
class EditorState:
    """Selected feature and edited values from sidebar controls."""

    selected_idx: int | None
    feature_type: str
    x: float
    y: float
    tolerance: float


def render_feature_editor(primitives: list[GeometryPrimitive]) -> EditorState:
    """Render sidebar controls for selected feature and return edited state."""
    if not primitives:
        return EditorState(selected_idx=None, feature_type="point", x=0.0, y=0.0, tolerance=0.0)

    with st.sidebar:
        st.header("Feature Editor")
        selected_idx = st.select_slider(
            "Selected Feature",
            options=list(range(len(primitives))),
            value=0,
            format_func=lambda i: f"Feature {i + 1} ({primitives[i].type})",
        )

        selected = primitives[selected_idx]
        default_tolerance = selected.radius if selected.type == "circle" and selected.radius is not None else 0.0

        feature_type = st.select_slider("Type", options=["point", "circle", "line"], value=selected.type)
        x = st.slider("Position X", min_value=-200.0, max_value=2000.0, value=float(selected.x), step=0.5)
        y = st.slider("Position Y", min_value=-200.0, max_value=2000.0, value=float(selected.y), step=0.5)
        tolerance = st.slider("Tolerance", min_value=0.0, max_value=50.0, value=float(default_tolerance), step=0.1)

    return EditorState(selected_idx=selected_idx, feature_type=feature_type, x=x, y=y, tolerance=tolerance)
