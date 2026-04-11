"""UI drawing panel helpers."""

from __future__ import annotations

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from app.mappers.geometry_mapper import GeometryPrimitive, parse_geometry_primitives

POINT_DISPLAY_RADIUS = 4


def resolve_canvas_mode(*, mode: str, draw_tool: str) -> str:
    """Map high-level interaction mode into drawable-canvas mode."""
    return draw_tool if mode == "Draw" else "transform"


def render_canvas(*, width: int, height: int, drawing_mode: str) -> list[GeometryPrimitive]:
    """Render drawable canvas and return normalized geometry primitives."""
    result = st_canvas(
        fill_color="rgba(56, 189, 248, 0.2)",
        stroke_color="#0284C7",
        stroke_width=2,
        point_display_radius=POINT_DISPLAY_RADIUS,
        background_color="#FFFFFF",
        width=width,
        height=height,
        drawing_mode=drawing_mode,
        update_streamlit=True,
        key="geometry_capture_canvas",
    )
    objects = result.json_data.get("objects", []) if result.json_data else []
    return parse_geometry_primitives(objects)


def render_geometry_table(primitives: list[GeometryPrimitive], *, selected_idx: int | None, edited: dict[str, float | str] | None) -> None:
    """Render compact geometry table with selection highlighting."""
    st.subheader("Captured Geometry")
    if not primitives:
        st.info("No objects drawn yet.")
        return

    rows: list[dict[str, object]] = []
    for idx, primitive in enumerate(primitives):
        x, y, t = primitive.x, primitive.y, primitive.type
        tolerance = primitive.radius if primitive.type == "circle" and primitive.radius is not None else 0.0

        if selected_idx is not None and edited is not None and idx == selected_idx:
            x = float(edited["x"])
            y = float(edited["y"])
            t = str(edited["type"])
            tolerance = float(edited["tolerance"])

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
