"""Interactive 2D geometry drawing utilities for Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from app.ui.types import CanvasAnchor


def _extract_anchor(obj: dict[str, Any], index: int, canvas_height: int) -> CanvasAnchor:
    """Convert a fabric.js object into a single normalized anchor point."""
    shape_type = str(obj.get("type", "unknown"))
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    radius = float(obj.get("radius", 0.0))
    width = float(obj.get("width", 0.0))
    height = float(obj.get("height", 0.0))

    if shape_type == "circle":
        x = left + radius
        y = top + radius
    elif shape_type == "rect":
        x = left + (width / 2.0)
        y = top + (height / 2.0)
    elif shape_type in {"line", "path"}:
        x = left
        y = top
    else:
        x = left
        y = top

    return CanvasAnchor(
        anchor_id=f"feature_{index + 1}",
        x=x,
        y=float(canvas_height) - y,
        source_shape=shape_type,
    )


def render_geometry_canvas(*, width: int, height: int) -> list[CanvasAnchor]:
    """Render drawing canvas and return extracted feature anchors."""
    st.subheader("Geometry Workspace")
    st.caption(
        "Draw points/shapes for features. Then drag to manipulate nominal positions before simulation."
    )

    drawing_mode = st.radio(
        "Draw Mode",
        options=["point", "circle", "rect", "line", "transform"],
        horizontal=True,
        index=0,
    )

    canvas_result = st_canvas(
        stroke_width=2,
        stroke_color="#0B84F3",
        fill_color="rgba(11,132,243,0.15)",
        background_color="#FFFFFF",
        width=width,
        height=height,
        drawing_mode=drawing_mode,
        update_streamlit=True,
        key="geometry_canvas",
    )

    objects = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
    anchors = [_extract_anchor(obj, index, height) for index, obj in enumerate(objects)]

    if anchors:
        st.dataframe(
            [
                {
                    "Feature": anchor.anchor_id,
                    "Nominal X (px)": round(anchor.x, 2),
                    "Nominal Y (px)": round(anchor.y, 2),
                    "Shape": anchor.source_shape,
                }
                for anchor in anchors
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No geometry drawn yet. Add a point/shape to create a feature.")

    return anchors
