"""Interactive geometry drawing app (UI-only, no simulation orchestration)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import streamlit as st
from streamlit_drawable_canvas import st_canvas

POINT_DISPLAY_RADIUS = 4


class GeometryObject(TypedDict):
    """Normalized geometry object extracted from the drawable canvas."""

    type: Literal["point", "circle", "line", "unknown"]
    x: float
    y: float
    radius: float | None


def parse_canvas_objects(objects: list[dict[str, Any]]) -> list[GeometryObject]:
    """Parse raw fabric.js objects into structured geometry data.

    Returns a list of dicts with:
    - ``type``: point, circle, line, or unknown
    - ``x``, ``y``: object position (center for circles/points, midpoint for lines)
    - ``radius``: circle radius, otherwise ``None``
    """
    parsed: list[GeometryObject] = []

    for obj in objects:
        shape_type = str(obj.get("type", "unknown"))
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))

        if shape_type == "circle":
            radius = float(obj.get("radius", 0.0))
            parsed_type: Literal["point", "circle", "line", "unknown"] = (
                "point" if radius <= float(POINT_DISPLAY_RADIUS) else "circle"
            )
            parsed.append(
                {
                    "type": parsed_type,
                    "x": left + radius,
                    "y": top + radius,
                    "radius": radius if parsed_type == "circle" else None,
                }
            )
            continue

        if shape_type == "line":
            x1 = float(obj.get("x1", 0.0))
            y1 = float(obj.get("y1", 0.0))
            x2 = float(obj.get("x2", 0.0))
            y2 = float(obj.get("y2", 0.0))
            parsed.append(
                {
                    "type": "line",
                    "x": left + ((x1 + x2) / 2.0),
                    "y": top + ((y1 + y2) / 2.0),
                    "radius": None,
                }
            )
            continue

        parsed.append(
            {
                "type": "unknown",
                "x": left,
                "y": top,
                "radius": None,
            }
        )

    return parsed


def main() -> None:
    """Render interactive canvas and output structured geometry only."""
    st.set_page_config(page_title="2D Geometry Capture", layout="wide")
    st.title("2D Geometry Capture")
    st.caption("Draw features and extract structured geometry data for downstream mapping.")

    col1, col2 = st.columns([3, 1])
    with col2:
        canvas_width = st.slider("Canvas Width", min_value=400, max_value=1400, value=900, step=50)
        canvas_height = st.slider("Canvas Height", min_value=300, max_value=900, value=550, step=50)
        drawing_mode = st.radio("Mode", options=["point", "circle", "line"], index=0)

    with col1:
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
    geometry_data = parse_canvas_objects(raw_objects)

    st.subheader("Parsed Geometry Objects")
    if geometry_data:
        st.dataframe(geometry_data, use_container_width=True, hide_index=True)
        st.json(geometry_data)
    else:
        st.info("No objects drawn yet.")


if __name__ == "__main__":
    main()
