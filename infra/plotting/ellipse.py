"""Plotly helpers for confidence ellipse overlays."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def add_confidence_ellipse(
    figure: go.Figure,
    center: tuple[float, float],
    major_axis: float,
    minor_axis: float,
    angle: float,
    n_points: int = 256,
) -> go.Figure:
    """Overlay a confidence ellipse on an existing figure and return it."""
    if major_axis < 0.0 or minor_axis < 0.0:
        raise ValueError("major_axis and minor_axis must be non-negative.")
    if n_points < 3:
        raise ValueError("n_points must be at least 3.")

    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    local_x = major_axis * np.cos(t)
    local_y = minor_axis * np.sin(t)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x = center[0] + (local_x * cos_a - local_y * sin_a)
    y = center[1] + (local_x * sin_a + local_y * cos_a)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Confidence Ellipse",
            line={"color": "royalblue", "width": 2},
        )
    )
    return figure
