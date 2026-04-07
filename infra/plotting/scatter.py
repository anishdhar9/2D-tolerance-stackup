"""Plotly scatter plotting utilities for 2D point clouds."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


def scatter_points(points: NDArray[np.float64], title: str = "2D Samples") -> go.Figure:
    """Create a scatter figure for points with shape ``(n_samples, 2)``."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2).")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            name="Samples",
            marker={"size": 5, "opacity": 0.6},
        )
    )
    figure.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def add_target_circle(
    figure: go.Figure,
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    n_points: int = 256,
) -> go.Figure:
    """Overlay a target circle on an existing figure and return it."""
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    if n_points < 3:
        raise ValueError("n_points must be at least 3.")

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Target Circle",
            line={"color": "crimson", "width": 2},
        )
    )
    return figure
