"""Interactive Plotly visualization for simulation results + geometry overlays."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

CHI2_95_DF2 = 5.991464547107979


def _validate_points(points: NDArray[np.float64]) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2).")


def _ellipse_points(points: NDArray[np.float64], n_points: int = 256) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return x/y coordinates for the 95% confidence ellipse from sample covariance."""
    mean = np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    axis_lengths = np.sqrt(np.maximum(eigenvalues, 0.0) * CHI2_95_DF2)

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
    transformed = eigenvectors @ np.diag(axis_lengths) @ unit_circle

    x = transformed[0, :] + mean[0]
    y = transformed[1, :] + mean[1]
    return x.astype(np.float64), y.astype(np.float64)


def _circle_points(center: tuple[float, float], radius: float, n_points: int = 256) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x.astype(np.float64), y.astype(np.float64)


def build_interactive_plot(
    points: NDArray[np.float64],
    failure_radius: float,
    *,
    failure_center: tuple[float, float] = (0.0, 0.0),
    geometry_overlay: Sequence[tuple[float, float]] | None = None,
    title: str = "Tolerance Simulation Results",
) -> go.Figure:
    """Create interactive plot with valid/failure points, mean, 95% ellipse, and failure boundary."""
    _validate_points(points)
    if failure_radius < 0.0:
        raise ValueError("failure_radius must be non-negative.")

    distances = np.linalg.norm(points - np.array(failure_center, dtype=np.float64), axis=1)
    is_failure = distances > failure_radius

    valid_points = points[~is_failure]
    failed_points = points[is_failure]
    mean = np.mean(points, axis=0)

    figure = go.Figure()

    if len(valid_points) > 0:
        figure.add_trace(
            go.Scatter(
                x=valid_points[:, 0],
                y=valid_points[:, 1],
                mode="markers",
                name="Valid",
                marker={"size": 5, "color": "#1f77b4", "opacity": 0.65},
            )
        )

    if len(failed_points) > 0:
        figure.add_trace(
            go.Scatter(
                x=failed_points[:, 0],
                y=failed_points[:, 1],
                mode="markers",
                name="Failure",
                marker={"size": 5, "color": "#d62728", "opacity": 0.75},
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[mean[0]],
            y=[mean[1]],
            mode="markers",
            name="Mean",
            marker={"size": 12, "color": "black", "symbol": "x"},
        )
    )

    ellipse_x, ellipse_y = _ellipse_points(points)
    figure.add_trace(
        go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode="lines",
            name="95% Confidence Ellipse",
            line={"color": "#9467bd", "width": 2},
        )
    )

    circle_x, circle_y = _circle_points(failure_center, failure_radius)
    figure.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="Failure Boundary",
            line={"color": "#d62728", "width": 2, "dash": "dash"},
        )
    )

    if geometry_overlay:
        gx = np.array([point[0] for point in geometry_overlay], dtype=np.float64)
        gy = np.array([point[1] for point in geometry_overlay], dtype=np.float64)
        figure.add_trace(
            go.Scatter(
                x=gx,
                y=gy,
                mode="markers+lines",
                name="Geometry",
                marker={"size": 9, "color": "#2ca02c", "symbol": "diamond"},
                line={"color": "#2ca02c", "width": 1},
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        legend_title="Layers",
        template="plotly_white",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure
