"""UI-specific typed models used to decouple Streamlit interactions from domain objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanvasAnchor:
    """Geometry anchor extracted from drawable-canvas objects."""

    anchor_id: str
    x: float
    y: float
    source_shape: str


@dataclass(frozen=True)
class UIFeatureSpec:
    """UI-level feature definition before conversion to domain Feature objects."""

    anchor_id: str
    nominal_x: float
    nominal_y: float
    tolerance_type: str
    sigma_x: float
    sigma_y: float
    radius: float
