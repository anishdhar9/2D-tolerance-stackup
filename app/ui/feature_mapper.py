"""Map UI interaction state into domain Feature entities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import streamlit as st

from app.ui.types import CanvasAnchor, UIFeatureSpec
from core.assembly import Feature
from core.tolerance.linear import LinearTolerance
from core.tolerance.position import CircularTolerance


def build_ui_feature_specs(anchors: Sequence[CanvasAnchor], *, units_per_px: float) -> list[UIFeatureSpec]:
    """Render per-feature controls and produce UI-level feature specs."""
    specs: list[UIFeatureSpec] = []

    st.subheader("Feature Definitions")
    st.caption("Define tolerance behavior for each drawn feature.")

    for index, anchor in enumerate(anchors):
        with st.expander(f"{anchor.anchor_id} ({anchor.source_shape})", expanded=index == 0):
            nominal_x = st.number_input(
                f"Nominal X - {anchor.anchor_id}",
                value=anchor.x * units_per_px,
                key=f"nominal_x_{anchor.anchor_id}",
            )
            nominal_y = st.number_input(
                f"Nominal Y - {anchor.anchor_id}",
                value=anchor.y * units_per_px,
                key=f"nominal_y_{anchor.anchor_id}",
            )

            tolerance_type = st.selectbox(
                f"Tolerance Type - {anchor.anchor_id}",
                options=["linear", "circular"],
                key=f"tol_type_{anchor.anchor_id}",
            )

            sigma_x = st.number_input(
                f"Sigma X - {anchor.anchor_id}",
                min_value=0.0,
                value=0.1,
                key=f"sigma_x_{anchor.anchor_id}",
            )
            sigma_y = st.number_input(
                f"Sigma Y - {anchor.anchor_id}",
                min_value=0.0,
                value=0.1,
                key=f"sigma_y_{anchor.anchor_id}",
            )
            radius = st.number_input(
                f"Radius - {anchor.anchor_id}",
                min_value=0.0,
                value=0.1,
                key=f"radius_{anchor.anchor_id}",
            )

            specs.append(
                UIFeatureSpec(
                    anchor_id=anchor.anchor_id,
                    nominal_x=float(nominal_x),
                    nominal_y=float(nominal_y),
                    tolerance_type=tolerance_type,
                    sigma_x=float(sigma_x),
                    sigma_y=float(sigma_y),
                    radius=float(radius),
                )
            )

    return specs


def to_domain_features(specs: Sequence[UIFeatureSpec]) -> list[Feature]:
    """Convert UI feature specs to immutable domain Feature objects."""
    domain_features: list[Feature] = []

    for spec in specs:
        tolerance = (
            LinearTolerance(sigma_x=spec.sigma_x, sigma_y=spec.sigma_y)
            if spec.tolerance_type == "linear"
            else CircularTolerance(radius=spec.radius)
        )
        nominal = np.array([spec.nominal_x, spec.nominal_y], dtype=np.float64)
        domain_features.append(Feature(nominal=nominal, tolerance=tolerance))

    return domain_features
