"""UI-layer modules for interactive geometry-driven simulation."""

from app.ui.feature_mapper import build_ui_feature_specs, to_domain_features
from app.ui.geometry_canvas import render_geometry_canvas

__all__ = [
    "build_ui_feature_specs",
    "render_geometry_canvas",
    "to_domain_features",
]
