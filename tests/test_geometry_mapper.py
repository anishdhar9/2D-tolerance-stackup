"""Tests for geometry-to-feature mapping layer."""

from __future__ import annotations

from app.mappers.geometry_mapper import map_geometry_to_features, parse_geometry_primitives
from core.tolerance.linear import LinearTolerance
from core.tolerance.position import CircularTolerance


def test_parse_geometry_primitives_handles_point_circle_line() -> None:
    canvas_objects = [
        {"type": "circle", "left": 10, "top": 20, "radius": 3},  # point
        {"type": "circle", "left": 100, "top": 200, "radius": 12},  # circle
        {"type": "line", "left": 5, "top": 7, "x1": 0, "y1": 0, "x2": 10, "y2": 6},
    ]

    primitives = parse_geometry_primitives(canvas_objects)

    assert [p.type for p in primitives] == ["point", "circle", "line"]
    assert primitives[0].x == 13
    assert primitives[0].y == 23
    assert primitives[1].radius == 12
    assert primitives[2].x == 10
    assert primitives[2].y == 10


def test_map_geometry_to_features_returns_backend_features() -> None:
    canvas_objects = [
        {"type": "circle", "left": 1, "top": 2, "radius": 2},
        {"type": "circle", "left": 10, "top": 20, "radius": 9},
        {"type": "line", "left": 0, "top": 0, "x1": 0, "y1": 0, "x2": 4, "y2": 4},
    ]

    features = map_geometry_to_features(canvas_objects)

    assert len(features) == 3
    assert isinstance(features[0].tolerance, LinearTolerance)
    assert isinstance(features[1].tolerance, CircularTolerance)
    assert isinstance(features[2].tolerance, LinearTolerance)
