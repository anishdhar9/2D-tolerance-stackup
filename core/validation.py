"""Shared validation helpers for core domain modules."""

from __future__ import annotations


def validate_positive_int(value: int, name: str) -> None:
    """Validate that an integer value is strictly positive."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
