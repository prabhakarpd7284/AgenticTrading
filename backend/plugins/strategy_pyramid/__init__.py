"""Pyramid options strategy plugin (post Phase 3 move).

The strategy.py module is the core simulator; PyramidStrategy below
wraps it in the agents_core plugin contract so the registry can pick
it up via the `alphadesk.strategies` entry-point declared in
backend/pyproject.toml.
"""
from .plugin import PyramidStrategy

__all__ = ["PyramidStrategy"]
