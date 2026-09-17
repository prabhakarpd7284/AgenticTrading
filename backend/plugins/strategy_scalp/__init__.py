"""Scalping options strategy plugin.

The engine (engine.py + profile.py + replay.py) is the pure-Python core;
ScalpStrategy wraps it in the agents_core plugin contract so the registry picks
it up via the ``alphadesk.strategies`` entry-point in backend/pyproject.toml.
"""
from .plugin import ScalpStrategy

__all__ = ["ScalpStrategy"]
