"""Swing plugin — Oliver Kell weekly cycle scanner.

Detects cycle phases (Wyckoff Accumulation, Cradle Buy, Reversal Extension,
etc.) on weekly bars derived from daily candles, and surfaces actionable
setups via OKAlertService.
"""
from plugins.strategy_swing.plugin import SwingScannerStrategy

__all__ = ["SwingScannerStrategy"]
