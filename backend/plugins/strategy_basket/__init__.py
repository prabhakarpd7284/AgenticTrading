"""Basket plugin — premarket directional basket builder.

Reads morning mood (gap, VIX, A/D), pulls OK-actionable cycle stocks
from the swing scanner, sizes each leg via risk-per-trade, and submits
the basket via BasketExecutor → broker. BasketPositionManager tracks
realised P&L until close.
"""
from plugins.strategy_basket.plugin import PremarketBasketStrategy

__all__ = ["PremarketBasketStrategy"]
