"""
P&L utilities — single source of truth for profit/loss calculations.

Avoids the scattered BUY/SELL P&L formulas across monitor.py, agent.py, data_layer.py.
"""
from typing import Tuple


def compute_equity_pnl(side: str, entry_price: float, exit_price: float, quantity: int) -> float:
    """
    Compute P&L for an equity trade.

    Args:
        side: "BUY" or "SELL"
        entry_price: Price at which position was opened
        exit_price: Current/exit price
        quantity: Number of shares

    Returns:
        P&L in INR (positive = profit, negative = loss)
    """
    if side == "BUY":
        return (exit_price - entry_price) * quantity
    else:
        return (entry_price - exit_price) * quantity


def compute_straddle_pnl(
    ce_sell: float,
    pe_sell: float,
    ce_current: float,
    pe_current: float,
    lot_size: int,
    lots: int = 1,
) -> Tuple[float, float]:
    """
    Compute short straddle P&L.

    Returns:
        (pnl_pts, pnl_inr) — points and INR
    """
    pnl_pts = (ce_sell + pe_sell) - (ce_current + pe_current)
    pnl_inr = pnl_pts * lot_size * lots
    return round(pnl_pts, 2), round(pnl_inr, 2)


def compute_risk_amount(entry_price: float, stop_loss: float, quantity: int) -> float:
    """Maximum loss if stop loss is hit."""
    return abs(entry_price - stop_loss) * quantity


def compute_risk_reward(entry_price: float, stop_loss: float, target: float) -> float:
    """Risk:Reward ratio. Higher is better."""
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return 0.0
    reward = abs(target - entry_price)
    return round(reward / risk, 2)


def calc_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    max_risk_pct: float = 1.0,
    max_position_pct: float = 10.0,
) -> int:
    """
    Calculate position size based on risk limits.

    Args:
        capital: Total capital in INR
        entry_price: Entry price per share
        stop_loss: Stop loss price
        max_risk_pct: Max risk per trade as % of capital (default: 1%)
        max_position_pct: Max position value as % of capital (default: 10%)

    Returns:
        Number of shares (quantity)
    """
    if entry_price <= 0 or stop_loss == entry_price:
        return 0

    risk_per_share = abs(entry_price - stop_loss)
    max_risk = capital * (max_risk_pct / 100)
    qty = int(max_risk / risk_per_share) if risk_per_share > 0 else 0

    # Cap by position size
    max_value = capital * (max_position_pct / 100)
    max_qty_by_value = int(max_value / entry_price) if entry_price > 0 else 0

    qty = min(qty, max_qty_by_value)
    return max(qty, 1) if qty > 0 else 0
