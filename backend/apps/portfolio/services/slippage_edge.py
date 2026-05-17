"""Slippage-vs-Edge calculator.

Given a candidate trade {symbol, qty, setup_avg_r_inr}, project:

  half_spread_inr   — live half-spread for the symbol (mid - bid)
  impact_inr        — qty × 5 bps proxy (until L2 depth lands)
  brokerage_inr     — flat 20 INR / order × 2 (entry + exit) for equity
  total_cost_inr    — sum of the above
  expected_edge_inr — setup_avg_r_inr × qty (caller-supplied)
  net_edge_inr      — expected_edge - total_cost
  verdict           — green / amber / red:
                       green  net_edge > 1.5× total_cost
                       amber  0 < net_edge ≤ 1.5× total_cost
                       red    net_edge ≤ 0
"""
from __future__ import annotations

from typing import Any


_BROKERAGE_PER_ORDER = 20.0
_IMPACT_BPS = 5.0


def _live_quote(symbol: str) -> tuple[float, float]:
    """Reuse the liquidity_map cache so this stays cheap."""
    try:
        from apps.market_data.services.liquidity_map import _batch_live_quotes
        q = _batch_live_quotes([symbol]).get(symbol, {})
        return float(q.get("bid", 0) or 0), float(q.get("ask", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def compute(payload: dict) -> dict[str, Any]:
    symbol = (payload.get("symbol") or "").upper()
    try:
        qty = int(payload.get("qty") or 0)
    except (TypeError, ValueError):
        qty = 0
    try:
        setup_r = float(payload.get("setup_avg_r_inr") or 0)
    except (TypeError, ValueError):
        setup_r = 0.0

    if not symbol or qty <= 0:
        return {"error": "symbol and qty>0 are required"}

    bid, ask = _live_quote(symbol)
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
    half_spread = max((ask - bid) / 2, 0.0)
    half_spread_inr = round(half_spread * qty, 2)

    impact_inr = round((mid * _IMPACT_BPS / 10_000.0) * qty, 2) if mid > 0 else 0.0
    brokerage_inr = round(_BROKERAGE_PER_ORDER * 2, 2)
    total_cost = round(half_spread_inr + impact_inr + brokerage_inr, 2)

    expected_edge = round(setup_r * qty, 2) if setup_r else 0.0
    net_edge = round(expected_edge - total_cost, 2)

    if expected_edge <= 0:
        verdict = "red"
    elif net_edge <= 0:
        verdict = "red"
    elif net_edge < total_cost * 0.5:    # net edge < 1.5× cost
        verdict = "amber"
    else:
        verdict = "green"

    return {
        "symbol": symbol, "qty": qty,
        "bid": bid, "ask": ask, "mid": round(mid, 2),
        "half_spread_inr": half_spread_inr,
        "impact_inr": impact_inr,
        "brokerage_inr": brokerage_inr,
        "total_cost_inr": total_cost,
        "expected_edge_inr": expected_edge,
        "net_edge_inr": net_edge,
        "edge_to_cost_ratio": round(expected_edge / total_cost, 2) if total_cost > 0 else 0.0,
        "verdict": verdict,
        "note": (
            "Impact is a flat 5-bps proxy until level-2 depth is wired. "
            "Brokerage is 20 INR/leg (Angel One equity flat)."
        ),
    }
