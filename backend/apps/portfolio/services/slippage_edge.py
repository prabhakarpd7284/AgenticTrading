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


# Historical avg-R per setup family (INR per share). Values were derived
# from typical published backtest expectations + the team's earlier runs;
# they're the defaults the FE pre-fills when a setup is picked. Override
# any of them per-call via the explicit `setup_avg_r_inr` field.
SETUP_FAMILIES: dict[str, dict] = {
    "ORB_BREAKOUT":         {"label": "Opening-Range Breakout",       "avg_r_inr": 5.5,  "description": "First 15-min OR breakout with vol confirmation"},
    "VWAP_REVERSION":       {"label": "VWAP Mean-Reversion",          "avg_r_inr": 3.2,  "description": "Fade ≥2σ stretches back toward VWAP"},
    "PIVOT_BREAKOUT":       {"label": "Pivot / Fresh-Breakout",       "avg_r_inr": 7.0,  "description": "Cup-with-handle / flat-base breakouts within 3% of pivot"},
    "PYRAMID_MOMENTUM":     {"label": "Pyramid Momentum",             "avg_r_inr": 9.0,  "description": "Add to winners on EMA-5 retest with trail"},
    "SHORT_STRADDLE":       {"label": "Short Index Straddle",         "avg_r_inr": 12.0, "description": "Weekly ATM straddle with delta-neutral hedges"},
    "VERTICAL_SPREAD":      {"label": "Vertical Debit Spread",        "avg_r_inr": 6.5,  "description": "Directional weekly bull/bear vertical"},
    "GAP_FADE":             {"label": "Gap-Fade",                     "avg_r_inr": 2.8,  "description": "Fade gaps with high historical fill-probability"},
    "FAILED_BREAKOUT_FADE": {"label": "Failed-Breakout Fade",         "avg_r_inr": 4.0,  "description": "Fade OR or pivot failures back to the midpoint"},
}


def setup_catalog() -> dict[str, Any]:
    """Return the SETUP_FAMILIES catalog for the FE picker."""
    return {
        "families": [
            {"id": k, **v} for k, v in SETUP_FAMILIES.items()
        ],
        "note": "Pick a family to pre-fill the expected R per trade. Override via the form if your backtest says different.",
    }


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
    setup_family = (payload.get("setup_family") or "").upper().strip()
    try:
        setup_r = float(payload.get("setup_avg_r_inr") or 0)
    except (TypeError, ValueError):
        setup_r = 0.0
    # Setup family pre-fill — explicit setup_avg_r_inr always wins.
    if not setup_r and setup_family and setup_family in SETUP_FAMILIES:
        setup_r = float(SETUP_FAMILIES[setup_family]["avg_r_inr"])

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
        "setup_family": setup_family or None,
        "setup_avg_r_inr": setup_r,
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
