"""Trade post-mortem auto-annotator.

For each closed TradeJournal entry in the requested month, classify the
outcome driver via simple heuristics over plan-vs-actual fields:

  sl_too_tight       — exited at SL, would have recovered on next bar
  exit_too_early     — closed in profit but well below max favourable
  sizing_too_small   — profitable trade but qty < median
  slippage           — abs(entry slippage) > 30 bps
  news_shock         — large gap or move outside ATR
  regime_mismatch    — strategy reasoning mentions a regime the day didn't see
  thesis_wrong       — default fallback when nothing else fits

The heuristics are deliberately rough — the goal is a starting taxonomy
for the monthly report, not a perfect classifier.
"""
from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any


_TAXONOMY = (
    "sl_too_tight", "exit_too_early", "sizing_too_small",
    "slippage", "news_shock", "regime_mismatch", "thesis_wrong",
)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _classify(trade, median_qty: float, atr_proxy: float) -> tuple[str, str]:
    """Return (cause, evidence) tuple."""
    entry = _safe_float(getattr(trade, "entry_price", 0))
    exit_p = _safe_float(getattr(trade, "fill_price", 0))
    sl = _safe_float(getattr(trade, "stop_loss", 0))
    qty = int(getattr(trade, "quantity", 0) or 0)
    pnl = _safe_float(getattr(trade, "pnl", 0))
    # TradeJournal doesn't have a stored slippage column — derive on the fly.
    if entry > 0 and exit_p > 0:
        slip = ((exit_p - entry) / entry) * 10_000.0
    else:
        slip = 0.0
    reason = (getattr(trade, "reasoning", "") or "").lower()

    if abs(slip) >= 30:
        return "slippage", f"entry slippage {slip:.0f} bps vs plan"

    if sl > 0 and exit_p > 0 and abs(exit_p - sl) / sl < 0.005:
        return "sl_too_tight", f"closed within 0.5% of stop ({sl}); ATR ~{atr_proxy:.2f}"

    if pnl > 0 and qty > 0 and qty < median_qty * 0.5:
        return "sizing_too_small", f"qty {qty} vs median {median_qty:.0f} — left money on the table"

    if pnl > 0 and entry > 0 and exit_p > 0:
        # Profit but exit < 50% of (entry+ATR*2) — proxy for early-exit.
        upside_proxy = entry + atr_proxy * 2
        if exit_p < entry + (upside_proxy - entry) * 0.4:
            return "exit_too_early", f"exited at {exit_p:.2f}; ATR-implied target ~{upside_proxy:.2f}"

    if entry > 0 and exit_p > 0 and abs(exit_p - entry) / entry > 0.04:
        return "news_shock", f"|move| {(exit_p - entry) / entry * 100:.1f}% exceeds typical band"

    if "trend" in reason or "breakout" in reason:
        if pnl < 0:
            return "regime_mismatch", "trend/breakout thesis in a chop day"

    return "thesis_wrong", "no specific signal; review setup"


def build_post_mortem_report(tenant=None, *, month: str | None = None) -> dict[str, Any]:
    from trading.models import TradeJournal

    closed_states = ("EXECUTED", "PAPER", "FILLED", "CLOSED")
    qs = TradeJournal.objects.filter(status__in=closed_states).exclude(fill_price__isnull=True)
    if month:
        try:
            y, m = month.split("-")
            qs = qs.filter(trade_date__year=int(y), trade_date__month=int(m))
        except (ValueError, AttributeError):
            pass

    trades = list(qs.order_by("-created_at")[:500])
    if not trades:
        return {
            "month": month, "count": 0, "by_cause": {}, "rows": [],
            "taxonomy": list(_TAXONOMY),
        }

    qtys = [int(t.quantity or 0) for t in trades if (t.quantity or 0) > 0]
    median_qty = statistics.median(qtys) if qtys else 0.0
    moves = [
        abs(_safe_float(t.fill_price) - _safe_float(t.entry_price))
        for t in trades
        if (t.fill_price and t.entry_price)
    ]
    atr_proxy = statistics.median(moves) if moves else 0.0

    rows = []
    counts: dict[str, int] = {k: 0 for k in _TAXONOMY}
    for t in trades:
        cause, evidence = _classify(t, median_qty, atr_proxy)
        counts[cause] += 1
        rows.append({
            "trade_id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry": _safe_float(t.entry_price),
            "exit": _safe_float(t.fill_price),
            "pnl": _safe_float(t.pnl),
            "cause": cause,
            "evidence": evidence,
            "timestamp": t.created_at.isoformat() if getattr(t, "created_at", None) else None,
        })

    return {
        "month": month,
        "count": len(rows),
        "by_cause": counts,
        "rows": rows,
        "taxonomy": list(_TAXONOMY),
    }
