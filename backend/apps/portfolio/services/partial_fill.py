"""Partial-Fill & Queue-Position Economics.

For every closed trade with both an entry signal and a fill, compute
per-trade execution-quality metrics:

  fill_ratio       fill_qty / requested_qty  (1.0 = full fill)
  slippage_bps     (fill - entry) / entry × 10_000   (signed; +ve = paid more)
  queue_score      proxy of where in the book the fill landed
                   (cheap when only fill_price + entry_price exist:
                    1.0 = mid; 0.5 = inside spread; 0 = touched far side)
  fill_latency_s   trade_date end of session — created_at  (approx, since
                   we don't store signal_time explicitly)
  cost_per_lot     total_cost / qty
  notes_flags      list of execution-quality warnings

Aggregates by side + symbol + strategy bucket.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any


_SPREAD_BPS_PROXY = 5.0


def _classify(reasoning: str) -> str:
    r = (reasoning or "").lower()
    if "straddle" in r:               return "short_straddle"
    if "pyramid" in r or "ema5" in r: return "pyramid"
    if "spread" in r:                 return "vertical_spread"
    if "intraday" in r or "vwap" in r or "structure" in r:
        return "directional"
    return "uncategorised"


def _queue_score(slippage_bps: float) -> float:
    """Map signed slippage into a 0..1 quality score (1 = mid, 0 = far side)."""
    abs_bps = abs(slippage_bps)
    if abs_bps <= _SPREAD_BPS_PROXY * 0.5:
        return 1.0
    if abs_bps <= _SPREAD_BPS_PROXY:
        return 0.5
    return max(0.0, 1.0 - abs_bps / (_SPREAD_BPS_PROXY * 4))


def _flags(slippage_bps: float, fill_ratio: float) -> list[str]:
    out: list[str] = []
    if fill_ratio < 0.6:
        out.append("low_fill_ratio")
    if abs(slippage_bps) > 25:
        out.append("high_slippage")
    if slippage_bps > 50:
        out.append("chased_market")
    return out


def build_partial_fill_report(tenant=None, *, limit: int = 200) -> dict[str, Any]:
    from trading.models import TradeJournal

    qs = (TradeJournal.objects
          .filter(status__in=("EXECUTED", "PAPER", "FILLED", "CLOSED"))
          .exclude(fill_price__isnull=True)
          .order_by("-created_at")[:limit])

    rows: list[dict] = []
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_symbol: dict[str, list[float]] = defaultdict(list)
    queue_scores: list[float] = []
    fill_ratios: list[float] = []

    for t in qs:
        entry = float(t.entry_price or 0)
        fill = float(t.fill_price or 0)
        qty = int(t.quantity or 0)
        if entry <= 0 or qty <= 0:
            continue

        slip_bps = ((fill - entry) / entry) * 10_000.0
        fill_qty = int(getattr(t, "fill_quantity", qty) or qty)
        fill_ratio = round(fill_qty / qty, 3) if qty else 0.0
        qs_score = round(_queue_score(slip_bps), 3)
        flags = _flags(slip_bps, fill_ratio)
        strat = _classify(t.reasoning or "")
        cost_per_lot = round(abs(slip_bps) * entry / 10_000.0, 3)

        row = {
            "trade_id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "qty_requested": qty,
            "qty_filled": fill_qty,
            "fill_ratio": fill_ratio,
            "entry": entry,
            "fill": fill,
            "slippage_bps": round(slip_bps, 1),
            "queue_score": qs_score,
            "cost_per_lot_inr": cost_per_lot,
            "strategy": strat,
            "flags": flags,
            "created_at": t.created_at.isoformat() if getattr(t, "created_at", None) else None,
        }
        rows.append(row)
        by_strategy[strat].append(qs_score)
        by_symbol[t.symbol].append(qs_score)
        queue_scores.append(qs_score)
        fill_ratios.append(fill_ratio)

    def _agg_scores(scores: list[float]) -> dict:
        if not scores:
            return {"count": 0}
        return {
            "count": len(scores),
            "avg_queue_score": round(statistics.mean(scores), 3),
            "median_queue_score": round(statistics.median(scores), 3),
        }

    return {
        "count": len(rows),
        "totals": {
            "trades": len(rows),
            "avg_queue_score": round(statistics.mean(queue_scores), 3) if queue_scores else 0.0,
            "avg_fill_ratio": round(statistics.mean(fill_ratios), 3) if fill_ratios else 0.0,
            "high_slippage_count": sum(1 for r in rows if "high_slippage" in r["flags"]),
            "low_fill_count":      sum(1 for r in rows if "low_fill_ratio" in r["flags"]),
            "chased_count":        sum(1 for r in rows if "chased_market" in r["flags"]),
        },
        "by_strategy": {k: _agg_scores(v) for k, v in by_strategy.items()},
        "by_symbol":   {k: _agg_scores(v) for k, v in sorted(by_symbol.items(), key=lambda kv: -len(kv[1]))[:15]},
        "rows": rows[:50],
        "note": (
            "queue_score ∈ [0,1]: 1.0 = filled at mid (queue priority worked), "
            "0 = filled at far side (chased). avg_queue_score < 0.5 across a "
            "strategy means you're paying spread; switch to limit-fragments."
        ),
    }
