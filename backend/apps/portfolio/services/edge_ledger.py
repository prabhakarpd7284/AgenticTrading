"""Per-trade Cost & Round-Trip Edge Ledger.

For every closed TradeJournal row, compute:

  gross_pnl_inr      sign * (fill - entry) * qty       (signed; +ve = winning trade)
  spread_cost_inr    estimated half-spread × qty × 2   (entry + exit)
  brokerage_inr      ₹20 × 2 (Angel One equity flat per leg)
  total_cost_inr     spread + brokerage
  net_edge_inr       gross_pnl − total_cost
  edge_to_cost       net_edge / total_cost  (ratio; >1 means trade pays for itself)
  edge_bps           net_edge / notional * 10_000

Aggregates by strategy + by symbol so the trader can see which set-ups
actually pay vs which churn cost.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


_BROKERAGE_PER_LEG = 20.0
_DEFAULT_HALF_SPREAD_BPS = 5.0


def _classify(reasoning: str) -> str:
    r = (reasoning or "").lower()
    if "straddle" in r:        return "short_straddle"
    if "pyramid" in r or "ema5" in r: return "pyramid"
    if "spread" in r:          return "vertical_spread"
    if "intraday" in r or "vwap" in r or "structure" in r or "5-min" in r:
        return "directional"
    return "uncategorised"


def build_edge_ledger(tenant=None, *, limit: int = 200) -> dict[str, Any]:
    from trading.models import TradeJournal

    closed_states = ("EXECUTED", "PAPER", "FILLED", "CLOSED")
    qs = (TradeJournal.objects
          .filter(status__in=closed_states)
          .exclude(fill_price__isnull=True)
          .order_by("-created_at")[:limit])

    rows: list[dict] = []
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    by_symbol: dict[str, list[dict]] = defaultdict(list)

    for t in qs:
        entry = float(t.entry_price or 0)
        fill = float(t.fill_price or 0)
        qty = int(t.quantity or 0)
        if entry <= 0 or qty <= 0:
            continue
        sign = 1 if t.side == "BUY" else -1
        gross = round(sign * (fill - entry) * qty, 2)
        spread_per_leg = entry * _DEFAULT_HALF_SPREAD_BPS / 10_000.0
        spread_cost = round(spread_per_leg * qty * 2, 2)
        brokerage = round(_BROKERAGE_PER_LEG * 2, 2)
        total_cost = round(spread_cost + brokerage, 2)
        net = round(gross - total_cost, 2)
        notional = round(entry * qty, 2)
        edge_bps = round((net / notional) * 10_000, 1) if notional > 0 else 0.0
        strat = _classify(t.reasoning or "")
        row = {
            "trade_id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "qty": qty,
            "entry": entry, "fill": fill,
            "strategy": strat,
            "gross_pnl_inr": gross,
            "spread_cost_inr": spread_cost,
            "brokerage_inr": brokerage,
            "total_cost_inr": total_cost,
            "net_edge_inr": net,
            "edge_to_cost": round(net / total_cost, 2) if total_cost > 0 else 0.0,
            "edge_bps": edge_bps,
            "created_at": t.created_at.isoformat() if getattr(t, "created_at", None) else None,
        }
        rows.append(row)
        by_strategy[strat].append(row)
        by_symbol[t.symbol].append(row)

    def _agg(items: list[dict]) -> dict:
        if not items:
            return {"trades": 0}
        gross = sum(i["gross_pnl_inr"] for i in items)
        cost = sum(i["total_cost_inr"] for i in items)
        net = sum(i["net_edge_inr"] for i in items)
        wins = sum(1 for i in items if i["net_edge_inr"] > 0)
        return {
            "trades": len(items),
            "gross_pnl_inr": round(gross, 2),
            "cost_inr": round(cost, 2),
            "net_edge_inr": round(net, 2),
            "cost_drag_pct": round((cost / abs(gross) * 100), 1) if gross else 0.0,
            "win_rate": round(wins / len(items) * 100, 1),
            "avg_net_edge_inr": round(net / len(items), 2),
        }

    strategy_summary = {k: _agg(v) for k, v in by_strategy.items()}
    symbol_summary = {k: _agg(v) for k, v in by_symbol.items()}
    # Top 15 symbols by trade count for the dashboard panel
    symbol_summary = dict(sorted(symbol_summary.items(),
                                  key=lambda kv: kv[1]["trades"], reverse=True)[:15])

    # Per-symbol scratch economics — how many trades land near net-zero
    # after costs. Scratch = abs(net_edge) <= total_cost. Surfaces the
    # symbols where you're paying to play.
    scratch_per_symbol: dict[str, dict] = {}
    for sym, items in by_symbol.items():
        scratch = sum(1 for r in items if abs(r["net_edge_inr"]) <= r["total_cost_inr"])
        n = len(items)
        scratch_per_symbol[sym] = {
            "trades": n,
            "scratch_count": scratch,
            "scratch_pct": round(scratch / n * 100, 1) if n else 0.0,
            "avg_tick_edge_bps": round(
                sum(r["edge_bps"] for r in items) / n, 1,
            ) if n else 0.0,
        }
    # Surface worst scratch offenders first (top-10)
    scratch_top = dict(sorted(
        scratch_per_symbol.items(), key=lambda kv: -kv[1]["scratch_pct"],
    )[:10])

    return {
        "count": len(rows),
        "totals": _agg(rows),
        "by_strategy": strategy_summary,
        "by_symbol": symbol_summary,
        "scratch_economics": scratch_top,
        "rows": rows[:50],   # cap payload
        "note": (
            "Spread cost = 5 bps × qty × 2 (entry + exit). Brokerage = ₹40 "
            "round-trip (Angel One equity flat). cost_drag_pct = how much of "
            "your gross P&L gets paid back to the broker + bid-ask."
        ),
    }
