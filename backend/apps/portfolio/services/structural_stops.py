"""Structural-stop suggester.

For every open swing/positional position, propose three stop levels —

  swing_low : last meaningful swing low on the daily chart
  ten_wma   : 10-week moving average
  atr_trail : 2.5 × ATR(14) trailed off the recent high

— plus a recommended pick based on which is tightest while still respecting
the structural-low rule, the R-distance from entry, and the resulting % loss
if the trader were to hold full size to that stop.

Pure-Python; reads daily candles via the legacy broker (cached 10 min).
"""
from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any

from django.core.cache import cache

_CANDLES_TTL = 600
_SWING_LOOKBACK = 5          # bars on each side of a candidate pivot


def _fetch_daily(symbol: str, days: int = 80) -> list[dict]:
    """Daily OHLC for `symbol`; cached 10 min. Each row: {h,l,c,d}."""
    key = f"stops:daily:{symbol}:{days}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service

        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, [], _CANDLES_TTL); return []

        today = date.today()
        start = (today - timedelta(days=days + 14)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        rows = [
            {"d": str(r[0]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4])}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, rows, _CANDLES_TTL)
        return rows
    except Exception:  # noqa: BLE001
        cache.set(key, [], _CANDLES_TTL)
        return []


def _swing_low(daily: list[dict], lookback: int = _SWING_LOOKBACK) -> float:
    """Last bar whose low is the lowest within ±lookback bars."""
    if len(daily) < 2 * lookback + 1:
        return 0.0
    for i in range(len(daily) - lookback - 1, lookback - 1, -1):
        win = daily[i - lookback:i + lookback + 1]
        if daily[i]["l"] == min(b["l"] for b in win):
            return daily[i]["l"]
    return min(b["l"] for b in daily[-20:]) if daily else 0.0


def _ten_week_ma(daily: list[dict]) -> float:
    """10W ≈ 50 trading-day SMA on closes."""
    if len(daily) < 50:
        return 0.0
    return round(sum(b["c"] for b in daily[-50:]) / 50.0, 2)


def _atr_trail(daily: list[dict], period: int = 14, mult: float = 2.5) -> float:
    """Most-recent-high − mult × ATR(period)."""
    if len(daily) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(daily)):
        h, l = daily[i]["h"], daily[i]["l"]
        pc = daily[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = statistics.mean(trs[-period:])
    recent_high = max(b["h"] for b in daily[-20:])
    return round(recent_high - atr * mult, 2)


def _empty_row(*, position_id, symbol, entry, qty) -> dict:
    return {
        "position_id": position_id, "symbol": symbol,
        "entry": entry, "qty": qty,
        "swing_low": 0.0, "ten_wma": 0.0, "atr_trail": 0.0,
        "recommended": None, "r_distance": 0.0, "pct_loss": 0.0,
        "ledger": [],
        "note": "no daily candles available",
    }


def build_structural_stops(tenant=None) -> dict[str, Any]:
    from trading.models import TradeJournal

    open_trades = list(
        TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER", "APPROVED"))
        .exclude(fill_price__isnull=True)
        .order_by("-created_at")[:50]
    )

    rows: list[dict] = []
    for t in open_trades:
        entry = float(t.entry_price or 0)
        qty = int(t.quantity or 0)
        daily = _fetch_daily(t.symbol)
        if not daily or entry <= 0:
            rows.append(_empty_row(
                position_id=t.id, symbol=t.symbol, entry=entry, qty=qty,
            ))
            continue

        sl = round(_swing_low(daily), 2)
        wma = _ten_week_ma(daily)
        atr = _atr_trail(daily)
        candidates = {k: v for k, v in (("swing_low", sl), ("ten_wma", wma), ("atr_trail", atr)) if v > 0}

        # Recommend the tightest stop that's still below entry; otherwise the
        # nearest (least-damaging) option.
        below = {k: v for k, v in candidates.items() if v < entry}
        if below:
            rec = max(below, key=lambda k: below[k])     # tightest still safe
        elif candidates:
            rec = min(candidates, key=lambda k: abs(entry - candidates[k]))
        else:
            rec = None

        rec_val = candidates.get(rec, 0.0) if rec else 0.0
        r_distance = abs(entry - rec_val) if rec_val > 0 else 0.0
        pct_loss = (r_distance / entry * 100.0) if entry > 0 and r_distance > 0 else 0.0

        # Build a 10-bar history so the trader can see how each candidate
        # stop has drifted — useful when deciding whether to ratchet up.
        history: list[dict] = []
        for i in range(max(0, len(daily) - 10), len(daily)):
            slice_ = daily[:i + 1]
            history.append({
                "date": slice_[-1]["d"][:10] if "d" in slice_[-1] else "",
                "swing_low": round(_swing_low(slice_), 2),
                "ten_wma": _ten_week_ma(slice_),
                "atr_trail": _atr_trail(slice_),
            })

        rows.append({
            "position_id": t.id, "symbol": t.symbol, "side": t.side,
            "entry": entry, "qty": qty,
            "swing_low": sl, "ten_wma": wma, "atr_trail": atr,
            "recommended": rec, "recommended_value": rec_val,
            "r_distance": round(r_distance, 2),
            "pct_loss": round(pct_loss, 2),
            "loss_at_stop_inr": round(r_distance * qty, 2),
            "ledger": history,
        })

    # ── Portfolio-aggregated open-risk ladder ────────────────────────────
    # Total ₹ on the line if every position hit its recommended stop, plus
    # a per-position rank-ordered "heat" list (biggest loss-at-stop first).
    capital = 0.0
    try:
        from trading.models import PortfolioSnapshot
        capital = float(PortfolioSnapshot.objects.latest().capital)
    except Exception:  # noqa: BLE001
        capital = 0.0
    total_loss = sum((r.get("loss_at_stop_inr") or 0.0) for r in rows)
    heat_list = sorted(
        [r for r in rows if (r.get("loss_at_stop_inr") or 0) > 0],
        key=lambda r: r["loss_at_stop_inr"], reverse=True,
    )
    ladder = [
        {
            "rank": i + 1,
            "symbol": r["symbol"],
            "loss_at_stop_inr": r["loss_at_stop_inr"],
            "pct_of_capital": round((r["loss_at_stop_inr"] / capital) * 100, 2) if capital > 0 else 0.0,
            "pct_loss_per_share": r["pct_loss"],
            "recommended": r.get("recommended"),
        }
        for i, r in enumerate(heat_list[:20])
    ]

    return {
        "count": len(rows),
        "rows": rows,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "open_risk_ladder": ladder,
        "total_loss_at_stop_inr": round(total_loss, 2),
        "total_loss_pct_of_capital": round((total_loss / capital) * 100, 2) if capital > 0 else 0.0,
        "capital": capital,
        "note": (
            "Stops are suggestions from 80 daily bars. The open-risk ladder "
            "ranks positions by ₹-at-stop so you can see which single name "
            "carries the most heat. total_loss_pct_of_capital > 3% means "
            "you're at the daily cap if everything goes wrong at once."
        ),
    }
