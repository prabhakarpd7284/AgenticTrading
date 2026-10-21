"""Breakeven-then-ATR trailing stop — docs/strategies/TRADING_RULES.md §2.

    * initial stop comes from the setup (``Trade.stop_loss``, defines 1R)
    * open profit ≥ 1R            → stop to breakeven (entry)
    * after that                  → stop trails 1.5 × ATR(14) behind the best
                                    price reached (never below breakeven)
    * the stop only ever moves in the trade's favour

``next_stop`` is pure. ``atr_for_symbol`` reads candles (cached 60 s).
The pyramid engine has its own trail (CLAUDE.md invariant #10) and does not
use this module.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULTS = {
    "TRAIL_BREAKEVEN_R": 1.0,
    "TRAIL_ATR_MULT": 1.5,
    "TRAIL_ATR_PERIOD": 14,
    "TRAIL_ATR_TIMEFRAME_MIN": 5,
}

INITIAL, BREAKEVEN, TRAILING = "initial", "breakeven", "trailing"


def _setting(key):
    from django.conf import settings
    return getattr(settings, "TRADING_RULES", {}).get(key, DEFAULTS[key])


@dataclass(frozen=True)
class TrailDecision:
    stop: float
    phase: str
    moved: bool


def next_stop(*, side: str, entry: float, initial_stop: float, best_price: float,
              atr: float | None, current_stop: float | None = None,
              breakeven_r: float = 1.0, atr_mult: float = 1.5) -> TrailDecision:
    """New stop for a trade. Ratchets only in the trade's favour.

    ``best_price`` is the most favourable price since entry (highest for a
    long, lowest for a short). ``atr`` ≤ 0 / None → breakeven move only.
    """
    long_side = side == "BUY"
    cur = float(current_stop if current_stop is not None else initial_stop)
    risk = abs(entry - initial_stop)
    if risk <= 0:
        return TrailDecision(cur, INITIAL, False)

    favour = (best_price - entry) if long_side else (entry - best_price)
    candidate = float(initial_stop)
    if favour >= breakeven_r * risk:
        candidate = float(entry)
        if atr and atr > 0:
            trail = best_price - atr_mult * atr if long_side else best_price + atr_mult * atr
            candidate = max(candidate, trail) if long_side else min(candidate, trail)

    new = max(cur, candidate) if long_side else min(cur, candidate)
    if new == entry:
        phase = BREAKEVEN
    elif (long_side and new > entry) or (not long_side and new < entry):
        phase = TRAILING
    else:
        phase = INITIAL
    return TrailDecision(round(new, 4), phase, new != cur)


def wilder_atr(bars: list[tuple[float, float, float]], period: int = 14) -> float | None:
    """ATR over (high, low, close) bars, oldest first. None if too few bars."""
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, _ = bars[i]  # noqa: E741
        pc = bars[i - 1][2]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def normalize_symbol(symbol: str) -> str:
    """'NSE:RELIANCE-EQ' → 'RELIANCE' (Fyers style); plain symbols unchanged."""
    s = (symbol or "").strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1]
    for suffix in ("-EQ", "-BE"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def atr_for_symbol(symbol: str) -> float | None:
    """ATR(period) on the trading timeframe, from 1m candles resampled to
    ``TRAIL_ATR_TIMEFRAME_MIN`` minutes. Cached 60 s. None when unknown."""
    from django.core.cache import cache

    sym = normalize_symbol(symbol)
    period = int(_setting("TRAIL_ATR_PERIOD"))
    tf = max(1, int(_setting("TRAIL_ATR_TIMEFRAME_MIN")))
    key = f"rules:atr:{sym}:{tf}:{period}"
    hit = cache.get(key)
    if hit is not None:
        return hit or None

    from apps.market_data.models import Candle

    need = (period + 2) * tf
    rows = list(
        Candle.objects.filter(symbol__tradingsymbol=sym, symbol__exchange="NSE", interval="1m")
        .order_by("-t").values_list("t", "h", "l", "c")[:need]
    )
    rows.reverse()
    bars: list[tuple[float, float, float]] = []
    bucket: list = []
    for t, h, l, c in rows:  # noqa: E741
        key_min = (t.hour * 60 + t.minute) // tf
        if bucket and (bucket[0][0].date() != t.date() or bucket[0][1] != key_min):
            bars.append((max(b[2] for b in bucket), min(b[3] for b in bucket), bucket[-1][4]))
            bucket = []
        bucket.append((t, key_min, float(h), float(l), float(c)))
    if bucket:
        bars.append((max(b[2] for b in bucket), min(b[3] for b in bucket), bucket[-1][4]))
    atr = wilder_atr(bars, period)
    cache.set(key, atr or 0.0, 60)
    return atr


def update_trailing_stop(trade, ltp: float) -> TrailDecision | None:
    """Advance ``trade.best_price`` / ``trade.trail_stop`` on a tick and persist.

    Returns the decision, or None when the trade isn't eligible. Emits a
    ``trade.trailed`` event when the stop moves (non-blocking).
    """
    from decimal import Decimal

    from apps.trading.models import Trade

    if ltp is None or ltp <= 0 or trade.stop_loss is None:
        return None
    entry = float(trade.fill_price or trade.entry_price)
    long_side = trade.side == "BUY"
    prev_best = float(trade.best_price) if trade.best_price is not None else entry
    best = max(prev_best, ltp) if long_side else min(prev_best, ltp)
    cur = float(trade.trail_stop) if trade.trail_stop is not None else None

    decision = next_stop(
        side=trade.side, entry=entry, initial_stop=float(trade.stop_loss),
        best_price=best, atr=atr_for_symbol(trade.symbol), current_stop=cur,
        breakeven_r=float(_setting("TRAIL_BREAKEVEN_R")),
        atr_mult=float(_setting("TRAIL_ATR_MULT")),
    )
    updates = {}
    if best != prev_best or trade.best_price is None:
        updates["best_price"] = Decimal(str(round(best, 4)))
    if decision.moved:
        updates["trail_stop"] = Decimal(str(decision.stop))
    if updates:
        Trade.objects.filter(pk=trade.pk).update(**updates)
        for k, v in updates.items():
            setattr(trade, k, v)
    if decision.moved:
        from apps.events.models import Event
        from apps.events.services.event_writer import emit

        emit(
            tenant=trade.tenant, type=Event.Type.TRADE_TRAILED, trade_id=trade.id,
            actor_kind=Event.ActorKind.SYSTEM, step_name="rules.trailing_stop",
            text=f"trail {trade.symbol} stop → {decision.stop:.2f} ({decision.phase})",
            payload={"stop": decision.stop, "phase": decision.phase,
                     "best_price": best, "previous_stop": cur or float(trade.stop_loss)},
        )
    return decision
