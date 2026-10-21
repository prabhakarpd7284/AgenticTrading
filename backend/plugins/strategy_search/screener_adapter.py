"""Replay the live screener's ENABLED strategies on DB candles (read-only).

Uses the real ``ScreenerEngine`` / ``CandleStore`` / ``IndicatorEngine`` /
conditions / ``STRATEGIES`` from ``plugins/strategy_screener`` unchanged —
nothing in that package is modified. Per (symbol, day) a fresh single-symbol
engine is built the way the live ``bootstrap()`` builds it before the open:

* ``prev_day_high/low/close`` = the PRIOR session's NSE bhavcopy H/L/C (live
  reads them from Angel's batch OHLC quote; the intent is "previous day OHLC").
* the prior session's 5-minute bars (resampled from DB 1m, 09:15-aligned) are
  seeded into the 5m buffer for warm-up, as live seeds Angel FIVE_MINUTE
  candles; then 5m/15m indicators are initialised, as in bootstrap step 3.
* the day's 1m bars are replayed as ticks: open, high, low, close of each bar
  with the bar's own timestamp (naive IST, as live uses), volume on the first
  tick. The engine's own ``CandleStore`` builds the 1m/5m/15m bars, VWAP,
  per-strategy cooldowns, the 20-bar symbol cooldown and the
  one-signal-per-tick dedupe — all live code paths.

A bar for minute m is completed by the first tick of minute m+1, so the
engine's signal timestamp is the start of m+1 = the moment the signal is
known. ``Intent.signal_t`` = that time → the simulator enters at the OPEN of
the m+1 bar (not at the engine's ``last_close`` entry).

Only performance change: the indicator engine skips recomputing timeframes no
enabled strategy reads (the 1m snapshot — every condition is 5m/15m and
``_build_signal`` reads 5m/15m). Bar counters, cooldowns and evaluation order
are untouched, so signals are identical (checked in the smoke run).

Exits — the live screener only ALERTS; it has no exit logic. Assumed:
bracket at the signal's own stoploss and target (absolute prices from the
engine), plus the owner trail (BE at +1R, then 1.5×ATR(14) of 5m bars) and
flat by 15:14. Per strategy (``strategies.py``):

    Breakout Long / Breakdown Short   stop = 5-bar 5m swing low/high ∓ ₹2 (fixed
                                      rupees, not %), target 2.0R, min RR 1.5,
                                      09:20–14:30, cooldown 30 bars
    VWAP Bounce Long / Rejection Short stop 1.2×ATR14(5m), target 1.5R,
                                      10:30–14:00, cooldown 60 bars
    Pivot Rejection Long / Short      stop 1.0×ATR14(5m), target 1.5R,
                                      10:00–14:00 (classic S1/R1 reclaim/fail)
    BB Squeeze Breakout (long)        stop 1.2×ATR14(5m), target 1.5R, 09:45–14:00
    EMA Crossover Trend (long)        stop 1.2×ATR14(5m), target 1.5R, 10:00–14:00
    (BB Fade, Morning Range Break: enabled=False → not run)

R targets are measured from the engine's ``last_close`` entry; with
next-bar-open fills the realised R:R differs slightly. A fill that opens
beyond the stop or target is rejected (phase-1 reject).
"""
from __future__ import annotations

from datetime import date

from plugins.strategy_pivot_boss.bars import ist_minute

from .data import SymbolData, epoch_of_ist, ist_dt, resample
from .sim import Intent


def enabled_strategies(names: list[str] | None = None):
    from plugins.strategy_screener.strategies import STRATEGIES
    out = [s for s in STRATEGIES if s.enabled]
    if names:
        want = {n.lower() for n in names}
        out = [s for s in out if s.name.lower() in want]
        missing = want - {s.name.lower() for s in out}
        if missing:
            raise ValueError(f"unknown/disabled screener strategies: {sorted(missing)}")
    return out


def _make_engine(symbol: str, strategies):
    from plugins.strategy_screener.engine import ScreenerEngine
    from plugins.strategy_screener.indicator_engine import IndicatorEngine, IndicatorSnapshot

    used = {c.timeframe for s in strategies for c in s.conditions} | {"5m", "15m"}

    class _SkipUnusedTF(IndicatorEngine):
        def update(self, sym, tf, store):  # noqa: D401 — perf only, see module doc
            if tf not in used:
                return self._cache.get((sym, tf)) or IndicatorSnapshot()
            return super().update(sym, tf, store)

    eng = ScreenerEngine([symbol], strategies)
    eng.indicator_engine = _SkipUnusedTF()
    return eng


class ScreenerSignalSource:
    """``generate(SymbolData) -> list[Intent]`` for the live screener strategies."""

    def __init__(self, strategy_names: list[str] | None = None, skip_unused_tf: bool = True):
        self.strategies = enabled_strategies(strategy_names)
        self.skip_unused_tf = skip_unused_tf
        self.name = "screener"

    def _engine(self, symbol):
        if self.skip_unused_tf:
            return _make_engine(symbol, self.strategies)
        from plugins.strategy_screener.engine import ScreenerEngine
        return ScreenerEngine([symbol], self.strategies)

    def day_signals(self, symbol: str, prev_session, session) -> list:
        """Raw screener ``Signal`` objects for one symbol-day."""
        eng = self._engine(symbol)
        store = eng.stores[symbol]
        pd = session.prev_daily
        if pd is not None:
            store.prev_day_high, store.prev_day_low, store.prev_day_close = pd.h, pd.l, pd.c
        if prev_session is not None and prev_session.bars:
            raw5 = [[ist_dt(b.t), b.o, b.h, b.l, b.c, int(b.v)]
                    for b in resample(prev_session.bars, 5)]
            store.seed_from_candles(raw5, "5m")
        for tf in ("1m", "5m", "15m"):
            if store.bar_count(tf) > 0:
                eng.indicator_engine.update(symbol, tf, store)
        got: list = []
        eng.add_output_handler(got.append)
        for b in session.bars:
            ts = ist_dt(b.t)
            eng.on_tick(symbol, b.o, int(b.v), ts)
            eng.on_tick(symbol, b.h, 0, ts)
            eng.on_tick(symbol, b.l, 0, ts)
            eng.on_tick(symbol, b.c, 0, ts)
        return got

    def generate(self, data: SymbolData) -> list[Intent]:
        out: list[Intent] = []
        prev = None
        for s in data.sessions:
            if s.in_period and s.bars and ist_minute(s.bars[0].t) == 9 * 60 + 15:
                for sig in self.day_signals(data.symbol, prev, s):
                    out.append(to_intent(sig, s.day))
            prev = s
        return out


def to_intent(sig, day: date) -> Intent:
    side = 1 if sig.side == "BUY" else -1
    return Intent(
        strategy=sig.strategy, symbol=sig.symbol, day=day, side=side,
        signal_t=epoch_of_ist(sig.timestamp), stop=float(sig.stoploss),
        target=float(sig.target), trail=True, trail_tf=5,
        priority=float(sig.confidence),
        meta={"engine_entry": round(float(sig.entry), 2), "rr": sig.risk_reward,
              "conf": sig.confidence},
    )
