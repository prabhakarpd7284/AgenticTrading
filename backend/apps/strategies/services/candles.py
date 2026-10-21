"""Historical candles for Studio backtests — read-only, one symbol at a time.

Source: ``market_data.Candle`` via the research harness's ``CandleSource``
(raw SQL, one symbol per query, never the whole table in memory):
  * ``1m`` — Angel One SmartAPI bars (historical backfill + live tick
    aggregation), NIFTY 50 and a few dozen more names + indices, from
    2026-04-20; cut at 15:14 IST (no bars 15:15–15:29 since 2026-08-03).
    Resampled to 5m/15m/60m buckets aligned to 09:15, never across sessions.
  * ``1d`` — NSE bhavcopy (official OHLC), ~2,900 stocks, same span,
    NOT adjusted for corporate actions.
Nothing synthetic is written to that table (the paper adapter synthesises
quotes, never candles), so results are stamped ``synthetic: false``.

``LoadedSymbol.stats`` feeds the provenance badge: bars, sessions, first/last
bar, and sessions with gaps (< 90 % of the expected 1m bars).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from plugins.strategy_backtest.types import Bar as EBar

from apps.strategies.services.rules import SeriesFrame
from apps.strategies.services.spec import TIMEFRAMES

INTRADAY_SOURCE = ("market_data.Candle 1m — Angel One SmartAPI (historical backfill + live tick "
                   "aggregation), cut at 15:14 IST")
DAILY_SOURCE = "market_data.Candle 1d — NSE bhavcopy official OHLC (not adjusted for corporate actions)"
EXPECTED_1M_PER_SESSION = 360          # 09:15 → 15:14
GAP_FRACTION = 0.9


@dataclass
class LoadedSymbol:
    symbol: str
    frame: SeriesFrame
    bars: list[EBar]
    stats: dict = field(default_factory=dict)
    raw: object = None                  # strategy_search SymbolData (1m) when kept


def iso_ist(t: int) -> str:
    from plugins.strategy_search.data import ist_dt
    return ist_dt(t).strftime("%Y-%m-%dT%H:%M:%S") + "+05:30"


def frame_from_bars(symbol: str, bars, timeframe: str, prev_day_by_session: dict[str, tuple | None]
                    ) -> tuple[SeriesFrame, list[EBar]]:
    """``bars``: pivot-boss Bars (epoch ``t`` = bar start) at ``timeframe``."""
    from plugins.strategy_pivot_boss.bars import ist_minute
    daily = timeframe == "1d"
    ts = [iso_ist(b.t) for b in bars]
    sess = [t[:10] for t in ts]
    frame = SeriesFrame(
        symbol=symbol, timestamps=ts,
        open=[b.o for b in bars], high=[b.h for b in bars], low=[b.l for b in bars],
        close=[b.c for b in bars], volume=[float(b.v) for b in bars],
        minute=[0 if daily else ist_minute(b.t) for b in bars], session=sess,
        prev_day=[prev_day_by_session.get(s) for s in sess], bar_minutes=TIMEFRAMES[timeframe],
    )
    ebars = [EBar(timestamp=t, open=b.o, high=b.h, low=b.l, close=b.c, volume=int(b.v))
             for t, b in zip(ts, bars)]
    return frame, ebars


class DbCandleLoader:
    """``load(symbol, timeframe, start, end, warmup_bars)``."""

    def __init__(self, end: date):
        from plugins.strategy_search.data import CandleSource
        # Studio runs are not research claims: the research OOS guard is
        # lifted here and the overlap is labelled in provenance instead.
        self.src = CandleSource(max_date=end, allow_oos=True)

    def source_label(self, timeframe: str) -> str:
        return DAILY_SOURCE if timeframe == "1d" else INTRADAY_SOURCE

    def sessions(self, start: date, end: date) -> list[date]:
        return self.src.sessions(start, end)

    def load(self, symbol: str, timeframe: str, start: date, end: date, warmup_bars: int = 60,
             keep_raw: bool = False) -> LoadedSymbol | None:
        from plugins.strategy_search.data import day_of, resample
        if timeframe == "1d":
            lo = start - timedelta(days=int(warmup_bars * 1.6) + 10)
            daily = self.src.daily_bars(symbol, end, start=lo)
            if not daily:
                return None
            prev = {}
            for k, b in enumerate(daily):
                prev[day_of(b.t).isoformat()] = ((daily[k - 1].h, daily[k - 1].l, daily[k - 1].c)
                                                 if k else None)
            frame, ebars = frame_from_bars(symbol, daily, "1d", prev)
            in_range = [s for s in frame.session if start.isoformat() <= s <= end.isoformat()]
            ref = {d.isoformat() for d in self.src.sessions(start, end)}
            missing = sorted(ref - set(in_range))
            stats = {"bars": len(in_range), "sessions": len(in_range),
                     "first": in_range[0] if in_range else None, "last": in_range[-1] if in_range else None,
                     "gap_sessions": len(missing), "gap_days": missing[:10],
                     "warmup_bars": len(frame) - len(in_range)}
            return LoadedSymbol(symbol, frame, ebars, stats)

        minutes = TIMEFRAMES[timeframe]
        per_session = max(1, math.ceil(EXPECTED_1M_PER_SESSION / minutes))
        warm_sessions = min(30, math.ceil(warmup_bars / per_session) + 1)
        data = self.src.load_symbol(symbol, start, end, warmup_sessions=warm_sessions)
        if not data.sessions:
            return None
        bars, prev, gaps, n_in, in_sess = [], {}, [], 0, []
        for s in data.sessions:
            key = s.day.isoformat()
            pd = s.prev_daily
            prev[key] = (pd.h, pd.l, pd.c) if pd is not None else None
            if s.in_period:
                in_sess.append(key)
                n_in += len(s.bars)
                if len(s.bars) < GAP_FRACTION * EXPECTED_1M_PER_SESSION:
                    gaps.append(key)
            bars.extend(resample(s.bars, minutes))
        frame, ebars = frame_from_bars(symbol, bars, timeframe, prev)
        in_set = set(in_sess)
        stats = {"bars_1m": n_in, "bars": sum(1 for s in frame.session if s in in_set),
                 "sessions": len(in_sess), "first": in_sess[0] if in_sess else None,
                 "last": in_sess[-1] if in_sess else None, "gap_sessions": len(gaps),
                 "gap_days": gaps[:10], "warmup_sessions": len(data.sessions) - len(in_sess)}
        return LoadedSymbol(symbol, frame, ebars, stats, raw=data if keep_raw else None)
