"""Unit tests for the Step 3 pre-registered rules (plugins/strategy_search/prereg_step3).

Pure Python — synthetic bars, no DB.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from plugins.strategy_pivot_boss.bars import Bar, ist_minute
from plugins.strategy_search.data import DaySession, SymbolData, epoch_at
from plugins.strategy_search.prereg_step3 import (
    ORB30,
    OR30FailFade,
    PPExtFade,
    XSReversal1D,
    f4_vetoes,
    or30,
    s0_first_per_day,
    xs_picks,
)
from plugins.strategy_search.sim import Candidate, Intent

D = date(2026, 6, 10)
SL = 1.0002  # 0.02 % slippage factor


def daily_hist(n=15, tr=10.0, close=100.0, start=date(2026, 5, 1)):
    """n daily bars with constant true range ``tr`` (h-l = tr, flat closes) → dATR = tr."""
    out = []
    for i in range(n):
        d = start + timedelta(days=i)
        t = epoch_at(d, 0, 0)
        out.append(Bar(t, close, close + tr / 2, close - tr / 2, close, 1))
    return out


def minute_bars(day, closes, start=(9, 15), spread=0.05):
    """1m bars with o = previous close (first o = closes[0]), h/l = max/min(o, c) ± spread."""
    t0 = epoch_at(day, *start)
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev
        out.append(Bar(t0 + 60 * i, o, max(o, c) + spread, min(o, c) - spread, c, 1))
        prev = c
    return out


def session(bars, hist, prev=None, day=D):
    return DaySession(day=day, bars=bars, prev_daily=prev or hist[-1], daily_history=hist)


def ist(t):
    m = ist_minute(t)
    return (m // 60, m % 60)


# ── C1 ──────────────────────────────────────────────────────────────────────

class TestPPExtFade:
    def test_fades_open_above_pp(self):
        hist = daily_hist()                      # PP = 100, dATR = 10
        bars = minute_bars(D, [107.0] * 30)      # open 107 → d = +0.7
        it = PPExtFade().generate(SymbolData("X", [session(bars, hist)]))
        assert len(it) == 1
        i = it[0]
        assert i.side == -1
        assert ist(i.signal_t) == (9, 16)
        assert i.target == pytest.approx(100.0)
        e = 107.0 / SL                           # short entry slipped down
        assert i.stop == pytest.approx(e + 10.0)
        assert i.priority == pytest.approx(0.7)

    @pytest.mark.parametrize("op,n", [(104.9, 0), (105.0, 1), (110.0, 1), (110.1, 0), (93.0, 1)])
    def test_band(self, op, n):
        hist = daily_hist()
        bars = minute_bars(D, [op] * 5)
        it = PPExtFade().generate(SymbolData("X", [session(bars, hist)]))
        assert len(it) == n
        if n and op < 100:
            assert it[0].side == 1

    def test_no_datr_skips(self):
        hist = daily_hist(n=14)                  # needs 15 bars
        bars = minute_bars(D, [107.0] * 5)
        assert PPExtFade().generate(SymbolData("X", [session(bars, hist)])) == []


# ── C2 ──────────────────────────────────────────────────────────────────────

def c2_day(path_after_or):
    """OR30: 30 bars between 99 and 101 → OR30 ≈ (101.05, 98.95); then 1m closes."""
    or_part = [100.0, 101.0, 99.0] + [100.0] * 27
    return minute_bars(D, or_part + path_after_or)


class TestOR30FailFade:
    def test_upside_failure_short(self):
        # 09:45..10:04 above OR (102.5, high 103), then back inside at 10:05..10:09
        path = [102.0] * 5 + [103.0] * 5 + [102.5] * 10 + [100.5] * 5 + [100.0] * 60
        bars = c2_day(path)
        hi, lo = or30(bars)
        it = OR30FailFade().generate(SymbolData("X", [session(bars, daily_hist())]))
        assert len(it) == 1
        i = it[0]
        assert i.side == -1
        assert ist(i.signal_t) == (10, 10)        # 10:05 5m bar closes inside
        assert i.target == pytest.approx(lo)
        sess_hi = max(b.h for b in bars if b.t < i.signal_t)
        e = bars[55].o / SL
        assert i.stop == pytest.approx(max(sess_hi + 1.0, e + 7.5))
        assert i.priority == pytest.approx((sess_hi - hi) / 10.0)

    def test_failure_before_1000_kills_day(self):
        # excursion 09:45 bar, back inside 09:50 bar (signal 09:55) → first failure out of
        # window → no trade even though a later excursion/failure happens
        path = [102.0] * 5 + [100.0] * 5 + [102.0] * 10 + [100.0] * 60
        bars = c2_day(path)
        assert OR30FailFade().generate(SymbolData("X", [session(bars, daily_hist())])) == []

    def test_downside_failure_long_and_stop_floor(self):
        path = [98.0] * 15 + [100.0] * 60
        bars = c2_day(path)
        it = OR30FailFade().generate(SymbolData("X", [session(bars, daily_hist())]))
        assert len(it) == 1 and it[0].side == 1
        assert ist(it[0].signal_t) == (10, 5)
        e = bars[50].o * SL
        sess_lo = min(b.l for b in bars if b.t < it[0].signal_t)
        assert it[0].stop == pytest.approx(min(sess_lo - 1.0, e - 7.5))

    def test_after_1330_no_signal(self):
        path = [100.0] * 210 + [102.0] * 20 + [100.0] * 30   # failure ~13:35+
        bars = c2_day(path)
        assert OR30FailFade().generate(SymbolData("X", [session(bars, daily_hist())])) == []


# ── C3 / C4 ─────────────────────────────────────────────────────────────────

class TestORB30:
    def test_short_first_15m_close_below(self):
        path = [99.5] * 15 + [98.0] * 15 + [97.0] * 60      # 10:00 15m bar closes 98 < OR low
        bars = c2_day(path)
        hi, lo = or30(bars)
        it = ORB30(-1).generate(SymbolData("X", [session(bars, daily_hist())]))
        assert len(it) == 1
        i = it[0]
        assert i.side == -1 and i.target is None
        assert ist(i.signal_t) == (10, 15)
        e = bars[60].o / SL
        assert i.stop == pytest.approx(e + max(hi - e, 7.5))
        assert ORB30(1).generate(SymbolData("X", [session(bars, daily_hist())])) == []

    def test_long_mirror_and_wide_stop_floor(self):
        path = [101.6] * 15 + [102.0] * 60
        bars = c2_day(path)
        hi, lo = or30(bars)
        it = ORB30(1).generate(SymbolData("X", [session(bars, daily_hist(tr=40.0))]))
        assert len(it) == 1
        e = bars[45].o * SL
        assert ist(it[0].signal_t) == (10, 0)
        assert it[0].stop == pytest.approx(e - max(e - lo, 30.0))

    def test_after_noon_ignored(self):
        path = [100.0] * 150 + [97.0] * 60     # breaks at 12:15 bar → signal 12:30
        bars = c2_day(path)
        assert ORB30(-1).generate(SymbolData("X", [session(bars, daily_hist())])) == []


# ── C5 ──────────────────────────────────────────────────────────────────────

class TestXSReversal:
    def _daily(self, last_move):
        h = daily_hist(n=16)
        b = h[-1]
        h[-1] = Bar(b.t, b.o, b.h + abs(last_move), b.l - abs(last_move), b.c + last_move, 1)
        return h

    def test_picks_extremes_with_threshold(self):
        day = date(2026, 6, 1)
        daily = {"UP": self._daily(+20), "DN": self._daily(-15), "FLAT": self._daily(0.5),
                 "MILD": self._daily(+2)}
        p = xs_picks(daily, [day])[day]
        syms = {s: side for s, side, *_ in p}
        assert syms == {"UP": -1, "DN": 1}
        z_up = next(z for s, _, z, _ in p if s == "UP")
        assert z_up >= 1.0

    def test_no_pick_below_threshold(self):
        day = date(2026, 6, 1)
        daily = {"A": self._daily(+1), "B": self._daily(-1)}
        assert xs_picks(daily, [day])[day] == []

    def test_uses_only_prior_bars(self):
        day = date(2026, 5, 16)                    # daily_hist covers 05-01..05-16
        daily = {"UP": self._daily(+20)}
        # the +20 bar is ON 05-16 → not visible on 05-16; 15 prior bars → dATR ok, z = 0
        assert xs_picks(daily, [day])[day] == []

    def test_generate_stop_one_datr(self):
        src = XSReversal1D({D: [("X", -1, 2.0, 10.0)]})
        bars = minute_bars(D, [100.0, 101.0, 101.0])
        it = src.generate(SymbolData("X", [session(bars, daily_hist())]))
        assert len(it) == 1 and it[0].side == -1
        assert it[0].stop == pytest.approx(100.0 / SL + 10.0)   # 09:16 open = 100
        assert src.generate(SymbolData("Y", [session(bars, daily_hist())])) == []


# ── filter block ────────────────────────────────────────────────────────────

def cand(sym, day, strat, sig_min, side=1):
    st = epoch_at(day, sig_min // 60, sig_min % 60)
    it = Intent(strategy=strat, symbol=sym, day=day, side=side, signal_t=st, stop=90.0)
    return Candidate(intent=it, entry_t=st, entry_raw=100, entry_fill=100, exit_t=st + 600,
                     exit_raw=101, exit_fill=101, reason="eod", risk_per_share=10, mfe_r=0,
                     mae_r=0, trail_moves=0, final_stop=90)


class TestFilters:
    def test_s0_keeps_first_per_symbol_day_strategy(self):
        cs = [cand("A", D, "Breakout Long", 600), cand("A", D, "Breakout Long", 570),
              cand("A", D, "Breakdown Short", 560), cand("B", D, "Breakout Long", 620),
              cand("A", D + timedelta(days=1), "Breakout Long", 700)]
        out = s0_first_per_day(cs, "Breakout Long")
        assert [(c.intent.symbol, ist_minute(c.intent.signal_t)) for c in out] == [
            ("A", 570), ("B", 620), ("A", 700)]

    def _bars(self, closes):
        return minute_bars(D, closes, spread=0.0)

    def test_f4_vetoes_gap_direction_untouched(self):
        bars = self._bars([100.5] * 30)                     # never back to 100
        assert f4_vetoes(1, epoch_at(D, 9, 40), 100.5, 100.0, bars) is True
        assert f4_vetoes(-1, epoch_at(D, 9, 40), 100.5, 100.0, bars) is False   # against gap

    @pytest.mark.parametrize("op", [100.2, 101.1])           # 0.2 % and 1.1 % gaps
    def test_f4_gap_band(self, op):
        bars = self._bars([op] * 30)
        assert f4_vetoes(1, epoch_at(D, 9, 40), op, 100.0, bars) is False

    def test_f4_after_1015_not_vetoed(self):
        bars = self._bars([100.5] * 80)
        assert f4_vetoes(1, epoch_at(D, 10, 15), 100.5, 100.0, bars) is True
        assert f4_vetoes(1, epoch_at(D, 10, 16), 100.5, 100.0, bars) is False

    def test_f4_touch_since_0916_unvetoes(self):
        closes = [100.5, 100.5, 100.0] + [100.5] * 30      # 09:17 bar touches 100
        bars = self._bars(closes)
        assert f4_vetoes(1, epoch_at(D, 9, 40), 100.5, 100.0, bars) is False
        # the touch is AFTER the signal bar → still vetoed
        assert f4_vetoes(1, epoch_at(D, 9, 17), 100.5, 100.0, bars) is True

    def test_f4_0915_touch_ignored(self):
        bars = [Bar(epoch_at(D, 9, 15), 100.5, 100.6, 99.9, 100.5, 1)] + self._bars([100.5] * 30)[1:]
        assert f4_vetoes(1, epoch_at(D, 9, 40), 100.5, 100.0, bars) is True
