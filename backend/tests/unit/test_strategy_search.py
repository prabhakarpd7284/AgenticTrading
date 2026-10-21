"""Unit tests for the strategy-search harness (plugins/strategy_search).

Pure Python — no DB, no broker. Cost examples are hand-computed.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from plugins.strategy_pivot_boss.bars import Bar, ist_minute
from plugins.strategy_search.costs import CostModel, slip
from plugins.strategy_search.data import epoch_at, epoch_of_ist, ist_dt, resample
from plugins.strategy_search.periods import (
    IS_END,
    OOSAccessError,
    check_date,
    resolve_period,
)
from plugins.strategy_search.rules import (
    COOLDOWN,
    NORMAL,
    REDUCED,
    CooldownTracker,
    nth_trading_day_after,
    trail_next_stop,
    wilder_atr,
)
from plugins.strategy_search.sim import (
    Candidate,
    Intent,
    PortfolioSimulator,
    SimConfig,
    resolve_intent,
    rr_gate_skip,
)

D = date(2026, 5, 6)  # a Wednesday


def bars_from(day, rows, start=(9, 15)):
    """rows = [(o, h, l, c)], one per minute from ``start``."""
    t0 = epoch_at(day, *start)
    return [Bar(t0 + 60 * i, o, h, lo, c, 1000) for i, (o, h, lo, c) in enumerate(rows)]


def weekdays(d):
    return d.weekday() < 5


NOSLIP = SimConfig(slippage_pct=0.0, trail=False, cooldown=False)


# ── costs ───────────────────────────────────────────────────────────────────

class TestCosts:
    def test_buy_leg_brokerage_capped(self):
        # 100 × ₹1000 = ₹1,00,000: brokerage min(20, 100) = 20; txn 2.97;
        # SEBI 0.10; GST 0.18 × 23.07 = 4.1526; stamp 3.00; STT 0 → 30.2226
        b = CostModel().breakdown(1000, 100, is_buy=True)
        assert b.brokerage == pytest.approx(20.0)
        assert b.txn == pytest.approx(2.97)
        assert b.sebi == pytest.approx(0.10)
        assert b.gst == pytest.approx(4.1526)
        assert b.stamp == pytest.approx(3.0)
        assert b.stt == 0
        assert b.total == pytest.approx(30.2226)

    def test_sell_leg_has_stt_no_stamp(self):
        # 100 × ₹1010 = ₹1,01,000: brokerage 20; txn 2.9997; SEBI 0.101;
        # GST 0.18 × 23.1007 = 4.158126; STT 25.25 → 52.508826
        assert CostModel().order_cost(1010, 100, is_buy=False) == pytest.approx(52.508826)

    def test_small_order_percent_brokerage(self):
        # 10 × ₹100 = ₹1,000: brokerage 1.00; txn 0.0297; SEBI 0.001;
        # GST 0.18 × 1.0307 = 0.185526; stamp 0.03 → 1.246226
        assert CostModel().order_cost(100, 10, is_buy=True) == pytest.approx(1.246226)

    def test_round_trip_short_sells_first(self):
        cm = CostModel()
        # short: sell @1010 (STT), buy back @1000 (stamp)
        assert cm.round_trip(1010, 1000, 100, side=-1) == pytest.approx(52.508826 + 30.2226)
        assert cm.round_trip(1000, 1010, 100, side=1) == pytest.approx(30.2226 + 52.508826)

    def test_slippage_direction(self):
        assert slip(100, 1, 0.02, entering=True) == pytest.approx(100.02)
        assert slip(100, 1, 0.02, entering=False) == pytest.approx(99.98)
        assert slip(100, -1, 0.02, entering=True) == pytest.approx(99.98)
        assert slip(100, -1, 0.02, entering=False) == pytest.approx(100.02)


# ── trailing ────────────────────────────────────────────────────────────────

class TestTrail:
    def test_no_move_below_1r(self):
        assert trail_next_stop(side=1, entry=100, initial_stop=98, best=101.9, atr=0.5,
                               current_stop=98) == 98

    def test_breakeven_at_1r_then_atr_trail(self):
        # best 102 = +1R → candidate max(100, 102 − 1.5×2=99) = 100 (breakeven)
        assert trail_next_stop(side=1, entry=100, initial_stop=98, best=102, atr=2.0,
                               current_stop=98) == 100
        # best 105 → 105 − 0.75 = 104.25
        assert trail_next_stop(side=1, entry=100, initial_stop=98, best=105, atr=0.5,
                               current_stop=100) == pytest.approx(104.25)

    def test_ratchet_only(self):
        # ATR widening must not loosen the stop
        assert trail_next_stop(side=1, entry=100, initial_stop=98, best=105, atr=3.0,
                               current_stop=104.25) == 104.25

    def test_short_mirror(self):
        assert trail_next_stop(side=-1, entry=100, initial_stop=102, best=98, atr=None,
                               current_stop=102) == 100
        assert trail_next_stop(side=-1, entry=100, initial_stop=102, best=95, atr=0.5,
                               current_stop=100) == pytest.approx(95.75)

    def test_matches_live_next_stop_when_available(self):
        ts = pytest.importorskip("apps.trading.services.trailing_stop")
        rnd = random.Random(7)
        for _ in range(500):
            side = rnd.choice([1, -1])
            entry = 100.0
            init = entry - side * rnd.uniform(0.5, 3)
            best = entry + side * rnd.uniform(-1, 8)
            atr = rnd.choice([None, rnd.uniform(0.1, 3)])
            cur = init if rnd.random() < 0.5 else trail_next_stop(
                side=side, entry=entry, initial_stop=init, best=best, atr=atr, current_stop=init)
            mine = trail_next_stop(side=side, entry=entry, initial_stop=init, best=best,
                                   atr=atr, current_stop=cur)
            live = ts.next_stop(side="BUY" if side > 0 else "SELL", entry=entry,
                                initial_stop=init, best_price=best, atr=atr, current_stop=cur)
            assert mine == pytest.approx(live.stop, abs=1e-4)

    def test_wilder_atr(self):
        # TRs: 2, 2, 4 ; period 2 → seed (2+2)/2 = 2 → (2×1 + 4)/2 = 3
        hlc = [(11, 9, 10), (11, 9, 10), (12, 10, 11), (13, 9, 10)]
        assert wilder_atr(hlc, 2) == pytest.approx(3.0)
        assert wilder_atr(hlc[:2], 2) is None


# ── cooldown ────────────────────────────────────────────────────────────────

class TestCooldown:
    def test_weekend_cooldown_and_half_risk(self):
        cd = CooldownTracker(weekdays)
        for d in (date(2026, 5, 6), date(2026, 5, 7), date(2026, 5, 8)):  # Wed Thu Fri
            assert cd.status(d) == (NORMAL, 1.0)
            cd.close_day(d, -100)
        assert cd.status(date(2026, 5, 11)) == (COOLDOWN, 0.0)   # Mon
        assert cd.status(date(2026, 5, 12)) == (COOLDOWN, 0.0)   # Tue
        assert cd.status(date(2026, 5, 13)) == (REDUCED, 0.5)    # Wed
        cd.close_day(date(2026, 5, 13), -50)                     # still reduced, streak 1
        assert cd.status(date(2026, 5, 14)) == (REDUCED, 0.5)
        cd.close_day(date(2026, 5, 14), 10)                      # first profitable day
        assert cd.status(date(2026, 5, 15)) == (NORMAL, 1.0)

    def test_holiday_skipped(self):
        hol = {date(2026, 5, 11)}
        cd = CooldownTracker(lambda d: weekdays(d) and d not in hol)
        for d in (date(2026, 5, 6), date(2026, 5, 7), date(2026, 5, 8)):
            cd.close_day(d, -1)
        assert cd.until == date(2026, 5, 13)
        assert nth_trading_day_after(date(2026, 5, 8), 2, weekdays) == date(2026, 5, 12)

    def test_zero_day_breaks_streak_no_trade_days_skipped(self):
        cd = CooldownTracker(weekdays)
        cd.close_day(date(2026, 5, 4), -1)
        cd.close_day(date(2026, 5, 5), -1)
        cd.close_day(date(2026, 5, 6), 0)       # flat day breaks the streak
        cd.close_day(date(2026, 5, 7), -1)
        cd.close_day(date(2026, 5, 11), -1)     # 8th had no trades → not a day
        assert cd.status(date(2026, 5, 12)) == (NORMAL, 1.0)
        cd.close_day(date(2026, 5, 12), -1)
        assert cd.status(date(2026, 5, 13))[0] == COOLDOWN

    def test_matches_live_evaluate_cooldown(self):
        cdm = pytest.importorskip("apps.trading.services.cooldown")
        rnd = random.Random(11)
        for _ in range(60):
            days, d = [], date(2026, 5, 4)
            for _ in range(40):
                d += timedelta(days=1)
                if not weekdays(d) or rnd.random() < 0.15:
                    continue
                days.append((d, rnd.choice([-1, -1, 1, 0, -2, 3])))
            tr = CooldownTracker(weekdays)
            for i, (dd, pnl) in enumerate(days):
                live = cdm.evaluate_cooldown([cdm.DayPnl(x, p) for x, p in days[:i]], dd,
                                             is_trading_day=weekdays, loss_days=3,
                                             cooldown_days=2, reduced_multiplier=0.5)
                st, mult = tr.status(dd)
                assert (st, mult) == (live.status, live.risk_multiplier), (dd, days[:i])
                if st != COOLDOWN:      # automated trades only happen outside cooldown
                    tr.close_day(dd, pnl)
                else:
                    days[i] = (dd, 0)   # no trades that day → excluded below
            # (days inside cooldown had no automated trades; the live evaluator skips them)


# ── single-trade path ───────────────────────────────────────────────────────

def _intent(side=1, stop=99.0, target=None, signal=(9, 16), **kw):
    return Intent(strategy="t", symbol="X", day=D, side=side,
                  signal_t=epoch_at(D, *signal), stop=stop, target=target, **kw)


class TestResolve:
    def test_entry_next_bar_open_and_target(self):
        bars = bars_from(D, [(100, 100.5, 99.8, 100.2), (100.3, 100.6, 100.1, 100.5),
                             (100.5, 102.2, 100.4, 102.0)])
        c = resolve_intent(_intent(target=102.0), bars, NOSLIP, None)
        assert c.entry_t == bars[1].t and c.entry_raw == 100.3
        assert c.reason == "target" and c.exit_raw == 102.0

    def test_both_hit_same_bar_stop_first(self):
        bars = bars_from(D, [(100, 100, 100, 100), (100, 100.2, 99.9, 100),
                             (100, 103, 98, 100)])
        c = resolve_intent(_intent(target=102.0), bars, NOSLIP, None)
        assert c.reason == "stop" and c.exit_raw == 99.0 and c.exit_t == bars[2].t

    def test_gap_through_stop_fills_at_open(self):
        bars = bars_from(D, [(100, 100, 100, 100), (100, 100.2, 99.9, 100),
                             (98.0, 98.5, 97.5, 98.2)])
        c = resolve_intent(_intent(), bars, NOSLIP, None)
        assert c.reason == "stop" and c.exit_raw == 98.0

    def test_entry_beyond_stop_rejected(self):
        bars = bars_from(D, [(100, 100, 100, 100), (98.5, 99, 98, 98.5)])
        assert resolve_intent(_intent(), bars, NOSLIP, None) == "entry_beyond_stop"

    def test_eod_exit_on_last_bar_close(self):
        bars = bars_from(D, [(100, 100, 100, 100), (100, 100.2, 99.9, 100.1),
                             (100.1, 100.3, 100, 100.25)])
        c = resolve_intent(_intent(), bars, NOSLIP, None)
        assert c.reason == "eod" and c.exit_raw == 100.25

    def test_trail_breakeven_then_exit(self):
        cfg = SimConfig(slippage_pct=0.0, trail=True, cooldown=False)
        bars = bars_from(D, [(100, 100, 100, 100), (100, 101.2, 99.9, 101),   # +1.2R
                             (100.8, 100.9, 99.95, 100.2)])                   # back to entry
        c = resolve_intent(_intent(), bars, cfg, lambda t: None)              # no ATR yet
        assert c.reason == "breakeven" and c.exit_raw == 100.0 and c.trail_moves == 1

    def test_trail_atr(self):
        cfg = SimConfig(slippage_pct=0.0, trail=True, cooldown=False)
        bars = bars_from(D, [(100, 100, 100, 100), (100, 103, 99.9, 102.9),
                             (102.9, 103, 101.9, 102)])
        c = resolve_intent(_intent(), bars, cfg, lambda t: 0.5)   # stop → 103 − 0.75
        assert c.reason == "trail" and c.exit_raw == pytest.approx(102.25)

    def test_short_with_slippage(self):
        cfg = SimConfig(slippage_pct=0.02, trail=False, cooldown=False)
        bars = bars_from(D, [(100, 100, 100, 100), (100, 100.1, 99.9, 100),
                             (99.5, 99.6, 98.9, 99)])
        c = resolve_intent(_intent(side=-1, stop=101, target=99.0), bars, cfg, None)
        assert c.entry_fill == pytest.approx(99.98) and c.reason == "target"
        assert c.exit_fill == pytest.approx(99.0 * 1.0002)

    def test_time_exit_and_last_entry(self):
        bars = bars_from(D, [(100, 100, 100, 100)] * 5)
        c = resolve_intent(_intent(time_exit_t=epoch_at(D, 9, 18)), bars, NOSLIP, None)
        assert c.reason == "time" and ist_minute(c.exit_t) == 9 * 60 + 18
        late = bars_from(D, [(100, 100, 100, 100)] * 3, start=(14, 30))
        assert resolve_intent(_intent(signal=(14, 31)), late, NOSLIP, None) == "after_last_entry"


# ── portfolio ───────────────────────────────────────────────────────────────

def _cand(sym="A", day=D, entry=100.0, stop=99.0, exit_=101.0, et=(9, 20), xt=(10, 0),
          strategy="s"):
    it = Intent(strategy=strategy, symbol=sym, day=day, side=1,
                signal_t=epoch_at(day, *et), stop=stop)
    return Candidate(intent=it, entry_t=epoch_at(day, *et), entry_raw=entry, entry_fill=entry,
                     exit_t=epoch_at(day, *xt), exit_raw=exit_, exit_fill=exit_,
                     reason="eod", risk_per_share=abs(entry - stop), mfe_r=0, mae_r=0,
                     trail_moves=0, final_stop=stop)


class TestPortfolio:
    def test_sizing_caps(self):
        sim = PortfolioSimulator(SimConfig(), weekdays)
        # risk ₹2,500 / ₹1 per share = 2,500 sh but 10 % cap = ₹50,000 / 100 = 500 sh
        assert sim.size(100, 1.0, 1.0) == (500, True)
        # wide stop: ₹2,500 / ₹10 = 250 sh < cap → risk-sized
        assert sim.size(100, 10.0, 1.0) == (250, False)
        assert sim.size(100, 10.0, 0.5) == (125, False)
        big = PortfolioSimulator(SimConfig(max_position_pct=500), weekdays)
        assert big.size(100, 1.0, 1.0) == (2500, False)
        # RiskEngine max risk 1 % never binds below risk_pct 0.5 but does above
        hi = PortfolioSimulator(SimConfig(risk_pct=2.0, max_position_pct=1000), weekdays)
        assert hi.size(100, 1.0, 1.0) == (5000, False)

    def test_max_open_symbol_busy_and_release(self):
        cfg = SimConfig(max_open_positions=2, cooldown=False)
        cands = [_cand("A", et=(9, 20), xt=(9, 40)), _cand("A", et=(9, 25), xt=(9, 50)),
                 _cand("B", et=(9, 21), xt=(10, 0)), _cand("C", et=(9, 22), xt=(10, 0)),
                 _cand("C", et=(9, 40), xt=(10, 0)),   # A exits in the 09:40 bar → still busy
                 _cand("D", et=(9, 41), xt=(10, 0))]   # slot free from 09:41
        res = PortfolioSimulator(cfg, weekdays).run(cands, [D])
        got = [(r.symbol, r.entry_time, r.status, r.skip_reason) for r in res.rows]
        assert got == [("A", "09:20", "taken", ""), ("B", "09:21", "taken", ""),
                       ("C", "09:22", "skipped", "max_open_positions"),
                       ("A", "09:25", "skipped", "symbol_busy"),
                       ("C", "09:40", "skipped", "max_open_positions"),
                       ("D", "09:41", "taken", "")]
        assert res.summary()["max_concurrent"] == 2

    def test_cooldown_shadow_and_half_risk(self):
        days = [date(2026, 5, d) for d in (6, 7, 8, 11, 12, 13, 14)]
        cands = []
        for d in days[:3]:
            cands.append(_cand(day=d, entry=100, stop=99, exit_=99))      # losers
        cands.append(_cand(day=days[3]))                                  # Mon: shadow
        cands.append(_cand(day=days[4]))                                  # Tue: shadow
        cands.append(_cand(day=days[5], entry=100, stop=90, exit_=101))   # Wed: half risk, win
        cands.append(_cand(day=days[6], entry=100, stop=90, exit_=101))   # Thu: full risk
        res = PortfolioSimulator(SimConfig(), weekdays).run(cands, days)
        st = [(r.day, r.status, r.qty) for r in res.rows]
        assert st[3][1] == "shadow" and st[4][1] == "shadow"
        assert st[5] == ("2026-05-13", "taken", 125)   # ₹1,250 / ₹10
        assert st[6] == ("2026-05-14", "taken", 250)
        s = res.summary()
        assert s["cooldown_days"] == 2 and s["reduced_risk_days"] == 1 and s["shadow_trades"] == 2

    def test_daily_loss_limit(self):
        cfg = SimConfig(max_position_pct=1000, cooldown=False, max_open_positions=10)
        # each loser: 2,500 sh × ₹6 loss = ₹15,000 = 3 % of capital
        cands = [_cand("A", entry=100, stop=99, exit_=94, et=(9, 20), xt=(9, 30)),
                 _cand("B", et=(9, 31), xt=(10, 0))]
        res = PortfolioSimulator(cfg, weekdays).run(cands, [D])
        assert [r.skip_reason for r in res.rows] == ["", "daily_loss_limit"]

    def test_costs_and_r_in_rows(self):
        res = PortfolioSimulator(SimConfig(cooldown=False), weekdays).run(
            [_cand(entry=1000, stop=990, exit_=1010)], [D])
        r = res.rows[0]
        assert r.qty == 50                                   # min(2500/10=250, 50000/1000=50)
        assert r.gross == pytest.approx(500)
        exp = CostModel().round_trip(1000, 1010, 50, 1)
        assert r.costs == pytest.approx(exp, abs=0.01)
        assert r.r_net == pytest.approx((500 - exp) / 500, abs=1e-3)


# ── periods / data helpers ──────────────────────────────────────────────────

class TestGuards:
    def test_is_ok_oos_refused(self):
        p = resolve_period("is")
        assert p.end == IS_END and not p.touches_oos
        with pytest.raises(OOSAccessError):
            resolve_period("oos")
        with pytest.raises(OOSAccessError):
            resolve_period("is", end=date(2026, 9, 5))
        with pytest.raises(OOSAccessError):
            resolve_period("all")
        assert resolve_period("oos", allow_oos=True).touches_oos
        with pytest.raises(OOSAccessError):
            check_date(date(2026, 9, 1), False)

    def test_candle_source_guard_before_any_query(self):
        from plugins.strategy_search.data import CandleSource
        with pytest.raises(OOSAccessError):
            CandleSource(max_date=date(2026, 10, 19))
        src = CandleSource()                     # IS only
        with pytest.raises(OOSAccessError):
            src.minute_bars("RELIANCE", date(2026, 8, 25), date(2026, 9, 2))
        with pytest.raises(OOSAccessError):
            src.load_symbol("RELIANCE", date(2026, 8, 25), date(2026, 9, 2))

    def test_resample_aligned_to_0915(self):
        bars = bars_from(D, [(i, i + 1, i - 1, i) for i in range(1, 12)])
        r5 = resample(bars, 5)
        assert [ist_minute(b.t) for b in r5] == [555, 560, 565]
        assert (r5[0].o, r5[0].h, r5[0].l, r5[0].c, r5[0].v) == (1, 6, 0, 5, 5000)

    def test_ist_roundtrip(self):
        t = epoch_at(D, 9, 15)
        assert ist_dt(t).hour == 9 and ist_dt(t).minute == 15
        assert epoch_of_ist(ist_dt(t)) == t


# ── Step 6 audit fixes ──────────────────────────────────────────────────────

def _cand_t(target, entry=100.0, stop=99.0, sym="A", et=(9, 20)):
    c = _cand(sym=sym, entry=entry, stop=stop, et=et)
    c.intent.target = target
    return c


class TestLiveRRGate:
    """Audit fix 1: the live RiskEngine rejects target-less drafts
    (``_check_fields``: target <= 0) and round(rr, 2) < MIN_RISK_REWARD_RATIO."""

    def test_default_off_keeps_old_behaviour(self):
        res = PortfolioSimulator(SimConfig(cooldown=False, max_open_positions=10), weekdays).run(
            [_cand_t(None, sym="A"), _cand_t(100.5, sym="B")], [D])
        assert [r.status for r in res.rows] == ["taken", "taken"]

    def test_live_gate_rejects_no_target_and_low_rr(self):
        cfg = SimConfig(cooldown=False, max_open_positions=10, live_rr_gate=True)
        cands = [_cand_t(None, sym="A"),          # trail-only → rejected live
                 _cand_t(101.0, sym="B"),         # rr 1.0
                 _cand_t(101.4951, sym="C"),      # rr 1.4951 → rounds to 1.50 → passes (live rounds)
                 _cand_t(101.494, sym="D"),       # rr 1.494 → 1.49 → rejected
                 _cand_t(102.0, sym="E")]         # rr 2.0
        res = PortfolioSimulator(cfg, weekdays).run(cands, [D])
        assert [(r.symbol, r.skip_reason) for r in res.rows] == [
            ("A", "no_target"), ("B", "min_rr"), ("C", ""), ("D", "min_rr"), ("E", "")]
        assert res.summary()["skipped"] == {"no_target": 1, "min_rr": 2}

    def test_live_gate_respects_explicit_min_rr(self):
        cfg = SimConfig(cooldown=False, live_rr_gate=True, min_rr=1.0)
        res = PortfolioSimulator(cfg, weekdays).run([_cand_t(101.0)], [D])
        assert res.rows[0].status == "taken"

    def test_matches_live_engine_threshold(self):
        from apps.trading.services.risk_engine import RiskEngine
        rng = random.Random(11)
        cfg = SimConfig(live_rr_gate=True)
        for _ in range(500):
            risk = rng.uniform(0.5, 20.0)
            rr = rng.choice([rng.uniform(0.5, 3.0), 1.5 + rng.uniform(-0.01, 0.01)])
            c = _cand_t(100.0 + rr * risk, entry=100.0, stop=100.0 - risk)
            live = RiskEngine._rr_below_minimum(abs(c.intent.target - 100.0) / risk, 1.5)
            assert (rr_gate_skip(c, cfg) == "min_rr") == live


class TestStats:
    """Audit fixes 3/4: power and break-even helpers used in Step 6."""

    def test_trades_for_t(self):
        from plugins.strategy_search.stats import trades_for_t
        assert trades_for_t(0.527, 0.037) == 812        # (2·0.527/0.037)² = 811.6
        assert trades_for_t(1.0, 0.1, t=2.0) == 400

    def test_p_mean_above(self):
        from plugins.strategy_search.stats import p_mean_above
        assert p_mean_above(0.0, 0.0, 0.5, 60) == pytest.approx(0.5)
        assert p_mean_above(0.0, 0.037, 0.527, 60) == pytest.approx(0.705, abs=0.01)

    def test_zero_crossing(self):
        from plugins.strategy_search.stats import zero_crossing
        assert zero_crossing([0.02, 0.05, 0.10], [0.03, 0.0, -0.05]) == pytest.approx(0.05)
        assert zero_crossing([0.02, 0.05], [0.04, -0.02]) == pytest.approx(0.04)
        assert zero_crossing([0.02, 0.05], [-0.01, -0.03]) is None
        # all positive, falling → linear extrapolation
        assert zero_crossing([0.02, 0.05], [0.04, 0.01]) == pytest.approx(0.06)
