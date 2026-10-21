"""Unit tests for the Pivot Boss toolkit (plugins/strategy_pivot_boss).

Pure Python — no DB, no broker. Formula tests use the book's worked example
(H=7743, L=7587, C=7618 → PP 7649, R1 7711, S1 7555).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from plugins.strategy_pivot_boss import levels as lv
from plugins.strategy_pivot_boss import research as rs
from plugins.strategy_pivot_boss.backtest import (
    BacktestConfig,
    CostModel,
    Signal,
    SizingConfig,
    simulate,
    summarize,
)
from plugins.strategy_pivot_boss.bars import Bar, ist_minute
from plugins.strategy_pivot_boss.relationships import (
    Relationship,
    classify_open,
    two_day_relationship,
)
from plugins.strategy_pivot_boss.setups import (
    DayContext,
    SetupParams,
    camarilla_h3l3_reversal,
    camarilla_h4l4_breakout,
)
from plugins.strategy_pivot_boss.triggers import (
    doji_reversal_series,
    extreme_reversal_series,
    outside_reversal_series,
    resample,
    wick_reversal_series,
)

IST = timezone(timedelta(hours=5, minutes=30))


def t_at(hh: int, mm: int, d: date = date(2026, 10, 19)) -> int:
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST).timestamp())


def mk(i: int, o, h, l, c, v=100.0, start=(9, 15)) -> Bar:  # noqa: E741
    """1m bar number ``i`` after ``start``."""
    return Bar(t_at(*start) + 60 * i, o, h, l, c, v)


def flat(n: int, px=100.0, rng=1.0, start=(9, 15)) -> list[Bar]:
    """n quiet bars with alternating small bodies (avg range = rng)."""
    out = []
    for i in range(n):
        o, c = (px - 0.1, px + 0.1) if i % 2 else (px + 0.1, px - 0.1)
        out.append(mk(i, o, px + rng / 2, px - rng / 2, c, start=start))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Levels
# ─────────────────────────────────────────────────────────────────────────────


def test_floor_pivots_book_worked_example():
    f = lv.floor_pivots(7743, 7587, 7618)
    assert int(f.pp) == 7649
    assert int(f.r1) == 7711
    assert int(f.s1) == 7555
    rng = 7743 - 7587
    assert f.r2 == pytest.approx(f.pp + rng)
    assert f.s2 == pytest.approx(f.pp - rng)
    assert f.r3 == pytest.approx(f.r1 + rng)
    assert f.s3 == pytest.approx(f.s1 - rng)
    assert f.r4 == pytest.approx(f.r3 + (f.r2 - f.r1))
    assert f.s4 == pytest.approx(f.s3 - (f.s1 - f.s2))
    assert f.resistance(3) == f.r3
    assert f.support(4) == f.s4


def test_cpr_normalises_tc_below_bc_and_width():
    # close near the low → raw TC < BC
    c = lv.cpr(7743, 7587, 7618)
    assert c.raw_tc < (7743 + 7587) / 2
    assert c.tc == pytest.approx((7743 + 7587) / 2)  # BC became the top
    assert c.bc == pytest.approx(c.raw_tc)
    assert c.width_pct == pytest.approx((c.tc - c.bc) / c.pp * 100)
    up = lv.cpr(110, 90, 108)
    assert up.tc == pytest.approx(2 * up.pp - 100)
    assert up.bc == 100
    assert lv.classify_cpr_width(0.1) == "narrow"
    assert lv.classify_cpr_width(0.9) == "wide"
    assert lv.classify_cpr_width(0.5) == "normal"


def test_camarilla_levels():
    h, l, c = 110.0, 90.0, 100.0  # noqa: E741
    cam = lv.camarilla(h, l, c)
    assert cam.h4 == pytest.approx(100 + 20 * 1.1 / 2)
    assert cam.l3 == pytest.approx(100 - 20 * 1.1 / 4)
    assert cam.h2 == pytest.approx(100 + 20 * 1.1 / 6)
    assert cam.l1 == pytest.approx(100 - 20 * 1.1 / 12)
    assert cam.h5 == pytest.approx(110 / 90 * 100)
    assert cam.l5 == pytest.approx(100 - (cam.h5 - 100))
    assert cam.width == pytest.approx(20 * 0.55)


def test_developing_cpr_and_dc3_end_equal_next_day_levels():
    bars = [mk(0, 100, 101, 99, 100.5), mk(1, 100.5, 103, 100, 102), mk(2, 102, 102.5, 98, 99)]
    dev = lv.developing_cpr(bars)
    assert dev[-1] == lv.cpr(103, 98, 99)
    assert dev[0] == lv.cpr(101, 99, 100.5)
    dc3 = lv.developing_camarilla3(bars)
    cam = lv.camarilla(103, 98, 99)
    assert dc3[-1] == (pytest.approx(cam.l3), pytest.approx(cam.h3))


def test_weekly_and_monthly_prior_period():
    daily = []
    d = date(2026, 9, 28)  # Monday
    for i in range(10):  # two weeks, Mon–Fri
        day = d + timedelta(days=i + (2 if i >= 5 else 0))
        daily.append((day, 100 + i, 105 + i, 95 + i, 101 + i))
    weeks = lv.aggregate_periods(daily, "week")
    assert len(weeks) == 2
    assert weeks[0].h == 109
    assert weeks[0].l == 95
    assert weeks[0].c == 105
    p = lv.prior_period_hlc(daily, date(2026, 10, 7), "week")
    assert (p.h, p.l, p.c) == (109, 95, 105)
    fp, cp, cam = lv.higher_tf_pivots(daily, date(2026, 10, 7), "week")
    assert fp == lv.floor_pivots(109, 95, 105)
    m = lv.prior_period_hlc(daily, date(2026, 10, 7), "month")
    assert m.key == (2026, 9)
    assert m.c == 103
    assert lv.prior_period_hlc(daily, date(2026, 9, 29), "week") is None


def test_volume_profile_spreads_volume_equally():
    prof = lv.volume_profile([mk(0, 10.0, 10.39, 10.0, 10.2, v=400)], bin_size=0.1)
    assert sorted(lv.volume_profile([mk(0, 10.0, 10.4, 10.0, 10.2, v=1)], 0.1)) == [100, 101, 102, 103]
    assert sorted(prof) == [100, 101, 102, 103]
    assert all(v == pytest.approx(100) for v in prof.values())


def test_value_area_expands_from_poc():
    prof = {0: 5, 1: 10, 2: 40, 3: 20, 4: 15, 5: 10}  # total 100
    poc, val, vah = lv.value_area(prof, 0.70)
    assert poc == 2
    # 40 → +20 (up) = 60 → +15 (up, vs 10 down) = 75 ≥ 70
    assert (val, vah) == (2, 4)


def test_money_zone_and_bin_size():
    assert lv.default_bin_size(1200) == pytest.approx(0.6)
    assert lv.default_bin_size(100) == pytest.approx(0.05)
    bars = [mk(i, 100, 100.5, 99.5, 100, v=1000) for i in range(30)]
    bars.append(mk(30, 101, 103, 101, 102, v=10))
    mz = lv.money_zone(bars, bin_size=0.5)
    assert 99.5 <= mz.vpoc <= 100.5
    assert mz.val <= mz.vpoc <= mz.vah
    assert mz.vah < 101
    assert mz.basis == "volume"
    no_vol = [Bar(b.t, b.o, b.h, b.l, b.c, 0) for b in bars]
    assert lv.money_zone(no_vol, bin_size=0.5).basis == "tpo"


def test_virgin_level_tracker():
    tr = lv.VirginLevelTracker(max_age=5)
    level = tr.add(100.0, "vpoc", session=1)
    assert tr.update(1, 101, 99) == []  # its own session never counts
    assert tr.update(2, 105, 101) == []
    assert tr.virgin(2) == [level]
    assert tr.update(3, 101, 99.5) == [level]
    assert tr.virgin(3) == []
    tr.add(200.0, "vah", session=3)
    assert tr.virgin(20) == []  # aged out


# ─────────────────────────────────────────────────────────────────────────────
# Relationships
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("today", "yday", "expected"), [
    ((105, 106), (100, 101), Relationship.HIGHER),
    ((95, 96), (100, 101), Relationship.LOWER),
    ((100.5, 102), (100, 101), Relationship.OVERLAPPING_HIGHER),
    ((99, 100.5), (100, 101), Relationship.OVERLAPPING_LOWER),
    ((100.2, 100.8), (100, 101), Relationship.INSIDE),
    ((99, 102), (100, 101), Relationship.OUTSIDE),
    ((100.01, 101.01), (100, 101), Relationship.UNCHANGED),
])
def test_two_day_relationships(today, yday, expected):
    assert two_day_relationship(*today, *yday) == expected


def test_unchanged_tolerance_is_configurable():
    assert two_day_relationship(100.05, 101.05, 100, 101, unchanged_tol_pct=0.0) \
        == Relationship.OVERLAPPING_HIGHER
    assert two_day_relationship(100.05, 101.05, 100, 101, unchanged_tol_pct=0.1) \
        == Relationship.UNCHANGED


def test_classify_open():
    assert classify_open(100, 105, 95, 98, 102).category == "in_range_in_value"
    assert classify_open(103, 105, 95, 98, 102).category == "in_range_out_of_value"
    loc = classify_open(107, 105, 95, 98, 102)
    assert loc.category == "out_of_range_out_of_value"
    assert loc.side == "above"
    assert loc.expected_day == "trend"


# ─────────────────────────────────────────────────────────────────────────────
# Triggers
# ─────────────────────────────────────────────────────────────────────────────


def test_wick_reversal_long_short_and_negative():
    base = flat(50)
    hammer = mk(50, 100.0, 100.3, 97.0, 100.2)  # body .2, lower wick 3.0, close top
    star = mk(51, 100.0, 103.0, 99.7, 99.8)  # body .2, upper wick 3.0, close bottom
    weak = mk(52, 100.0, 100.3, 99.5, 100.2)  # lower wick .5 < 2.5×.2
    s = wick_reversal_series(base + [hammer, star, weak])
    assert s[50] == 1
    assert s[51] == -1
    assert s[52] == 0
    assert all(x == 0 for x in s[:49])  # not enough history / quiet bars


def test_wick_reversal_bodyless_needs_avg_range():
    base = flat(50, rng=1.0)
    big = mk(50, 100.5, 100.5, 98.0, 100.5)  # O=C=H, range 2.5 ≥ avg
    tiny = mk(51, 100.5, 100.5, 100.3, 100.5)  # O=C=H, range .2 < avg
    s = wick_reversal_series(base + [big, tiny])
    assert s[50] == 1
    assert s[51] == 0


def test_extreme_reversal():
    base = flat(50, rng=1.0)
    big_red = mk(50, 103.0, 103.2, 97.0, 97.2)  # range 6.2 > 2×avg, body 5.8
    green = mk(51, 97.2, 98.0, 97.0, 97.8)
    s = extreme_reversal_series(base + [big_red, green])
    assert s[51] == 1
    small_red = mk(50, 100.4, 100.5, 99.5, 99.6)
    assert extreme_reversal_series(base + [small_red, green])[51] == 0
    red_again = mk(51, 97.2, 97.5, 96.0, 96.5)
    assert extreme_reversal_series(base + [big_red, red_again])[51] == 0


def test_outside_reversal():
    base = flat(50, rng=1.0)
    prev = mk(50, 100, 100.5, 99.5, 99.8)
    bull = mk(51, 99.8, 101.0, 99.0, 100.8)  # L<L1, C>H1, range 2 ≥ 1.25
    assert outside_reversal_series(base + [prev, bull])[51] == 1
    bear = mk(51, 100.2, 101.0, 99.0, 99.2)
    assert outside_reversal_series(base + [prev, bear])[51] == -1
    narrow = mk(51, 99.8, 100.6, 99.4, 100.55)  # outside but range 1.2 < 1.25
    assert outside_reversal_series(base + [prev, narrow])[51] == 0


def test_doji_reversal_short_and_two_bar_variant():
    rising = [mk(i, 100 + i, 100.6 + i, 99.6 + i, 100.5 + i) for i in range(12)]
    doji = mk(12, 113.0, 113.5, 112.5, 113.05)  # above SMA(10)
    red = mk(13, 113.0, 113.1, 112.0, 112.2)  # red, closes below doji low
    s = doji_reversal_series(rising + [doji, red])
    assert s[13] == -1
    green = mk(13, 112.9, 113.3, 112.7, 113.2)
    s2 = doji_reversal_series(rising + [doji, green, Bar(red.t + 60, 113.0, 113.1, 112.0, 112.2)])
    assert s2[13] == 0
    assert s2[14] == -1
    late = [doji, green, green, Bar(red.t + 120, 113.0, 113.1, 112.0, 112.2)]
    assert doji_reversal_series(rising + late)[15] == 0  # 3rd bar: too late


def test_doji_reversal_long_mirror_and_negative():
    falling = [mk(i, 200 - i, 200.4 - i, 199.4 - i, 199.5 - i) for i in range(12)]
    doji = mk(12, 187.0, 187.5, 186.5, 187.05)
    green = mk(13, 187.0, 188.0, 186.9, 187.8)
    assert doji_reversal_series(falling + [doji, green])[13] == 1
    not_doji = mk(12, 186.5, 187.5, 186.5, 187.4)
    assert doji_reversal_series(falling + [not_doji, green])[13] == 0


def test_resample_aligns_to_0915_and_respects_days():
    bars = [mk(i, 100 + i, 101 + i, 99 + i, 100.5 + i, v=10) for i in range(20)]
    other = [Bar(t_at(9, 15, date(2026, 10, 21)), 1, 2, 0.5, 1.5, 5)]
    out = resample(bars + other, 15)
    assert len(out) == 3
    assert ist_minute(out[0].t) == 9 * 60 + 15
    assert ist_minute(out[1].t) == 9 * 60 + 30
    assert out[0].o == 100
    assert out[0].c == 114.5
    assert out[0].h == 115
    assert out[0].l == 99
    assert out[0].v == 150
    assert out[1].v == 50
    assert out[2].o == 1


# ─────────────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────────────


def test_costs_match_nse_schedule():
    cm = CostModel()
    buy = cm.order_cost(1000, 100, is_buy=True)  # turnover 1,00,000
    brokerage, txn, sebi = 20, 2.97, 0.1
    assert buy == pytest.approx(brokerage + txn + sebi + 0.18 * (brokerage + txn + sebi) + 3.0)
    sell = cm.order_cost(1000, 100, is_buy=False)
    assert sell == pytest.approx(brokerage + txn + sebi + 0.18 * (brokerage + txn + sebi) + 25.0)
    small = cm.order_cost(100, 10, is_buy=True)  # turnover 1,000 → brokerage 1.0
    assert small < 2.0
    assert cm.round_trip(1000, 1000, 100, side=-1) == pytest.approx(buy + sell)


def test_sizing_risk_and_notional_cap():
    s = SizingConfig()
    assert s.qty(1000, 990) == 100  # 0.5% × 5L = 2,500 / 10 = 250 → cap 100,000/1000 = 100
    assert s.qty(1000, 950) == 50  # 2,500 / 50
    assert s.qty(1000, 1000) == 0


def _day(prices):
    return [mk(i, *p) for i, p in enumerate(prices)]


def test_simulate_stop_first_when_bar_hits_both():
    bars = _day([(100, 100.2, 99.9, 100), (100, 103, 97, 100)])
    cfg = BacktestConfig(slippage_pct=0.0)
    r = simulate("X", "d", bars, Signal("s", 1, 1, stop=98, target=102), cfg)
    assert r.reason == "stop"
    assert r.exit_px == 98
    assert r.gross == pytest.approx((98 - 100) * r.qty)
    assert r.net < r.gross


def test_simulate_target_gap_and_eod_and_slippage():
    cfg = BacktestConfig(slippage_pct=0.0)
    bars = _day([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (103, 104, 102.5, 103.5)])
    r = simulate("X", "d", bars, Signal("s", 1, 1, stop=98, target=102), cfg)
    assert r.reason == "target"
    assert r.exit_px == 103
    bars2 = _day([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (97, 97.5, 96, 97)])
    r2 = simulate("X", "d", bars2, Signal("s", 1, 1, stop=98, target=105), cfg)
    assert r2.reason == "stop"
    assert r2.exit_px == 97
    bars3 = _day([(100, 100, 100, 100), (100, 100.5, 99.5, 100.2), (100.2, 100.6, 100, 100.4)])
    r3 = simulate("X", "d", bars3, Signal("s", -1, 1, stop=102, target=95),
                  BacktestConfig(slippage_pct=0.1))
    assert r3.reason == "eod"
    assert r3.entry_px == pytest.approx(99.9)
    assert r3.exit_px == pytest.approx(100.4 * 1.001)


def test_simulate_rejects_untakeable_and_trails_up():
    cfg = BacktestConfig(slippage_pct=0.0)
    bars = _day([(100, 100, 100, 100), (97, 97, 97, 97)])
    assert simulate("X", "d", bars, Signal("s", 1, 1, stop=98), cfg) is None
    prices = [(100, 100, 100, 100), (100, 101, 99.8, 101), (101, 105, 101, 105),
              (105, 105, 103.4, 104), (104, 104, 103, 103)]
    bars = _day(prices)
    atr = [None, None, 1.0, None, None]  # ATR published at the close of bar 2
    sig = Signal("s", 1, 1, stop=99, trail_mult=1.5, atr_at=atr)
    r = simulate("X", "d", bars, sig, cfg)
    assert r.reason == "trail"
    assert r.exit_px == pytest.approx(105 - 1.5)
    assert r.stop == 99  # initial stop reported


def test_summarize():
    cfg = BacktestConfig(slippage_pct=0.0)
    win = simulate("X", "d", _day([(100, 100, 100, 100), (100, 102.5, 99.9, 102)]),
                   Signal("s", 1, 1, stop=99, target=102), cfg)
    loss = simulate("X", "d", _day([(100, 100, 100, 100), (100, 100.1, 98.5, 99)]),
                    Signal("s", 1, 1, stop=99, target=102), cfg)
    s = summarize([win, loss], sessions=2)
    assert s["trades"] == 2
    assert s["win_rate"] == 0.5
    assert s["net"] == pytest.approx(win.net + loss.net)
    assert s["max_drawdown"] >= 0
    assert s["trades_per_day"] == 1.0
    assert summarize([], 5)["trades"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Research primitives + setups
# ─────────────────────────────────────────────────────────────────────────────


def test_first_hit_and_trend_day():
    bars = _day([(100, 100.5, 99.5, 100), (100, 101, 99, 100), (100, 103, 100, 102)])
    assert rs.first_hit(bars, 0, up=102, down=98) == ("up", 2)
    assert rs.first_hit(bars, 0, up=100.8, down=99.2) == ("both", 1)
    assert rs.first_hit(bars, 0, up=200) == ("none", -1)
    assert rs.trend_day(100, 110, 99, 109) == 1
    assert rs.trend_day(109, 110, 99, 100) == -1
    assert rs.trend_day(105, 110, 99, 106) == 0


def test_floor_row_and_gap_fill():
    f = lv.floor_pivots(110, 90, 100)
    cp = lv.cpr(110, 90, 100)
    r = rs.floor_row(101, 121, 99, 120.5, f, cp)
    assert r["pp_touch"]
    assert r["l1_touch"]
    assert r["r2_touch"]
    assert r["l1_beyond"]
    assert not r["s1_touch"]
    assert r["open_above_pp"]
    bars = [mk(i, 102 - 0.1 * i, 102.2 - 0.1 * i, 101.8 - 0.1 * i, 102 - 0.1 * i) for i in range(70)]
    g = rs.gap_fill(bars, prev_close=100.0, cpr=cp)
    assert g["gap_pct"] == pytest.approx(2.0)
    assert g["open_vs_cpr"] == "above"
    assert g["cpr_fill_1015"]
    assert g["pc_fill_eod"]


def _ctx(bars1m, cam, prev=(110.0, 90.0, 100.0)):
    h, l, c = prev  # noqa: E741
    return DayContext(
        symbol="X", day="2026-10-19", bars1m=bars1m, bars5=resample(bars1m, 5),
        bars15=resample(bars1m, 15), trig15=[0] * 30, prev_h=h, prev_l=l, prev_c=c,
        floor=lv.floor_pivots(h, l, c), cpr=lv.cpr(h, l, c), cam=cam, cpr_prev=None,
        prev_mz=None, atr_daily=10.0, atr15_at=[None] * len(bars1m))


def test_h4_breakout_needs_close_cross():
    cam = lv.camarilla(110, 90, 100)  # H4 = 111, H3 = 105.5, H5 ≈ 122.2
    bars = [mk(i, 108, 109, 107.5, 108.5) for i in range(5)]  # 09:15–09:19 below H4
    bars += [mk(5 + i, 110, 112, 109.8, 111.5) for i in range(5)]  # 09:20 bar closes > H4
    bars += [mk(10 + i, 111.5, 112, 111, 111.8) for i in range(10)]
    sig = camarilla_h4l4_breakout(_ctx(bars, cam), SetupParams())
    assert sig.side == 1
    assert sig.stop == cam.h3
    assert sig.target == cam.h5
    assert ist_minute(bars[sig.entry_idx].t) == 9 * 60 + 25
    wick_only = [mk(i, 108, 112, 107.5, 108.5) for i in range(20)]  # touches, never closes above
    assert camarilla_h4l4_breakout(_ctx(wick_only, cam), SetupParams()) is None


def test_h3_reversal_wick_confirm_entry():
    cam = lv.camarilla(110, 90, 100)  # H3 = 105.5
    bars = [mk(i, 104, 104.5, 103.5, 104.2) for i in range(5)]
    bars += [mk(5, 104.2, 106.5, 104, 105)] + [mk(6 + i, 105, 105.3, 104.5, 105.0) for i in range(4)]
    bars += [mk(10 + i, 104.8, 105.0, 104.0, 104.3) for i in range(5)]  # confirm bar
    bars += [mk(15 + i, 104.3, 104.4, 103.5, 103.8) for i in range(10)]
    sig = camarilla_h3l3_reversal(_ctx(bars, cam), SetupParams())
    assert sig.side == -1
    assert sig.target == cam.l3
    assert sig.stop > cam.h4
    assert ist_minute(bars[sig.entry_idx].t) == 9 * 60 + 30
    gapped = [mk(0, 115, 116, 114, 115)] + bars[1:]  # open outside prior range
    assert camarilla_h3l3_reversal(_ctx(gapped, cam), SetupParams()) is None
