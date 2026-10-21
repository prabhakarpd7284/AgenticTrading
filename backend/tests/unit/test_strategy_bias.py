"""Unit tests for the Step-1 bias study (plugins/strategy_search/bias.py). Pure Python."""
from __future__ import annotations

from datetime import date

import pytest

from plugins.strategy_pivot_boss.bars import Bar
from plugins.strategy_pivot_boss.levels import camarilla, floor_pivots
from plugins.strategy_search.bias import (
    BiasSources,
    daily_atr,
    day_features,
    day_t,
    mech_null,
    net_pct,
)
from plugins.strategy_search.data import DaySession, SymbolData, epoch_at

D = date(2026, 5, 14)
PREV = Bar(epoch_at(date(2026, 5, 13), 0, 0), 100.0, 102.0, 98.0, 100.0)   # PP = 100


def _session(path, start=(9, 15)):
    """1m bars 09:15 → 15:14; ``path(i)`` gives the close of bar i, o = previous close."""
    t0 = epoch_at(D, *start)
    bars, prev = [], path(-1)
    for i in range(360):
        c = path(i)
        bars.append(Bar(t0 + 60 * i, prev, max(prev, c) + 0.01, min(prev, c) - 0.01, c))
        prev = c
    return DaySession(day=D, bars=bars, prev_daily=PREV, daily_history=[PREV])


def test_daily_atr_needs_n_plus_one_bars():
    hist = [Bar(i, 10, 11, 9, 10) for i in range(14)]
    assert daily_atr(hist) is None
    hist.append(Bar(99, 10, 12, 9, 11))
    assert daily_atr(hist) == pytest.approx((13 * 2 + 3) / 14)


def test_day_features_side_and_returns():
    s = _session(lambda i: 101.0 if i < 0 else 101.0 + 0.001 * i)
    f = day_features("X", s, bhav_close=101.5)
    assert f["side_pp"] == 1 and f["pp"] == pytest.approx(100.0)
    assert f["pp_touched_1015"] is False
    assert f["r_oc_bhav"] == pytest.approx((101.5 / 101.0 - 1) * 100)
    assert f["e916"] == s.bars[1].o
    assert f["c1514"] == s.bars[-1].c


def test_day_features_rejects_late_start():
    assert day_features("X", _session(lambda i: 101.0, start=(9, 20)), None) is None


def test_mech_null_is_share_of_pool_returns_staying_on_side():
    rows = [{"symbol": "X", "day": D, "r_oc_bhav": r, "side_pp": 1, "open": 101.0, "pp": 100.0}
            for r in (-2.0, -0.5, 0.5)]
    nl = mech_null(rows)
    # threshold to reach PP from 101 is −0.990 %: −2 crosses, −0.5 and +0.5 stay above
    assert nl[("X", D)] == pytest.approx(2 / 3)


def test_net_pct_subtracts_slippage_and_costs():
    # 0 % move at ₹50k: −0.04 % slippage − costs (brokerage ₹40 dominates)
    v = net_pct(1000.0, 0.0, 1, 50_000)
    assert -0.20 < v < -0.12
    assert net_pct(1000.0, 0.0, 1, 500_000) > v   # fixed brokerage shrinks with size


def test_day_t_clusters_by_date():
    t, nd = day_t([(date(2026, 5, d), v) for d, v in ((1, 1.0), (1, 3.0), (2, 1.0), (3, 2.0))])
    assert nd == 3 and t is not None and t > 0


def test_sources_emit_expected_geometry():
    # open 100.5 (> PP 100, inside cam L3/H3), drift up through H3 by mid-morning
    cam = camarilla(PREV.h, PREV.l, PREV.c)
    s = _session(lambda i: 100.5 if i < 0 else 100.5 + 0.02 * i)
    src = BiasSources(lambda sym: {})
    intents = src.generate(SymbolData("X", [s]))
    names = {i.strategy for i in intents}
    assert {"pp_open_side", "cam_touch_tgtH4", "cam_touch_stopL3", "gapfill_60m"} <= names
    by = {i.strategy: i for i in intents}
    assert by["pp_open_side"].stop == pytest.approx(floor_pivots(102, 98, 100).pp)
    assert by["pp_open_side"].signal_t == s.bars[0].t + 60
    assert by["cam_touch_tgtH4"].side == 1
    assert by["cam_touch_tgtH4"].target == pytest.approx(cam.h4)
    assert by["cam_touch_tgtH4"].stop == pytest.approx(PREV.c)
    assert by["cam_touch_stopL3"].stop == pytest.approx(cam.l3)
    g = by["gapfill_60m"]                      # gap +0.5 % → fade short toward 100
    assert g.side == -1
    assert g.target == 100.0
    assert g.stop == pytest.approx(101.0)
    assert len(src.days) == 1
