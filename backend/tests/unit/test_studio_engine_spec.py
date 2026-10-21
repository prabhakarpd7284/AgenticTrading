"""Strategy Studio P1: spec validation, indicators, rule compiler, TIMELINE engine."""
from __future__ import annotations

import copy
from datetime import date

import pytest

from plugins.strategy_backtest.engine import BacktestEngine, EngineConfig
from plugins.strategy_backtest.exits import StoplossExit, TargetExit
from plugins.strategy_backtest.types import Bar, EntrySignal, PnLMode, TimeframeMode, TradeSide

from apps.strategies.services import indicators as ind
from apps.strategies.services.adapter import PrecomputedDetector, RatchetStop, spec_signals
from apps.strategies.services.backtest_service import RunRequest, compute_metrics, execute
from apps.strategies.services.rules import CompiledSpec
from apps.strategies.services.spec import explain, live_gate, spec_hash, validate_spec
from apps.strategies.services.templates import RULE_TEMPLATES
from tests.studio_helpers import FakeLoader, simple_spec


# ── spec validation ─────────────────────────────────────────────────────────

def test_valid_spec_normalises_and_hashes_stably():
    a = validate_spec(simple_spec())
    assert a.ok, a.errors
    b = validate_spec(copy.deepcopy(a.spec))
    assert spec_hash(a.spec) == spec_hash(b.spec)
    assert a.spec["sizing"]["risk_pct"] == 0.5                # default filled
    assert any("crosses above" in line for line in explain(a.spec))


@pytest.mark.parametrize("patch,path", [
    ({"timeframe": "2m"}, "timeframe"),
    ({"entry": {"long": None, "short": None}}, "entry"),
    ({"entry": {"long": {"left": {"ref": "nope"}, "op": ">", "right": {"value": 1}}}}, "entry.long.left.ref"),
    ({"entry": {"long": {"left": {"price": "close"}, "op": "~", "right": {"value": 1}}}}, "entry.long.op"),
    ({"sizing": {"risk_pct": 5}}, "sizing.risk_pct"),                    # above live 1 %
    ({"sizing": {"max_positions": 9}}, "sizing.max_positions"),          # above live 3
    ({"limits": {"daily_loss_pct": 10}}, "limits.daily_loss_pct"),
    ({"exits": {"eod_time": "15:25"}}, "exits.eod_time"),                # past auto square-off
    ({"entry": {"time_from": "09:00"}}, "entry.time_from"),
])
def test_spec_errors_name_the_path(patch, path):
    spec = simple_spec()
    for k, v in patch.items():
        spec[k] = {**spec[k], **v} if isinstance(spec.get(k), dict) and isinstance(v, dict) else v
    res = validate_spec(spec)
    assert not res.ok
    assert any(e["path"].startswith(path) for e in res.errors), res.errors


def test_daily_spec_rejects_shorts_and_intraday_indicators():
    spec = simple_spec(timeframe="1d")
    spec["entry"]["short"] = {"left": {"price": "close"}, "op": "<", "right": {"ind": "vwap"}}
    res = validate_spec(spec)
    paths = {e["path"] for e in res.errors}
    assert "entry.short" in paths
    assert any("vwap" in e["message"] or "intraday-only" in e["message"] for e in res.errors)


def test_live_gate_flags_missing_or_small_targets():
    spec = validate_spec(simple_spec()).spec
    assert live_gate(spec)["passes"] is True
    spec["exits"]["target"] = {"kind": "none"}
    assert live_gate(spec)["passes"] is False
    spec["exits"]["target"] = {"kind": "r", "value": 1.2}
    g = live_gate(spec)
    assert g["passes"] is False and "1.5" in g["message"]
    res = validate_spec({**simple_spec(), "exits": {"target": {"kind": "none"}}})
    assert res.ok and any(w["path"] == "exits.target" for w in res.warnings)


def test_every_rule_template_is_valid():
    for t in RULE_TEMPLATES:
        res = validate_spec(copy.deepcopy(t["spec"]))
        assert res.ok, (t["key"], res.errors)


# ── indicators match trading.utils (the screener's library) ─────────────────

def test_indicator_series_match_legacy_library():
    from trading.utils import indicators as legacy
    closes = [100 + (i % 7) * 1.3 - (i % 3) * 0.7 + i * 0.05 for i in range(80)]
    highs = [c + 1 for c in closes]
    lows = [c - 1.2 for c in closes]
    assert ind.ema(closes, 9)[-1] == pytest.approx(legacy.ema(closes, 9), abs=0.01)
    assert ind.sma(closes, 20)[-1] == pytest.approx(legacy.sma(closes, 20), abs=0.01)
    assert ind.rsi(closes, 14)[-1] == pytest.approx(legacy.rsi(closes, 14), abs=0.01)
    cd = [{"high": h, "low": lo, "close": c} for h, lo, c in zip(highs, lows, closes)]
    assert ind.atr(highs, lows, closes, 14)[-1] == pytest.approx(legacy.atr(cd, 14), abs=0.01)
    up, mid, lo = ind.bollinger(closes, 20)
    bb = legacy.bollinger_bands(closes, 20)
    assert (up[-1], mid[-1], lo[-1]) == pytest.approx((bb["upper"], bb["middle"], bb["lower"]), abs=0.01)


def test_rules_have_no_lookahead():
    """Masks on a truncated history equal the prefix of masks on the full one."""
    loader = FakeLoader()
    spec = validate_spec(simple_spec()).spec
    full = loader.load("AAA", "5m", date(2026, 5, 4), date(2026, 5, 15))
    compiled = CompiledSpec(spec)
    long_full, _ = compiled.entry_masks(full.frame)
    assert any(long_full)
    cut = 300
    trunc = loader.load("AAA", "5m", date(2026, 5, 4), date(2026, 5, 15))
    f = trunc.frame
    for name in ("timestamps", "open", "high", "low", "close", "volume", "minute", "session", "prev_day"):
        setattr(f, name, getattr(f, name)[:cut])
    long_cut, _ = CompiledSpec(spec).entry_masks(f)
    # the last bar of a truncated frame can't fill (no next bar) — compare before it
    assert long_cut[:cut - 1] == long_full[:cut - 1]


def test_spec_signals_are_relative_and_need_atr_warmup():
    loader = FakeLoader()
    spec = validate_spec(simple_spec()).spec
    loaded = loader.load("AAA", "5m", date(2026, 5, 4), date(2026, 5, 15))
    sigs = spec_signals(CompiledSpec(spec), loaded.frame)
    assert sigs and min(sigs) >= 14
    s = next(iter(sigs.values()))
    assert s.metadata["levels"] == "relative"
    assert s.target - s.entry_price == pytest.approx(2 * (s.entry_price - s.stoploss))


# ── TIMELINE engine ─────────────────────────────────────────────────────────

def _bars(prices, day="2026-05-04", start_min=9 * 60 + 15, step=5):
    out = []
    for k, (o, h, lo, c) in enumerate(prices):
        m = start_min + k * step
        out.append(Bar(f"{day}T{m // 60:02d}:{m % 60:02d}:00+05:30", o, h, lo, c, 1000))
    return out


def _engine(det, **cfg):
    base = dict(pnl_mode=PnLMode.RUPEES, mode=TimeframeMode.TIMELINE, capital=500_000, max_risk_pct=1.0,
                max_position_pct=100.0, max_positions=3, slippage_pct=0.0, session_flat=True,
                eod_minute=15 * 60 + 14, bar_minutes=5)
    base.update(cfg)
    return BacktestEngine(EngineConfig(**base), det, [StoplossExit(gap_aware=True), TargetExit()])


def _sig(sym, bar, stop, target, levels="relative"):
    return EntrySignal(symbol=sym, timestamp=bar.timestamp, side=TradeSide.BUY, entry_price=bar.close,
                       stoploss=stop, target=target, risk_points=bar.close - stop,
                       metadata={"levels": levels})


def test_timeline_fills_at_next_bar_open_and_reanchors_levels():
    bars = _bars([(100, 101, 99, 100), (102, 103, 101, 102), (102, 110, 101, 109), (109, 109, 108, 108)])
    det = PrecomputedDetector()
    det.set("AAA", {0: _sig("AAA", bars[0], 98, 104)})          # 2-pt stop, 4-pt target off close 100
    eng = _engine(det)
    eng.run({"AAA": bars})
    (t,) = eng._closed
    assert t.entry_price == 102                                   # bar 1 open, not bar 0 close
    assert t.stoploss == 100 and t.target == 106                  # distances kept from the fill
    assert t.exit_reason == "Target hit" and t.exit_price == 106


def test_timeline_absolute_levels_reject_fill_beyond_stop():
    bars = _bars([(100, 101, 99, 100), (97, 98, 96, 97), (97, 98, 96, 97)])
    det = PrecomputedDetector()
    det.set("AAA", {0: _sig("AAA", bars[0], 98, 104, levels="absolute")})
    eng = _engine(det)
    eng.run({"AAA": bars})
    assert eng._closed == []
    assert eng.rejections == {"opened_beyond_levels": 1}


def test_timeline_flat_at_session_end_and_costs_deducted():
    bars = _bars([(100, 100.5, 99.5, 100), (100, 100.5, 99.8, 100.2), (100.2, 100.6, 100, 100.4)])
    det = PrecomputedDetector()
    det.set("AAA", {0: _sig("AAA", bars[0], 95, 130)})
    eng = _engine(det, cost_fn=lambda t: 50.0)
    eng.run({"AAA": bars})
    (t,) = eng._closed
    assert t.exit_reason == "EOD close" and t.exit_time == bars[-1].timestamp
    assert t.metadata["costs"] == 50.0
    assert t.pnl == pytest.approx(t.metadata["gross_pnl"] - 50.0)


def test_timeline_respects_max_positions_across_symbols():
    det = PrecomputedDetector()
    data = {}
    for s in ("A", "B", "C"):
        bars = _bars([(100, 101, 99, 100)] * 6)
        data[s] = bars
        det.set(s, {0: _sig(s, bars[0], 95, 130)})
    eng = _engine(det, max_positions=2)
    eng.run(data)
    assert len(eng._closed) == 2
    assert eng.rejections.get("max_positions") == 1


def test_gap_aware_stop_fills_at_open():
    bars = _bars([(100, 101, 99, 100), (100, 101, 99.5, 100), (94, 95, 93, 94)])
    det = PrecomputedDetector()
    det.set("AAA", {0: _sig("AAA", bars[0], 98, 130)})
    eng = _engine(det)
    eng.run({"AAA": bars})
    (t,) = eng._closed
    assert t.exit_price == 94 and t.exit_reason == "SL hit"


def test_ratchet_stop_uses_prior_bars_only():
    bars = _bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 103, 100, 102.9),
                  (102.9, 103, 99.9, 100.1)])
    det = PrecomputedDetector()
    sig = _sig("AAA", bars[0], 98, 0.0)
    sig.atr = 1.0
    det.set("AAA", {0: sig})
    eng = BacktestEngine(EngineConfig(pnl_mode=PnLMode.RUPEES, mode=TimeframeMode.TIMELINE, capital=500_000,
                                      max_position_pct=100.0, slippage_pct=0.0, session_flat=True,
                                      eod_minute=15 * 60 + 14, bar_minutes=5),
                         det, [RatchetStop(1.0, 1.5), StoplossExit(gap_aware=True)])
    eng.run({"AAA": bars})
    (t,) = eng._closed
    # best 103 seen on bar 2 → stop ratchets to max(100, 103 − 1.5) = 101.5 on bar 3, hit there.
    assert t.exit_reason == "Trail SL" and t.exit_price == pytest.approx(101.5)


# ── execute() end to end on fake candles ────────────────────────────────────

def test_execute_spec_end_to_end_with_split_and_provenance():
    req = RunRequest(kind="spec", strategy_name="spec", definition=simple_spec(),
                     from_date=date(2026, 5, 4), to_date=date(2026, 5, 29), capital=500_000)
    res = execute(req, FakeLoader(missing={"BBB"}))
    m, pv = res["metrics"], res["provenance"]
    assert m["trades"] > 0 and m["trades"] == len(res["trades"])
    assert m["net_pnl"] == pytest.approx(sum(t["net_pnl"] for t in res["trades"]), abs=0.05)
    assert m["costs"] > 0
    assert pv["synthetic"] is False and pv["symbols_missing"] == ["BBB"]
    assert pv["split"]["oos"][0] == res["split"]["oos_start"]
    assert res["split"]["is"]["trades"] + res["split"]["oos"]["trades"] == m["trades"]
    assert all(t["segment"] == ("oos" if t["entry_time"][:10] >= res["split"]["oos_start"] else "is")
               for t in res["trades"])
    assert pv["sample"]["too_few"] == (m["trades"] < 30)
    # fills strictly after the signal bar
    assert all(t["meta"]["signal_time"] < t["entry_time"] for t in res["trades"])


def test_execute_refuses_llm_and_options_plugins():
    for name in ("directional", "pyramid"):
        req = RunRequest(kind="plugin", strategy_name=name, definition={}, from_date=date(2026, 5, 4),
                         to_date=date(2026, 5, 8), symbols=["AAA"])
        with pytest.raises(ValueError, match="can't be backtested"):
            execute(req, FakeLoader())


def test_compute_metrics_profit_factor_and_drawdown():
    rows = [{"seq": i, "net_pnl": p, "gross_pnl": p + 10, "costs": 10, "r_multiple": p / 100,
             "exit_time": f"2026-05-0{i}T10:00", "bars_held": 2, "side": "BUY"}
            for i, p in enumerate([200, -100, -100, 300], start=1)]
    m = compute_metrics(rows, 100_000)
    assert m["profit_factor"] == 2.5 and m["max_drawdown"] == 200 and m["win_rate"] == 50.0
    assert m["avg_r"] == pytest.approx(0.75)
