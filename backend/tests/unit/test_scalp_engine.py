"""Unit tests for the scalping engine (plugins/strategy_scalp).

Pure-Python, no DB. Covers the bin/pressure profile, the both-direction state
machine (long/short/pyramid/reverse/trail-exit), determinism, and tick synthesis.
"""
from __future__ import annotations

import pytest

from plugins.strategy_scalp.engine import Candle, ScalpConfig, ScalpEngine, Tick
from plugins.strategy_scalp.profile import VolumeProfile
from plugins.strategy_scalp.replay import (
    aggregate_decision_candles,
    generate_scalp_sample,
    run_scalp,
    synthesize_ticks,
)


# ──────────────────────────────────────────────
# Profile
# ──────────────────────────────────────────────

def test_profile_pressure_positive_on_rising_prices():
    prof = VolumeProfile(ScalpConfig(bin_width=20.0, window_secs=300))
    snap = None
    px = 280.0
    for i in range(120):
        snap = prof.update(ts=i * 2.0, ltp=px, vol=0)
        px += 1.0  # steady climb
    assert snap is not None
    assert snap.pressure > 0.2, snap.pressure
    assert snap.val <= snap.poc <= snap.vah


def test_profile_pressure_negative_on_falling_prices():
    prof = VolumeProfile(ScalpConfig(bin_width=20.0, window_secs=300))
    snap = None
    px = 420.0
    for i in range(120):
        snap = prof.update(ts=i * 2.0, ltp=px, vol=0)
        px -= 1.0
    assert snap.pressure < -0.2, snap.pressure


def test_profile_pressure_bounded():
    prof = VolumeProfile(ScalpConfig())
    for i in range(50):
        snap = prof.update(ts=i, ltp=300 + (i % 7), vol=0)
        assert -1.0 <= snap.pressure <= 1.0


def test_value_area_handles_gaps_without_hanging():
    # Non-contiguous bins (a gap) must not trip the value-area growth loop.
    prof = VolumeProfile(ScalpConfig(bin_width=10.0, window_secs=10_000))
    for px in (100, 100, 100, 300, 300, 500):
        prof.update(ts=prof._prev_ts + 1 if prof._prev_ts else 0, ltp=px, vol=0)
    snap = prof.update(ts=999, ltp=300, vol=0)
    assert snap.val <= snap.vah


# ──────────────────────────────────────────────
# Tick synthesis / aggregation
# ──────────────────────────────────────────────

def test_synthesize_ticks_ohlc_emits_four_per_bar():
    c = Candle(timestamp="t", open=100, high=110, low=95, close=105, volume=400, epoch=0.0)
    ticks = synthesize_ticks([c], resolution_secs=5, mode="ohlc")
    assert len(ticks) == 4
    prices = [t.ltp for t in ticks]
    assert prices[0] == 100 and prices[-1] == 105        # O … C
    assert 110 in prices and 95 in prices                # H and L visited
    assert ticks[-1].vol == pytest.approx(400)           # cumulative volume conserved


def test_decision_candle_aggregation_buckets_by_10min():
    candles = generate_scalp_sample(5)  # full session 09:15–15:30
    decision = aggregate_decision_candles(candles, bucket_secs=600)
    assert len(decision) == 38                           # 09:15..15:25 in 10-min bars
    end_ts, first = decision[0]
    # decision candles are session-anchored to the 09:15 open (not clock-aligned)
    assert first.timestamp.endswith("T09:15:00+05:30")
    assert first.high >= first.low
    assert end_ts - first.epoch == 600


# ──────────────────────────────────────────────
# Engine state machine
# ──────────────────────────────────────────────

def _feed_rising(engine, start=280.0, n=120, step=1.0, t0=0.0, dt=2.0):
    px = start
    for i in range(n):
        engine.feed_tick(Tick(ts=t0 + i * dt, ltp=px, vol=0))
        px += step
    return px


def test_engine_enters_long_on_buying_pressure():
    eng = ScalpEngine("CE", ScalpConfig(require_bias_alignment=False, trend_only=False, min_sl_distance=1.0))
    _feed_rising(eng)
    assert eng.st.side == "LONG"
    assert eng.st.lots >= 1
    assert any(e.side == "LONG" for e in eng.entries)


def test_engine_enters_short_on_selling_pressure():
    eng = ScalpEngine("CE", ScalpConfig(require_bias_alignment=False, trend_only=False, min_sl_distance=1.0))
    px = 420.0
    for i in range(120):
        eng.feed_tick(Tick(ts=i * 2.0, ltp=px, vol=0))
        px -= 1.0
    assert eng.st.side == "SHORT"


def test_engine_pyramids_then_holds_multiple_lots():
    eng = ScalpEngine("CE", ScalpConfig(require_bias_alignment=False, trend_only=False, min_sl_distance=1.0,
                                        pyramid_cooldown_secs=0.0))
    _feed_rising(eng, n=200)
    assert eng.st.side == "LONG"
    assert eng.st.lots >= 2          # at least one pyramid add
    assert eng.st.pyramids >= 1


def test_engine_trail_sl_only_ratchets_up_for_long():
    eng = ScalpEngine("CE", ScalpConfig(require_bias_alignment=False, trend_only=False, min_sl_distance=1.0))
    _feed_rising(eng)
    seen = []
    px = 360.0
    for i in range(40):
        eng.feed_tick(Tick(ts=1000 + i, ltp=px, vol=0))
        if eng.st.side == "LONG":
            seen.append(eng.st.trail_sl)
        px += 0.5
    assert seen == sorted(seen)      # never relaxes


def test_engine_deterministic():
    candles = generate_scalp_sample(5)
    a = run_scalp(candles, "CE", ScalpConfig(), resolution_secs=5)
    b = run_scalp(candles, "CE", ScalpConfig(), resolution_secs=5)
    assert a.realized_pnl == b.realized_pnl
    assert len(a.entries) == len(b.entries)
    assert a.log == b.log


def test_sample_session_trades_and_profits():
    """The bullish-then-fade sample should enter long, pyramid, reverse short,
    and net positive — the smoke test for the whole pipeline."""
    res = run_scalp(generate_scalp_sample(5), "NIFTY 23800 CE", ScalpConfig(), resolution_secs=5)
    assert res.trades >= 2
    assert len(res.entries) >= 3
    assert res.peak_lots >= 2
    assert res.realized_pnl > 0


def test_bin_reversal_entry_on_sample():
    """With bin_reversal_entry (default), entries are pullback-to-bin reversals —
    the reason names the bin and the SL sits below/above it."""
    res = run_scalp(generate_scalp_sample(5), "CE", ScalpConfig(), resolution_secs=5)
    assert res.entries, "no entries on the uptrend sample"
    rev = [e for e in res.entries if "reversal off" in e.reason]
    assert rev, "expected at least one bin-reversal entry"
    long_rev = next((e for e in rev if e.side == "LONG"), None)
    if long_rev:  # SL sits below the support bin that held
        assert long_rev.sl_at_entry < long_rev.price


def test_manual_override_wins():
    eng = ScalpEngine("CE", ScalpConfig(require_bias_alignment=False))
    _feed_rising(eng, n=60)
    # force-exit whatever the engine did
    eng.apply_manual("exit", ts=10_000)
    assert eng.st.side == "FLAT"
    # force a manual short with explicit lots + SL
    eng.apply_manual("short", ts=10_001, lots=3, price=350.0, sl=370.0)
    assert eng.st.side == "SHORT"
    assert eng.st.lots == 3
    assert eng.st.trail_sl == 370.0
