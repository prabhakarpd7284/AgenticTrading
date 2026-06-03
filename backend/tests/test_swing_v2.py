"""
v2 swing engine — unit + integration tests (no broker IO).

Validates the parts that v1 got wrong: the regime gate, ATR-normalised
extension, contraction-base detection, freshness decay, and the money model.
"""
from __future__ import annotations

import pytest

from plugins.strategy_swing.v2 import factors as F
from plugins.strategy_swing.v2.config import SwingTier, get_tier_config
from plugins.strategy_swing.v2.engine import (
    SwingEngineV2, Regime, Phase, Action,
)


# ══════════════════════════════════════════════════════════════════════
# synthetic candle builders
# ══════════════════════════════════════════════════════════════════════

def _bar(o, h, l, c, v, i):
    return {"timestamp": f"2026-01-01 {9 + i // 60:02d}:{i % 60:02d}",
            "open": o, "high": h, "low": l, "close": c, "volume": v}


def uptrend_with_base_breakout():
    """50 rising bars → 14-bar tight base ~140 → 1 breakout bar."""
    bars, i = [], 0
    price = 100.0
    for _ in range(50):                       # steady rise 100 → ~140
        nxt = price + 0.8
        bars.append(_bar(price, nxt + 0.6, price - 0.4, nxt, 1000, i)); price = nxt; i += 1
    base = price
    for k in range(14):                       # tight base, low volume
        c = base + (0.4 if k % 2 else -0.4)
        bars.append(_bar(base, base + 0.7, base - 0.7, c, 800, i)); i += 1
    # breakout bar: clears base high, volume expansion, modest extension
    bars.append(_bar(base, base + 4.2, base - 0.2, base + 4.0, 3200, i))
    return bars


def downtrend_with_base_breakdown():
    bars, i = [], 0
    price = 200.0
    for _ in range(50):
        nxt = price - 0.8
        bars.append(_bar(price, price + 0.4, nxt - 0.6, nxt, 1000, i)); price = nxt; i += 1
    base = price
    for k in range(14):
        c = base + (0.4 if k % 2 else -0.4)
        bars.append(_bar(base, base + 0.7, base - 0.7, c, 800, i)); i += 1
    bars.append(_bar(base, base + 0.2, base - 4.2, base - 4.0, 3200, i))
    return bars


def parabolic_blowoff():
    """Uptrend then a vertical spike many ATRs above the EMA."""
    bars, i = [], 0
    price = 100.0
    for _ in range(55):
        nxt = price + 0.5
        bars.append(_bar(price, nxt + 0.4, price - 0.3, nxt, 1000, i)); price = nxt; i += 1
    # blowoff: 4 huge bars
    for _ in range(4):
        nxt = price + 8.0
        bars.append(_bar(price, nxt + 1, price, nxt, 5000, i)); price = nxt; i += 1
    return bars


# ══════════════════════════════════════════════════════════════════════
# factor unit tests (pure, deterministic)
# ══════════════════════════════════════════════════════════════════════

def test_regime_gate_blocks_counter_trend():
    # long wants bullish; bearish primary → 0 (hard gate)
    assert F.regime_score("bearish", "bearish", "long") == 0.0
    assert F.regime_score("bullish", "bullish", "long") == 1.0
    assert F.regime_score("bullish", "bullish", "short") == 0.0
    assert F.regime_score("bearish", "bearish", "short") == 1.0
    # primary leads HTF (counter-trend) → low but nonzero
    assert 0.0 < F.regime_score("bullish", "bearish", "long") < 0.5


def test_freshness_decays_to_zero():
    cfg = get_tier_config(SwingTier.MEDIUM)
    assert F.freshness_score(0, cfg.freshness_max_bars) == 1.0
    assert F.freshness_score(cfg.freshness_max_bars, cfg.freshness_max_bars) == 0.0
    mid = F.freshness_score(cfg.freshness_max_bars // 2, cfg.freshness_max_bars)
    assert 0.3 < mid < 0.7


def test_extension_penalises_chasing():
    near = F.extension_score(0.5, "long", ideal=1.5, chase=4.0)   # near EMA
    far = F.extension_score(5.0, "long", ideal=1.5, chase=4.0)    # 5 ATRs out
    assert near > 0.8
    assert far < 0.2
    assert near > far


def test_base_rewards_tight_and_long():
    loose = F.base_score(0.95, 3, min_bars=4, target_ratio=0.6)   # too short
    tight = F.base_score(0.4, 10, min_bars=4, target_ratio=0.6)
    assert tight > loose
    assert tight > 0.6


def test_risk_geometry_prefers_tight_stop_high_rr():
    good = F.risk_geometry_score(risk_pct=0.02, rr=3.0, rr_target=2.5)
    bad = F.risk_geometry_score(risk_pct=0.10, rr=1.0, rr_target=2.5)
    assert good > bad


# ══════════════════════════════════════════════════════════════════════
# engine primitives
# ══════════════════════════════════════════════════════════════════════

def test_regime_at_requires_stack_and_slope():
    assert SwingEngineV2.regime_at(12, 11, 10, 0.5) == Regime.BULLISH
    assert SwingEngineV2.regime_at(10, 11, 12, -0.5) == Regime.BEARISH
    assert SwingEngineV2.regime_at(11, 12, 10, 0.5) == Regime.NEUTRAL  # not stacked
    # stacked up but slow EMA rolling over → not bullish
    assert SwingEngineV2.regime_at(12, 11, 10, -1.0) == Regime.NEUTRAL


def test_base_metrics_detects_contraction():
    eng = SwingEngineV2(get_tier_config(SwingTier.MEDIUM))
    df = eng.compute(uptrend_with_base_breakout())
    ratio, base_len = eng._base_metrics(df, len(df) - 1)
    assert ratio < 0.6          # base is much tighter than the prior rise
    assert base_len >= 4


# ══════════════════════════════════════════════════════════════════════
# integration
# ══════════════════════════════════════════════════════════════════════

def test_insufficient_data_returns_error():
    eng = SwingEngineV2(get_tier_config(SwingTier.MEDIUM))
    sig = eng.analyze("TEST", [{"timestamp": "t", "open": 1, "high": 1, "low": 1,
                                "close": 1, "volume": 1}])
    assert sig.error
    assert sig.phase == Phase.NONE


def test_uptrend_breakout_yields_consistent_long():
    eng = SwingEngineV2(get_tier_config(SwingTier.MEDIUM), capital=500_000)
    sig = eng.analyze("TEST", uptrend_with_base_breakout())
    assert sig.regime_primary == Regime.BULLISH
    assert sig.phase in (Phase.BASIN_BREAK, Phase.WEDGE_POP, Phase.EMA_CROSSBACK)
    assert sig.direction == "long"
    assert sig.action in (Action.BUY, Action.ADD)
    # money model sanity
    assert sig.stop < sig.entry < sig.target
    assert sig.qty > 0
    assert sig.rr > 0
    assert sig.grade in ("A", "B", "C")
    assert sig.actionable


def test_downtrend_breakdown_yields_short():
    eng = SwingEngineV2(get_tier_config(SwingTier.MEDIUM), capital=500_000)
    sig = eng.analyze("TEST", downtrend_with_base_breakdown())
    assert sig.regime_primary == Regime.BEARISH
    assert sig.direction == "short"
    assert sig.action == Action.SHORT
    assert sig.target < sig.entry < sig.stop
    assert sig.qty > 0


def test_exhaustion_is_trim_not_short():
    eng = SwingEngineV2(get_tier_config(SwingTier.MEDIUM))
    candles = parabolic_blowoff()
    sig = eng.analyze("TEST", candles)
    if sig.phase != Phase.EXHAUSTION_EXTENSION:
        import json, pandas as pd
        df = eng.compute(candles)
        last = len(df) - 1
        with open("/tmp/v2_diag.json", "w") as fh:
            json.dump({
                "phase": sig.phase.value,
                "pandas_options": {
                    "mode.use_inf_as_na": str(pd.get_option("mode.use_inf_as_na"))
                    if "use_inf_as_na" in str(pd.describe_option("mode.use_inf_as_na", _print_desc=False) or "") else "n/a",
                },
                "tail": [
                    {"i": i, "close": float(df["close"].iloc[i]),
                     "ema_fast": float(df["ema_fast"].iloc[i]),
                     "ema_slow": float(df["ema_slow"].iloc[i]),
                     "atr": float(df["atr"].iloc[i]),
                     "ext_atr": None if pd.isna(df["ext_atr"].iloc[i]) else float(df["ext_atr"].iloc[i])}
                    for i in range(last - 4, last + 1)
                ],
            }, fh, indent=2)
    assert sig.phase == Phase.EXHAUSTION_EXTENSION
    assert sig.action == Action.TRIM        # never auto-short a strong uptrend
    assert not sig.actionable


def test_analyze_is_deterministic():
    eng = SwingEngineV2(get_tier_config(SwingTier.MEDIUM))
    candles = uptrend_with_base_breakout()
    a = eng.analyze("TEST", candles).to_dict()
    b = eng.analyze("TEST", candles).to_dict()
    assert a == b


# ══════════════════════════════════════════════════════════════════════
# DAO layer — the persist path must be bulk + idempotent
# ══════════════════════════════════════════════════════════════════════

def _fake_long(symbol: str):
    from plugins.strategy_swing.v2.engine import SwingSignalV2, Regime
    s = SwingSignalV2(symbol=symbol, tier="medium")
    s.phase = Phase.BASIN_BREAK
    s.action = Action.BUY
    s.direction = "long"
    s.regime_primary = Regime.BULLISH
    s.regime_htf = Regime.BULLISH
    s.aligned = True
    s.grade = "A"
    s.score = 0.81
    s.entry, s.stop, s.target = 100.0, 96.0, 110.0
    s.risk_per_share, s.rr, s.qty = 4.0, 2.5, 125
    s.factors = {"regime": 1.0}
    s.reasons = ["test"]
    return s


@pytest.mark.django_db
def test_persist_dao_is_single_bulk_insert_and_idempotent():
    """The optimised DAO path must: (1) read existing keys in ONE query,
    (2) write all new signals in ONE bulk INSERT, (3) be idempotent."""
    from django.db import connection
    from django.test.utils import CaptureQueriesContext
    from tests.factories import TenantFactory, UserFactory, MembershipFactory
    from apps.strategies.models import Signal
    from trading.management.commands.run_ok_scanner_v2 import Command

    tenant = TenantFactory()
    user = UserFactory()
    MembershipFactory(user=user, tenant=tenant, role="owner")

    sigs = [_fake_long(f"SYM{i}") for i in range(8)]
    cmd = Command()

    with CaptureQueriesContext(connection) as ctx:
        created = cmd._persist(sigs, "medium")

    assert created == 8
    inserts = [q for q in ctx.captured_queries
               if 'insert into "strategies_signal"' in q["sql"].lower()]
    assert len(inserts) == 1, f"expected ONE bulk insert, got {len(inserts)}"
    assert Signal.objects.filter(tenant=tenant, source=Signal.Source.OK_SCANNER).count() == 8

    # idempotent re-run — existing keys are skipped, nothing new written
    assert cmd._persist(sigs, "medium") == 0
    assert Signal.objects.filter(tenant=tenant, source=Signal.Source.OK_SCANNER).count() == 8
