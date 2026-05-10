"""Stage 5 — setup preview service.

Covers the deterministic plan-builder + per-criterion breakdown in
``setup_service.build_setup`` and the standalone ``evaluate_criteria``.

The production risk gate at ``trading.services.risk_engine.validate_trade``
is the authoritative verdict — we don't re-test that contract here, we just
prove that:
  1. We feed it a well-formed plan.
  2. The breakdown rows are consistent with the engine's view.
  3. Each gate's pass/fail surfaces independently (so the UI can show the
     full X-ray, not just the first failure).

Network-free: a tiny ``_StubDataPort`` replaces ``DefaultMarketData``.
"""
from __future__ import annotations

import pytest
from django.core.cache import cache

from apps.market_data.services import setup_service
from apps.market_data.services.setup_service import (
    DEFAULT_CONFIDENCE,
    MAX_RISK_PER_TRADE_PCT,
    MIN_RISK_REWARD_RATIO,
    build_setup,
    evaluate_criteria,
)
from trading.services.risk_engine import PULSE_CACHE_KEY


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_pulse_cache():
    """Most tests want a deterministic regime — either a tradeable pulse or
    a cold cache (which soft-skips the regime gate in dev mode)."""
    cache.delete(PULSE_CACHE_KEY)
    yield
    cache.delete(PULSE_CACHE_KEY)


def _seed_tradeable_pulse():
    """Warm the pulse cache with a plain-dict payload that satisfies the
    duck-typed ``_check_regime`` in trading.services.risk_engine."""
    cache.set(
        PULSE_CACHE_KEY,
        {
            "regime": {
                "vol": "moderate",
                "trend": "up",
                "global_tone": "risk-on",
                "tradeable": True,
                "summary": "Moderate vol, trend up — tradeable.",
            }
        },
        60,
    )


class _StubDataPort:
    """Pluggable replacement for DefaultMarketData. No DB, no Redis.

    ``candles`` is a flat list of close prices; we synthesise OHLC by
    treating each close as both high and low ± a small bar — keeps ATR
    deterministic and predictable.
    """
    def __init__(self, *, last: float | None, closes: list[float] | None,
                 bar_range: float = 5.0):
        self._last = last
        self._closes = closes or []
        self._bar = bar_range

    def ltp(self, _symbol: str) -> float:
        return float(self._last or 0.0)

    def candles(self, _symbol: str, _interval: str, n: int) -> list[dict]:
        out = []
        for c in self._closes[-n:]:
            out.append({
                "t": "2026-04-21T09:15:00+00:00",
                "o": c,
                "h": c + self._bar,
                "l": c - self._bar,
                "c": c,
                "v": 100_000,
            })
        return out


# ---------------------------------------------------------------------------
# build_setup — happy path
# ---------------------------------------------------------------------------
def test_build_setup_happy_path_buy_approved():
    """A 1.5×ATR stop + 3×ATR target on TCS with 5L capital should clear
    every gate the engine has — overall ``approved=True``."""
    _seed_tradeable_pulse()
    closes = [4100 + i * 2 for i in range(30)]  # gentle uptrend
    port = _StubDataPort(last=4150.0, closes=closes, bar_range=20.0)

    payload = build_setup(
        symbol="TCS",
        side="BUY",
        capital=500_000,
        daily_loss=0.0,
        open_positions=0,
        data_port=port,
    )

    # Plan was built from real data
    assert payload.plan is not None
    assert payload.plan.entry_price == pytest.approx(4150.0, rel=0.01)
    assert payload.plan.stop_loss < payload.plan.entry_price
    assert payload.plan.target > payload.plan.entry_price
    assert payload.plan.quantity > 0
    assert payload.plan.confidence == pytest.approx(DEFAULT_CONFIDENCE)
    assert payload.plan.risk_reward_ratio >= MIN_RISK_REWARD_RATIO

    # 1% risk-per-trade discipline holds
    risk_cap = 500_000 * MAX_RISK_PER_TRADE_PCT / 100
    assert payload.plan.risk_amount <= risk_cap + 1  # allow flooring slack

    # Authoritative engine verdict + breakdown both present
    assert payload.risk.approved is True, payload.risk.reason
    keys = [c.key for c in payload.risk.criteria]
    assert keys == [
        "regime", "fields", "stop_direction", "target_direction",
        "risk_per_trade", "daily_loss", "position_size", "risk_reward",
        "confidence", "open_positions",
    ]
    assert all(c.passed for c in payload.risk.criteria), \
        [c for c in payload.risk.criteria if not c.passed]


# ---------------------------------------------------------------------------
# build_setup — SELL side mirrors BUY
# ---------------------------------------------------------------------------
def test_build_setup_sell_side_mirrors_geometry():
    _seed_tradeable_pulse()
    closes = [4100 + i for i in range(30)]
    port = _StubDataPort(last=4130.0, closes=closes, bar_range=15.0)

    payload = build_setup(
        symbol="INFY",
        side="SELL",
        capital=500_000,
        data_port=port,
    )
    assert payload.plan is not None
    # SELL: SL above entry, target below entry
    assert payload.plan.stop_loss > payload.plan.entry_price
    assert payload.plan.target < payload.plan.entry_price
    assert payload.risk.approved is True


# ---------------------------------------------------------------------------
# Insufficient market data → no plan, errors surfaced, breakdown still useful
# ---------------------------------------------------------------------------
def test_build_setup_no_data_returns_safe_payload():
    _seed_tradeable_pulse()
    port = _StubDataPort(last=None, closes=[])
    payload = build_setup(
        symbol="UNKNOWN", side="BUY", capital=500_000, data_port=port,
    )
    assert payload.plan is None
    assert any("insufficient market data" in e for e in payload.errors)
    # Still emits a breakdown so the UI doesn't show a blank card
    assert payload.risk.criteria
    assert payload.risk.approved is False


# ---------------------------------------------------------------------------
# Regime gate — extreme vol should reject everything cleanly
# ---------------------------------------------------------------------------
def test_build_setup_extreme_regime_blocks_trade(monkeypatch):
    cache.set(
        PULSE_CACHE_KEY,
        {
            "regime": {
                "vol": "extreme", "trend": "down", "global_tone": "risk-off",
                "tradeable": False,
                "summary": "Extreme vol — flat the desk.",
            }
        },
        60,
    )
    closes = [4100 + i for i in range(30)]
    port = _StubDataPort(last=4150.0, closes=closes, bar_range=20.0)
    payload = build_setup(
        symbol="TCS", side="BUY", capital=500_000, data_port=port,
    )

    # Engine short-circuits on the regime gate
    assert payload.risk.approved is False
    assert "EXTREME" in payload.risk.reason or "extreme" in payload.risk.reason.lower()
    # The breakdown explicitly fails the regime row
    regime_row = next(c for c in payload.risk.criteria if c.key == "regime")
    assert regime_row.passed is False
    assert regime_row.severity == "danger"
    # Other rows still computed (operator wants to see all of them)
    assert len(payload.risk.criteria) == 10


# ---------------------------------------------------------------------------
# evaluate_criteria — direct unit tests for fail paths
# ---------------------------------------------------------------------------
def test_evaluate_criteria_flags_bad_stop_direction():
    """A BUY with SL above entry has to fail the stop-direction gate."""
    _seed_tradeable_pulse()
    rows = evaluate_criteria(
        plan={
            "symbol": "TCS", "side": "BUY",
            "entry_price": 4100, "stop_loss": 4150,  # WRONG side
            "target": 4300, "quantity": 10,
            "confidence": 0.7,
        },
        capital=500_000,
    )
    sl_row = next(r for r in rows if r.key == "stop_direction")
    assert sl_row.passed is False
    assert "WRONG" in sl_row.detail


def test_evaluate_criteria_flags_low_rr():
    _seed_tradeable_pulse()
    rows = evaluate_criteria(
        plan={
            "symbol": "TCS", "side": "BUY",
            "entry_price": 4100, "stop_loss": 4080,  # 20 risk
            "target": 4110,                          # 10 reward → R:R 0.5
            "quantity": 10, "confidence": 0.7,
        },
        capital=500_000,
    )
    rr_row = next(r for r in rows if r.key == "risk_reward")
    assert rr_row.passed is False
    assert "0.50" in rr_row.detail


def test_evaluate_criteria_flags_low_confidence():
    _seed_tradeable_pulse()
    rows = evaluate_criteria(
        plan={
            "symbol": "TCS", "side": "BUY",
            "entry_price": 4100, "stop_loss": 4080, "target": 4180,
            "quantity": 10, "confidence": 0.30,
        },
        capital=500_000,
    )
    conf_row = next(r for r in rows if r.key == "confidence")
    assert conf_row.passed is False


def test_evaluate_criteria_flags_too_many_open_positions():
    _seed_tradeable_pulse()
    rows = evaluate_criteria(
        plan={
            "symbol": "TCS", "side": "BUY",
            "entry_price": 4100, "stop_loss": 4080, "target": 4180,
            "quantity": 10, "confidence": 0.7,
        },
        capital=500_000,
        open_positions=3,   # at the cap
    )
    pos_row = next(r for r in rows if r.key == "open_positions")
    assert pos_row.passed is False


def test_evaluate_criteria_returns_all_ten_gates_in_order():
    """The UI relies on a stable row order for animation + keyboard nav."""
    _seed_tradeable_pulse()
    rows = evaluate_criteria(
        plan={
            "symbol": "TCS", "side": "BUY",
            "entry_price": 4100, "stop_loss": 4080, "target": 4180,
            "quantity": 10, "confidence": 0.7,
        },
        capital=500_000,
    )
    assert [r.key for r in rows] == [
        "regime", "fields", "stop_direction", "target_direction",
        "risk_per_trade", "daily_loss", "position_size", "risk_reward",
        "confidence", "open_positions",
    ]
