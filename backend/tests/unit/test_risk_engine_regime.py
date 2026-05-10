"""Cascade Stage 1 coupling — `@RiskGuard` regime gate.

Covers criterion #10 of `trading/services/risk_engine.py::validate_trade`:
before any plan is inspected, we consult the live pulse payload in the
Django cache (shared with `apps.market_data.services.pulse_service`) and
reject if the regime is not tradeable.

Why this matters: the LLM agents can and will propose trades during
extreme vol regimes; the deterministic gate is what actually prevents a
fat-finger on a high-VIX day.  These tests lock in the contract.
"""
from __future__ import annotations

from typing import Any

import pytest
from django.core.cache import cache

from trading.services import risk_engine
from trading.services.risk_engine import PULSE_CACHE_KEY, validate_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _good_plan() -> dict[str, Any]:
    """A plan that would pass every check EXCEPT the regime gate.

    Entry 1600, SL 1580, target 1650 → risk 20, reward 50 → R:R = 2.5.
    qty=10 → notional 16,000 (3.2% of 500k, within 10% cap), risk 200 INR
    (0.04% of capital, within 1% cap).  Confidence 0.6 clears 0.55.
    """
    return {
        "symbol": "HDFCBANK",
        "side": "BUY",
        "entry_price": 1600,
        "stop_loss": 1580,
        "target": 1650,
        "quantity": 10,
        "confidence": 0.6,
    }


def _pulse(*, vol: str = "normal", tradeable: bool = True,
           trend: str = "range", summary: str = "ok") -> dict[str, Any]:
    return {
        "regime": {
            "vol": vol,
            "trend": trend,
            "global_tone": "neutral",
            "tradeable": tradeable,
            "summary": summary,
        },
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    cache.delete(PULSE_CACHE_KEY)
    yield
    cache.delete(PULSE_CACHE_KEY)


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------
def test_cache_miss_soft_skips_in_dev(monkeypatch):
    """By default (RISK_REGIME_STRICT=0), a cold pulse cache must NOT block
    trades — otherwise dev + test flows all break on boot."""
    monkeypatch.setattr(risk_engine, "RISK_REGIME_STRICT", False)
    ok, reason, details = validate_trade(_good_plan(), capital=500_000)
    assert ok is True, reason
    assert details["regime"] == {"cache": "miss"}


def test_cache_miss_hard_rejects_in_strict_mode(monkeypatch):
    """In production (RISK_REGIME_STRICT=1) a cold cache is unsafe — we
    refuse to act blind."""
    monkeypatch.setattr(risk_engine, "RISK_REGIME_STRICT", True)
    ok, reason, _ = validate_trade(_good_plan(), capital=500_000)
    assert ok is False
    assert "cold" in reason.lower()


# ---------------------------------------------------------------------------
# Regime tradeable
# ---------------------------------------------------------------------------
def test_tradeable_normal_regime_passes():
    cache.set(PULSE_CACHE_KEY, _pulse(vol="normal", tradeable=True), 30)
    ok, reason, details = validate_trade(_good_plan(), capital=500_000)
    assert ok is True, reason
    assert details["regime"]["vol"] == "normal"
    assert details["regime"]["tradeable"] is True


# ---------------------------------------------------------------------------
# Regime not tradeable — hard reject
# ---------------------------------------------------------------------------
def test_regime_not_tradeable_blocks_trade():
    cache.set(PULSE_CACHE_KEY, _pulse(
        vol="high", tradeable=False, summary="First 30 min read-only"
    ), 30)
    ok, reason, details = validate_trade(_good_plan(), capital=500_000)
    assert ok is False
    assert "not tradeable" in reason.lower()
    assert "read-only" in reason  # summary flows through to the operator
    assert details["regime"]["tradeable"] is False


# ---------------------------------------------------------------------------
# Vol extreme — always blocks even if tradeable=True (defense-in-depth)
# ---------------------------------------------------------------------------
def test_vol_extreme_always_blocks():
    # Deliberately set tradeable=True to prove the vol=extreme rule is
    # independent — a buggy classifier shouldn't be able to unlock trading.
    cache.set(PULSE_CACHE_KEY, _pulse(
        vol="extreme", tradeable=True, summary="VIX crossed 35"
    ), 30)
    ok, reason, details = validate_trade(_good_plan(), capital=500_000)
    assert ok is False
    assert "extreme" in reason.lower()
    assert details["regime"]["vol"] == "extreme"


# ---------------------------------------------------------------------------
# Regime gate short-circuits before any other check
# ---------------------------------------------------------------------------
def test_regime_gate_runs_before_field_validation():
    """Even with a malformed plan, the regime gate should fire first —
    that's how a fail-fast top-level gate works."""
    cache.set(PULSE_CACHE_KEY, _pulse(vol="extreme", tradeable=False), 30)

    # plan is missing `target` — would normally fail basic-field check
    bad = {"symbol": "X", "side": "BUY", "entry_price": 100,
           "stop_loss": 95, "quantity": 1, "confidence": 0.9}
    ok, reason, _ = validate_trade(bad, capital=500_000)
    assert ok is False
    # The reason should mention the regime, not the missing field
    assert "extreme" in reason.lower() or "not tradeable" in reason.lower()


# ---------------------------------------------------------------------------
# Dataclass payload works identically to the dict fallback
# ---------------------------------------------------------------------------
def test_accepts_pulsepayload_dataclass():
    """`pulse_service.build_pulse()` stores a `PulsePayload` dataclass —
    our getter must handle both that and the plain dict used in tests."""
    from apps.market_data.services.pulse_service import PulsePayload

    payload = PulsePayload(
        as_of="2026-04-20T10:00:00Z",
        session_phase="open",
        is_market_open=True,
        regime={"vol": "normal", "trend": "range", "global_tone": "neutral",
                "tradeable": True, "summary": "ok"},
    )
    cache.set(PULSE_CACHE_KEY, payload, 30)
    ok, reason, details = validate_trade(_good_plan(), capital=500_000)
    assert ok is True, reason
    assert details["regime"]["vol"] == "normal"
