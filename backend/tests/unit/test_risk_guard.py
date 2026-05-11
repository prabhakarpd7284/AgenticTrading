"""Deterministic RiskGuard — DEPRECATED tests.

This test module exercises the legacy `DeterministicRiskGuard` shim which now
delegates to the canonical `apps.trades.services.risk_engine.RiskEngine`. The
new engine has stricter 10-criterion validation, so drafts must carry full
trade economics (entry_price + stop_loss + target) — not just `price`.

The fresh test suite against the canonical engine lives in
`tests/unit/test_risk_engine.py`. Once the deprecation window closes (Phase 6
of the redesign-v2 migration), this file gets deleted.

Rules still covered by this shim:
  1. qty ≤ 0 → reject
  2. TRADING_MODE == "halt" → reject (kill-switch)
  3-10. delegated to the canonical engine when portfolio_id is present
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from django.test import override_settings

from apps.orders.services.risk_guard import (
    DeterministicRiskGuard,
    PortfolioSnapshot,
)


class StubPortfolioProvider:
    """In-memory snapshot provider — lets the guard run without any DB."""

    def __init__(self, snap: PortfolioSnapshot):
        self.snap = snap

    def get(self, _portfolio_id) -> PortfolioSnapshot:  # noqa: D401
        return self.snap


@pytest.fixture
def paper_snap() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        capital=Decimal("500000"),
        used_capital=Decimal("0"),
        day_pnl=Decimal("0"),
        open_positions=0,
    )


@pytest.fixture
def guard(paper_snap):
    return DeterministicRiskGuard(StubPortfolioProvider(paper_snap))


def _draft(**overrides):
    # The new engine requires entry/SL/target/confidence; defaults keep R:R 1.74 and
    # confidence above the threshold so size/daily-loss tests can vary one knob at a time.
    base = {
        "portfolio_id": "00000000-0000-4000-a000-000000000001",
        "symbol": "HDFCBANK",
        "qty": 10,
        "price": 1_600,
        "entry_price": 1_600,
        "stop_loss": 1_592,
        "target": 1_614,
        "confidence": 0.72,
        "side": "BUY",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Rule 1 — qty must be positive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qty", [0, -1, -100])
def test_rejects_non_positive_qty(guard, qty):
    decision = guard.validate(_draft(qty=qty))
    assert decision.approved is False
    assert "qty" in decision.reason


# ---------------------------------------------------------------------------
# Rule 2 — kill-switch
# ---------------------------------------------------------------------------
@override_settings(TRADING_MODE="halt")
def test_rejects_when_trading_halted(guard):
    decision = guard.validate(_draft())
    assert decision.approved is False
    assert "halt" in decision.reason.lower() or "kill" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Rule 3 — position size cap (10% of capital by default)
# ---------------------------------------------------------------------------
def test_rejects_position_over_size_cap(guard):
    # 100 × 1,600 = 160,000 ≈ 32% of 500,000 → over 10% cap
    decision = guard.validate(_draft(qty=100, price=1_600))
    assert decision.approved is False
    # Canonical engine phrases the failure as "Position value … exceeds …% of capital"
    assert "Position value" in decision.reason or "position size" in decision.reason


def test_accepts_position_at_or_under_size_cap(guard):
    # 30 × 1,600 = 48,000 → 9.6% of 500,000 → under 10% cap
    decision = guard.validate(_draft(qty=30, price=1_600))
    assert decision.approved is True, decision.reason


# ---------------------------------------------------------------------------
# Rule 4 — daily loss cap (3% of capital by default)
# ---------------------------------------------------------------------------
def test_rejects_when_daily_loss_already_breached():
    snap = PortfolioSnapshot(
        capital=Decimal("500000"),
        used_capital=Decimal("0"),
        # 3% of 500k = 15,000. Anything below -15k trips the rule.
        day_pnl=Decimal("-16000"),
        open_positions=2,
    )
    guard = DeterministicRiskGuard(StubPortfolioProvider(snap))
    decision = guard.validate(_draft())
    assert decision.approved is False
    assert "daily loss" in decision.reason.lower()


def test_accepts_when_daily_loss_under_cap():
    snap = PortfolioSnapshot(
        capital=Decimal("500000"),
        used_capital=Decimal("0"),
        day_pnl=Decimal("-5000"),  # 1% loss
        open_positions=1,
    )
    guard = DeterministicRiskGuard(StubPortfolioProvider(snap))
    decision = guard.validate(_draft(qty=10, price=1_600))
    assert decision.approved is True


# ---------------------------------------------------------------------------
# Determinism smoke — identical inputs always produce identical decisions.
# ---------------------------------------------------------------------------
def test_validate_is_deterministic(guard):
    draft = _draft(qty=100, price=1_600)
    a = guard.validate(draft)
    b = guard.validate(draft)
    assert a.approved == b.approved
    assert a.reason == b.reason
