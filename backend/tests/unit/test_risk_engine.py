"""Canonical RiskEngine — 10-criterion gate ported from the legacy engine.

Each test focuses on one criterion; failing any of these means the system
is unsafe to ship. The portfolio + regime ports are stubbed so tests run
without a database or live market data.
"""
from __future__ import annotations

import pytest

from apps.trading.services.risk_engine import (
    PortfolioSnapshot,
    RegimeSnapshot,
    RiskEngine,
    TradeDraft,
)


# ── Stub providers ────────────────────────────────────────────────────

class StubPortfolio:
    def __init__(self, snap: PortfolioSnapshot):
        self.snap = snap

    def get(self, _pid):
        return self.snap


class StubRegime:
    def __init__(self, snap: RegimeSnapshot | None):
        self.snap = snap

    def get(self):
        return self.snap


@pytest.fixture
def healthy_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        capital=500_000, used_capital=0, day_pnl=0, open_positions=0,
    )


@pytest.fixture
def healthy_regime() -> RegimeSnapshot:
    return RegimeSnapshot(vol="normal", tradeable=True, summary="OK")


@pytest.fixture
def engine(healthy_portfolio, healthy_regime):
    return RiskEngine(
        portfolio_provider=StubPortfolio(healthy_portfolio),
        regime_provider=StubRegime(healthy_regime),
    )


def _draft(**overrides) -> TradeDraft:
    # Default: R:R 1.74, risk = 73 INR (well under 1% of 500k = 5000)
    base = dict(
        symbol="HDFCBANK", side="BUY",
        entry_price=1_492.30, stop_loss=1_485.00, target=1_505.00,
        quantity=10, confidence=0.72,
    )
    base.update(overrides)
    return TradeDraft(**base)


# ── Criterion 0: regime gate ──────────────────────────────────────────

def test_rejects_when_vol_extreme(engine):
    engine.regime_provider = StubRegime(RegimeSnapshot(vol="extreme", tradeable=True))
    d = engine.validate(_draft(), portfolio_id="any")
    assert d.approved is False
    assert "EXTREME" in d.reason


def test_rejects_when_regime_not_tradeable(engine):
    engine.regime_provider = StubRegime(RegimeSnapshot(vol="normal", tradeable=False, summary="halt"))
    d = engine.validate(_draft(), portfolio_id="any")
    assert d.approved is False
    assert "not tradeable" in d.reason


def test_soft_skips_when_regime_cache_cold(engine):
    engine.regime_provider = StubRegime(None)
    d = engine.validate(_draft(), portfolio_id="any")
    assert d.approved is True  # soft-skip in non-strict mode


# ── Criterion 1: field validation ─────────────────────────────────────

def test_rejects_zero_qty(engine):
    d = engine.validate(_draft(quantity=0), portfolio_id="any")
    assert d.approved is False
    assert "Quantity" in d.reason


def test_rejects_negative_price(engine):
    d = engine.validate(_draft(entry_price=-1), portfolio_id="any")
    assert d.approved is False
    assert "positive" in d.reason


# ── Criteria 2-3: SL/target direction ─────────────────────────────────

def test_rejects_buy_with_sl_above_entry(engine):
    d = engine.validate(_draft(stop_loss=1_500), portfolio_id="any")
    assert d.approved is False
    assert "stop_loss" in d.reason


def test_rejects_sell_with_sl_below_entry(engine):
    d = engine.validate(_draft(side="SELL", stop_loss=1_480, target=1_470), portfolio_id="any")
    assert d.approved is False
    assert "stop_loss" in d.reason


def test_rejects_buy_with_target_below_entry(engine):
    d = engine.validate(_draft(target=1_480), portfolio_id="any")
    assert d.approved is False
    assert "target" in d.reason


# ── Criterion 4: risk per trade ───────────────────────────────────────

def test_rejects_when_risk_exceeds_pct(engine, healthy_portfolio):
    # 5000 / 500000 = 1% cap; risk-per-share 100 * qty 100 = 10000
    d = engine.validate(
        _draft(stop_loss=1_392.30, target=1_692.30, quantity=100),
        portfolio_id="any",
    )
    assert d.approved is False
    assert "Risk" in d.reason


# ── Criterion 5: daily loss limit ─────────────────────────────────────

def test_rejects_when_daily_loss_breached(engine, healthy_portfolio):
    healthy_portfolio.day_pnl = -16_000  # > 3% of 500k = 15000
    d = engine.validate(_draft(), portfolio_id="any")
    assert d.approved is False
    assert "Daily loss" in d.reason


def test_rejects_when_trade_would_breach_daily_loss(engine, healthy_portfolio):
    # Already lost 14k; new trade risks 8k more; 1.5 × 15000 = 22500 — exceeded
    healthy_portfolio.day_pnl = -14_000
    d = engine.validate(
        _draft(stop_loss=1_400, target=1_700, quantity=100),
        portfolio_id="any",
    )
    assert d.approved is False


# ── Criterion 6: position size ────────────────────────────────────────

def test_rejects_position_over_10pct(engine):
    # 100 × 1492 = 149200, > 10% of 500k = 50000
    d = engine.validate(
        _draft(stop_loss=1_491.50, target=1_493.00, quantity=100),
        portfolio_id="any",
    )
    assert d.approved is False
    assert "Position value" in d.reason


# ── Criterion 7: risk:reward ratio ────────────────────────────────────

def test_rejects_bad_rr(engine):
    # entry 100, SL 90 (risk 10), target 105 (reward 5) → R:R 0.5
    d = engine.validate(
        _draft(entry_price=100, stop_loss=90, target=105),
        portfolio_id="any",
    )
    assert d.approved is False
    assert "Risk:Reward" in d.reason


# ── Criterion 8: confidence ───────────────────────────────────────────

def test_rejects_low_confidence(engine):
    d = engine.validate(_draft(confidence=0.30), portfolio_id="any")
    assert d.approved is False
    assert "Confidence" in d.reason


# ── Criterion 9: max open positions ───────────────────────────────────

def test_rejects_when_max_open_reached(engine, healthy_portfolio):
    healthy_portfolio.open_positions = 3  # MAX_OPEN_POSITIONS default = 3
    d = engine.validate(_draft(), portfolio_id="any")
    assert d.approved is False
    assert "Max open positions" in d.reason


# ── Happy path ────────────────────────────────────────────────────────

def test_approves_well_formed_trade(engine):
    d = engine.validate(_draft(), portfolio_id="any")
    assert d.approved is True
    assert d.reason == "Approved"
    # criteria dict carries the per-rule breakdown for the UI
    assert "rr" in d.criteria
    assert "risk" in d.criteria
    assert "position_size" in d.criteria


# ── as_risk_port adapter (used by LangGraph plugin nodes) ─────────────

def test_adapter_validates_with_legacy_draft_dict(engine):
    adapter = engine.as_risk_port(portfolio_id="any")
    d = adapter.validate({
        "symbol": "HDFCBANK", "side": "BUY",
        "entry_price": 1_492.30, "stop_loss": 1_485, "target": 1_505,
        "quantity": 10, "confidence": 0.72,
    })
    assert d.approved is True


def test_adapter_rejects_invalid_draft_dict(engine):
    adapter = engine.as_risk_port(portfolio_id="any")
    d = adapter.validate({"symbol": "X"})  # missing required fields
    assert d.approved is False
