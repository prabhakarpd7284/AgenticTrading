"""DEPRECATED shim — kept only for backward compatibility during the
unification migration. New code should import from
`apps.trading.services.risk_engine.RiskEngine` directly.

This shim preserves the lightweight `DeterministicRiskGuard.validate(draft)`
contract (returns the pydantic `RiskDecision` from agents_core contracts)
while delegating to the canonical 10-criterion engine where possible.

Deletion target: Phase 6 of the redesign-v2 migration.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from django.conf import settings

from apps.agents_core.domain.contracts import RiskDecision


@dataclass
class PortfolioSnapshot:
    capital: Decimal
    used_capital: Decimal
    day_pnl: Decimal
    open_positions: int


class PortfolioProvider(Protocol):
    def get(self, portfolio_id) -> PortfolioSnapshot: ...


class DeterministicRiskGuard:
    """DEPRECATED. Use apps.trading.services.risk_engine.RiskEngine instead.

    The new engine has the stricter (10-criterion) validation and consumes
    the same settings.ALPHADESK limits. This class delegates to it when
    portfolio_id is in the draft; otherwise it falls back to a minimal
    qty/halt check so existing call sites with no portfolio still pass.
    """

    def __init__(self, portfolio_provider: PortfolioProvider | None = None):
        warnings.warn(
            "DeterministicRiskGuard is deprecated; "
            "use apps.trading.services.risk_engine.RiskEngine.",
            DeprecationWarning, stacklevel=2,
        )
        self.portfolio_provider = portfolio_provider
        self.limits = settings.ALPHADESK

    def validate(self, draft: dict) -> RiskDecision:
        # Cheap pre-checks that don't need a portfolio
        if draft.get("qty", 0) <= 0:
            return RiskDecision(approved=False, reason="qty must be > 0")
        if getattr(settings, "TRADING_MODE", "paper") == "halt":
            return RiskDecision(approved=False, reason="trading is halted by kill-switch")

        # Delegate the full 10-criterion check to the canonical engine.
        # If the caller supplied a stub portfolio_provider (legacy test path),
        # bridge it through; otherwise the engine uses DjangoPortfolioProvider.
        portfolio_id = draft.get("portfolio_id")
        if portfolio_id is None:
            return RiskDecision(approved=True)  # caller didn't bind a portfolio; soft-pass

        from apps.trading.services.risk_engine import (
            PortfolioSnapshot as EnginePortfolioSnapshot,
            RiskEngine,
        )

        engine_provider = None
        if self.portfolio_provider is not None:
            stub = self.portfolio_provider  # captured for closure

            class _BridgeProvider:
                def get(self, pid):
                    legacy_snap = stub.get(pid)
                    return EnginePortfolioSnapshot(
                        capital=float(legacy_snap.capital),
                        used_capital=float(legacy_snap.used_capital),
                        day_pnl=float(legacy_snap.day_pnl),
                        open_positions=legacy_snap.open_positions,
                    )

            engine_provider = _BridgeProvider()

        engine = RiskEngine(portfolio_provider=engine_provider)
        adapter = engine.as_risk_port(portfolio_id=portfolio_id)
        return adapter.validate(draft)
