"""Deterministic RiskGuard. No LLM. No exceptions to the rules.

Ports the logic shape from trading/services/risk_engine.py but expresses it as a pure
domain service, decoupled from Django models. Repositories hand over what it needs.
"""
from __future__ import annotations

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
    def __init__(self, portfolio_provider: PortfolioProvider | None = None):
        self.portfolio_provider = portfolio_provider
        self.limits = settings.ALPHADESK

    # Public contract ------------------------------------------------
    def validate(self, draft: dict) -> RiskDecision:
        """`draft` must contain: portfolio_id, symbol, qty, price or sl, side."""
        errors: list[str] = []
        if draft.get("qty", 0) <= 0:
            errors.append("qty must be > 0")
        trading_mode = getattr(settings, "TRADING_MODE", "paper")
        if trading_mode == "halt":
            errors.append("trading is halted by kill-switch")

        # Size check
        price = draft.get("price") or 0
        notional = Decimal(str(price)) * Decimal(draft.get("qty", 0))
        if self.portfolio_provider:
            snap = self.portfolio_provider.get(draft["portfolio_id"])
            max_pos = snap.capital * Decimal(self.limits["MAX_POSITION_SIZE_PCT"]) / 100
            if notional > max_pos:
                errors.append(
                    f"position size {notional} exceeds {self.limits['MAX_POSITION_SIZE_PCT']}% "
                    f"of capital ({max_pos})"
                )
            max_daily = snap.capital * Decimal(self.limits["MAX_DAILY_LOSS_PCT"]) / 100
            if snap.day_pnl <= -max_daily:
                errors.append("daily loss limit already hit — new entries blocked")

        if errors:
            return RiskDecision(approved=False, reason="; ".join(errors))
        return RiskDecision(approved=True)
