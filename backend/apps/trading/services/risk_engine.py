"""Canonical Risk Engine.

THE single risk gate in AlphaDesk. Replaces both:
  - legacy `trading/services/risk_engine.py` (kept for backward compat
    during migration; v2 code should use this module)
  - `apps/orders/services/risk_guard.py` (the lightweight v2 stub,
    deleted in Phase 4)

Ten criteria, evaluated in order. Regime gate runs first (fail-fast).
Failure reasons + per-criterion details flow into the Event log and
the React risk-breakdown panel.

Ports:
  PortfolioProvider — exposes capital, daily P&L, open position count
  RegimeProvider    — exposes the current market regime (or None)
Both have default implementations that read from Django models / cache.

Settings (settings.ALPHADESK):
  MAX_RISK_PER_TRADE_PCT   default 1.0
  MAX_DAILY_LOSS_PCT       default 3.0
  MAX_POSITION_SIZE_PCT    default 10.0
  MIN_RISK_REWARD_RATIO    default 1.5
  MIN_CONFIDENCE           default 0.55
  MAX_OPEN_POSITIONS       default 3
  RISK_REGIME_STRICT       default False  (env: 0/1)

Usage:
    engine = RiskEngine()
    decision = engine.validate(draft, portfolio_id=portfolio.id)
    if not decision.approved:
        ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

import structlog
from django.conf import settings

log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Value objects
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TradeDraft:
    """Minimal info the engine needs about a proposed trade.

    Field names mirror apps.trading.Trade for ease of conversion.
    Caller is responsible for sanity-converting Decimal/str/int.
    """
    symbol: str
    side: str                  # "BUY" | "SELL"
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    confidence: float = 0.0


@dataclass
class PortfolioSnapshot:
    capital: float
    used_capital: float = 0.0
    day_pnl: float = 0.0       # negative = losses (matches legacy `daily_loss` polarity inverted)
    open_positions: int = 0


@dataclass
class RegimeSnapshot:
    vol: str = "normal"        # complacent | low | normal | elevated | high | extreme
    trend: str = "neutral"
    global_tone: str = "neutral"
    tradeable: bool = True
    summary: str = ""


@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""
    criteria: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Provider ports
# ──────────────────────────────────────────────────────────────────────

class PortfolioProvider(Protocol):
    def get(self, portfolio_id) -> PortfolioSnapshot: ...


class RegimeProvider(Protocol):
    def get(self) -> RegimeSnapshot | None: ...


# ── Default implementations ──

class DjangoPortfolioProvider:
    """Reads from apps.trading.Portfolio + counts open Trade rows."""

    def get(self, portfolio_id) -> PortfolioSnapshot:
        # Lazy imports — avoid module-level Django ORM access in tests
        from apps.trading.models import Portfolio
        from apps.trading.models import Trade

        p = Portfolio.objects.get(id=portfolio_id)
        open_count = Trade.objects.filter(
            portfolio_id=portfolio_id,
            status__in=[Trade.Status.SENT, Trade.Status.PARTIAL, Trade.Status.FILLED],
        ).count()
        return PortfolioSnapshot(
            capital=float(p.capital),
            used_capital=float(p.used_capital),
            day_pnl=float(p.day_pnl),
            open_positions=open_count,
        )


class CachedRegimeProvider:
    """Reads the live Market Pulse payload from Django cache.

    Matches the legacy `trading.services.risk_engine` regime check —
    this is the single source of truth for "what regime are we in?".
    """
    CACHE_KEY = "market:pulse:v1"

    def get(self) -> RegimeSnapshot | None:
        try:
            from django.core.cache import cache
        except Exception:  # pragma: no cover
            return None
        payload = cache.get(self.CACHE_KEY)
        if payload is None:
            return None
        regime = getattr(payload, "regime", None) or (
            payload.get("regime") if isinstance(payload, dict) else None
        )
        if not regime:
            return None
        if not isinstance(regime, dict):
            # Could be a dataclass — duck-type
            regime = {
                "vol": getattr(regime, "vol", "normal"),
                "trend": getattr(regime, "trend", "neutral"),
                "global_tone": getattr(regime, "global_tone", "neutral"),
                "tradeable": getattr(regime, "tradeable", True),
                "summary": getattr(regime, "summary", ""),
            }
        return RegimeSnapshot(
            vol=regime.get("vol", "normal"),
            trend=regime.get("trend", "neutral"),
            global_tone=regime.get("global_tone", "neutral"),
            tradeable=regime.get("tradeable", True),
            summary=regime.get("summary", ""),
        )


# ──────────────────────────────────────────────────────────────────────
# The engine
# ──────────────────────────────────────────────────────────────────────

class RiskEngine:
    def __init__(
        self,
        *,
        portfolio_provider: PortfolioProvider | None = None,
        regime_provider: RegimeProvider | None = None,
    ):
        self.portfolio_provider = portfolio_provider or DjangoPortfolioProvider()
        self.regime_provider = regime_provider or CachedRegimeProvider()
        self.limits = settings.ALPHADESK
        self.regime_strict = os.getenv("RISK_REGIME_STRICT", "0") == "1"

    def validate(self, draft: TradeDraft | dict, *, portfolio_id) -> RiskDecision:
        """Run all 10 criteria against the draft. Returns first failure or
        an approve if everything passes."""
        if isinstance(draft, dict):
            draft = TradeDraft(**{k: draft[k] for k in TradeDraft.__dataclass_fields__ if k in draft})

        criteria: dict[str, Any] = {}

        # 0. Regime — fail fast
        regime_ok, regime_reason, regime_details = self._check_regime()
        criteria["regime"] = regime_details
        if not regime_ok:
            log.warning("risk.rejected.regime", reason=regime_reason)
            return RiskDecision(approved=False, reason=regime_reason, criteria=criteria)

        # 1. Field validation
        err = self._check_fields(draft)
        if err:
            return RiskDecision(approved=False, reason=err, criteria=criteria)

        # 2. SL direction
        if draft.side == "BUY" and draft.stop_loss >= draft.entry_price:
            return RiskDecision(
                approved=False,
                reason=f"BUY stop_loss ({draft.stop_loss}) must be below entry ({draft.entry_price})",
                criteria=criteria,
            )
        if draft.side == "SELL" and draft.stop_loss <= draft.entry_price:
            return RiskDecision(
                approved=False,
                reason=f"SELL stop_loss ({draft.stop_loss}) must be above entry ({draft.entry_price})",
                criteria=criteria,
            )

        # 3. Target direction
        if draft.side == "BUY" and draft.target <= draft.entry_price:
            return RiskDecision(
                approved=False,
                reason=f"BUY target ({draft.target}) must be above entry ({draft.entry_price})",
                criteria=criteria,
            )
        if draft.side == "SELL" and draft.target >= draft.entry_price:
            return RiskDecision(
                approved=False,
                reason=f"SELL target ({draft.target}) must be below entry ({draft.entry_price})",
                criteria=criteria,
            )

        portfolio = self.portfolio_provider.get(portfolio_id)
        capital = portfolio.capital
        if capital <= 0:
            return RiskDecision(approved=False, reason="Portfolio capital is zero", criteria=criteria)

        # 4. Risk per trade
        risk_per_share = abs(draft.entry_price - draft.stop_loss)
        risk_amount = risk_per_share * draft.quantity
        max_risk = capital * (float(self.limits["MAX_RISK_PER_TRADE_PCT"]) / 100)
        risk_pct = (risk_amount / capital) * 100 if capital > 0 else 0
        criteria["risk"] = {
            "amount": round(risk_amount, 2),
            "max_allowed": round(max_risk, 2),
            "pct_of_capital": round(risk_pct, 3),
        }
        if risk_amount > max_risk:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Risk {risk_amount:.0f} INR exceeds "
                    f"{self.limits['MAX_RISK_PER_TRADE_PCT']}% of capital ({max_risk:.0f} INR)"
                ),
                criteria=criteria,
            )

        # 5. Daily loss limit
        daily_loss = max(0.0, -portfolio.day_pnl)  # convert negative pnl → positive loss
        max_daily_loss = capital * (float(self.limits["MAX_DAILY_LOSS_PCT"]) / 100)
        criteria["daily_loss"] = {
            "so_far": round(daily_loss, 2),
            "max_allowed": round(max_daily_loss, 2),
        }
        if daily_loss >= max_daily_loss:
            return RiskDecision(
                approved=False,
                reason=f"Daily loss limit reached ({daily_loss:.0f} >= {max_daily_loss:.0f} INR)",
                criteria=criteria,
            )
        if (daily_loss + risk_amount) > max_daily_loss * 1.5:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Trade could breach daily loss limit. Loss so far: {daily_loss:.0f}, "
                    f"trade risk: {risk_amount:.0f}, limit: {max_daily_loss:.0f}"
                ),
                criteria=criteria,
            )

        # 6. Position size
        position_value = draft.entry_price * draft.quantity
        max_position = capital * (float(self.limits["MAX_POSITION_SIZE_PCT"]) / 100)
        criteria["position_size"] = {
            "value": round(position_value, 2),
            "max_allowed": round(max_position, 2),
        }
        if position_value > max_position:
            return RiskDecision(
                approved=False,
                reason=(
                    f"Position value {position_value:.0f} INR exceeds "
                    f"{self.limits['MAX_POSITION_SIZE_PCT']}% of capital ({max_position:.0f} INR)"
                ),
                criteria=criteria,
            )

        # 7. Risk:Reward
        reward_per_share = abs(draft.target - draft.entry_price)
        rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        min_rr = float(self.limits.get("MIN_RISK_REWARD_RATIO", 1.5))
        criteria["rr"] = {"ratio": round(rr, 2), "min_required": min_rr}
        if rr < min_rr:
            return RiskDecision(
                approved=False,
                reason=f"Risk:Reward ratio {rr:.2f} below minimum {min_rr}",
                criteria=criteria,
            )

        # 8. Confidence
        min_conf = float(self.limits.get("MIN_CONFIDENCE", 0.55))
        criteria["confidence"] = {"value": round(draft.confidence, 3), "min_required": min_conf}
        if draft.confidence < min_conf:
            return RiskDecision(
                approved=False,
                reason=f"Confidence {draft.confidence:.2f} below minimum {min_conf}",
                criteria=criteria,
            )

        # 9. Max open positions
        max_open = int(self.limits.get("MAX_OPEN_POSITIONS", 3))
        criteria["open_positions"] = {"current": portfolio.open_positions, "max_allowed": max_open}
        if portfolio.open_positions >= max_open:
            return RiskDecision(
                approved=False,
                reason=f"Max open positions reached ({portfolio.open_positions}/{max_open})",
                criteria=criteria,
            )

        # 10. (Kill switch is below regime; we also check TRADING_MODE here for safety)
        trading_mode = getattr(settings, "TRADING_MODE", "paper")
        if trading_mode == "halt":
            return RiskDecision(
                approved=False,
                reason="Trading halted by kill-switch",
                criteria=criteria,
            )

        log.info(
            "risk.approved",
            symbol=draft.symbol, side=draft.side, qty=draft.quantity,
            risk_pct=round(risk_pct, 2), rr=round(rr, 2),
        )
        return RiskDecision(approved=True, reason="Approved", criteria=criteria)

    # ── Internals ──

    def _check_fields(self, d: TradeDraft) -> str:
        if d.quantity <= 0:
            return "Quantity must be positive"
        if d.entry_price <= 0 or d.stop_loss <= 0 or d.target <= 0:
            return "Prices must be positive"
        return ""

    # ── RiskPort adapter ──

    def as_risk_port(self, portfolio_id) -> "RiskPortAdapter":
        """Return an object satisfying agents_core.domain.contracts.RiskPort.

        Plugins (LangGraph nodes) call `ctx.risk.validate(draft_dict)` which
        doesn't carry portfolio_id. The adapter binds it from the workflow's
        AgentContext at construction time.
        """
        return RiskPortAdapter(engine=self, portfolio_id=portfolio_id)

    def _check_regime(self) -> tuple[bool, str, dict[str, Any]]:
        regime = self.regime_provider.get()
        if regime is None:
            if self.regime_strict:
                return False, "Regime cache cold — refusing to trade blind", {"cache": "miss"}
            return True, "Approved (regime cache cold, soft-skip)", {"cache": "miss"}

        details = {
            "vol": regime.vol,
            "trend": regime.trend,
            "global_tone": regime.global_tone,
            "tradeable": regime.tradeable,
            "summary": regime.summary,
        }
        if regime.vol == "extreme":
            return False, f"Vol regime EXTREME — no new entries. {regime.summary}", details
        if not regime.tradeable:
            return False, f"Regime not tradeable: {regime.summary}", details
        return True, "Approved (regime OK)", details


# ──────────────────────────────────────────────────────────────────────
# RiskPort adapter — bridges the engine to the agents_core plugin contract
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RiskPortAdapter:
    """Thin adapter that lets LangGraph plugin nodes call
    `ctx.risk.validate(draft_dict)` while internally delegating to the
    canonical RiskEngine with a bound portfolio_id.

    The plugin contract `agents_core.domain.contracts.RiskPort` uses
    pydantic's `RiskDecision(approved, reason, adjustments)`; we map our
    dataclass equivalent to that on return so existing plugins don't
    need to change.
    """
    engine: "RiskEngine"
    portfolio_id: Any

    def validate(self, draft: dict):
        # Lazy import to avoid agents_core ↔ trades cycle at import time
        from apps.agents_core.domain.contracts import RiskDecision as PluginRiskDecision

        # Coerce the loose dict into TradeDraft fields
        try:
            td = TradeDraft(
                symbol=draft["symbol"],
                side=draft["side"],
                entry_price=float(draft.get("entry_price") or draft.get("price") or 0),
                stop_loss=float(draft.get("stop_loss") or draft.get("sl") or 0),
                target=float(draft.get("target") or draft.get("tp") or 0),
                quantity=int(draft.get("quantity") or draft.get("qty") or 0),
                confidence=float(draft.get("confidence", 0.55)),
            )
        except (KeyError, TypeError, ValueError) as e:
            return PluginRiskDecision(approved=False, reason=f"Invalid draft: {e}")

        decision = self.engine.validate(td, portfolio_id=self.portfolio_id)
        return PluginRiskDecision(
            approved=decision.approved,
            reason=decision.reason,
            adjustments=decision.criteria,
        )
