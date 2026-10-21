"""Owner-approved trading rules gate — docs/strategies/TRADING_RULES.md.

Runs in ``PlaceOrder`` for *automated* (non-manual) origins, after the
operator controls (kill switch / AI pause) and BEFORE the 10-criterion
RiskEngine — which stays the last gate (CLAUDE.md invariant #1).

  §1 losing-streak cooldown  → block entries (shadow-record them) / ×0.5 risk
  §3 nightly sentiment filter → block BUY on strong_negative, SELL on
                                strong_positive, both on an event tomorrow

Deterministic; no LLM decides anything here (the LLM only scored headlines
the night before, and its output can only *block*, never size or force).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import structlog
from django.utils import timezone

from apps.trading.services.cooldown import evaluate_for_portfolio
from apps.trading.services.trailing_stop import normalize_symbol

log = structlog.get_logger(__name__)


@dataclass
class RulesDecision:
    allowed: bool
    reason: str = ""
    rule: str = ""
    risk_multiplier: float = 1.0
    tags: dict = field(default_factory=dict)


def check_sentiment(symbol: str, side: str, trading_day: date) -> dict:
    """Sentiment-filter verdict as a plain dict. Never raises: any failure is
    reported as ``{"allowed": True, "status": "error", ...}`` (fail open,
    TRADING_RULES.md §3 "if sentiment couldn't be computed, don't block")."""
    try:
        from apps.news.services.sentiment_filter import check_direction
        return check_direction(normalize_symbol(symbol), side, trading_day)
    except Exception as e:  # noqa: BLE001
        log.warning("rules.sentiment_filter_failed", symbol=symbol, error=str(e))
        return {"allowed": True, "status": "error", "reason": f"filter error: {e}"}


def check_automated_entry(portfolio, draft, *, today: date | None = None) -> RulesDecision:
    today = today or timezone.localdate()
    cd = evaluate_for_portfolio(portfolio, today)
    tags: dict = {"cooldown": cd.status, "risk_multiplier": cd.risk_multiplier,
                  "loss_streak": cd.loss_streak}
    if cd.blocks_entries:
        return RulesDecision(False, reason=f"Losing-streak cooldown: {cd.reason}",
                             rule="cooldown", risk_multiplier=0.0,
                             tags={**tags, "cooldown_detail": cd.as_dict()})

    sent = check_sentiment(draft.symbol, draft.side, today)
    tags["sentiment"] = sent
    if not sent.get("allowed", True):
        return RulesDecision(False, reason=f"Sentiment filter: {sent.get('reason', '')}",
                             rule="sentiment", risk_multiplier=cd.risk_multiplier, tags=tags)
    return RulesDecision(True, risk_multiplier=cd.risk_multiplier, tags=tags)


def record_block(tenant, portfolio, draft, decision: RulesDecision, *, user=None) -> None:
    """Audit a blocked automated entry. During a cooldown the event is the
    *shadow trade* — the would-be order, kept so we can later measure whether
    the cooldown helped. Non-blocking (``emit`` swallows its own failures)."""
    from apps.events.models import Event
    from apps.events.services.event_writer import emit

    shadow = decision.rule == "cooldown"
    emit(
        tenant=tenant,
        type=Event.Type.RISK_REJECTED,
        severity=Event.Severity.WARN,
        actor_kind=Event.ActorKind.RISK_ENGINE,
        step_name="rules.cooldown.shadow_trade" if shadow else "rules.sentiment_filter",
        text=(f"{'shadow trade (cooldown)' if shadow else 'blocked'}: "
              f"{draft.side} {draft.qty} {draft.symbol} — {decision.reason}")[:500],
        payload={
            "rule": decision.rule,
            "shadow": shadow,
            "reason": decision.reason,
            "portfolio_id": str(portfolio.id),
            "draft": draft.model_dump(),
            "tags": decision.tags,
        },
    )


def tag_accepted(tenant, portfolio, draft, order_id: str, decision: RulesDecision) -> None:
    """Tag an accepted automated order with the rule state it passed through
    (cooldown status, multiplier, sentiment verdict — incl. "no scan")."""
    from apps.events.models import Event
    from apps.events.services.event_writer import emit

    emit(
        tenant=tenant,
        type=Event.Type.ORDER_QUEUED,
        actor_kind=Event.ActorKind.RISK_ENGINE,
        step_name="rules.gate",
        text=(f"rules passed: {draft.side} {draft.qty} {draft.symbol} "
              f"(cooldown={decision.tags.get('cooldown')}, "
              f"sentiment={(decision.tags.get('sentiment') or {}).get('status')})"),
        payload={"order_id": order_id, "portfolio_id": str(portfolio.id),
                 "tags": decision.tags},
    )
