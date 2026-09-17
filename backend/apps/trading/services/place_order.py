"""Use-case: place an order. Uses Outbox pattern — returns HTTP 202 + order id.

Risk validation goes through the canonical RiskEngine in
apps.trading.services.risk_engine (10-criterion gate ported from the
legacy trading.services.risk_engine).

Before the risk gate this use-case enforces the operator controls that the
10-criterion engine can't see on its own:
  * kill switch / AI-pause (emergency halt)               — #14
  * live-placement gate (TRADING_MODE + per-tenant flag)  — #13
  * fail-loud when a live broker link can't actually place — #16
  * an absolute notional/qty backstop using a server-side
    reference price, so a tiny-price + huge-qty MARKET order
    can't slip under the percentage-of-capital caps          — #12
"""
from __future__ import annotations

from dataclasses import dataclass, field

from django.conf import settings
from django.db import IntegrityError, transaction

from apps.agents_core.registry import broker_registry
from apps.common.exceptions import RiskRejected
from apps.system.services.flags import (
    is_ai_paused,
    is_kill_switch_on,
    is_live_trading_enabled,
)
from apps.trading.domain.entities import OrderDraft, OrderResult
from apps.trading.models import Order, OutboxEvent
from apps.trading.services.risk_engine import RiskEngine, TradeDraft

# Origins that count as a human/manual action (AI-pause does not apply to them).
_MANUAL_ORIGINS = {"ui", "manual"}


def reference_price(symbol: str, tenant_id) -> float:
    """Non-blocking server-side price for the sizing guards.

    Redis ``ltp:<symbol>`` → most-recent local Candle close → 0.0. It never
    touches the broker, so it cannot wedge the single ASGI sync thread in the
    request path (see the broker-wedge incident). 0.0 means "unknown".
    """
    from django.core.cache import cache

    v = cache.get(f"ltp:{symbol}")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    from apps.market_data.models import Candle

    row = Candle.objects.filter(symbol__tradingsymbol=symbol).order_by("-t").first()
    return float(row.c) if row else 0.0


@dataclass
class PlaceOrder:
    risk: RiskEngine = field(default_factory=RiskEngine)

    @transaction.atomic
    def execute(self, tenant, user, portfolio, draft: OrderDraft,
                idempotency_key: str = "") -> OrderResult:
        # ── Emergency halt + AI pause (operator kill switch) ───────────────
        if is_kill_switch_on(tenant.id):
            raise RiskRejected("Trading halted — operator kill switch is ON")
        if draft.origin not in _MANUAL_ORIGINS and is_ai_paused(tenant.id):
            raise RiskRejected("AI trading is paused by the operator")

        # ── Idempotency: a repeated key returns the existing order ─────────
        if idempotency_key:
            existing = Order.objects.filter(
                tenant=tenant, idempotency_key=idempotency_key,
            ).first()
            if existing:
                return OrderResult(id=str(existing.id), status=existing.status)

        # ── Live-placement gate (defence in depth; saga re-checks) ─────────
        if portfolio.mode != "paper":
            self._assert_live_allowed(tenant, portfolio)

        # ── Server-side sizing backstop (#12) ──────────────────────────────
        ref_price = reference_price(draft.symbol, tenant.id)
        entry_for_risk = float(draft.price or 0)
        if draft.order_type == "MARKET" and ref_price > 0:
            # Client price is advisory for a MARKET order — size on the real LTP.
            entry_for_risk = ref_price
        self._assert_notional_cap(draft, ref_price)

        # ── Canonical 10-criterion risk gate ───────────────────────────────
        trade_draft = TradeDraft(
            symbol=draft.symbol,
            side=draft.side,
            entry_price=entry_for_risk,
            stop_loss=float(draft.sl or 0),
            target=float(draft.tp or 0),
            quantity=int(draft.qty),
            # Human-initiated orders carry full conviction — don't gate them on
            # the AI-confidence criterion. (0.55 == MIN_CONFIDENCE meant raising
            # the threshold silently rejected every UI order.) (#48)
            confidence=1.0 if draft.origin in _MANUAL_ORIGINS else 0.55,
        )
        decision = self.risk.validate(trade_draft, portfolio_id=portfolio.id)
        if not decision.approved:
            raise RiskRejected(decision.reason)

        # ── Create order + outbox (race-safe on the unique key) ────────────
        try:
            with transaction.atomic():  # savepoint so the outer txn survives a clash
                order = Order.objects.create(
                    tenant=tenant,
                    portfolio=portfolio,
                    created_by=user,
                    broker_link=portfolio.broker_link,
                    symbol=draft.symbol,
                    side=draft.side,
                    qty=draft.qty,
                    order_type=draft.order_type,
                    product=draft.product,
                    price=draft.price,
                    sl=draft.sl,
                    tp=draft.tp,
                    status=Order.Status.QUEUED,
                    idempotency_key=idempotency_key,
                    origin=draft.origin,
                )
                OutboxEvent.objects.create(order=order, payload=draft.model_dump())
        except IntegrityError:
            # A concurrent POST with the same key won the race — return its order.
            existing = Order.objects.filter(
                tenant=tenant, idempotency_key=idempotency_key,
            ).first()
            if existing:
                return OrderResult(id=str(existing.id), status=existing.status)
            raise
        return OrderResult(id=str(order.id), status=order.status)

    # ── Guards ─────────────────────────────────────────────────────────────
    def _assert_live_allowed(self, tenant, portfolio) -> None:
        """Real-money placement must be deliberate AND actually possible."""
        if getattr(settings, "TRADING_MODE", "paper") != "live":
            raise RiskRejected("Live trading is disabled (TRADING_MODE is not 'live')")
        if not is_live_trading_enabled(tenant.id):
            raise RiskRejected("Live trading is not enabled for this tenant")
        if not self._broker_can_place(portfolio):
            # #16 — fail loudly at enqueue rather than letting a live order
            # retry 5× and silently land in the DLQ marked FAILED.
            raise RiskRejected(
                "Live broker link has no order-placement capability "
                f"({getattr(portfolio.broker_link, 'broker_name', 'none')})"
            )

    @staticmethod
    def _broker_can_place(portfolio) -> bool:
        link = portfolio.broker_link
        if link is None:
            return False
        try:
            adapter = broker_registry.get(link.broker_name)
        except LookupError:
            return False
        from apps.market_data.adapters.base import BrokerAdapterBase

        # The registry stores an instance, OR the class itself for adapters
        # that need constructor args (e.g. Angel needs credentials). Resolve to
        # the class either way before checking whether place() is overridden.
        cls = adapter if isinstance(adapter, type) else type(adapter)
        return cls.place is not BrokerAdapterBase.place

    def _assert_notional_cap(self, draft: OrderDraft, ref_price: float) -> None:
        max_qty = int(settings.ALPHADESK["MAX_ORDER_QTY"])
        max_notional = float(settings.ALPHADESK["MAX_ORDER_NOTIONAL"])
        qty = int(draft.qty)
        if qty > max_qty:
            raise RiskRejected(f"Order qty {qty} exceeds the absolute cap of {max_qty}")
        # Prefer the server LTP; fall back to the client price only to bound the
        # notional. If neither is known the qty cap above is the backstop.
        px = ref_price if ref_price > 0 else float(draft.price or 0)
        if px > 0 and px * qty > max_notional:
            raise RiskRejected(
                f"Order notional {px * qty:.0f} exceeds the absolute cap of "
                f"{max_notional:.0f} INR"
            )
