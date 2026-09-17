"""Order execution saga — processes a single OutboxEvent."""
from __future__ import annotations

from datetime import timedelta

import structlog
from django.conf import settings
from django.db import transaction
from django.db.models import F
from django.utils import timezone

from apps.agents_core.registry import broker_registry
from apps.system.services.flags import is_kill_switch_on, is_live_trading_enabled
from apps.trading.models import Order, OutboxEvent

log = structlog.get_logger()

MAX_ATTEMPTS = 5


class OrderSaga:
    def handle(self, event: OutboxEvent) -> None:
        order = event.order

        # Emergency halt — never place while the kill switch is on. Hold the
        # event (NO retry attempt consumed) so it resumes when the operator
        # clears the switch; without this a queued order would still fire. (#14)
        if is_kill_switch_on(order.tenant_id):
            OutboxEvent.objects.filter(
                pk=event.pk, status=OutboxEvent.Status.PENDING,
            ).update(
                next_run_at=timezone.now() + timedelta(seconds=30),
                last_error="held: operator kill switch is ON",
            )
            return

        # Atomically CLAIM the event: PENDING -> IN_FLIGHT in a single UPDATE.
        # Two concurrent `process_outbox` runs can both SELECT the same PENDING
        # row (the SKIP-LOCKED lock is released when its txn commits, before the
        # broker call), so the claim — not the lock — guarantees exactly one
        # placement. The loser's UPDATE matches 0 rows and bails. Terminal
        # events (succeeded/dlq/failed) also match 0 rows and are skipped.
        claimed = OutboxEvent.objects.filter(
            pk=event.pk, status=OutboxEvent.Status.PENDING,
        ).update(status=OutboxEvent.Status.IN_FLIGHT, attempts=F("attempts") + 1)
        if not claimed:
            return
        event.refresh_from_db(fields=["status", "attempts", "last_error", "next_run_at"])

        is_live = order.portfolio.mode != "paper"

        # Live-placement gate — real money only when deliberate. Re-checked here
        # (not just at enqueue) to cover a flag flipped after the order queued.
        if is_live and not self._live_allowed(order.tenant_id):
            self._fail(
                event, order,
                "live trading disabled (TRADING_MODE/flag) — not placed",
                alert="order.live_blocked",
            )
            return

        try:
            broker_name = order.broker_link.broker_name if is_live else "paper"
            adapter = broker_registry.get(broker_name)
            broker_order_id = adapter.place(event.payload)
        except NotImplementedError as exc:
            # #16 — the broker link has no write capability. Non-retryable: fail
            # loudly now instead of burning 5 retries into a silent DLQ.
            self._fail(
                event, order, f"broker cannot place orders: {exc}",
                alert="order.place_unsupported",
            )
            return
        except Exception as exc:  # noqa: BLE001
            event.last_error = str(exc)
            if event.attempts >= MAX_ATTEMPTS:
                self._dlq(event, order, str(exc))
            else:
                event.status = OutboxEvent.Status.PENDING
                event.next_run_at = timezone.now() + timedelta(seconds=2 ** event.attempts)
                event.save()
            return

        with transaction.atomic():
            order.broker_order_id = getattr(broker_order_id, "id", str(broker_order_id))
            order.status = Order.Status.SENT
            order.save(update_fields=["broker_order_id", "status"])
            event.status = OutboxEvent.Status.SUCCEEDED
            event.save(update_fields=["status"])

    @staticmethod
    def _live_allowed(tenant_id) -> bool:
        return (
            getattr(settings, "TRADING_MODE", "paper") == "live"
            and is_live_trading_enabled(tenant_id)
        )

    def _fail(self, event: OutboxEvent, order: Order, reason: str, *, alert: str) -> None:
        """Terminal, non-retryable failure — marks the order FAILED and alerts."""
        log.error(
            alert, order_id=str(order.id), tenant_id=str(order.tenant_id), reason=reason
        )
        with transaction.atomic():
            order.status = Order.Status.FAILED
            order.error = reason
            order.save(update_fields=["status", "error"])
            event.status = OutboxEvent.Status.FAILED
            event.last_error = reason
            event.save(update_fields=["status", "last_error"])

    def _dlq(self, event: OutboxEvent, order: Order, reason: str) -> None:
        """Retries exhausted — dead-letter the event and alert the operator."""
        log.error(
            "order.dlq", order_id=str(order.id), tenant_id=str(order.tenant_id),
            attempts=event.attempts, reason=reason,
        )
        event.status = OutboxEvent.Status.DLQ
        with transaction.atomic():
            order.status = Order.Status.FAILED
            order.error = reason
            order.save(update_fields=["status", "error"])
        event.save()
