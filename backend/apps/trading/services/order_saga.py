"""Order execution saga — processes a single OutboxEvent."""
from __future__ import annotations

from datetime import timedelta

from django.db import transaction
from django.db.models import F
from django.utils import timezone

from apps.agents_core.registry import broker_registry
from apps.trading.models import Order, OutboxEvent


MAX_ATTEMPTS = 5


class OrderSaga:
    def handle(self, event: OutboxEvent) -> None:
        # Atomically CLAIM the event: PENDING -> IN_FLIGHT in a single UPDATE.
        # Two concurrent `process_outbox` runs can both SELECT the same PENDING
        # row (the SKIP-LOCKED lock is released when its txn commits, before the
        # broker call), so the claim — not the lock — is what guarantees exactly
        # one placement. The loser's UPDATE matches 0 rows (status is no longer
        # PENDING) and bails, so the broker is never hit twice. Terminal events
        # (succeeded/dlq) also match 0 rows and are skipped.
        claimed = OutboxEvent.objects.filter(
            pk=event.pk, status=OutboxEvent.Status.PENDING,
        ).update(status=OutboxEvent.Status.IN_FLIGHT, attempts=F("attempts") + 1)
        if not claimed:
            return
        event.refresh_from_db(fields=["status", "attempts", "last_error", "next_run_at"])
        order = event.order

        try:
            broker_name = "paper" if order.portfolio.mode == "paper" else order.broker_link.broker_name
            adapter = broker_registry.get(broker_name)
            broker_order_id = adapter.place(event.payload)
        except Exception as exc:  # noqa: BLE001
            event.last_error = str(exc)
            if event.attempts >= MAX_ATTEMPTS:
                event.status = OutboxEvent.Status.DLQ
                with transaction.atomic():
                    order.status = Order.Status.FAILED
                    order.error = str(exc)
                    order.save(update_fields=["status", "error"])
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
