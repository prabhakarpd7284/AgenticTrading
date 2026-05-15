"""Celery beat task — drains OutboxEvents by executing the saga."""
from __future__ import annotations

from celery import shared_task
from django.db import transaction
from django.utils import timezone

from apps.trading.models import OutboxEvent
from apps.trading.services.order_saga import OrderSaga


@shared_task(queue="orders")
def process_outbox(batch: int = 50) -> int:
    """Pull up to `batch` pending events FOR UPDATE SKIP LOCKED and run the saga."""
    processed = 0
    saga = OrderSaga()
    with transaction.atomic():
        qs = (
            OutboxEvent.objects.select_for_update(skip_locked=True)
            .filter(status=OutboxEvent.Status.PENDING, next_run_at__lte=timezone.now())
            .order_by("next_run_at")[:batch]
        )
        events = list(qs)
    for ev in events:
        saga.handle(ev)
        processed += 1
    return processed
