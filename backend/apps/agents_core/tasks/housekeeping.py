from datetime import timedelta

from celery import shared_task
from django.utils import timezone

from apps.agents_core.models import AgentRun


@shared_task
def expire_runs() -> int:
    cutoff = timezone.now() - timedelta(minutes=30)
    n = AgentRun.objects.filter(
        status__in=[AgentRun.Status.QUEUED, AgentRun.Status.RUNNING],
        created_at__lt=cutoff,
    ).update(status=AgentRun.Status.FAILED, error="expired", completed_at=timezone.now())
    return n
