from datetime import timedelta

from celery import shared_task
from django.utils import timezone

from apps.agents_core.models import AgentRun


@shared_task(ignore_result=True)
def expire_runs() -> int:
    cutoff = timezone.now() - timedelta(minutes=30)
    qs = AgentRun.objects.filter(
        status__in=[AgentRun.Status.QUEUED, AgentRun.Status.RUNNING],
        created_at__lt=cutoff,
    )
    ids = list(qs.values_list("id", flat=True))
    if not ids:
        return 0
    n = AgentRun.objects.filter(id__in=ids).update(
        status=AgentRun.Status.FAILED, error="expired", completed_at=timezone.now())
    # .update() skips post_save, so mirror the expiry into the run receipts
    # here — an expired run must show up as a failed receipt (and alert).
    from apps.system.services.detail_receipts import sync_agent_run

    for run in AgentRun.objects.filter(id__in=ids):
        sync_agent_run(run)
    return n
