from celery import shared_task

from apps.agents_core.models import AgentRun
from apps.agents_core.services.run_agent import run_agent as _run


@shared_task(queue="agents")
def execute_run(run_id: str) -> None:
    run = AgentRun.objects.select_related("portfolio", "triggered_by").get(id=run_id)
    _run(run)
