from celery import shared_task

from apps.system.services.run_receipts import tracked_task

TASK_NAME = "apps.trading.tasks.reconcile.reconcile_stale_intraday"


def _describe(run, result) -> None:
    if isinstance(result, dict):
        run.output(**{k: v for k, v in result.items() if isinstance(v, (int, float, str))})
        if result.get("failed"):
            run.swallowed_error(f"{result['failed']} stale trade(s) could not be reconciled",
                                count=int(result["failed"]))


@shared_task(name=TASK_NAME, ignore_result=True)
@tracked_task("job", "reconcile_stale_intraday", task_name=TASK_NAME, max_seconds=1800,
              post=_describe)
def reconcile_stale_intraday() -> dict:
    """Sweep INTRADAY trades left open by a dead session.

    Scheduled twice: after the close (tidy books same day) and premarket. The
    premarket run is the one that matters — if the worker was down at 15:45
    (or for a month), the EOD sweep never fired, and without the premarket run
    the stale rows would hold the open-positions cap shut for the whole day.
    """
    from apps.trading.services.reconcile import reconcile_stale_intraday as run

    return run()
