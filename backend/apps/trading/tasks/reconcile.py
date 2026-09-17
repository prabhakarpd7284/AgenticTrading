from celery import shared_task


@shared_task(ignore_result=True)
def reconcile_stale_intraday() -> dict:
    """Sweep INTRADAY trades left open by a dead session.

    Scheduled twice: after the close (tidy books same day) and premarket. The
    premarket run is the one that matters — if the worker was down at 15:45
    (or for a month), the EOD sweep never fired, and without the premarket run
    the stale rows would hold the open-positions cap shut for the whole day.
    """
    from apps.trading.services.reconcile import reconcile_stale_intraday as run

    return run()
