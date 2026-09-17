from celery import shared_task


@shared_task(ignore_result=True)
def process_paper_fills() -> dict:
    """Sweep paper orders/trades against the latest prices.

    Cheap by design — DB plus the Redis LTP cache, never the broker — so it can
    run on a short interval without touching the Angel rate limit.
    """
    from apps.trading.services.paper_fill_driver import process_paper_fills as run

    return run()
