"""Studio backtest task (queue ``backtests``).

Replaces the old placeholder that wrote all-zero metrics. The work is done by
``apps.strategies.services.backtest_service.run_backtest_job``; the Backtest
row's post_save keeps its RunReceipt in sync (apps.system.detail_receipts).
"""
from celery import shared_task


@shared_task(queue="backtests", acks_late=False)
def run_backtest(backtest_id: str) -> str:
    from apps.strategies.services.backtest_service import run_backtest_job
    return run_backtest_job(backtest_id)
