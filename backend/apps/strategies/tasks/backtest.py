from celery import shared_task

from apps.strategies.models import Backtest


@shared_task(queue="backtests")
def run_backtest(backtest_id: str) -> None:
    bt = Backtest.objects.get(id=backtest_id)
    bt.status = Backtest.Status.RUNNING
    bt.save(update_fields=["status"])
    try:
        # Placeholder metrics — real impl drives the backtester engine.
        bt.metrics = {"sharpe": 0.0, "maxdd": 0.0, "cagr": 0.0, "trades": 0}
        bt.equity_curve = []
        bt.status = Backtest.Status.DONE
    except Exception as e:  # noqa: BLE001
        bt.status = Backtest.Status.FAILED
        bt.error = str(e)
    bt.save()
