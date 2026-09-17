# Celery's `autodiscover_tasks()` imports the `tasks` package — re-export the
# submodule tasks here so the @shared_task decorators register on worker boot.
# Without this, `process_outbox` and `refresh_all` never appear in the worker's
# task registry and the celery-beat schedule entries silently no-op.
from apps.trading.tasks.outbox import process_outbox  # noqa: F401
from apps.trading.tasks.paper_fills import process_paper_fills  # noqa: F401
from apps.trading.tasks.reconcile import reconcile_stale_intraday  # noqa: F401
from apps.trading.tasks.snapshots import refresh_all  # noqa: F401
