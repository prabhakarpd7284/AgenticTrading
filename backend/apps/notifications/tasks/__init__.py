# Celery's autodiscover_tasks() imports the `tasks` package — re-export the
# task functions here so the @shared_task decorators register on worker boot.
# Same pattern as apps.trading.tasks / apps.agents_core.tasks.
from apps.notifications.tasks.watchlists import refresh_auto_watchlists  # noqa: F401
