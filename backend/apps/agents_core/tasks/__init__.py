# Celery's `autodiscover_tasks()` imports the `tasks` package — re-export the
# submodule tasks here so the @shared_task decorators register on worker boot.
# Without this, `expire_runs` never appears in the worker's task registry and
# the celery-beat schedule entry silently no-ops.
from apps.agents_core.tasks.housekeeping import expire_runs  # noqa: F401
from apps.agents_core.tasks.run import execute_run            # noqa: F401
