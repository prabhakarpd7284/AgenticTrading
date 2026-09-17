"""Celery autodiscover only recurses one level into apps.<app>/tasks/, so
re-export every shared_task from this package's submodules here. Without
this, the beat scheduler silently never finds them."""
from apps.market_data.tasks.broker_refresh import (  # noqa: F401
    prune_old_snapshots,
    refresh_broker_positions,
    refresh_link_snapshot,
)
