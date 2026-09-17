"""Celery autodiscover only recurses one level into apps.<app>/tasks/, so
re-export every shared_task from this package's submodules here. Without
this, the beat scheduler silently never finds them."""
from apps.strategies.tasks.daily_pipeline import (  # noqa: F401
    run_eod_enrichment,
    run_premarket_basket,
    run_screener_session,
    run_swing_scan,
)
