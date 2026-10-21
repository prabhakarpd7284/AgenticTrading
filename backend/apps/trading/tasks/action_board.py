"""Action Board Celery tasks (UX redesign R3) — both write a RunReceipt.

  09:35 IST (settings.ACTION_BOARD_TIME)  build_action_board
  16:10 IST                               evaluate_action_outcomes (after the 15:50 candle capture)

Mon–Fri on beat; each self-skips NSE holidays. The management commands
``build_action_board`` / ``evaluate_action_outcomes`` call the same runners.
"""
from __future__ import annotations

from datetime import date

import structlog
from celery import shared_task

log = structlog.get_logger(__name__)

BUILD_TASK = "apps.trading.tasks.action_board.build_action_board"
OUTCOMES_TASK = "apps.trading.tasks.action_board.evaluate_action_outcomes"


def run_build(*, day: str | None = None, max_planner_calls: int | None = None,
              telegram: bool | None = None, trigger: str = "beat") -> dict:
    from apps.system.services.run_receipts import tracked_run
    from apps.trading.services.action_board import build_action_board

    inputs = {"date": day, "max_planner_calls": max_planner_calls, "telegram": telegram}
    with tracked_run("job", "action_board", inputs, trigger=trigger, task_name=BUILD_TASK,
                     primary_metric="planned", max_seconds=3600) as run:
        out = build_action_board(date.fromisoformat(day) if day else None,
                                 max_planner_calls=max_planner_calls, telegram=telegram, run=run)
        log.info("action_board.built", **{k: v for k, v in out.items() if k != "per_symbol"})
        out["receipt_id"] = str(run.receipt.id) if run.receipt is not None else None
        return out


def run_outcomes(*, day: str | None = None, force: bool = False, trigger: str = "beat") -> dict:
    from django.utils import timezone

    from apps.market_data.services.market_calendar import is_trading_day
    from apps.system.services.run_receipts import tracked_run
    from apps.trading.services.action_outcomes import evaluate_outcomes

    with tracked_run("job", "action_board_outcomes", {"date": day, "force": force}, trigger=trigger,
                     task_name=OUTCOMES_TASK, primary_metric="evaluated", max_seconds=1800) as run:
        d = date.fromisoformat(day) if day else timezone.localdate()
        if not day and not is_trading_day(d):
            run.skip("not a trading day")
            return {"skipped": "not a trading day"}
        out = evaluate_outcomes(d, run=run, force=force)
        out["receipt_id"] = str(run.receipt.id) if run.receipt is not None else None
        return out


@shared_task(name=BUILD_TASK, ignore_result=True, soft_time_limit=3300, time_limit=3600)
def build_action_board(day: str | None = None, max_planner_calls: int | None = None,
                       trigger: str | None = None) -> dict:
    return run_build(day=day, max_planner_calls=max_planner_calls,
                     trigger=trigger if trigger in ("beat", "manual", "api") else "beat")


@shared_task(name=OUTCOMES_TASK, ignore_result=True, soft_time_limit=1500, time_limit=1800)
def evaluate_action_outcomes(day: str | None = None, force: bool = False,
                             trigger: str | None = None) -> dict:
    return run_outcomes(day=day, force=force,
                        trigger=trigger if trigger in ("beat", "manual", "api") else "beat")
