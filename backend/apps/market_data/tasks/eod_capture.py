"""After-close market-data capture — Celery tasks (beat, Mon–Fri, holiday-aware).

  15:50 IST  capture_session_candles   Angel 1m candles, NIFTY 50 + indices, the session only
  18:30 IST  capture_bhavcopy(attempt=1)  NSE daily bhavcopy (partial if not yet published)
  20:15 IST  capture_bhavcopy(attempt=2)  retry; fails if NSE still has not published

Each run writes a RunReceipt (``tracked_run``) with checks; the schedule
registry turns a missed slot into a ``not_run`` receipt + alert.
See ``apps.market_data.services.session_capture`` for why this exists.
"""
from __future__ import annotations

from datetime import date

import structlog
from celery import shared_task

log = structlog.get_logger(__name__)

CANDLES_TASK = "apps.market_data.tasks.eod_capture.capture_session_candles"
BHAV_TASK = "apps.market_data.tasks.eod_capture.capture_bhavcopy"


def _resolve_session(session: str | None, force: bool, run) -> date | None:
    from apps.market_data.services.market_calendar import is_trading_day
    from apps.market_data.services.session_capture import session_to_capture

    if session:
        return date.fromisoformat(session)
    if not force and not is_trading_day():
        run.skip("not a trading day")
        return None
    return session_to_capture()


@shared_task(name=CANDLES_TASK, ignore_result=True, soft_time_limit=1800, time_limit=2100)
def capture_session_candles(force: bool = False, session: str | None = None,
                            trigger: str | None = None) -> dict:
    from apps.market_data.services import session_capture
    from apps.system.services.run_receipts import infer_trigger, tracked_run

    kw = {"force": force, "trigger": trigger}
    with tracked_run("import", "eod_candle_capture", {"force": force, "session": session},
                     trigger=infer_trigger(kw), task_name=CANDLES_TASK, primary_metric="bars_stored",
                     max_seconds=2400) as run:
        d = _resolve_session(session, force, run)
        if d is None:
            return {"skipped": run.skipped_reason}
        out = session_capture.capture_session_candles(d, run=run)
        log.info("eod_capture.candles.done", session=str(d), symbols=out["symbols"])
        return {"session": out["session"], "symbols": out["symbols"]}


@shared_task(name=BHAV_TASK, ignore_result=True, soft_time_limit=900, time_limit=1200)
def capture_bhavcopy(attempt: int = 1, force: bool = False, session: str | None = None,
                     trigger: str | None = None) -> dict:
    from apps.market_data.services import session_capture
    from apps.system.services.run_receipts import infer_trigger, tracked_run

    kw = {"force": force, "trigger": trigger}
    with tracked_run("import", "eod_bhavcopy_capture",
                     {"attempt": attempt, "force": force, "session": session},
                     trigger=infer_trigger(kw), task_name=BHAV_TASK, primary_metric="rows",
                     max_seconds=1800) as run:
        d = _resolve_session(session, force, run)
        if d is None:
            return {"skipped": run.skipped_reason}
        out = session_capture.capture_bhavcopy(d, attempt=attempt, run=run)
        log.info("eod_capture.bhavcopy.done", **out)
        return out
