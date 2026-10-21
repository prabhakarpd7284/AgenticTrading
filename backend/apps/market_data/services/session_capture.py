"""After-close market-data capture (UX redesign R2 follow-up).

Why this exists: 1-minute candles are written live only by the screener
session (09:15 beat, ``daily_pipeline.run_screener_session``) through its tick
aggregator, and that task only runs while the machine and the Celery worker
are up during market hours — a restart after 15:30 skips it ("market already
closed"). Before this job, the only other writer was the 21:00 nightly scan's
refresh step, so on a day the laptop was off during the session the status
strip read "stale" from 15:30 until 21:00.

Two captures, both resumable/idempotent (duplicates are ignored on insert):

* ``capture_session_candles`` — Angel 1m candles for NIFTY 50 + the indices,
  one ``getCandleData`` call per symbol for the session only (~58 calls,
  shared Redis rate limiter, waits on the breaker).
* ``capture_bhavcopy`` — NSE's daily CM bhavcopy (every NSE equity, 1 file,
  no Angel budget).

Checks: every expected symbol has ≥ ``MIN_BARS`` 1m bars for the session; the
bhavcopy row count is close to the previous session's.
"""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# A full NSE session is 375 one-minute bars (09:15–15:29); Angel sometimes
# returns a few less for indices. Below this a symbol's day is incomplete.
MIN_BARS = 360
# Bhavcopy size vs the previous session: within 5 % pass, within 15 % warn.
BHAV_PASS_PCT = 5.0
BHAV_WARN_PCT = 15.0


def session_bounds(d: date) -> tuple[datetime, datetime]:
    return (datetime.combine(d, time(9, 15), tzinfo=IST),
            datetime.combine(d, time(15, 30), tzinfo=IST))


def bars_for_session(d: date, symbol_ids: list[int]) -> dict[int, int]:
    from django.db.models import Count

    from apps.market_data.models import Candle

    lo, hi = session_bounds(d)
    rows = (Candle.objects.filter(interval="1m", symbol_id__in=symbol_ids, t__gte=lo, t__lt=hi)
            .values("symbol_id").annotate(n=Count("id")))
    return {r["symbol_id"]: r["n"] for r in rows}


def check_bars(counts: dict[str, int], *, minimum: int = MIN_BARS) -> dict:
    """Every expected symbol must have ≥ ``minimum`` bars for the session."""
    from apps.system.services import run_checks as rc

    if not counts:
        return rc.make("bars_per_symbol", rc.FAIL, "no symbols captured")
    short = {s: n for s, n in counts.items() if n < minimum}
    lowest = min(counts.values())
    if not short:
        return rc.make("bars_per_symbol", rc.PASS,
                       f"all {len(counts)} symbols have ≥ {minimum} bars (lowest {lowest})",
                       symbols=len(counts), lowest=lowest)
    worst = sorted(short.items(), key=lambda kv: kv[1])[:10]
    return rc.make("bars_per_symbol", rc.FAIL,
                   f"{len(short)} of {len(counts)} symbols below {minimum} bars: "
                   + ", ".join(f"{s}={n}" for s, n in worst),
                   short=dict(worst), symbols=len(counts))


def capture_session_candles(session: date, *, run=None) -> dict:
    """Fetch and store the session's 1m candles for NIFTY 50 + indices."""
    from trading.services.data_service import BrokerClient

    from apps.market_data.management.commands.backfill_candles import (
        ensure_symbol_row,
        parse_rows,
        resolve_targets,
        write_rows,
    )
    from apps.system.services.run_receipts import current_run

    run = run or current_run()
    targets, missing = resolve_targets("nifty50")
    for name in missing:
        run.warn(f"{name}: no NSE token in the scrip master — not captured")
    broker = BrokerClient.get_instance()
    lo, hi = session_bounds(session)
    syms: dict[str, int] = {}
    offered = calls = 0
    empty: list[str] = []
    for name, exchange, token in targets:
        try:
            sym = ensure_symbol_row(name, exchange, token)
            syms[name] = sym.id
            rows = broker.fetch_candles(token, f"{session} 09:15", f"{session} 15:30", "ONE_MINUTE",
                                        exchange=exchange, wait_on_breaker=True)
            calls += 1
            if not rows:
                empty.append(name)
                continue
            parsed = [r for r in parse_rows(rows) if lo <= r[0] < hi]
            offered += write_rows(sym, "1m", parsed)
            run.unit_ok()
        except Exception as e:  # noqa: BLE001 — one symbol never sinks the capture
            run.unit_failed(error=f"{name}: {type(e).__name__}: {e}")
            log.warning("session_capture.symbol_failed", symbol=name, error=str(e))

    stored = bars_for_session(session, list(syms.values()))
    counts = {name: stored.get(sid, 0) for name, sid in syms.items()}
    run.output(session=session.isoformat(), symbols=len(targets), angel_calls=calls,
               rows_offered=offered, bars_stored=sum(counts.values()),
               empty_responses=len(empty), lowest_bars=min(counts.values()) if counts else 0)
    if empty:
        run.warn("Angel returned no rows for: " + ", ".join(empty[:20]))
    run.check(check_bars(counts))
    run.set_provenance(source="Angel One getCandleData (ONE_MINUTE)", session=session.isoformat(),
                       universe="NIFTY 50 + indices")
    run.link("Runs", "/runs")
    return {"session": session.isoformat(), "symbols": len(targets), "bars": counts, "empty": empty}


def check_bhav_count(rows: int, prev_rows: int | None, prev_day: date | None) -> dict:
    from apps.system.services import run_checks as rc

    if rows == 0:
        return rc.make("bhavcopy_rows", rc.FAIL, "no bhavcopy rows stored for the session")
    if not prev_rows:
        return rc.make("bhavcopy_rows", rc.INFO, f"{rows} rows; no previous session to compare",
                       rows=rows)
    diff = abs(rows - prev_rows) / prev_rows * 100
    status = rc.PASS if diff <= BHAV_PASS_PCT else (rc.WARN if diff <= BHAV_WARN_PCT else rc.FAIL)
    return rc.make("bhavcopy_rows", status,
                   f"{rows} rows vs {prev_rows} on {prev_day} ({diff:.1f}% apart)",
                   rows=rows, previous=prev_rows, diff_pct=round(diff, 1))


def capture_bhavcopy(session: date, *, attempt: int = 1, run=None) -> dict:
    """Download + store the session's bhavcopy. ``attempt`` 1 tolerates "not
    published yet" (receipt ``partial``); the retry run fails on it."""
    from apps.market_data.management.commands.backfill_bhavcopy import (
        day_row_count,
        download,
        has_day,
        parse,
        write_day,
    )
    from apps.system.services import run_checks as rc
    from apps.system.services.run_receipts import current_run

    run = run or current_run()
    run.set_provenance(source="NSE CM bhavcopy archive (UDiFF)", session=session.isoformat())
    if has_day(session):
        mode, written = "already present", 0
    else:
        blob = download(session)
        if blob is None:
            msg = f"NSE has not published the {session} bhavcopy yet"
            run.output(session=session.isoformat(), mode="not published", attempt=attempt)
            if attempt <= 1:
                run.mark_partial(msg + " — the 20:15 retry will try again")
                run.check(rc.make("bhavcopy_published", rc.WARN, msg))
            else:
                run.check(rc.make("bhavcopy_published", rc.FAIL, msg + " (retry also found nothing)"))
            return {"session": session.isoformat(), "published": False}
        written = write_day(session, parse(blob))
        mode = "downloaded"

    rows = day_row_count(session)
    prev_day = rc.previous_trading_day(session)
    prev_rows = day_row_count(prev_day)
    run.output(session=session.isoformat(), mode=mode, attempt=attempt, rows=rows, rows_written=written,
               previous_session=prev_day.isoformat(), previous_rows=prev_rows)
    run.check(check_bhav_count(rows, prev_rows, prev_day))
    return {"session": session.isoformat(), "published": True, "rows": rows, "mode": mode}


def session_to_capture(now: datetime | None = None) -> date:
    """The latest NSE session whose close has passed (today after 15:30)."""
    from apps.system.services.run_checks import expected_session

    return expected_session(now)

