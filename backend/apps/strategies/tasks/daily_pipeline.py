"""Daily trading pipeline — Celery tasks that make AlphaDesk produce data
on its own every trading day.

Three beat-scheduled tasks (see ``CELERY_BEAT_SCHEDULE`` in settings):

  * ``run_swing_scan``       — premarket Oliver Kell daily/weekly scan.
  * ``run_screener_session`` — intraday live screener, runs 09:15 → 15:30.
  * ``run_eod_enrichment``   — post-close signal outcome enrichment.

Each task is a no-op on weekends / NSE holidays (``is_trading_day``), so
beat can fire every weekday without a holiday calendar of its own. Pass
``force=True`` (the /pipeline debugger's "Run now" button) to bypass that
guard for ad-hoc testing.

Every actual execution writes an ``apps.system.PipelineRun`` row so the
/pipeline page can show what ran, when, and what it produced.

These deliberately do NOT auto-execute trades — the premarket *basket*
executor and the agentic planner stay manual. The pipeline produces
*data* (signals + outcomes) that feed the watchlists, the Now feed, and
the monthly feedback report.

Heavy imports are deferred into the task bodies so worker startup and
beat registration don't pay the cost (mirrors the management commands).
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)

# Screener session bounds. Market is 09:15–15:30 IST = 6h15m; 7h hard
# limit leaves room for the candle bootstrap. Soft limit fires 10 min
# earlier so we can stop the tick stream cleanly before the hard kill.
_SCREENER_HARD_LIMIT = 7 * 3600
_SCREENER_SOFT_LIMIT = _SCREENER_HARD_LIMIT - 600

# Redis lock guarding the screener session — refreshed every loop so a
# dead worker's lock self-expires and a redelivered task can take over.
_SCREENER_LOCK_KEY = "alphadesk:pipeline:screener:lock"
_SCREENER_LOCK_TTL = 1800  # 30 min — must exceed the loop sleep below


def _redis():
    """Process-shared Redis client (same instance as Celery / Channels)."""
    import redis
    from django.conf import settings

    return redis.Redis.from_url(settings.REDIS_URL, socket_timeout=2)


@contextmanager
def _record_run(task_key: str, force: bool):
    """Open a PipelineRun row, mark it RUNNING, and finalise on exit.

    Yields the row so the task can set ``run.summary``. On an unhandled
    exception the row is marked FAILED with the error text; on clean exit
    it becomes SUCCESS. ``force`` distinguishes a manual /pipeline-UI run
    from a beat-scheduled one.
    """
    from django.utils import timezone

    from apps.system.models import PipelineRun

    run = PipelineRun.objects.create(
        task=task_key,
        trigger=PipelineRun.Trigger.MANUAL if force else PipelineRun.Trigger.BEAT,
        status=PipelineRun.Status.RUNNING,
    )
    try:
        yield run
    except Exception as e:
        run.status = PipelineRun.Status.FAILED
        run.error = f"{type(e).__name__}: {e}"[:2000]
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error", "finished_at", "summary"])
        raise
    else:
        run.status = PipelineRun.Status.SUCCESS
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "finished_at", "summary"])


# ─── Premarket: Oliver Kell swing scan ────────────────────────────────

@shared_task(name="apps.strategies.tasks.daily_pipeline.run_swing_scan")
def run_swing_scan(force: bool = False) -> dict:
    """Run the Oliver Kell daily/weekly scanner across NIFTY 100.

    One-shot, ~minutes. Writes swing signals; Telegram alerts go out if
    the bot is configured. Beat fires this premarket (~08:20 IST).
    """
    from apps.market_data.services.market_calendar import is_trading_day

    if not force and not is_trading_day():
        logger.info("pipeline.swing_scan.skipped reason=not_a_trading_day")
        return {"skipped": "not a trading day"}

    from django.core.management import call_command

    logger.info("pipeline.swing_scan.start force=%s", force)
    with _record_run("swing_scan", force) as run:
        call_command("run_ok_scanner", actionable_only=True, telegram=True)
        run.summary = {"ok": True, "scanner": "oliver_kell"}
    logger.info("pipeline.swing_scan.done")
    return {"ok": True}


# ─── Intraday: live screener session ──────────────────────────────────

@shared_task(
    name="apps.strategies.tasks.daily_pipeline.run_screener_session",
    time_limit=_SCREENER_HARD_LIMIT,
    soft_time_limit=_SCREENER_SOFT_LIMIT,
)
def run_screener_session(force: bool = False) -> dict:
    """Run the live intraday screener until market close.

    Beat fires this at 09:15 IST. The task builds the screener engine,
    bootstraps candle history, starts the tick stream, then idles until
    15:30 IST — every fired signal is persisted to ``apps.strategies.Signal``
    + ``apps.events.Event`` via the screener's own persist handler.

    A Redis lock (refreshed each loop) ensures only one session runs at a
    time; if the worker dies the lock self-expires so a redelivered task
    can resume.
    """
    from apps.market_data.services.market_calendar import is_trading_day
    from trading.utils.time_utils import MARKET_CLOSE

    if not force and not is_trading_day():
        logger.info("pipeline.screener.skipped reason=not_a_trading_day")
        return {"skipped": "not a trading day"}

    if datetime.now().time() >= MARKET_CLOSE:
        logger.info("pipeline.screener.skipped reason=market_already_closed")
        return {"skipped": "market already closed"}

    r = _redis()
    try:
        got_lock = r.set(_SCREENER_LOCK_KEY, "1", nx=True, ex=_SCREENER_LOCK_TTL)
    except Exception as e:
        logger.warning("pipeline.screener.lock_unavailable err=%s", e)
        got_lock = True  # Redis down — proceed rather than skip the whole day.
    if not got_lock:
        logger.info("pipeline.screener.skipped reason=session_already_running")
        return {"skipped": "screener session already running"}

    try:
        with _record_run("screener_session", force) as run:
            result = _run_screener_until_close(r)
            run.summary = result
        return result
    finally:
        try:
            r.delete(_SCREENER_LOCK_KEY)
        except Exception:
            pass


def _run_screener_until_close(r) -> dict:
    """Build the engine, stream ticks until close. Split out so the lock
    handling above stays readable."""
    from dotenv import load_dotenv

    load_dotenv()  # TELEGRAM_BOT_TOKEN / SMARTAPI_* for the screener.

    from apps.market_data.constants import SCREENER_UNIVERSE
    from plugins.strategy_screener.engine import ScreenerEngine
    from plugins.strategy_screener.strategies import STRATEGIES
    from plugins.strategy_screener.telegram import TelegramAlertService
    from plugins.strategy_screener.tick_stream import TickStream
    from trading.utils.time_utils import MARKET_CLOSE

    symbols = list(SCREENER_UNIVERSE)
    strategies = list(STRATEGIES)
    logger.info(
        "pipeline.screener.start symbols=%d strategies=%d",
        len(symbols), len(strategies),
    )

    engine = ScreenerEngine(symbols, strategies)
    # Persist every signal — this is the whole point: data forming.
    engine.add_output_handler(lambda sig: sig.persist("SCREENER"))

    telegram = TelegramAlertService()
    if telegram.is_configured:
        telegram.set_engine(engine)
        engine.add_output_handler(telegram.send_signal)
        logger.info("pipeline.screener.telegram_enabled")

    engine.bootstrap(fetch_candles_fn=True)
    logger.info("pipeline.screener.bootstrapped")

    tick_stream = TickStream(symbols=symbols, on_tick=engine.on_tick, poll_interval=5.0)
    tick_stream.start()
    logger.info("pipeline.screener.streaming mode=%s", tick_stream.mode)

    try:
        # Idle until market close, refreshing the lock so it never goes
        # stale while we're alive.
        while datetime.now().time() < MARKET_CLOSE:
            time.sleep(30)
            try:
                r.expire(_SCREENER_LOCK_KEY, _SCREENER_LOCK_TTL)
            except Exception:
                pass
    except SoftTimeLimitExceeded:
        logger.warning("pipeline.screener.soft_time_limit — stopping early")
    finally:
        tick_stream.stop()

    stats = engine.get_stats()
    logger.info(
        "pipeline.screener.done bars=%s signals=%s",
        stats.get("bars_processed"), stats.get("signals_emitted"),
    )
    return {
        "ok": True,
        "bars_processed": stats.get("bars_processed"),
        "signals_emitted": stats.get("signals_emitted"),
    }


# ─── Premarket: morning basket scan ───────────────────────────────────

@shared_task(name="apps.strategies.tasks.daily_pipeline.run_premarket_basket")
def run_premarket_basket(force: bool = False) -> dict:
    """Run the morning basket in scan mode and persist its signals.

    Assesses market mood, generates the equity + index-option basket
    legs, and writes each as an ``apps.strategies.Signal`` row with
    ``source=PREMARKET`` — so the basket finally feeds the signal ledger
    and the monthly capture report.

    Scan-only: this NEVER executes trades. The basket executor stays a
    manual decision. Beat fires this premarket (~08:45 IST).
    """
    from apps.market_data.services.market_calendar import is_trading_day

    if not force and not is_trading_day():
        logger.info("pipeline.basket.skipped reason=not_a_trading_day")
        return {"skipped": "not a trading day"}

    logger.info("pipeline.basket.start force=%s", force)
    with _record_run("premarket_basket", force) as run:
        result = _run_basket_scan()
        run.summary = result
    logger.info("pipeline.basket.done %s", result)
    return result


def _run_basket_scan() -> dict:
    """Mood → signal generation → persist. No execution."""
    from dotenv import load_dotenv

    load_dotenv()

    from plugins.strategy_basket.config import BasketConfig
    from plugins.strategy_basket.mood import MarketMood, MarketMoodAssessor
    from plugins.strategy_basket.signals import BasketSignalGenerator

    cfg = BasketConfig()
    mood = MarketMoodAssessor(cfg).assess()
    if mood.mood == MarketMood.NEUTRAL:
        logger.info("pipeline.basket.neutral — no basket today")
        return {"ok": True, "mood": "NEUTRAL", "signals": 0}

    signals = BasketSignalGenerator(cfg).generate(mood) or []
    persisted = _persist_basket_signals(signals)
    return {"ok": True, "mood": mood.mood.value, "signals": persisted}


def _persist_basket_signals(signals) -> int:
    """Write BasketSignal objects to apps.strategies.Signal (source=PREMARKET).

    The basket dataclass has no explicit target — we derive a 2R target
    from entry ± 2×risk_points so the row satisfies the Signal schema and
    the monthly capture matrix can score it.
    """
    from django.utils import timezone

    from apps.strategies.models import Signal
    from apps.tenants.models import Tenant

    tenant = Tenant.objects.order_by("id").first()
    if tenant is None:
        logger.warning("pipeline.basket.no_tenant — cannot persist signals")
        return 0

    now = timezone.now()
    today = timezone.localdate()
    count = 0
    for bs in signals:
        risk = getattr(bs, "risk_points", 0.0) or 0.0
        entry = getattr(bs, "entry_price", 0.0) or 0.0
        if bs.side == "BUY":
            target = entry + 2 * risk
        else:
            target = entry - 2 * risk
        Signal.objects.create(
            tenant=tenant,
            symbol=getattr(bs, "option_symbol", "") or bs.symbol,
            signal_date=today,
            signal_time=now,
            source=Signal.Source.PREMARKET,
            strategy=f"Basket {bs.leg_type}".title()[:40],
            side=bs.side,
            entry_price=entry,
            stoploss=getattr(bs, "stoploss", 0.0) or 0.0,
            target=round(target, 2),
            confidence=getattr(bs, "confluence", 0.0) or 0.0,
            risk_reward=2.0 if risk else 0.0,
            reasons=[f"phase {bs.phase}"] if getattr(bs, "phase", "") else [],
            indicators={
                "ema5": getattr(bs, "ema5", 0.0),
                "bb_mid": getattr(bs, "bb_mid", 0.0),
                "bb_lower": getattr(bs, "bb_lower", 0.0),
                "bb_upper": getattr(bs, "bb_upper", 0.0),
            },
        )
        count += 1
    return count


# ─── EOD: derive intraday trades (replay the just-closed session) ──────

@shared_task(name="apps.strategies.tasks.daily_pipeline.derive_intraday_trades")
def derive_intraday_trades(force: bool = False, date: str | None = None) -> dict:
    """Derive paper trades by replaying a completed intraday session.

    Unlike the live agent (which only trades during market hours and stamps
    today's date), this replays a *finished* day's candles through the same
    monitor, so the Monthly report's trade-driven sections stay current. It
    runs EOD by default (~16:30 IST) over the day that just closed.

    Gated two ways so the beat pipeline stays scan-only until opted in:
      * ``is_trading_day()`` — skip weekends / holidays.
      * ``is_auto_execute_enabled()`` — operator opt-in (UI toggle).
    Kill-switch / AI-pause halt it like every other automated action.
    ``force=True`` (the /pipeline "Run now" button) bypasses all three so the
    debugger can back-fill any day on demand.
    """
    from apps.market_data.services.market_calendar import is_trading_day
    from apps.system.services.flags import (
        is_auto_execute_enabled, is_ai_paused, is_kill_switch_on,
    )

    if not force:
        if not is_trading_day():
            logger.info("pipeline.derive_trades.skipped reason=not_a_trading_day")
            return {"skipped": "not a trading day"}
        if not is_auto_execute_enabled():
            logger.info("pipeline.derive_trades.skipped reason=auto_execute_disabled")
            return {"skipped": "auto-execute disabled — enable it on the Pipeline page"}
        if is_kill_switch_on() or is_ai_paused():
            logger.info("pipeline.derive_trades.skipped reason=halted")
            return {"skipped": "kill switch / AI pause active"}

    from datetime import date as _date

    from django.core.management import call_command

    target = date or _date.today().isoformat()
    logger.info("pipeline.derive_trades.start date=%s force=%s", target, force)
    with _record_run("intraday_agent", force) as run:
        call_command("derive_trades", date=target)
        from apps.trading.models import Trade
        closed = Trade.objects.filter(
            trade_date=target, status=Trade.Status.CLOSED,
        ).count()
        run.summary = {"ok": True, "date": target, "trades_closed": closed}
    logger.info("pipeline.derive_trades.done date=%s closed=%s", target, closed)
    return {"ok": True, "date": target, "trades_closed": closed}


# ─── EOD: signal outcome enrichment ───────────────────────────────────

@shared_task(name="apps.strategies.tasks.daily_pipeline.run_eod_enrichment")
def run_eod_enrichment(
    force: bool = False,
    date: str | None = None,
    backfill: bool = False,
) -> dict:
    """Enrich signals with EOD price + outcome.

    Reads ``apps.strategies.Signal`` rows, fetches the day's 5-min candles,
    computes eod_price / max_favorable / max_adverse / outcome, and links
    them to trades. Feeds the monthly feedback report.

    Modes:
      * ``backfill=True``  — enrich *every* un-enriched signal, all dates.
      * ``date="YYYY-MM-DD"`` — enrich that one day's signals.
      * neither — enrich today's signals (the beat default, ~16:00 IST).
    """
    from apps.market_data.services.market_calendar import is_trading_day

    # Manual date / backfill runs from the debugger always proceed; only
    # the plain beat-default run honours the trading-day guard.
    if not force and not backfill and not date and not is_trading_day():
        logger.info("pipeline.eod_enrichment.skipped reason=not_a_trading_day")
        return {"skipped": "not a trading day"}

    from django.core.management import call_command

    mode = "backfill-all" if backfill else (date or "today")
    logger.info("pipeline.eod_enrichment.start mode=%s force=%s", mode, force)
    with _record_run("eod_enrichment", force) as run:
        if backfill:
            call_command("enrich_signals", all=True)
        elif date:
            call_command("enrich_signals", date=date)
        else:
            call_command("enrich_signals")
        run.summary = {"ok": True, "mode": mode}
    logger.info("pipeline.eod_enrichment.done mode=%s", mode)
    return {"ok": True, "mode": mode}
