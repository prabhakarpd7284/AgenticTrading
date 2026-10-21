"""Checks library for run receipts (UX redesign plan §8.2).

A check is a plain dict:

    {"name": "candles_fresh", "status": "pass" | "warn" | "fail" | "info",
     "message": "human sentence", ...extra data}

A receipt is ``success`` only if no check failed. Every function here is
defensive: a check that cannot evaluate returns an ``info``/``warn`` check
describing why — it never raises into the job being tracked.
"""
from __future__ import annotations

import statistics
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MARKET_CLOSE = time(15, 30)

PASS, WARN, FAIL, INFO = "pass", "warn", "fail", "info"

# Anomaly detection needs some history before it is meaningful.
MIN_HISTORY = 3
HISTORY_WINDOW = 20
DURATION_FACTOR = 3.0
DURATION_FLOOR_SECS = 60.0


def make(name: str, status: str, message: str, **data: Any) -> dict:
    out = {"name": name, "status": status, "message": message}
    out.update({k: v for k, v in data.items() if v is not None})
    return out


def worst(checks: Iterable[dict]) -> str:
    order = {FAIL: 3, WARN: 2, INFO: 1, PASS: 0}
    w = PASS
    for c in checks:
        if order.get(c.get("status"), 0) > order[w]:
            w = c["status"]
    return w


# ─── Counts ──────────────────────────────────────────────────────────────

def output_count(name: str, actual: int | None, *, expected: int | None = None,
                 minimum: int | None = None, label: str = "rows",
                 severity: str = FAIL) -> dict:
    """Output size vs the expected universe (exact) or a floor (minimum)."""
    if actual is None:
        return make(name, WARN, f"{label}: count unavailable")
    if expected is not None:
        if actual >= expected:
            return make(name, PASS, f"{actual} of {expected} {label}", actual=actual, expected=expected)
        if actual == 0:
            return make(name, severity, f"0 of {expected} {label}", actual=actual, expected=expected)
        return make(name, WARN, f"only {actual} of {expected} {label}", actual=actual, expected=expected)
    if minimum is not None and actual < minimum:
        return make(name, severity, f"{actual} {label} (expected at least {minimum})",
                    actual=actual, expected=minimum)
    return make(name, PASS, f"{actual} {label}", actual=actual)


def rows_written(label: str, n: int | None, *, minimum: int = 1, severity: str = FAIL) -> dict:
    """Downstream rows actually written (not just "the function returned")."""
    return output_count("rows_written", n, minimum=minimum, label=f"{label} written",
                        severity=severity)


def swallowed_errors(count: int, examples: list[str] | None = None) -> dict:
    """Errors a job caught and carried on past — reported, never hidden."""
    if not count:
        return make("swallowed_errors", PASS, "no errors swallowed")
    return make("swallowed_errors", WARN, f"{count} error(s) caught and skipped",
                count=count, examples=(examples or [])[:5])


# ─── History-based anomalies ─────────────────────────────────────────────

def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.median(vals) if len(vals) >= MIN_HISTORY else None


def zero_output_anomaly(metric: str, value: Any, history: list[Any]) -> dict:
    """A run producing 0 where the last ~20 runs' median is > 0."""
    med = _median(history)
    if med is None:
        return make("zero_output_anomaly", INFO,
                    f"not enough history to judge '{metric}' ({len(history)} prior runs)",
                    metric=metric, value=value)
    if (value or 0) == 0 and med > 0:
        return make("zero_output_anomaly", WARN,
                    f"'{metric}' is 0; typical is {med:g} (median of last {len(history)} runs)",
                    metric=metric, value=value, median=med)
    return make("zero_output_anomaly", PASS,
                f"'{metric}' = {value} (typical {med:g})", metric=metric, value=value, median=med)


def duration_anomaly(seconds: float | None, history: list[float]) -> dict:
    med = _median(history)
    if seconds is None:
        return make("duration_anomaly", INFO, "duration unknown")
    if med is None:
        return make("duration_anomaly", INFO,
                    f"took {seconds:.0f}s; not enough history to judge ({len(history)} prior runs)",
                    seconds=round(seconds, 1))
    if seconds > max(med * DURATION_FACTOR, DURATION_FLOOR_SECS):
        return make("duration_anomaly", WARN,
                    f"took {seconds:.0f}s vs typical {med:.0f}s",
                    seconds=round(seconds, 1), median=round(med, 1))
    if med > DURATION_FLOOR_SECS and seconds < med / 10:
        return make("duration_anomaly", WARN,
                    f"finished in {seconds:.0f}s vs typical {med:.0f}s — suspiciously fast",
                    seconds=round(seconds, 1), median=round(med, 1))
    return make("duration_anomaly", PASS, f"took {seconds:.0f}s (typical {med:.0f}s)",
                seconds=round(seconds, 1), median=round(med, 1))


# ─── Data freshness ─────────────────────────────────────────────────────

def freshness(name: str, latest: datetime | date | None, *, max_age: timedelta,
              now: datetime | None = None, label: str = "data", severity: str = FAIL) -> dict:
    """Generic age check: ``latest`` must be newer than ``now - max_age``."""
    from django.utils import timezone

    now = now or timezone.now()
    if latest is None:
        return make(name, severity, f"no {label} found")
    if isinstance(latest, date) and not isinstance(latest, datetime):
        latest = datetime.combine(latest, time(0), tzinfo=IST)
    age = now - latest
    hours = round(age.total_seconds() / 3600, 1)
    if age <= max_age:
        return make(name, PASS, f"{label} is {hours}h old", age_hours=hours,
                    latest=latest.isoformat())
    return make(name, severity, f"{label} is {hours}h old (limit {max_age.total_seconds() / 3600:g}h)",
                age_hours=hours, latest=latest.isoformat())


def previous_trading_day(d: date) -> date:
    from apps.market_data.services.market_calendar import is_trading_day

    p = d - timedelta(days=1)
    while not is_trading_day(p):
        p -= timedelta(days=1)
    return p


def expected_session(now: datetime | None = None) -> date:
    """Latest NSE session whose close has passed (the data we should have)."""
    from django.utils import timezone

    from apps.market_data.services.market_calendar import is_trading_day

    now = (now or timezone.now()).astimezone(IST)
    today = now.date()
    if is_trading_day(today) and now.time() >= MARKET_CLOSE:
        return today
    return previous_trading_day(today)


def trading_days_between(older: date, newer: date) -> int:
    """Number of trading sessions after ``older`` up to and including ``newer``."""
    from apps.market_data.services.market_calendar import is_trading_day

    n, d = 0, older
    while d < newer and n < 60:
        d += timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n


def latest_candle(interval: str = "1d") -> datetime | None:
    from django.db.models import Max

    from apps.market_data.models import Candle

    return Candle.objects.filter(interval=interval).aggregate(m=Max("t"))["m"]


def candle_freshness(interval: str = "1d", *, now: datetime | None = None,
                     severity: str = FAIL) -> dict:
    """Latest stored candle vs the latest completed NSE session.

    0 sessions behind → pass, 1 → warn, more → ``severity``.
    """
    try:
        latest = latest_candle(interval)
        if latest is None:
            return make("candles_fresh", severity, f"no {interval} candles stored", interval=interval)
        have = latest.astimezone(IST).date()
        want = expected_session(now)
        behind = trading_days_between(have, want)
        data = {"interval": interval, "latest": latest.isoformat(), "expected_session": want.isoformat(),
                "sessions_behind": behind}
        if behind == 0:
            return make("candles_fresh", PASS, f"{interval} candles up to {have}", **data)
        status = WARN if behind == 1 else severity
        return make("candles_fresh", status,
                    f"{interval} candles end {have}; expected {want} ({behind} session(s) behind)", **data)
    except Exception as e:  # noqa: BLE001
        return make("candles_fresh", WARN, f"could not evaluate candle freshness: {e}")


# ─── Broker / synthetic data ────────────────────────────────────────────

SYNTHETIC_BROKERS = {"paper"}


def live_broker_links(tenant=None):
    """Active, non-synthetic broker links (for one tenant, or any tenant)."""
    from apps.market_data.models import BrokerLink

    qs = BrokerLink.objects.filter(status=BrokerLink.Status.ACTIVE).exclude(
        broker_name__in=SYNTHETIC_BROKERS)
    if tenant is not None:
        qs = qs.filter(tenant=tenant)
    return qs


def broker_live(tenant=None, *, required: bool = True) -> dict:
    """Broker linked and NOT synthetic.

    Without an active real link, ``data_port`` falls back to
    ``PaperBrokerAdapter`` and strategies run on synthesised prices that
    look completely normal (see the dev-stack skill, "Check 2").
    """
    try:
        links = list(live_broker_links(tenant).values_list("broker_name", flat=True))
        if links:
            return make("broker_live", PASS, f"active broker link: {', '.join(sorted(set(links)))}",
                        brokers=sorted(set(links)))
        return make("broker_live", FAIL if required else WARN,
                    "no active real broker link — market data may be synthetic")
    except Exception as e:  # noqa: BLE001
        return make("broker_live", WARN, f"could not evaluate broker links: {e}")
