"""Stale-NAV detection: schemes whose latest NAV is more than N trading days old.

Data-quality only — it reports which schemes the daily AMFI pull has stopped
updating (merged, wound up, matured FMPs). Nothing here gives advice.
"""
from __future__ import annotations

from datetime import date, timedelta

from django.db.models import Max

from apps.market_data.services.market_calendar import is_trading_day
from apps.mutual_funds.models import MFScheme

DEFAULT_MAX_TRADING_DAYS = 3


def trading_days_back(today: date, n: int) -> date:
    """The date ``n`` trading days before ``today`` (``today`` itself excluded)."""
    d = today
    while n > 0:
        d -= timedelta(days=1)
        if is_trading_day(d):
            n -= 1
    return d


def stale_schemes(today: date | None = None, max_trading_days: int = DEFAULT_MAX_TRADING_DAYS):
    """Returns ``[(scheme, latest_nav_date | None), ...]`` for schemes whose
    newest NAV is older than ``max_trading_days`` trading days (or that have
    no NAV at all), oldest first."""
    today = today or date.today()
    cutoff = trading_days_back(today, max_trading_days)
    rows = (
        MFScheme.objects.annotate(latest=Max("navs__nav_date"))
        .filter(latest__lt=cutoff)
        .order_by("latest")
    )
    no_nav = MFScheme.objects.annotate(latest=Max("navs__nav_date")).filter(latest__isnull=True)
    return [(s, s.latest) for s in rows] + [(s, None) for s in no_nav]
