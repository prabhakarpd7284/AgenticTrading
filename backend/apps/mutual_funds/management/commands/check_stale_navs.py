"""Report schemes whose latest NAV is more than N trading days old.

Usage:
    .venv/bin/python manage.py check_stale_navs
    .venv/bin/python manage.py check_stale_navs --days 5 --limit 50
"""
from __future__ import annotations

from collections import Counter

import structlog
from django.core.management.base import BaseCommand

from apps.mutual_funds.services.staleness import DEFAULT_MAX_TRADING_DAYS, stale_schemes

log = structlog.get_logger(__name__)


class Command(BaseCommand):
    help = "List schemes with no NAV update in more than N trading days."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=DEFAULT_MAX_TRADING_DAYS)
        parser.add_argument("--limit", type=int, default=20, help="Rows to print (0 = counts only)")

    def handle(self, *args, **options):
        stale = stale_schemes(max_trading_days=options["days"])
        self.stats = {"stale_schemes": len(stale)}
        self.stdout.write(f"{len(stale)} scheme(s) with no NAV in >{options['days']} trading days.")
        by_amc = Counter(s.amc_name for s, _ in stale)
        for amc, n in by_amc.most_common(5):
            self.stdout.write(f"  {n:5d}  {amc}")
        for scheme, latest in stale[: options["limit"]]:
            self.stdout.write(f"  {scheme.scheme_code:>8}  last NAV {latest or 'never'}  {scheme.scheme_name[:70]}")
        log.info("stale_navs.checked", count=len(stale), max_trading_days=options["days"])
