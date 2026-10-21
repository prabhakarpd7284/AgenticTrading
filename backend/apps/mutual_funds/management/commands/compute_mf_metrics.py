"""Compute MFMetricSnapshot rows for the live screening universe (S3).

Usage:
    .venv/bin/python manage.py compute_mf_metrics                       # as of the catalog date
    .venv/bin/python manage.py compute_mf_metrics --as-of 2026-10-16
    .venv/bin/python manage.py compute_mf_metrics --category "Equity Scheme - Flexi Cap Fund" --dry-run

Processes one category at a time (memory-bounded). Dead schemes are loaded
only to build peer medians. Scheduled weekly via celery beat
(``mf-metrics-weekly``). Run under ``nice -n 10`` by hand on a small machine.
"""
from __future__ import annotations

import time
from datetime import date

import structlog
from django.core.management.base import BaseCommand, CommandError

from apps.mutual_funds.services import screener_config as cfg
from apps.mutual_funds.services.metrics_runner import compute_snapshots
from apps.mutual_funds.services.nav_backfill import catalog_date, refresh_active_flags

log = structlog.get_logger(__name__)


class Command(BaseCommand):
    help = "Compute NAV-based metric snapshots for every universe scheme."

    def add_arguments(self, parser):
        parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: catalog NAV date)")
        parser.add_argument("--category", action="append", default=None, help="Canonical category (repeatable)")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **o):
        t0 = time.monotonic()
        as_of = date.fromisoformat(o["as_of"]) if o["as_of"] else catalog_date()
        if as_of is None:
            raise CommandError("No NAV data — run pull_amfi_nav / backfill_mf_nav_history first.")
        if not o["dry_run"] and not o["as_of"]:
            refresh_active_flags(as_of)
        rf = cfg.RISK_FREE_RATE
        self.stdout.write(f"Metrics as of {as_of}; risk-free {rf['value']:.2%} ({rf['instrument']}, as of {rf['as_of']})")

        def progress(cat, n):
            self.stdout.write(f"  {cat[:70]:70} {n:>4} schemes")
            self.stdout.flush()

        summary = compute_snapshots(as_of, o["category"], dry_run=o["dry_run"], progress=progress)
        self.stats = {"as_of": as_of.isoformat(), **summary}
        self.stdout.write(self.style.SUCCESS(
            f"Done in {time.monotonic() - t0:.0f}s: {summary['snapshots']} snapshots written "
            f"({summary['schemes_live']} live schemes, {summary['categories']} categories, "
            f"{summary['dead_in_medians']} dead schemes used in peer medians)"
            + (" [dry-run]" if o["dry_run"] else "")))
        log.info("mf_metrics.complete", as_of=str(as_of), **summary)
