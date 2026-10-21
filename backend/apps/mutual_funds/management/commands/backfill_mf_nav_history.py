"""Resumable NAV-history backfill for the whole screening universe (S1).

Usage:
    .venv/bin/python manage.py backfill_mf_nav_history                 # 5y, live + dead schemes
    .venv/bin/python manage.py backfill_mf_nav_history --years 14      # deeper (S5 backtest)
    .venv/bin/python manage.py backfill_mf_nav_history --scheme 118989 --force
    .venv/bin/python manage.py backfill_mf_nav_history --no-dead --limit 50 --dry-run

Source: mfapi.in, one throttled call per scheme (see services/mfapi_client).
Progress is tracked per scheme in MFBackfillJob, so an interrupted run is
harmless: re-run and finished schemes are skipped. Raw responses are archived
gzipped under $MF_RAW_DIR (default ~/.local/share/alphadesk/mf_raw) for
reproducibility. Run long backfills detached under nice, e.g.:

    setsid nohup nice -n 10 .venv/bin/python manage.py backfill_mf_nav_history \
        --years 14 >> ../logs/mf_backfill.log 2>&1 &
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import structlog
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.mutual_funds.services.mfapi_client import DEFAULT_MIN_INTERVAL_S, MfapiClient
from apps.mutual_funds.services.nav_backfill import (
    BackfillStats,
    BackfillTarget,
    backfill_one,
    build_targets,
    refresh_active_flags,
)

log = structlog.get_logger(__name__)

DEFAULT_YEARS = int(getattr(settings, "MF_HISTORY_YEARS", 5))


def default_raw_dir() -> Path:
    return Path(os.environ.get("MF_RAW_DIR") or Path.home() / ".local/share/alphadesk/mf_raw")


class Command(BaseCommand):
    help = "Backfill MFNav history for the screening universe from mfapi.in (resumable)."

    def add_arguments(self, parser):
        parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                            help=f"History depth in years (default {DEFAULT_YEARS})")
        parser.add_argument("--scheme", action="append", default=None, help="Only these scheme codes (repeatable)")
        parser.add_argument("--no-dead", action="store_true", help="Skip dead schemes from mfapi's list")
        parser.add_argument("--limit", type=int, default=0, help="Stop after N targets (0 = all)")
        parser.add_argument("--force", action="store_true", help="Refetch even if the job is already done")
        parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_S,
                            help="Seconds between HTTP calls (throttle)")
        parser.add_argument("--no-raw", action="store_true", help="Do not archive raw responses")
        parser.add_argument("--dry-run", action="store_true", help="Fetch + parse only, no DB writes")

    def handle(self, *args, **opts):
        t0 = time.monotonic()
        from_date = date.today() - timedelta(days=round(365.25 * opts["years"]))
        client = MfapiClient(min_interval_s=opts["min_interval"],
                             raw_dir=None if opts["no_raw"] else default_raw_dir())

        if opts["scheme"]:
            from apps.mutual_funds.models import MFScheme
            known = set(MFScheme.objects.filter(scheme_code__in=opts["scheme"]).values_list("scheme_code", flat=True))
            targets = [BackfillTarget(c, c in known) for c in opts["scheme"]]
        else:
            targets = build_targets(client, include_dead=not opts["no_dead"])
        if opts["limit"]:
            targets = targets[: opts["limit"]]

        stats = BackfillStats(targets=len(targets))
        self.stdout.write(f"Backfill: {len(targets)} target scheme(s) "
                          f"({sum(t.known for t in targets)} in AMFI catalog, "
                          f"{sum(not t.known for t in targets)} dead candidates), from {from_date}"
                          f"{' [dry-run]' if opts['dry_run'] else ''}")
        for i, target in enumerate(targets, 1):
            backfill_one(client, target, from_date, force=opts["force"], dry_run=opts["dry_run"], stats=stats)
            if i % 50 == 0 or i == len(targets):
                self.stdout.write(
                    f"  [{i}/{len(targets)}] fetched={stats.fetched} skipped={stats.skipped_done} "
                    f"empty={stats.empty} out_of_universe={stats.out_of_universe} failed={stats.failed} "
                    f"rows+={stats.rows_inserted} MB={client.bytes_downloaded / 1e6:.1f} "
                    f"elapsed={time.monotonic() - t0:.0f}s"
                )
                self.stdout.flush()

        if not opts["dry_run"]:
            active, inactive = refresh_active_flags()
            self.stdout.write(f"Active flags refreshed: {active} active / {inactive} inactive schemes.")
        elapsed = time.monotonic() - t0
        self.stdout.write(self.style.SUCCESS(
            f"Done in {elapsed:.0f}s. fetched={stats.fetched} skipped={stats.skipped_done} empty={stats.empty} "
            f"out_of_universe={stats.out_of_universe} failed={stats.failed} new_dead_schemes={stats.new_dead_schemes} "
            f"rows_inserted={stats.rows_inserted} source_mismatches={stats.mismatches} "
            f"data_issues={stats.issues} downloaded={client.bytes_downloaded / 1e6:.1f} MB "
            f"in {client.calls} HTTP calls."
        ))
        if stats.failures:
            self.stderr.write(f"Failed schemes (re-run to retry): {', '.join(stats.failures[:50])}")
        log.info("mf_backfill.complete", elapsed_s=round(elapsed), fetched=stats.fetched,
                 failed=stats.failed, rows=stats.rows_inserted, bytes=client.bytes_downloaded)
