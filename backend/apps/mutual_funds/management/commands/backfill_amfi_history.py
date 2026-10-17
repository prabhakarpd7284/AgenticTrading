"""Backfill historical NAVs for specific schemes from AMFI's history report.

Usage:
    .venv/bin/python manage.py backfill_amfi_history --scheme 148921
    .venv/bin/python manage.py backfill_amfi_history --scheme 148921 --scheme 135762 --years 3
    .venv/bin/python manage.py backfill_amfi_history --scheme 148921 --amc-id 3   # skip AMC-id probing

AMFI only filters by AMC, so each (AMC, year) is a ~15 MB download. Pass the
schemes you actually hold, not the whole catalog. Idempotent: existing
(scheme, date) rows are left alone.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import requests
import structlog
from django.core.management.base import BaseCommand, CommandError

from apps.mutual_funds.models import MFNav, MFScheme
from apps.mutual_funds.services.amfi_history import (
    fetch_history_text,
    parse_history_text,
    resolve_amc_id,
    year_chunks,
)

log = structlog.get_logger(__name__)


class Command(BaseCommand):
    help = "Backfill historical MFNav rows for the given scheme codes from AMFI."

    def add_arguments(self, parser):
        parser.add_argument("--scheme", action="append", required=True, help="AMFI scheme code (repeatable)")
        parser.add_argument("--years", type=int, default=2, help="How many years back (default 2)")
        parser.add_argument("--amc-id", type=int, default=None, help="AMFI AMC id; skips probing (single AMC only)")
        parser.add_argument("--dry-run", action="store_true", help="Download + parse, no DB writes")

    def handle(self, *args, **options):
        codes = set(options["scheme"])
        schemes = {s.scheme_code: s for s in MFScheme.objects.filter(scheme_code__in=codes)}
        missing = codes - set(schemes)
        if missing:
            raise CommandError(
                f"Unknown scheme code(s) {sorted(missing)} — run pull_amfi_nav first to load the catalog."
            )

        by_amc: dict[str, set[str]] = defaultdict(set)
        for s in schemes.values():
            by_amc[s.amc_name].add(s.scheme_code)
        if options["amc_id"] is not None and len(by_amc) > 1:
            raise CommandError("--amc-id only makes sense when all schemes belong to one AMC.")

        end = date.today()
        start = end - timedelta(days=365 * options["years"])

        total_new = 0
        for amc_name, amc_codes in by_amc.items():
            amc_id = options["amc_id"] or resolve_amc_id(amc_name)
            if amc_id is None:
                self.stderr.write(self.style.ERROR(f"Could not resolve AMFI id for '{amc_name}' — skipping."))
                log.error("amfi_backfill.amc_unresolved", amc=amc_name)
                continue
            self.stdout.write(f"{amc_name} (mf={amc_id}): {len(amc_codes)} scheme(s), {start} -> {end}")

            for chunk_start, chunk_end in year_chunks(start, end):
                try:
                    text = fetch_history_text(amc_id, chunk_start, chunk_end)
                except requests.RequestException as exc:
                    self.stderr.write(self.style.ERROR(f"  {chunk_start}..{chunk_end}: fetch failed: {exc}"))
                    log.error("amfi_backfill.fetch_failed", amc=amc_name, start=str(chunk_start), error=str(exc))
                    continue
                rows = parse_history_text(text, amc_codes)
                if options["dry_run"]:
                    self.stdout.write(f"  {chunk_start}..{chunk_end}: {len(rows)} rows [dry-run]")
                    continue
                before = MFNav.objects.filter(scheme_id__in=amc_codes).count()
                MFNav.objects.bulk_create(
                    [MFNav(scheme_id=r.scheme_code, nav_date=r.nav_date, nav=r.nav) for r in rows],
                    batch_size=2000,
                    ignore_conflicts=True,
                )
                added = MFNav.objects.filter(scheme_id__in=amc_codes).count() - before
                total_new += added
                self.stdout.write(f"  {chunk_start}..{chunk_end}: {len(rows)} parsed, {added} new")

        self.stdout.write(self.style.SUCCESS(f"Done. {total_new} new NAV rows."))
        log.info("amfi_backfill.complete", schemes=sorted(codes), navs_created=total_new)
