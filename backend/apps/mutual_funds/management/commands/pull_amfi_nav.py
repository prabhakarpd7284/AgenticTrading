"""Daily AMFI NAV pull — populates MFScheme + MFNav.

Usage:
    .venv/bin/python manage.py pull_amfi_nav
    .venv/bin/python manage.py pull_amfi_nav --dry-run    # parse + report counts, no DB writes

Intended to run once daily after NAVs are published (AMFI typically
updates by ~11 PM IST for the current trading day). Schedule it the same
way enrich_signals/run_ok_scanner are scheduled — see config/settings
crontab/beat config.
"""
from __future__ import annotations

import structlog
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.mutual_funds.models import MFNav, MFScheme
from apps.mutual_funds.services.amfi_nav_collector import fetch_nav_text, parse_nav_text

log = structlog.get_logger(__name__)


class Command(BaseCommand):
    help = "Fetch AMFI's daily NAV file and upsert MFScheme + MFNav rows."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Parse only, skip DB writes")

    def handle(self, *args, **options):
        # Machine-readable result for callers that pass a Command instance to
        # call_command (the Celery task reads it into its run receipt).
        self.stats = {"rows_parsed": 0}
        self.stdout.write("Fetching AMFI NAV file...")
        try:
            text = fetch_nav_text()
        except RuntimeError as exc:
            self.stderr.write(self.style.ERROR(str(exc)))
            log.error("amfi_nav_pull.fetch_failed", error=str(exc))
            self.stats["error"] = f"fetch failed: {exc}"[:500]
            return

        rows = parse_nav_text(text)
        self.stats["rows_parsed"] = len(rows)
        self.stdout.write(f"Parsed {len(rows)} scheme/NAV rows.")
        if not rows:
            self.stderr.write(self.style.WARNING("Zero rows parsed — AMFI likely changed format. Not writing."))
            self.stats["error"] = "zero rows parsed (AMFI format change?)"
            return

        if options["dry_run"]:
            schemes = {r.scheme_code for r in rows}
            self.stdout.write(f"[dry-run] {len(schemes)} distinct schemes, {len(rows)} NAV rows. No writes made.")
            return

        schemes_upserted = 0
        navs_created = 0
        navs_skipped = 0

        with transaction.atomic():
            for row in rows:
                _, created_scheme = MFScheme.objects.update_or_create(
                    scheme_code=row.scheme_code,
                    defaults=dict(
                        isin_growth=row.isin_growth,
                        isin_div_reinvestment=row.isin_div_reinvestment,
                        amc_name=row.amc_name,
                        scheme_name=row.scheme_name,
                        plan=row.plan,
                        option=row.option,
                        category=row.category,
                        is_direct="direct" in row.plan.lower(),
                    ),
                )
                schemes_upserted += 1

                _, nav_created = MFNav.objects.get_or_create(
                    scheme_id=row.scheme_code,
                    nav_date=row.nav_date,
                    defaults={"nav": row.nav},
                )
                if nav_created:
                    navs_created += 1
                else:
                    navs_skipped += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. {schemes_upserted} schemes upserted, {navs_created} new NAV rows "
                f"({navs_skipped} already present for their date)."
            )
        )
        self.stats.update(schemes_upserted=schemes_upserted, navs_created=navs_created,
                          navs_skipped=navs_skipped)
        log.info(
            "amfi_nav_pull.complete",
            schemes=schemes_upserted,
            navs_created=navs_created,
            navs_skipped=navs_skipped,
        )
