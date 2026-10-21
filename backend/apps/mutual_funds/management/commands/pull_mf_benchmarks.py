"""Pull benchmark TRI history (niftyindices.com) into MFBenchmarkNav.

Usage:
    .venv/bin/python manage.py pull_mf_benchmarks                   # all indices, 14 years
    .venv/bin/python manage.py pull_mf_benchmarks --index "NIFTY 500" --years 1
Idempotent (bulk_create ignore_conflicts); safe to re-run to top up.
"""
from __future__ import annotations

from datetime import date, timedelta

from django.core.management.base import BaseCommand
from django.db.models import Max

from apps.mutual_funds.models import MFBenchmarkNav
from apps.mutual_funds.services.benchmarks import BENCHMARK_INDICES, fetch_tri_range


class Command(BaseCommand):
    help = "Fetch Nifty TRI history for the screener's benchmarks."

    def add_arguments(self, parser):
        parser.add_argument("--index", action="append", default=None)
        parser.add_argument("--years", type=int, default=14)
        parser.add_argument("--incremental", action="store_true", help="Start from each index's last stored date")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **o):
        end = date.today()
        total = 0
        for name in o["index"] or BENCHMARK_INDICES:
            start = end - timedelta(days=round(365.25 * o["years"]))
            if o["incremental"]:
                last = MFBenchmarkNav.objects.filter(index_name=name).aggregate(m=Max("nav_date"))["m"]
                if last:
                    start = last + timedelta(days=1)
            if start > end:
                continue
            rows, errors = fetch_tri_range(name, start, end)
            for e in errors:
                self.stderr.write(self.style.WARNING(e))
            if o["dry_run"]:
                self.stdout.write(f"{name}: {len(rows)} rows [dry-run]")
                continue
            before = MFBenchmarkNav.objects.filter(index_name=name).count()
            MFBenchmarkNav.objects.bulk_create(
                [MFBenchmarkNav(index_name=name, nav_date=d, value=v, is_tri=True, source="niftyindices")
                 for d, v in rows], batch_size=2000, ignore_conflicts=True)
            added = MFBenchmarkNav.objects.filter(index_name=name).count() - before
            total += added
            first = rows[0][0] if rows else None
            self.stdout.write(f"{name}: {len(rows)} rows fetched ({first} .. {rows[-1][0] if rows else None}), {added} new")
        self.stdout.write(self.style.SUCCESS(f"Done. {total} new benchmark rows."))
