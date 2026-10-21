"""Report NAV-history coverage of the screening universe (S1 acceptance).

Usage:
    .venv/bin/python manage.py mf_data_coverage
    .venv/bin/python manage.py mf_data_coverage --by-category
    .venv/bin/python manage.py mf_data_coverage --json

Prints: universe size (live) and dead schemes kept for survivorship, rows per
scheme, % of the live universe with >= 1/3/5/10 years of history, data-quality
issues by kind, backfill job status, total download size and wall-clock runtime
of the backfill (first job start -> last job finish).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import median

from django.core.management.base import BaseCommand
from django.db.models import Count, Max, Min, Sum

from apps.mutual_funds.models import MFBackfillJob, MFDataIssue, MFNav
from apps.mutual_funds.services.universe import canonical_category, universe_schemes

THRESHOLDS = (1, 3, 5, 10)


def coverage_report(by_category: bool = False) -> dict:
    all_u = universe_schemes(include_inactive=True)
    live = [s for s in all_u if s.is_active]
    dead = [s for s in all_u if not s.is_active]
    codes = [s.scheme_code for s in all_u]
    stats = {
        r["scheme_id"]: r
        for r in MFNav.objects.filter(scheme_id__in=codes).values("scheme_id")
        .annotate(n=Count("id"), first=Min("nav_date"), last=Max("nav_date"))
    }

    def years(code):
        r = stats.get(code)
        return 0.0 if not r else (r["last"] - r["first"]).days / 365.25

    def block(schemes):
        n = len(schemes)
        ys = [years(s.scheme_code) for s in schemes]
        rows = [stats.get(s.scheme_code, {}).get("n", 0) for s in schemes]
        out = {"schemes": n, "nav_rows": sum(rows),
               "rows_per_scheme_median": median(rows) if rows else 0,
               "rows_per_scheme_min": min(rows) if rows else 0,
               "rows_per_scheme_max": max(rows) if rows else 0,
               "with_history_only_1_row": sum(1 for r in rows if r <= 1)}
        for t in THRESHOLDS:
            k = sum(1 for y in ys if y >= t - 0.02)   # ~1 week slack for holidays at the edges
            out[f"ge_{t}y"] = k
            out[f"ge_{t}y_pct"] = round(k / n * 100, 1) if n else 0.0
        return out

    report = {"live_universe": block(live), "dead_schemes_kept": block(dead)}
    if by_category:
        cats = defaultdict(list)
        for s in live:
            cats[canonical_category(s.category, s.scheme_name)].append(s)
        report["by_category"] = {c: block(v) for c, v in sorted(cats.items())}

    issues = Counter(dict(MFDataIssue.objects.filter(scheme_id__in=codes).values_list("kind")
                          .annotate(n=Count("id")).values_list("kind", "n")))
    report["data_issues"] = dict(issues)
    report["schemes_with_issues"] = MFDataIssue.objects.filter(scheme_id__in=codes).values("scheme_id").distinct().count()
    jobs = MFBackfillJob.objects.filter(source="mfapi")
    agg = jobs.aggregate(bytes=Sum("bytes_downloaded"), start=Min("started_at"), end=Max("finished_at"),
                         fetched=Sum("rows_fetched"), inserted=Sum("rows_inserted"))
    report["backfill_jobs"] = dict(jobs.values_list("status").annotate(n=Count("id")).values_list("status", "n"))
    report["backfill_download_mb"] = round((agg["bytes"] or 0) / 1e6, 1)
    report["backfill_rows_fetched"] = agg["fetched"] or 0
    report["backfill_rows_inserted"] = agg["inserted"] or 0
    report["backfill_wallclock_s"] = (
        round((agg["end"] - agg["start"]).total_seconds()) if agg["start"] and agg["end"] else None
    )
    report["total_nav_rows_all_schemes"] = MFNav.objects.count()
    return report


class Command(BaseCommand):
    help = "NAV-history coverage of the MF screening universe."

    def add_arguments(self, parser):
        parser.add_argument("--by-category", action="store_true")
        parser.add_argument("--json", action="store_true")

    def handle(self, *args, **o):
        r = coverage_report(by_category=o["by_category"])
        if o["json"]:
            self.stdout.write(json.dumps(r, indent=2, default=str))
            return
        w = self.stdout.write
        for name in ("live_universe", "dead_schemes_kept"):
            b = r[name]
            w(f"{name}: {b['schemes']} schemes, {b['nav_rows']:,} NAV rows "
              f"(rows/scheme median {b['rows_per_scheme_median']}, min {b['rows_per_scheme_min']}, "
              f"max {b['rows_per_scheme_max']}; {b['with_history_only_1_row']} with <=1 row)")
            w("   history: " + "  ".join(f">={t}y {b[f'ge_{t}y']} ({b[f'ge_{t}y_pct']}%)" for t in THRESHOLDS))
        if "by_category" in r:
            w("\nby category (live):")
            for c, b in r["by_category"].items():
                w(f"  {c[:60]:60} n={b['schemes']:>4}  >=3y {b['ge_3y_pct']:>5}%  >=5y {b['ge_5y_pct']:>5}%")
        w(f"\ndata issues: {r['data_issues']} across {r['schemes_with_issues']} schemes")
        w(f"backfill jobs: {r['backfill_jobs']}; downloaded {r['backfill_download_mb']} MB; "
          f"rows fetched {r['backfill_rows_fetched']:,}, inserted {r['backfill_rows_inserted']:,}; "
          f"wall-clock {r['backfill_wallclock_s']} s")
        w(f"MFNav rows (all schemes): {r['total_nav_rows_all_schemes']:,}")
