"""Resumable NAV-history backfill for the screening universe (S1).

Source: mfapi.in, one call per scheme (see mfapi_client). Progress lives in
``MFBackfillJob`` (source="mfapi", key=scheme code): re-running skips jobs
already ``done``/``empty`` at the requested depth, so an interrupted run just
continues. Rows go in with ``bulk_create(ignore_conflicts=True)`` in batches;
existing (scheme, date) rows — e.g. from the AMFI daily pull — are never
overwritten. Where both sources have the same date and disagree, a
``source_mismatch`` MFDataIssue is recorded (AMFI's value is kept).

Collection code: no ranking/advice imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import structlog
from django.db.models import Count, Max
from django.utils import timezone

from apps.mutual_funds.models import MFBackfillJob, MFDataIssue, MFNav, MFScheme
from apps.mutual_funds.services import data_quality
from apps.mutual_funds.services.mfapi_client import MfapiClient, SchemeHistory
from apps.mutual_funds.services.universe import (
    active_cutoff,
    looks_like_direct_growth_name,
    passes_static_rules,
)

log = structlog.get_logger(__name__)

SOURCE = "mfapi"
INSERT_BATCH = 2000
MISMATCH_REL_TOL = 1e-4     # 0.01 %: rounding differences between sources are below this


@dataclass
class BackfillTarget:
    scheme_code: str
    known: bool          # already in MFScheme (from AMFI's daily file)


@dataclass
class BackfillStats:
    targets: int = 0
    skipped_done: int = 0
    fetched: int = 0
    empty: int = 0
    out_of_universe: int = 0
    failed: int = 0
    rows_inserted: int = 0
    new_dead_schemes: int = 0
    mismatches: int = 0
    issues: int = 0
    bytes: int = 0
    failures: list[str] = field(default_factory=list)


def catalog_date() -> date | None:
    """The date most schemes are priced at (liquid funds forward-date
    weekend NAVs, so MAX(nav_date) can sit in the future)."""
    mx = MFNav.objects.aggregate(m=Max("nav_date"))["m"]
    if mx is None:
        return None
    row = (
        MFNav.objects.filter(nav_date__gte=mx - timedelta(days=10))
        .values("nav_date").annotate(n=Count("id")).order_by("-n", "-nav_date").first()
    )
    return row["nav_date"] if row else mx


def refresh_active_flags(as_of: date | None = None) -> tuple[int, int]:
    """is_active = latest NAV within ACTIVE_NAV_DAYS of the catalog date.
    Returns (active, inactive) counts."""
    as_of = as_of or catalog_date()
    if as_of is None:
        return 0, 0
    cutoff = active_cutoff(as_of)
    latest = MFScheme.objects.annotate(latest=Max("navs__nav_date"))
    active_codes = set(latest.filter(latest__gte=cutoff).values_list("scheme_code", flat=True))
    MFScheme.objects.filter(scheme_code__in=active_codes, is_active=False).update(is_active=True)
    MFScheme.objects.exclude(scheme_code__in=active_codes).filter(is_active=True).update(is_active=False)
    total = MFScheme.objects.count()
    return len(active_codes), total - len(active_codes)


def build_targets(client: MfapiClient | None, include_dead: bool = True) -> list[BackfillTarget]:
    """Universe for the backfill: every catalog scheme passing the static
    universe rules (live or stale), plus — for survivorship — direct-growth
    looking schemes that mfapi lists but AMFI's current file no longer has."""
    targets = [
        BackfillTarget(s.scheme_code, True)
        for s in MFScheme.objects.filter(category__istartswith="Open Ended").only(
            "scheme_code", "plan", "option", "scheme_name", "category", "is_direct")
        if passes_static_rules(plan=s.plan, option=s.option, scheme_name=s.scheme_name,
                               category=s.category, is_direct=s.is_direct)
    ]
    if include_dead and client is not None:
        known = set(MFScheme.objects.values_list("scheme_code", flat=True))
        for row in client.scheme_list():
            code = str(row.get("schemeCode") or "")
            if code and code not in known and looks_like_direct_growth_name(row.get("schemeName") or ""):
                targets.append(BackfillTarget(code, False))
    targets.sort(key=lambda t: (not t.known, t.scheme_code))
    return targets


def _create_dead_scheme(h: SchemeHistory) -> MFScheme:
    return MFScheme.objects.create(
        scheme_code=h.scheme_code,
        isin_growth=h.isin_growth[:20],
        isin_div_reinvestment=h.isin_div_reinvestment[:20],
        amc_name=h.fund_house[:200] or "Unknown",
        scheme_name=h.scheme_name[:300],
        plan="Direct Plan",
        option="Growth",
        category=h.amfi_category[:200],
        is_direct=True,
        is_active=False,
        catalog_source=SOURCE,
    )


def record_issues(scheme_code: str, navs: list[tuple[date, Decimal]]) -> int:
    """Re-derive gap/jump/nonpositive issues for one scheme (replaces old ones)."""
    found = data_quality.find_issues(navs)
    MFDataIssue.objects.filter(
        scheme_id=scheme_code,
        kind__in=[MFDataIssue.KIND_GAP, MFDataIssue.KIND_JUMP, MFDataIssue.KIND_NONPOSITIVE],
    ).delete()
    MFDataIssue.objects.bulk_create(
        [MFDataIssue(scheme_id=scheme_code, kind=i.kind, start_date=i.start_date, end_date=i.end_date,
                     detail=i.detail) for i in found],
        ignore_conflicts=True,
    )
    return len(found)


def _check_mismatches(scheme_code: str, navs: list[tuple[date, Decimal]]) -> int:
    if not navs:
        return 0
    fetched = dict(navs)
    existing = MFNav.objects.filter(
        scheme_id=scheme_code, nav_date__gte=navs[0][0], nav_date__lte=navs[-1][0],
        nav_date__in=list(fetched),
    ).values_list("nav_date", "nav")
    n = 0
    for d, v in existing:
        other = fetched.get(d)
        if other is None or v == 0:
            continue
        rel = abs(float(other) / float(v) - 1)
        if rel > MISMATCH_REL_TOL:
            MFDataIssue.objects.update_or_create(
                scheme_id=scheme_code, kind=MFDataIssue.KIND_SOURCE_MISMATCH, start_date=d,
                defaults={"end_date": d, "detail": {"kept": float(v), "mfapi": float(other),
                                                    "rel_diff": round(rel, 6)}},
            )
            n += 1
    return n


def backfill_one(client: MfapiClient, target: BackfillTarget, from_date: date, *,
                 force: bool = False, dry_run: bool = False, stats: BackfillStats | None = None) -> str:
    """Returns the resulting job status (or "skipped")."""
    stats = stats or BackfillStats()
    job, _ = MFBackfillJob.objects.get_or_create(source=SOURCE, key=target.scheme_code)
    done_states = (MFBackfillJob.STATUS_DONE, MFBackfillJob.STATUS_EMPTY)
    if (not force and job.status in done_states and job.from_date is not None and job.from_date <= from_date):
        stats.skipped_done += 1
        return "skipped"

    job.attempts += 1
    job.started_at = timezone.now()
    try:
        h = client.scheme_history(target.scheme_code, start=from_date)
    except Exception as exc:  # noqa: BLE001 — one scheme must never kill the run
        job.status = MFBackfillJob.STATUS_FAILED
        job.error = str(exc)[:2000]
        job.finished_at = timezone.now()
        if not dry_run:
            job.save()
        stats.failed += 1
        stats.failures.append(target.scheme_code)
        log.warning("mf_backfill.failed", scheme=target.scheme_code, error=str(exc))
        return job.status

    stats.bytes += h.raw_bytes
    job.bytes_downloaded = h.raw_bytes
    job.rows_fetched = len(h.navs)
    job.from_date = from_date
    job.error = ""
    if not h.found or not h.navs:
        job.status = MFBackfillJob.STATUS_EMPTY
        job.error = "source returned no data"
        stats.empty += 1
    elif not target.known and not passes_static_rules(
        plan="", option="", scheme_name=h.scheme_name, category=h.amfi_category,
        scheme_type=h.scheme_type,
    ):
        job.status = MFBackfillJob.STATUS_EMPTY
        job.error = f"outside universe: {h.scheme_type} / {h.scheme_category}"
        stats.out_of_universe += 1
    else:
        job.first_nav_date, job.last_nav_date = h.navs[0][0], h.navs[-1][0]
        if not dry_run:
            if not target.known and not MFScheme.objects.filter(scheme_code=h.scheme_code).exists():
                _create_dead_scheme(h)
                stats.new_dead_schemes += 1
            stats.mismatches += _check_mismatches(h.scheme_code, h.navs)
            before = MFNav.objects.filter(scheme_id=h.scheme_code).count()
            for i in range(0, len(h.navs), INSERT_BATCH):
                MFNav.objects.bulk_create(
                    [MFNav(scheme_id=h.scheme_code, nav_date=d, nav=v) for d, v in h.navs[i:i + INSERT_BATCH]],
                    ignore_conflicts=True,
                )
            job.rows_inserted = MFNav.objects.filter(scheme_id=h.scheme_code).count() - before
            stats.rows_inserted += job.rows_inserted
            series = list(MFNav.objects.filter(scheme_id=h.scheme_code).order_by("nav_date")
                          .values_list("nav_date", "nav"))
            stats.issues += record_issues(h.scheme_code, series)
        job.status = MFBackfillJob.STATUS_DONE
        stats.fetched += 1
    job.finished_at = timezone.now()
    if not dry_run:
        job.save()
    return job.status
