"""Forward-enrich multi-day (swing) signals with daily candles.

Companion to the intraday ``enrich_signals``: that one measures same-day 5-minute
MFE/MAE (right for screener/scalp), this one measures the **forward N-trading-day**
excursion (right for swing/positional ideas like StockEdge composite momentum).
The two never overlap — the intraday command excludes ``SWING_SOURCES``.

Runs after the holding window elapses; fills ``max_favorable_move`` /
``max_adverse_move`` / ``eod_price`` + per-horizon snapshots so the monthly
capture matrix scores swing signals meaningfully.

Examples
--------
    ./manage.py enrich_swing_signals                       # STOCKEDGE, 20d window
    ./manage.py enrich_swing_signals --horizon 10
    ./manage.py enrich_swing_signals --partial --as-of 2026-07-10
"""
from __future__ import annotations

from datetime import datetime

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Forward-enrich swing signals (StockEdge momentum, …) with daily candles."

    def add_arguments(self, parser):
        parser.add_argument("--source", default="STOCKEDGE",
                            help="Signal source to enrich (default: STOCKEDGE).")
        parser.add_argument("--horizon", type=int, default=None,
                            help="Holding window in trading days (default: 20).")
        parser.add_argument("--as-of", default="",
                            help="Evaluate as of this date (YYYY-MM-DD, default: today).")
        parser.add_argument("--tenant", default="",
                            help="Restrict to one tenant (slug or name). Default: all.")
        parser.add_argument("--partial", action="store_true",
                            help="Also enrich windows that haven't fully elapsed yet "
                                 "(keeps them PENDING for a later refresh).")

    def handle(self, *args, **o):
        from apps.strategies.services.swing_enrichment import (
            DEFAULT_HORIZON_DAYS, enrich_swing_signals,
        )
        from apps.tenants.models import Tenant

        horizon = o["horizon"] or DEFAULT_HORIZON_DAYS
        as_of = None
        if o["as_of"]:
            try:
                as_of = datetime.strptime(o["as_of"], "%Y-%m-%d").date()
            except ValueError as exc:
                raise CommandError(f"Bad --as-of {o['as_of']!r}: {exc}")

        if o["tenant"]:
            tenants = list(
                Tenant.objects.filter(slug=o["tenant"])
                or Tenant.objects.filter(name=o["tenant"])
            )
            if not tenants:
                raise CommandError(f"Tenant {o['tenant']!r} not found.")
        else:
            tenants = list(Tenant.objects.all())
        if not tenants:
            raise CommandError("No tenants found.")

        total_enriched = total_skipped = 0
        for tenant in tenants:
            res = enrich_swing_signals(
                tenant, source=o["source"], horizon=horizon,
                as_of=as_of, partial=o["partial"],
            )
            total_enriched += res["enriched"]
            total_skipped += res["skipped"]
            if res["enriched"] or res["skipped"]:
                self.stdout.write(
                    f"[{tenant.slug or tenant.name}] enriched {res['enriched']}, "
                    f"skipped {res['skipped']}"
                )

        self.stdout.write(self.style.SUCCESS(
            f"\nSwing enrichment ({o['source']}, {horizon}d window): "
            f"{total_enriched} enriched, {total_skipped} skipped."
        ))
