"""Run the MF advisor over a tenant's holdings (personal use — see services/advisor.py).

Usage:
    .venv/bin/python manage.py run_mf_advisor --tenant <uuid|slug>
    .venv/bin/python manage.py run_mf_advisor --tenant my-slug --portfolio <uuid|name> \
        --target-file target.json --assumed-drag-pct 0.75

target.json maps AMFI category -> percent, e.g.
    {"Equity Scheme - Large Cap Fund": 30, "Equity Scheme - Flexi Cap Fund": 40, "Debt Scheme - Liquid Fund": 30}

Findings are also logged to the event log (step_name "mf_advisor.<kind>");
pass --no-events to print only.
"""
from __future__ import annotations

import json
import uuid

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q

from apps.mutual_funds.services.advisor import (
    DEFAULT_ASSUMED_DRAG_PCT,
    DEFAULT_DRIFT_THRESHOLD_PCT,
    run_advisor,
)
from apps.tenants.models import Tenant
from apps.trading.models import Portfolio

SECTIONS = [
    ("direct_vs_regular", "Direct vs Regular (drag is an ESTIMATE from an assumed spread)"),
    ("allocation", "Category allocation vs target"),
    ("overlap", "Overlap (category + AMC only)"),
    ("sip_continuity", "SIP continuity"),
]


def _is_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except ValueError:
        return False


class Command(BaseCommand):
    help = "Run the deterministic MF advisor (direct-vs-regular, allocation, overlap, SIP continuity)."

    def add_arguments(self, parser):
        parser.add_argument("--tenant", required=True, help="Tenant id (uuid) or slug")
        parser.add_argument("--portfolio", default=None, help="Portfolio id (uuid) or name; default: all of the tenant's")
        parser.add_argument("--target-file", default=None, help="JSON file: {category: percent}")
        parser.add_argument("--assumed-drag-pct", type=float, default=DEFAULT_ASSUMED_DRAG_PCT,
                            help=f"ASSUMED regular-minus-direct spread, %%/yr (default {DEFAULT_ASSUMED_DRAG_PCT})")
        parser.add_argument("--drift-threshold-pct", type=float, default=DEFAULT_DRIFT_THRESHOLD_PCT)
        parser.add_argument("--no-events", action="store_true", help="Do not write findings to the event log")

    def handle(self, *args, **opts):
        tq = Q(id=opts["tenant"]) if _is_uuid(opts["tenant"]) else Q(slug=opts["tenant"])
        tenant = Tenant.objects.filter(tq).first()
        if tenant is None:
            raise CommandError(f"Tenant '{opts['tenant']}' not found.")

        portfolio = None
        if opts["portfolio"]:
            pq = Q(id=opts["portfolio"]) if _is_uuid(opts["portfolio"]) else Q(name=opts["portfolio"])
            portfolio = Portfolio.objects.filter(pq, tenant=tenant).first()
            if portfolio is None:
                raise CommandError(f"Portfolio '{opts['portfolio']}' not found for tenant {tenant.id}.")

        target = None
        if opts["target_file"]:
            try:
                with open(opts["target_file"]) as fh:
                    target = json.load(fh)
                if not isinstance(target, dict) or not all(isinstance(v, (int, float)) for v in target.values()):
                    raise ValueError("expected an object of {category: percent}")
            except (OSError, ValueError) as exc:
                raise CommandError(f"Bad --target-file: {exc}") from exc

        report = run_advisor(
            tenant, portfolio, target=target,
            assumed_drag_pct=opts["assumed_drag_pct"],
            drift_threshold_pct=opts["drift_threshold_pct"],
            emit_events=not opts["no_events"],
        )

        self.stdout.write(f"MF advisor — tenant {tenant.id}, portfolio value INR {report.total_value:,.2f}")
        for note in report.notes:
            self.stdout.write(self.style.WARNING(f"  note: {note}"))
        for name in report.unvalued:
            self.stdout.write(self.style.WARNING(f"  no NAV, excluded from value: {name}"))
        for kind, title in SECTIONS:
            items = report.by_kind(kind)
            if not items:
                continue
            self.stdout.write(f"\n{title}")
            for f in items:
                style = self.style.WARNING if f.severity == "warn" else (lambda s: s)
                self.stdout.write(style(f"  - {f.text}"))
        self.stdout.write(self.style.SUCCESS(f"\n{len(report.findings)} finding(s)."))
