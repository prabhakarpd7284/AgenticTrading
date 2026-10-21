"""Replay skipped / expired / blocked Action Board cards on the day's 1m candles.

    manage.py evaluate_action_outcomes                    # today (run after the 15:50 capture)
    manage.py evaluate_action_outcomes --date 2026-10-21 --force
"""
from __future__ import annotations

import json
from datetime import date

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Expire open Action Board cards and compute what skipped/expired cards would have done."

    def add_arguments(self, parser):
        parser.add_argument("--date", default="", help="Trading day YYYY-MM-DD (default today).")
        parser.add_argument("--force", action="store_true", help="Recompute cards already evaluated.")

    def handle(self, *args, **opts):
        from apps.trading.tasks.action_board import run_outcomes

        day = opts["date"] or None
        if day:
            try:
                date.fromisoformat(day)
            except ValueError as e:
                raise CommandError(f"--date must be YYYY-MM-DD: {e}") from e
        out = run_outcomes(day=day, force=opts["force"], trigger="manual")
        self.stdout.write(json.dumps(out, indent=2, default=str))
