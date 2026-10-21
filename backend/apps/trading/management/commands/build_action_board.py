"""Build the Action Board by hand (UX redesign R3).

    manage.py build_action_board                          # today, live (after 09:20 IST)
    manage.py build_action_board --date 2026-10-21 --max-calls 2   # replay a past session at 09:35
    manage.py build_action_board --telegram               # force the Telegram summary

Writes a RunReceipt (trigger=manual). Replays never send Telegram unless
--telegram is given. PAPER only; never places an order.
"""
from __future__ import annotations

import json
from datetime import date

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Build today's Action Board (directional planner, PAPER, unproven)."

    def add_arguments(self, parser):
        parser.add_argument("--date", default="", help="Trading day YYYY-MM-DD (past days replay at the board time).")
        parser.add_argument("--max-calls", type=int, default=None,
                            help="Cap on Claude planner calls (default settings.ACTION_BOARD MAX_PLANNER_CALLS).")
        tg = parser.add_mutually_exclusive_group()
        tg.add_argument("--telegram", action="store_true", help="Send the Telegram summary.")
        tg.add_argument("--no-telegram", action="store_true", help="Never send the Telegram summary.")

    def handle(self, *args, **opts):
        from apps.trading.tasks.action_board import run_build

        day = opts["date"] or None
        if day:
            try:
                date.fromisoformat(day)
            except ValueError as e:
                raise CommandError(f"--date must be YYYY-MM-DD: {e}") from e
        telegram = True if opts["telegram"] else (False if opts["no_telegram"] else None)
        out = run_build(day=day, max_planner_calls=opts["max_calls"], telegram=telegram, trigger="manual")
        self.stdout.write(json.dumps(out, indent=2, default=str))
