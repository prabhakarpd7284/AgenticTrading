"""python manage.py run_ai_trader_user — generate feature requests
from a trader's POV. Writes into the shared mind palace.

Profiles change the persona of the trader:
  default   — generalist quant
  options   — NIFTY/BANKNIFTY index options
  futures   — index/stock futures
  equity    — swing/positional equities
  intraday  — intraday equity scalper

Pass --profile <id> to switch, or --all to run every profile in sequence.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.agents_core.tester import trader_user
from apps.agents_core.tester.trader_profiles import PROFILES


class Command(BaseCommand):
    help = "Run the AI Trader-User agent (generates feature requests)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--profile", default="default",
            choices=list(PROFILES.keys()),
            help="Persona to adopt when generating requests.",
        )
        parser.add_argument(
            "--all", action="store_true",
            help="Run every profile in sequence (ignores --profile).",
        )

    def handle(self, *args, **opts):
        if opts["all"]:
            results = trader_user.run_all_profiles()
            total = sum(results.values())
            self.stdout.write(self.style.SUCCESS(
                f"trader_user (all profiles): {total} feature request(s) added/updated across "
                f"{len(results)} personas."
            ))
            for pid, count in results.items():
                self.stdout.write(f"  {pid}: {count}")
            return

        palace, added = trader_user.run(profile=opts["profile"])
        self.stdout.write(self.style.SUCCESS(
            f"trader_user [{opts['profile']}]: {added} feature request"
            f"{'s' if added != 1 else ''} added/updated."
        ))
        if palace.feature_requests:
            self.stdout.write("\nLatest feature requests:")
            for f in palace.feature_requests[-5:]:
                rb = getattr(f, "requested_by", "trader_user")
                self.stdout.write(f"  [{f.category}] ({rb}) {f.title}")
                self.stdout.write(f"    rationale: {f.rationale[:200]}")
