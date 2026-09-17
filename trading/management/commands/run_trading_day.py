"""run_trading_day — deprecated full-day orchestrator.

The orchestrator's handlers were tightly coupled to the legacy SQLite
schema (StraddlePosition.ce_symbol/pe_symbol/management_log,
TradeJournal.pnl/order_id, PortfolioSnapshot.capital/daily_loss/etc.)
that the v1→v2 migration retired. Field-level reconciliation would mean
a rewrite of every cycle handler; the v2-native path already exists as
individual plugin commands wired into the Ops Console.

This command now prints a clear migration map and exits cleanly so the
Ops Console doesn't show opaque AttributeError cascades when an operator
clicks Run on it.
"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run the full trading day (deprecated — see migration map)"

    def add_arguments(self, parser):
        # Args preserved so callers passing them don't argparse-error;
        # the deprecation banner runs regardless.
        parser.add_argument("--universe", default="high_volume")
        parser.add_argument("--skip-llm", action="store_true", dest="skip_llm")
        parser.add_argument("--dry-run", action="store_true", dest="dry_run")
        parser.add_argument("--max-positions", type=int, default=3, dest="max_positions")
        parser.add_argument("--straddle-interval", type=int, default=10, dest="straddle_interval")
        parser.add_argument("--equity-interval", type=int, default=5, dest="equity_interval")

    def handle(self, *args, **options):
        del args, options  # required by BaseCommand interface; intentionally unused.
        bar = "═" * 72
        self.stdout.write(self.style.WARNING(
            f"\n{bar}\n"
            f"  run_trading_day is decomposed in v2.\n"
            f"  The legacy orchestrator was glued to retired SQLite-era\n"
            f"  field shapes (StraddlePosition.ce_symbol, TradeJournal.pnl,\n"
            f"  PortfolioSnapshot.capital, …). Each cycle is now its own\n"
            f"  Ops Console command:\n"
            f"{bar}\n"
            f"\n"
            f"  Pre-market scan         python manage.py run_morning_basket --dry-run\n"
            f"  Intraday opportunity    python manage.py run_screener\n"
            f"  Cycle scanner           python manage.py run_ok_scanner --actionable-only\n"
            f"  Trade planner           python manage.py run_trading_agent 'Plan a BUY trade for X'\n"
            f"  Pyramid options         python manage.py run_pyramid --strike X --type CE\n"
            f"  Straddle lifecycle      use the short_straddle plugin via the strategies\n"
            f"                          workflow runtime (apps.strategies.runtime); the\n"
            f"                          legacy manage_straddle CLI's --register/--list\n"
            f"                          still work for one-shot ops\n"
            f"  EOD signal enrichment   python manage.py enrich_signals --all\n"
            f"\n"
            f"  All of the above stream into /ws/ops/ so the React UI's\n"
            f"  Ops Console (and the per-page OpButtons) drives them.\n"
            f"{bar}\n"
        ))
