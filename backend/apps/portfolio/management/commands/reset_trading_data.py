"""python manage.py reset_trading_data — wipe trading state so you can
start fresh and watch new trades flow in.

By default does NOTHING unless --confirm is passed. Flags pick what to
wipe; --all wipes everything (still requires --confirm).

  --journal      legacy TradeJournal
  --straddles    legacy StraddlePosition
  --snapshots    legacy + v2 PortfolioSnapshot
  --audit        legacy AuditLog
  --signals      legacy SignalLog
  --watchlist    legacy WatchlistEntry
  --orders       v2 Order + OutboxEvent
  --runs         v2 AgentRun + AgentStep
  --positions    v2 Position
  --cache        Flush the entire Django cache (clears cockpit caches)
  --capital N    Seed today's snapshot with N rupees of capital
  --all          All of the above flags
  --keep-watchlist  Don't wipe watchlist even with --all

Example:
  python manage.py reset_trading_data --all --confirm --capital 500000
"""
from __future__ import annotations

from datetime import date

from django.core.cache import cache
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Reset trading state (journal / straddles / runs / orders / caches)."

    def add_arguments(self, parser):
        parser.add_argument("--journal", action="store_true")
        parser.add_argument("--straddles", action="store_true")
        parser.add_argument("--snapshots", action="store_true")
        parser.add_argument("--audit", action="store_true")
        parser.add_argument("--signals", action="store_true")
        parser.add_argument("--watchlist", action="store_true")
        parser.add_argument("--orders", action="store_true")
        parser.add_argument("--runs", action="store_true")
        parser.add_argument("--positions", action="store_true")
        parser.add_argument("--cache", action="store_true")
        parser.add_argument("--all", action="store_true")
        parser.add_argument("--keep-watchlist", action="store_true")
        parser.add_argument("--capital", type=float, default=None,
                            help="Seed today's snapshot with N rupees.")
        parser.add_argument("--confirm", action="store_true",
                            help="Required — destructive operation.")

    def handle(self, *args, **opts):
        if not opts["confirm"]:
            self.stdout.write(self.style.WARNING(
                "Refusing to run without --confirm. This deletes data."
            ))
            return

        # When --all is set, light up every flag (with optional opt-out for
        # the watchlist since that's annoying to re-seed by hand).
        if opts["all"]:
            for k in ("journal", "straddles", "snapshots", "audit", "signals",
                      "watchlist", "orders", "runs", "positions", "cache"):
                opts[k] = True
            if opts["keep_watchlist"]:
                opts["watchlist"] = False

        wiped: list[str] = []

        if opts["journal"]:
            from trading.models import TradeJournal
            n, _ = TradeJournal.objects.all().delete()
            wiped.append(f"TradeJournal: {n}")

        if opts["straddles"]:
            from trading.models import StraddlePosition
            n, _ = StraddlePosition.objects.all().delete()
            wiped.append(f"StraddlePosition: {n}")

        if opts["snapshots"]:
            from trading.models import PortfolioSnapshot as LegacySnap
            n_legacy, _ = LegacySnap.objects.all().delete()
            wiped.append(f"legacy PortfolioSnapshot: {n_legacy}")
            try:
                from apps.portfolio.models import PortfolioSnapshot as V2Snap
                n_v2, _ = V2Snap.objects.all().delete()
                wiped.append(f"v2 PortfolioSnapshot: {n_v2}")
            except Exception as e:  # noqa: BLE001
                wiped.append(f"v2 PortfolioSnapshot skipped: {e}")

        if opts["audit"]:
            from trading.models import AuditLog
            n, _ = AuditLog.objects.all().delete()
            wiped.append(f"AuditLog: {n}")

        if opts["signals"]:
            from trading.models import SignalLog
            n, _ = SignalLog.objects.all().delete()
            wiped.append(f"SignalLog: {n}")

        if opts["watchlist"]:
            from trading.models import WatchlistEntry
            n, _ = WatchlistEntry.objects.all().delete()
            wiped.append(f"WatchlistEntry: {n}")

        if opts["orders"]:
            try:
                from apps.orders.models import Order, OutboxEvent
                n_o, _ = Order.objects.all().delete()
                n_oe, _ = OutboxEvent.objects.all().delete()
                wiped.append(f"Order: {n_o}  OutboxEvent: {n_oe}")
            except Exception as e:  # noqa: BLE001
                wiped.append(f"Order skipped: {e}")

        if opts["runs"]:
            try:
                from apps.agents_core.models import AgentRun, AgentStep
                n_s, _ = AgentStep.objects.all().delete()   # children first
                n_r, _ = AgentRun.objects.all().delete()
                wiped.append(f"AgentRun: {n_r}  AgentStep: {n_s}")
            except Exception as e:  # noqa: BLE001
                wiped.append(f"AgentRun skipped: {e}")

        if opts["positions"]:
            try:
                from apps.portfolio.models import Position
                n, _ = Position.objects.all().delete()
                wiped.append(f"v2 Position: {n}")
            except Exception as e:  # noqa: BLE001
                wiped.append(f"v2 Position skipped: {e}")

        if opts["cache"]:
            try:
                cache.clear()
                wiped.append("Django cache: flushed")
            except Exception as e:  # noqa: BLE001
                wiped.append(f"cache flush skipped: {e}")

        if opts["capital"] is not None:
            from trading.models import PortfolioSnapshot as LegacySnap
            snap, created = LegacySnap.objects.get_or_create(
                snapshot_date=date.today(),
                defaults={
                    "capital": opts["capital"],
                    "available_cash": opts["capital"],
                    "invested": 0.0,
                },
            )
            if not created:
                snap.capital = opts["capital"]
                snap.available_cash = opts["capital"]
                snap.invested = 0.0
                snap.save(update_fields=["capital", "available_cash", "invested", "last_updated"])
            wiped.append(f"PortfolioSnapshot seeded: capital={opts['capital']}")

        if not wiped:
            self.stdout.write(self.style.WARNING(
                "Nothing wiped — pass at least one flag (or --all)."
            ))
            return

        self.stdout.write(self.style.MIGRATE_HEADING("Reset summary"))
        for line in wiped:
            self.stdout.write(f"  {line}")
        self.stdout.write(self.style.SUCCESS("Done."))
