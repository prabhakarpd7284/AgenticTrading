"""python manage.py reset_trading_data — wipe trading state so you can
start fresh and watch new trades flow in.

By default does NOTHING unless --confirm is passed. Flags pick what to
wipe; --all wipes the listed targets; --nuke does a full DB flush.

Per-table flags:
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

Bulk:
  --all          All per-table flags above
  --keep-watchlist  Don't wipe watchlist even with --all
  --capital N    Seed today's snapshot with N rupees of capital

Nuclear:
  --nuke         flush() every table in BOTH databases (v2 default + legacy
                 sqlite). Schema + migrations preserved, but ALL rows go.
                 Auto-runs --reseed afterwards unless --no-reseed.
  --reseed       After wipe, re-create the smoke user + tenant + smoke
                 portfolio so you can log back in immediately. Implied
                 by --nuke.
  --no-reseed    Skip the re-seed step on --nuke (you'll need to recreate
                 a user manually).

Examples:
  # Soft reset: trading data only, keep watchlist + smoke user
  python manage.py reset_trading_data --all --keep-watchlist --capital 500000 --confirm

  # Full DB nuke: wipe everything, re-seed smoke user
  python manage.py reset_trading_data --nuke --confirm

  # Full DB nuke without re-seeding (you'll need to create a user)
  python manage.py reset_trading_data --nuke --no-reseed --confirm
"""
from __future__ import annotations

from datetime import date

from io import StringIO

from django.conf import settings
from django.core.cache import cache
from django.core.management import call_command
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
        parser.add_argument("--nuke", action="store_true",
                            help="Full DB flush across both databases.")
        parser.add_argument("--reseed", action="store_true",
                            help="Re-create the smoke user + tenant after a nuke.")
        parser.add_argument("--no-reseed", action="store_true",
                            help="Skip the auto-reseed that --nuke does by default.")
        parser.add_argument("--confirm", action="store_true",
                            help="Required — destructive operation.")

    def handle(self, *args, **opts):
        if not opts["confirm"]:
            self.stdout.write(self.style.WARNING(
                "Refusing to run without --confirm. This deletes data."
            ))
            return

        # --nuke short-circuits the per-table flags and flushes both DBs.
        if opts["nuke"]:
            return self._nuke(opts)

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

    # ---------------------------------------------------------------------
    # Full DB nuke — Django's `flush` on every configured database.
    # Schema + migrations stay; every row goes.
    # ---------------------------------------------------------------------
    def _nuke(self, opts):
        wiped: list[str] = []
        databases = list(settings.DATABASES.keys())
        self.stdout.write(self.style.WARNING(
            f"NUKE: flushing {len(databases)} database(s): {', '.join(databases)}"
        ))
        for db in databases:
            buf = StringIO()
            try:
                call_command("flush", "--no-input", database=db,
                             stdout=buf, verbosity=1)
                wiped.append(f"flushed {db}: {buf.getvalue().strip() or 'ok'}")
            except Exception as e:  # noqa: BLE001
                wiped.append(f"flush {db} failed: {e}")

        try:
            cache.clear()
            wiped.append("Django cache: flushed")
        except Exception as e:  # noqa: BLE001
            wiped.append(f"cache flush skipped: {e}")

        # Re-seed unless explicitly opted out. --no-reseed wins over --reseed
        # (and over the implicit re-seed --nuke does by default) so paranoid
        # operators can wipe and decide later.
        should_reseed = not opts["no_reseed"]
        if should_reseed:
            try:
                reseed_summary = self._reseed_smoke_user(
                    capital=opts.get("capital") or 500_000,
                )
                wiped.extend(reseed_summary)
            except Exception as e:  # noqa: BLE001
                wiped.append(f"re-seed failed: {e}")
        else:
            wiped.append("re-seed skipped (--no-reseed). Create a user with `manage.py createsuperuser`.")

        self.stdout.write(self.style.MIGRATE_HEADING("Nuke summary"))
        for line in wiped:
            self.stdout.write(f"  {line}")
        self.stdout.write(self.style.SUCCESS("Done."))

    def _reseed_smoke_user(self, *, capital: float = 500_000.0) -> list[str]:
        """Idempotent re-creation of the smoke user/tenant/portfolio.

        Mirrors apps.agents_core.tester.runner.ensure_smoke_user so the
        same credentials still work after a nuke.
        """
        from datetime import date
        out: list[str] = []
        from apps.accounts.models import User
        from apps.tenants.models import Tenant, Membership
        from apps.portfolio.models import Portfolio

        email = "smoke@alphadesk.local"
        user, _ = User.objects.get_or_create(email=email, defaults={"is_active": True})
        user.set_password("smoke-1234")
        user.save()
        out.append(f"smoke user: {email}  (password: smoke-1234)")

        tenant, _ = Tenant.objects.get_or_create(
            slug="smoke",
            defaults={"name": "Smoke Tenant", "kind": "retail"},
        )
        Membership.objects.get_or_create(
            tenant=tenant, user=user, defaults={"role": "owner"},
        )
        out.append(f"tenant: {tenant.slug}")

        portfolio, _ = Portfolio.objects.get_or_create(
            tenant=tenant, name="smoke-portfolio",
            defaults={"capital": capital, "mode": "paper"},
        )
        out.append(f"portfolio: {portfolio.name}  capital={capital}")

        # Seed a legacy snapshot too so RiskGuard has a capital to size off.
        try:
            from trading.models import PortfolioSnapshot as LegacySnap
            LegacySnap.objects.update_or_create(
                snapshot_date=date.today(),
                defaults={
                    "capital": capital,
                    "available_cash": capital,
                    "invested": 0.0,
                },
            )
            out.append(f"legacy PortfolioSnapshot seeded: {capital}")
        except Exception as e:  # noqa: BLE001
            out.append(f"legacy snapshot skipped: {e}")
        return out
