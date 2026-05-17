"""python manage.py seed_indices — populate WatchlistEntry with every
index of interest + top liquid equity constituents so the cockpits and
strategies have a non-empty universe to fan out across.

Why this command exists:
  Most cockpit panels (ORB tracker, VWAP bands, base-quality, MTF
  stages, breakout classifier, etc.) iterate the legacy WatchlistEntry
  table to pick which symbols to score. After a DB nuke that table is
  empty, so every panel returns 0 rows. This seeds a sensible default
  universe so the operator can run strategies immediately.

  Idempotent — re-running just refreshes scan_date to today and leaves
  existing rows alone. Safe to run after every nuke.

Usage:
  python manage.py seed_indices                  # full universe
  python manage.py seed_indices --indices-only   # just NIFTY/BANKNIFTY/etc.
  python manage.py seed_indices --equity-top 25  # top 25 equity names only
  python manage.py seed_indices --clear          # wipe + re-seed
"""
from __future__ import annotations

from datetime import date

from django.core.management.base import BaseCommand


# Major Indian indices the strategies care about — spots that resolve
# via ticker_service.INDEX_SPOT_TOKENS.
INDICES = [
    ("NIFTY",       "NSE", "INDEX"),
    ("BANKNIFTY",   "NSE", "INDEX"),
    ("FINNIFTY",    "NSE", "INDEX"),
    ("MIDCPNIFTY",  "NSE", "INDEX"),
    ("SENSEX",      "BSE", "INDEX"),
    ("BANKEX",      "BSE", "INDEX"),
    ("INDIAVIX",    "NSE", "INDEX"),
]

# Sector-leader equity constituents — chosen so every NIFTY sector has
# at least 2 names, enough for sector_dispersion + sector_heatmap +
# stock_rrg + base_quality etc. to render something useful.
EQUITY_UNIVERSE = [
    # Banks (highest weight in NIFTY)
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
    # IT
    "TCS", "INFY", "WIPRO", "TECHM", "HCLTECH", "LTIM",
    # Energy
    "RELIANCE", "ONGC", "BPCL", "IOC", "POWERGRID", "NTPC",
    # Auto
    "TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT",
    # Pharma
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP",
    # FMCG
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    # Metals
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA",
    # NBFC / Insurance
    "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
    # Cement / Infra
    "ULTRACEMCO", "GRASIM", "LT", "ADANIENT", "ADANIPORTS",
    # Telecom + Misc large caps
    "BHARTIARTL", "ASIANPAINT", "TITAN", "MARICO", "HEROMOTOCO",
]


class Command(BaseCommand):
    help = "Seed WatchlistEntry with major indices + sector-leader equity universe."

    def add_arguments(self, parser):
        parser.add_argument("--indices-only", action="store_true",
                            help="Skip the equity universe — seed only the index spots.")
        parser.add_argument("--equity-top", type=int, default=None,
                            help="Limit to top-N equity names (default: all 50ish).")
        parser.add_argument("--clear", action="store_true",
                            help="Wipe WatchlistEntry first, then re-seed.")

    def handle(self, *args, **opts):
        from trading.models import WatchlistEntry

        if opts["clear"]:
            n, _ = WatchlistEntry.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Cleared {n} existing rows."))

        today = date.today()
        rows = []

        # Indices first — these always seed.
        for symbol, _exchange, _kind in INDICES:
            rows.append({
                "symbol": symbol,
                "scan_date": today,
                "score": 95.0,        # high default — these are always in scope
                "bias": "NEUTRAL",
                "setups": ["INDEX_SPOT"],
            })

        # Equity universe unless --indices-only
        if not opts["indices_only"]:
            equities = EQUITY_UNIVERSE
            if opts["equity_top"]:
                equities = equities[:opts["equity_top"]]
            for sym in equities:
                rows.append({
                    "symbol": sym,
                    "scan_date": today,
                    "score": 75.0,
                    "bias": "NEUTRAL",
                    "setups": ["ORB_LONG", "PDH_BREAK", "VWAP_RECLAIM"],
                })

        added = 0
        refreshed = 0
        for row in rows:
            obj, created = WatchlistEntry.objects.update_or_create(
                symbol=row["symbol"], scan_date=row["scan_date"],
                defaults={
                    "score": row["score"],
                    "bias": row["bias"],
                    "setups": row["setups"],
                },
            )
            if created:
                added += 1
            else:
                refreshed += 1

        self.stdout.write(self.style.SUCCESS(
            f"Seeded {added} new + refreshed {refreshed} existing watchlist rows for {today.isoformat()}."
        ))
        self.stdout.write(f"Universe size: {WatchlistEntry.objects.filter(scan_date=today).count()} for today.")
        self.stdout.write(f"Total rows in table: {WatchlistEntry.objects.count()}.")
        self.stdout.write("")
        self.stdout.write("Next steps:")
        self.stdout.write("  /cockpits → every panel that iterates the watchlist now has data.")
        self.stdout.write("  python manage.py run_screener — fires real signals on this universe.")
        self.stdout.write("  python manage.py run_ai_team --all-profiles — agents see the universe.")
