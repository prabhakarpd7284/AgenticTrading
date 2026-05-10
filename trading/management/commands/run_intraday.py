"""
Management command: Run the Intraday Trading Agent.

Usage:
    # Full pipeline: premarket scan → LLM analysis → monitor → trade
    python manage.py run_intraday

    # Premarket scan only (no trading)
    python manage.py run_intraday --scan-only

    # Skip LLM analysis (fast mode, structure-only decisions)
    python manage.py run_intraday --skip-llm

    # Custom date (for testing with historical data)
    python manage.py run_intraday --date 2026-03-14

    # Use full NIFTY 50 universe
    python manage.py run_intraday --universe nifty50

    # Show today's watchlist
    python manage.py run_intraday --show-watchlist

    # Show today's intraday trades
    python manage.py run_intraday --show-trades
"""
import os
from datetime import date

from django.core.management.base import BaseCommand
from logzero import logger


class Command(BaseCommand):
    help = "Run the Intraday Trading Agent — premarket scan, monitor, and trade"

    def add_arguments(self, parser):
        parser.add_argument(
            "--date", type=str, default=None,
            help="Trading date (YYYY-MM-DD). Default: today",
        )
        parser.add_argument(
            "--universe", type=str, default="high_volume",
            choices=["nifty50", "high_volume", "banknifty"],
            help="Stock universe to scan. Default: high_volume (top 20)",
        )
        parser.add_argument(
            "--capital", type=float, default=None,
            help="Override capital. Default: from portfolio or DEFAULT_CAPITAL env",
        )
        parser.add_argument(
            "--max-positions", type=int, default=3,
            help="Max concurrent positions. Default: 3",
        )
        parser.add_argument(
            "--scan-only", action="store_true",
            help="Run premarket scan only, don't enter monitoring/trading",
        )
        parser.add_argument(
            "--skip-llm", action="store_true",
            help="Skip LLM analysis (faster, structure-only decisions)",
        )
        parser.add_argument(
            "--show-watchlist", action="store_true",
            help="Show today's watchlist from DB and exit",
        )
        parser.add_argument(
            "--show-trades", action="store_true",
            help="Show today's intraday trades from DB and exit",
        )

    def handle(self, *args, **options):
        trading_date = options["date"] or date.today().strftime("%Y-%m-%d")

        # ── Show watchlist ──
        if options["show_watchlist"]:
            self._show_watchlist(trading_date)
            return

        # ── Show trades ──
        if options["show_trades"]:
            self._show_trades(trading_date)
            return

        # ── Scan only ──
        if options["scan_only"]:
            self._run_scan_only(trading_date, options["universe"])
            return

        # ── Full pipeline ──
        capital = options["capital"] or float(os.getenv("DEFAULT_CAPITAL", "500000"))

        from trading.intraday.agent import run_intraday_agent

        state = run_intraday_agent(
            trading_date=trading_date,
            capital=capital,
            universe=options["universe"],
            skip_llm=options["skip_llm"],
            max_positions=options["max_positions"],
        )

        # Save watchlist to DB
        self._save_watchlist(state)

        # Print summary
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("INTRADAY AGENT COMPLETE"))
        self.stdout.write("=" * 60)
        self.stdout.write(f"Date: {trading_date}")
        self.stdout.write(f"Watchlist: {len(state.watchlist)} stocks")
        self.stdout.write(f"Trades: {len(state.trades_today)}")
        self.stdout.write(f"Open positions: {state.open_positions}")

        if state.premarket_analysis and not options["skip_llm"]:
            self.stdout.write(f"\nPremarket Analysis:\n{state.premarket_analysis[:800]}")

        if state.trades_today:
            self.stdout.write("\nTrades:")
            for t in state.trades_today:
                self.stdout.write(
                    f"  {t['time']} | {t['symbol']} {t['side']} {t['qty']}x "
                    f"@ {t['entry']:.2f} | SL {t['sl']:.2f} | "
                    f"Target {t['target']:.2f} | {t['setup']} | {t['status']}"
                )

    def _run_scan_only(self, trading_date: str, universe: str):
        """Run premarket scan and display watchlist."""
        from trading.intraday.scanner import PremarketScanner

        scanner = PremarketScanner(lookback_days=5, top_n=10)
        watchlist = scanner.scan(universe=universe, trading_date=trading_date)

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS(f"PREMARKET SCAN: {trading_date}"))
        self.stdout.write("=" * 60)

        if not watchlist:
            self.stdout.write(self.style.WARNING("No stocks found in scan."))
            return

        for i, s in enumerate(watchlist, 1):
            self.stdout.write(
                f"\n{i}. {self.style.SUCCESS(s.symbol)} — Score: {s.score:.0f}/100 | "
                f"Bias: {s.bias.value}"
            )
            self.stdout.write(f"   PDH: {s.prev_high:.2f} | PDL: {s.prev_low:.2f} | "
                              f"Close: {s.prev_close:.2f} | ATR: {s.prev_atr:.2f}")
            self.stdout.write(f"   Setups: {[st.value for st in s.setups]}")
            self.stdout.write(f"   Reason: {s.reason}")

        # Save to DB
        from trading.intraday.state import IntradayState
        state = IntradayState(trading_date=trading_date, watchlist=watchlist)
        self._save_watchlist(state)
        self.stdout.write(f"\nWatchlist saved to DB ({len(watchlist)} entries)")

    def _save_watchlist(self, state):
        """Persist watchlist to WatchlistEntry model."""
        from trading.models import WatchlistEntry
        from datetime import datetime

        scan_date = datetime.strptime(state.trading_date, "%Y-%m-%d").date()

        for setup in state.watchlist:
            WatchlistEntry.objects.update_or_create(
                symbol=setup.symbol,
                scan_date=scan_date,
                defaults={
                    "score": setup.score,
                    "bias": setup.bias.value,
                    "setups": [s.value for s in setup.setups],
                    "prev_high": setup.prev_high,
                    "prev_low": setup.prev_low,
                    "prev_close": setup.prev_close,
                    "prev_atr": setup.prev_atr,
                    "orb_high": setup.orb_high if setup.orb_high > 0 else None,
                    "orb_low": setup.orb_low if setup.orb_low > 0 else None,
                    "vwap": setup.vwap if setup.vwap > 0 else None,
                    "reason": setup.reason,
                },
            )

    def _show_watchlist(self, trading_date: str):
        """Display today's watchlist from DB."""
        from trading.models import WatchlistEntry
        from datetime import datetime

        scan_date = datetime.strptime(trading_date, "%Y-%m-%d").date()
        entries = WatchlistEntry.objects.filter(scan_date=scan_date).order_by("-score")

        self.stdout.write(f"\nWatchlist for {trading_date}: {entries.count()} stocks\n")

        if not entries.exists():
            self.stdout.write(self.style.WARNING("No watchlist found. Run --scan-only first."))
            return

        for i, e in enumerate(entries, 1):
            self.stdout.write(
                f"{i}. {self.style.SUCCESS(e.symbol)} — Score: {e.score:.0f} | "
                f"Bias: {e.bias} | Outcome: {e.outcome}"
            )
            self.stdout.write(f"   PDH: {e.prev_high:.2f} | PDL: {e.prev_low:.2f} | "
                              f"Close: {e.prev_close:.2f} | ATR: {e.prev_atr:.2f}")
            self.stdout.write(f"   Setups: {e.setups}")
            self.stdout.write(f"   {e.reason}")

    def _show_trades(self, trading_date: str):
        """Display today's intraday trades from journal."""
        from trading.models import TradeJournal
        from datetime import datetime

        trade_date = datetime.strptime(trading_date, "%Y-%m-%d").date()
        trades = TradeJournal.objects.filter(trade_date=trade_date).order_by("created_at")

        self.stdout.write(f"\nIntraday trades for {trading_date}: {trades.count()} trades\n")

        if not trades.exists():
            self.stdout.write(self.style.WARNING("No trades found for this date."))
            return

        for t in trades:
            status_style = self.style.SUCCESS if t.status == "PAPER" else (
                self.style.ERROR if t.status == "REJECTED" else self.style.WARNING
            )
            self.stdout.write(
                f"  {t.created_at.strftime('%H:%M:%S')} | "
                f"{t.symbol} {t.side} {t.quantity}x @ {t.entry_price:.2f} | "
                f"SL {t.stop_loss:.2f} | Target {t.target:.2f} | "
                f"{status_style(t.status)}"
            )
            if t.reasoning:
                self.stdout.write(f"    Reason: {t.reasoning[:100]}")
