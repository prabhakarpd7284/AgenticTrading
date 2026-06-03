"""
Live Screener — find intraday trading opportunities in real-time.

Usage:
    # Live screener (websocket → polling fallback)
    python manage.py run_screener

    # Live with Telegram alerts
    python manage.py run_screener --telegram

    # Backtest strategies on historical data
    python manage.py run_screener --backtest --from 2026-03-10 --to 2026-03-20

    # Custom symbols (default: NIFTY 50)
    python manage.py run_screener --symbols RELIANCE,TCS,HDFCBANK

    # List available strategies
    python manage.py run_screener --list-strategies

    # Disable specific strategy
    python manage.py run_screener --disable "VWAP Bounce Long"

    # Polling only (skip websocket)
    python manage.py run_screener --poll-only --poll-interval 5
"""
import signal
import sys
import time
from datetime import datetime

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from logzero import logger

load_dotenv()  # Load .env for TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, etc.

from apps.market_data.constants import NIFTY_50_SYMBOLS, SCREENER_UNIVERSE


class Command(BaseCommand):
    help = "Run the live screener for intraday opportunity detection"

    def add_arguments(self, parser):
        # Mode
        parser.add_argument("--backtest", action="store_true", help="Run backtest instead of live")
        parser.add_argument("--from", dest="from_date", help="Backtest start date (YYYY-MM-DD)")
        parser.add_argument("--to", dest="to_date", help="Backtest end date (YYYY-MM-DD)")
        parser.add_argument(
            "--persist-signals", action="store_true",
            help="Backtest only: write every fired signal to apps.strategies.Signal "
                 "(populates the monthly capture matrix / signal audit / rejections).",
        )

        # Symbols
        parser.add_argument("--symbols", help="Comma-separated symbols")
        parser.add_argument("--universe", choices=["nifty50", "nifty100", "all"], default="all",
                          help="Symbol universe: nifty50, nifty100/all (default: all = NIFTY 50 + Next 50)")

        # Strategies
        parser.add_argument("--list-strategies", action="store_true", help="List all strategies")
        parser.add_argument("--disable", action="append", default=[], help="Disable a strategy by name")
        parser.add_argument("--only", help="Run only this strategy")

        # Telegram
        parser.add_argument("--telegram", action="store_true", help="Enable Telegram alerts")
        parser.add_argument("--digest", action="store_true", help="Send digest instead of individual alerts")
        parser.add_argument("--digest-interval", type=int, default=15, help="Digest interval in minutes")

        # Data source
        parser.add_argument("--poll-only", action="store_true", help="Skip websocket, use REST polling")
        parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval in seconds")

    def handle(self, *args, **options):
        from plugins.strategy_screener.strategies import STRATEGIES

        # List strategies
        if options["list_strategies"]:
            self._list_strategies(STRATEGIES)
            return

        # Resolve symbols
        symbols = self._resolve_symbols(options.get("symbols"), options.get("universe", "all"))

        # Filter strategies
        strategies = list(STRATEGIES)
        for name in options["disable"]:
            strategies = [s for s in strategies if s.name != name]
        if options.get("only"):
            strategies = [s for s in strategies if s.name == options["only"]]
            if not strategies:
                self.stderr.write(f"Strategy '{options['only']}' not found")
                return

        # Backtest mode
        if options["backtest"]:
            self._run_backtest(symbols, strategies, options)
            return

        # Live mode
        self._run_live(symbols, strategies, options)

    def _list_strategies(self, strategies):
        self.stdout.write("\n═══ Available Strategies ═══\n")
        for i, s in enumerate(strategies, 1):
            status = "ON" if s.enabled else "OFF"
            window = f"{s.active_window[0].strftime('%H:%M')}-{s.active_window[1].strftime('%H:%M')}"
            self.stdout.write(
                f"  {i}. {s.name} [{status}]\n"
                f"     {s.description}\n"
                f"     Side: {s.side} | Window: {window} | "
                f"Min R:R: {s.min_rr} | Cooldown: {s.cooldown_bars} bars\n"
                f"     Conditions: {len(s.conditions)}\n"
            )

    def _resolve_symbols(self, symbols_str, universe="all"):
        if symbols_str:
            return [s.strip() for s in symbols_str.split(",")]
        if universe == "nifty50":
            return list(NIFTY_50_SYMBOLS)
        # nifty100 / all = full screener universe
        return list(SCREENER_UNIVERSE)

    def _run_backtest(self, symbols, strategies, options):
        from plugins.strategy_backtest.compat import run_screener_backtest
        from plugins.strategy_backtest.report import ReportFormatter
        from plugins.strategy_backtest.types import PnLMode

        from_date = options.get("from_date")
        to_date = options.get("to_date")

        if not from_date or not to_date:
            self.stderr.write("Backtest requires --from and --to dates")
            return

        self.stdout.write(f"\nBacktest (v2 engine): {len(symbols)} symbols, {from_date} → {to_date}\n")
        self.stdout.write(f"Strategies: {len(strategies)}\n")

        stats = run_screener_backtest(
            symbols, from_date, to_date, strategies,
            persist_signals=options.get("persist_signals", False),
        )
        fmt = ReportFormatter("Screener Backtest", PnLMode.POINTS)
        self.stdout.write(f"\n{fmt.cli_summary(stats, {'from_date': from_date, 'to_date': to_date})}\n")

        # Send to Telegram if enabled
        if options.get("telegram"):
            from plugins.strategy_screener.telegram import TelegramAlertService
            telegram = TelegramAlertService()
            if telegram.is_configured:
                pf = "∞" if stats.profit_factor == float("inf") else f"{stats.profit_factor:.2f}"
                lines = [
                    f"📊 *Backtest Results*",
                    f"_{from_date} → {to_date} | {len(symbols)} symbols_\n",
                    f"Signals: {stats.total_signals}",
                    f"Trades: {stats.total_trades}",
                    f"Win Rate: *{stats.win_rate:.0%}* ({stats.winners}W / {stats.losers}L)",
                    f"Total P&L: *{stats.total_pnl:+.1f} pts*",
                    f"Profit Factor: *{pf}*",
                    f"Avg R:R: {stats.avg_rr:.2f}",
                    f"Max DD: {stats.max_drawdown:.1f} pts\n",
                ]
                if stats.per_phase:
                    lines.append("*Per Strategy:*")
                    for name, ps in stats.per_phase.items():
                        emoji = "✅" if ps.pnl > 0 else "❌"
                        lines.append(
                            f"{emoji} {name}: {ps.trades}T "
                            f"{ps.win_rate:.0%}W "
                            f"`{ps.pnl:+.1f}` pts"
                        )

                msg = "\n".join(lines)
                telegram._do_send(msg, "Markdown")  # sync send for backtest
                self.stdout.write("\nBacktest results sent to Telegram.\n")
            else:
                self.stderr.write("Telegram not configured\n")

    def _run_live(self, symbols, strategies, options):
        from plugins.strategy_screener.engine import ScreenerEngine
        from plugins.strategy_screener.tick_stream import TickStream
        from plugins.strategy_screener.telegram import TelegramAlertService

        self.stdout.write(f"\n═══ Live Screener ═══")
        self.stdout.write(f"\nSymbols: {len(symbols)} | Strategies: {len(strategies)}")
        self.stdout.write(f"\nMode: {'polling' if options['poll_only'] else 'websocket (polling fallback)'}\n")

        # Build engine
        engine = ScreenerEngine(symbols, strategies)

        # CLI output handler
        def cli_handler(sig):
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"\n🔔 SIGNAL @ {sig.timestamp.strftime('%H:%M:%S')}")
            self.stdout.write(f"\n{sig.format_cli()}")
            self.stdout.write(f"\n{'='*60}\n")

        engine.add_output_handler(cli_handler)

        # Persist every signal to SignalLog for monthly feedback report
        engine.add_output_handler(lambda sig: sig.persist("SCREENER"))

        # Telegram handler
        telegram = None
        if options["telegram"]:
            telegram = TelegramAlertService()
            if telegram.is_configured:
                telegram.set_engine(engine)  # enables chart rendering
                if options["digest"]:
                    engine.add_output_handler(telegram.buffer_for_digest)
                    self.stdout.write("Telegram: DIGEST mode\n")
                else:
                    engine.add_output_handler(telegram.send_signal)
                    self.stdout.write("Telegram: LIVE alerts (with charts)\n")
                telegram.send_status(
                    f"Screener started\n"
                    f"Symbols: {len(symbols)}\n"
                    f"Strategies: {len(strategies)}"
                )
            else:
                self.stderr.write("Telegram not configured (set TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)\n")

        # Bootstrap historical candles
        self.stdout.write("\nBootstrapping candle stores...\n")
        engine.bootstrap(fetch_candles_fn=True)
        self.stdout.write("Bootstrap complete.\n")

        # Start tick stream
        tick_stream = TickStream(
            symbols=symbols,
            on_tick=engine.on_tick,
            poll_interval=options["poll_interval"],
        )

        # Graceful shutdown
        running = True

        def shutdown(signum, frame):
            nonlocal running
            running = False
            tick_stream.stop()
            if telegram and telegram.is_configured:
                stats = engine.get_stats()
                telegram.send_status(
                    f"Screener stopped\n"
                    f"Bars: {stats['bars_processed']}\n"
                    f"Signals: {stats['signals_emitted']}"
                )
            self.stdout.write("\nScreener stopped.\n")
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Start streaming
        if options["poll_only"]:
            tick_stream._running = True
            tick_stream._resolve_tokens()
            tick_stream._start_polling()
        else:
            tick_stream.start()  # websocket first, polling fallback

        self.stdout.write(f"\nScreener running ({tick_stream.mode})... Press Ctrl+C to stop.\n\n")

        # Main loop — just keep alive and print periodic stats
        last_stats = time.monotonic()
        last_digest = time.monotonic()
        digest_interval = options["digest_interval"] * 60

        while running:
            time.sleep(1)

            now = time.monotonic()

            # Periodic stats (every 5 minutes)
            if now - last_stats > 300:
                stats = engine.get_stats()
                self.stdout.write(
                    f"[{datetime.now().strftime('%H:%M')}] "
                    f"Bars: {stats['bars_processed']} | "
                    f"Signals: {stats['signals_emitted']} | "
                    f"Stream: {tick_stream.mode}\n"
                )
                last_stats = now

            # Digest send
            if telegram and options["digest"] and now - last_digest > digest_interval:
                telegram.send_digest()
                last_digest = now
