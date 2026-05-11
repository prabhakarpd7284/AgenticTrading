"""
Morning Basket Strategy — assess mood, generate signals, execute with scale-in.

Usage:
    python manage.py run_morning_basket                    # Live/paper mode
    python manage.py run_morning_basket --dry-run          # Signals only
    python manage.py run_morning_basket --telegram         # Send report
    python manage.py run_morning_basket --capital 500000   # Custom capital
"""
from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from logzero import logger

load_dotenv()


class Command(BaseCommand):
    help = "Run morning basket strategy — mood assessment + equity/options signals + execution"

    def add_arguments(self, parser):
        parser.add_argument("--symbols", help="Comma-separated symbols (default: NIFTY 50)")
        parser.add_argument("--dry-run", action="store_true", help="Generate signals without executing")
        parser.add_argument("--capital", type=float, default=500000, help="Capital for position sizing")
        parser.add_argument("--telegram", action="store_true", help="Send report to Telegram")
        # Backtest mode
        parser.add_argument("--backtest", action="store_true", help="Run backtest on historical data")
        parser.add_argument("--from", dest="from_date", help="Backtest start date YYYY-MM-DD")
        parser.add_argument("--to", dest="to_date", help="Backtest end date YYYY-MM-DD")

    def handle(self, *args, **options):
        # ── Backtest mode ──
        if options.get("backtest"):
            self._run_backtest(options)
            return

        from trading.basket.config import BasketConfig
        from trading.basket.mood import MarketMoodAssessor, MarketMood
        from trading.basket.signals import BasketSignalGenerator
        from trading.basket.executor import BasketExecutor
        from trading.basket.manager import BasketPositionManager
        from trading.backtester.sizing import PositionSizer
        from trading.backtester.types import PnLMode

        cfg = BasketConfig()
        capital = options["capital"]
        dry_run = options["dry_run"]

        self.stdout.write(
            f"\n{'═' * 65}\n"
            f"  Morning Basket Strategy {'(DRY RUN)' if dry_run else ''}\n"
            f"  Capital: ₹{capital:,.0f}\n"
            f"{'═' * 65}\n"
        )

        # ── 1. Mood Assessment ──
        self.stdout.write("\n▸ Step 1: Market Mood Assessment")
        assessor = MarketMoodAssessor(cfg)
        mood = assessor.assess()

        mood_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(mood.mood.value, "⚪")
        self.stdout.write(f"  {mood_icon} {mood.mood.value} ({mood.confidence:.0%})")
        self.stdout.write(f"  A/D: {mood.advance}/{mood.decline} ({mood.ad_ratio:.2f})")
        self.stdout.write(f"  NIFTY: {mood.nifty_spot} (gap: {mood.gap_pct:+.2f}%)")
        self.stdout.write(f"  VIX: {mood.vix} ({mood.vix_tier})")
        for r in mood.reasons:
            self.stdout.write(f"    • {r}")

        if mood.mood == MarketMood.NEUTRAL:
            self.stdout.write(self.style.WARNING("\n  Mood is NEUTRAL — no basket today."))
            if options["telegram"]:
                self._send_telegram(mood, [], None, capital)
            return

        # ── 2. Signal Generation ──
        self.stdout.write("\n▸ Step 2: Signal Generation")
        gen = BasketSignalGenerator(cfg)
        signals = gen.generate(mood)

        if not signals:
            self.stdout.write(self.style.WARNING("  No signals generated."))
            if options["telegram"]:
                self._send_telegram(mood, [], None, capital)
            return

        self.stdout.write(f"  {len(signals)} signals:")
        for s in signals:
            self.stdout.write(
                f"    {s.leg_type:>6} {s.side:>5} {s.symbol:<25} "
                f"entry={s.entry_price:>10.2f} SL={s.stoploss:>10.2f} "
                f"risk={s.risk_points:.2f} phase={s.phase} conf={s.confluence}"
            )

        if dry_run:
            self.stdout.write(self.style.SUCCESS("\n  DRY RUN — no execution."))
            if options["telegram"]:
                self._send_telegram(mood, signals, None, capital)
            return

        # ── 3. Execution ──
        self.stdout.write("\n▸ Step 3: Execution (scale-in)")
        sizer = PositionSizer(capital, cfg.max_risk_per_leg_pct, 15.0, PnLMode.RUPEES)
        executor = BasketExecutor(cfg)
        manager = BasketPositionManager(cfg, executor)

        for sig in signals:
            size = sizer.compute(sig.entry_price, sig.risk_points)
            if size.quantity <= 0:
                self.stdout.write(f"  Skip {sig.symbol}: qty=0 (risk too small)")
                continue

            leg = executor.execute_signal(sig, size.quantity)
            manager.add_leg(leg)

            self.stdout.write(
                f"  {sig.symbol}: filled {leg.filled_qty}/{size.quantity} "
                f"avg={leg.avg_entry:.2f}"
            )

        summary = manager.summary()
        self.stdout.write(f"\n{'─' * 65}")
        self.stdout.write(f"  Basket open: {summary['open_legs']} legs")
        for leg in summary["legs"]:
            self.stdout.write(
                f"    {leg['symbol']:<20} {leg['side']:>5} {leg['type']:>6} "
                f"entry={leg['entry']:.2f} qty={leg['filled_qty']}"
            )
        self.stdout.write(f"{'═' * 65}\n")

        if options["telegram"]:
            self._send_telegram(mood, signals, summary, capital)

    def _send_telegram(self, mood, signals, summary, capital):
        from plugins.strategy_swing.ok_alerts import OKAlertService
        import time

        svc = OKAlertService()
        if not svc.is_configured:
            self.stdout.write(self.style.WARNING("Telegram not configured."))
            return

        mood_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(mood.mood.value, "⚪")

        lines = [
            f"<b>🛒 Morning Basket</b>",
            f"{mood_icon} <b>{mood.mood.value}</b> ({mood.confidence:.0%})",
            f"A/D: {mood.advance}/{mood.decline} | VIX: {mood.vix} | Gap: {mood.gap_pct:+.2f}%",
            "",
        ]

        if signals:
            lines.append(f"<b>Signals ({len(signals)}):</b>")
            for s in signals:
                emoji = "📈" if s.side == "BUY" else "📉"
                lines.append(
                    f"  {emoji} {s.symbol} {s.side} @ {s.entry_price:.2f} "
                    f"SL:{s.stoploss:.2f} ({s.phase})"
                )
        else:
            lines.append("No signals today.")

        if summary:
            lines.append(f"\n<b>Execution:</b> {summary['open_legs']} legs open")

        svc._send("\n".join(lines), parse_mode="HTML")
        time.sleep(2)
        self.stdout.write(self.style.SUCCESS("Telegram sent."))

    def _run_backtest(self, options):
        """Run basket backtest on historical intraday data."""
        from datetime import date
        from trading.backtester.compat import run_basket_backtest
        from trading.backtester.report import ReportFormatter
        from trading.backtester.types import PnLMode
        from dashboard_utils.market_scanner import NIFTY_50_SYMBOLS

        from_date = options.get("from_date")
        to_date = options.get("to_date") or date.today().strftime("%Y-%m-%d")

        if not from_date:
            self.stderr.write("Backtest requires --from date")
            return

        if options.get("symbols"):
            symbols = [s.strip().upper() for s in options["symbols"].split(",")]
        else:
            symbols = list(NIFTY_50_SYMBOLS)
        capital = options["capital"]

        self.stdout.write(
            f"\n{'═' * 65}\n"
            f"  Morning Basket Backtest\n"
            f"  {from_date} → {to_date} | {len(symbols)} symbols\n"
            f"  Capital: ₹{capital:,.0f}\n"
            f"{'═' * 65}\n"
        )

        stats = run_basket_backtest(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            capital=capital,
        )

        fmt = ReportFormatter("Morning Basket Backtest", PnLMode.RUPEES)
        meta = {
            "from_date": from_date, "to_date": to_date,
            "capital": capital, "symbols": len(symbols),
        }
        self.stdout.write(fmt.cli_summary(stats, meta))

        if options.get("telegram"):
            self._send_backtest_telegram(stats, meta)

    def _send_backtest_telegram(self, stats, meta):
        from plugins.strategy_swing.ok_alerts import OKAlertService
        from trading.backtester.report import ReportFormatter
        from trading.backtester.types import PnLMode
        import time

        svc = OKAlertService()
        if not svc.is_configured:
            self.stdout.write(self.style.WARNING("Telegram not configured."))
            return

        fmt = ReportFormatter("Morning Basket Backtest", PnLMode.RUPEES)
        msg = fmt.telegram_html(stats, meta)
        svc._send(msg, parse_mode="HTML")
        time.sleep(2)
        self.stdout.write(self.style.SUCCESS("Telegram backtest report sent."))
