"""
Oliver Kell Cycle Backtest — simulate swing trades on historical data.

Usage:
    # Last month
    python manage.py run_ok_backtest --from 2026-04-01 --to 2026-04-30

    # Last week
    python manage.py run_ok_backtest --from 2026-04-25 --to 2026-05-01

    # Specific symbols
    python manage.py run_ok_backtest --from 2026-04-01 --to 2026-04-30 --symbols RELIANCE,TCS

    # Send report to Telegram
    python manage.py run_ok_backtest --from 2026-04-01 --to 2026-04-30 --telegram

    # Custom risk settings
    python manage.py run_ok_backtest --from 2026-04-01 --to 2026-04-30 --capital 500000 --risk 1.0 --rr 2.0
"""
import json
from datetime import date

from django.core.management.base import BaseCommand
from dotenv import load_dotenv

load_dotenv()


class Command(BaseCommand):
    help = "Run Oliver Kell Cycle backtest on historical daily candles"

    def add_arguments(self, parser):
        parser.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
        parser.add_argument("--to", dest="to_date", help="End date YYYY-MM-DD (default: today)")
        parser.add_argument("--symbols", help="Comma-separated symbols (default: NIFTY 100)")
        parser.add_argument("--universe", choices=["nifty50", "nifty100"], default="nifty100")
        parser.add_argument("--capital", type=float, help="Starting capital (default: 500000)")
        parser.add_argument("--risk", type=float, help="Max risk per trade %% (default: 1.0)")
        parser.add_argument("--rr", type=float, default=2.0, help="Min risk:reward ratio (default: 2.0)")
        parser.add_argument("--max-hold", type=int, default=10, help="Max hold days (default: 10)")
        parser.add_argument("--slippage", type=float, default=0.1, help="Slippage %% (default: 0.1)")
        parser.add_argument("--telegram", action="store_true", help="Send report to Telegram")
        parser.add_argument("--json", action="store_true", help="Output as JSON")

    def handle(self, *args, **options):
        from dashboard_utils.market_scanner import NIFTY_50_SYMBOLS, SCREENER_UNIVERSE
        from trading.backtester.compat import run_ok_backtest
        from trading.backtester.report import ReportFormatter
        from trading.backtester.types import PnLMode

        # Resolve symbols
        if options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",")]
        elif options["universe"] == "nifty50":
            symbols = list(NIFTY_50_SYMBOLS)
        else:
            symbols = list(SCREENER_UNIVERSE)

        from_date = options["from_date"]
        to_date = options.get("to_date") or date.today().strftime("%Y-%m-%d")

        self.stdout.write(
            f"\n{'═' * 60}\n"
            f"  Oliver Kell Cycle Backtest (v2 Engine)\n"
            f"  {from_date} → {to_date}\n"
            f"  Symbols: {len(symbols)} | R:R: {options['rr']} | "
            f"Max hold: {options['max_hold']}d\n"
            f"{'═' * 60}\n"
        )

        stats = run_ok_backtest(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            capital=options.get("capital"),
            max_risk_pct=options.get("risk"),
            min_rr=options["rr"],
            max_hold_bars=options["max_hold"],
            slippage_pct=options["slippage"],
        )

        fmt = ReportFormatter("Oliver Kell Cycle Backtest", PnLMode.RUPEES)
        meta = {
            "from_date": from_date, "to_date": to_date,
            "capital": options.get("capital") or 500000,
            "symbols": len(symbols),
        }

        if options["json"]:
            output = {
                "from": from_date, "to": to_date,
                "capital": meta["capital"],
                "total_trades": stats.total_trades,
                "win_rate": stats.win_rate,
                "total_pnl": stats.total_pnl,
                "profit_factor": stats.profit_factor,
                "max_drawdown": stats.max_drawdown,
                "per_phase": {k: {"trades": v.trades, "win_rate": v.win_rate, "pnl": v.pnl}
                              for k, v in stats.per_phase.items()},
                "weekly_pnl": stats.weekly_pnl,
            }
            self.stdout.write(json.dumps(output, indent=2))
        else:
            self.stdout.write(fmt.cli_summary(stats, meta))

        # Telegram
        if options["telegram"]:
            self._send_telegram(fmt, stats, meta)

    def _send_telegram(self, fmt, stats, meta):
        from trading.swing.ok_alerts import OKAlertService
        import time

        service = OKAlertService()
        if not service.is_configured:
            self.stdout.write(self.style.WARNING(
                "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            ))
            return

        message = fmt.telegram_html(stats, meta)
        service._send(message, parse_mode="HTML")
        time.sleep(2)
        self.stdout.write(self.style.SUCCESS("Telegram report sent."))
