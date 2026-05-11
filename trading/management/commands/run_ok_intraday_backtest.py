"""
OK Intraday Multi-TF Backtest — compare 3m/5m/15m with risk optimization.

Usage:
    # Full optimization grid (3 TFs × 3 SL × 3 RR = 27 combos)
    python manage.py run_ok_intraday_backtest --from 2026-04-25 --to 2026-05-01

    # Specific TF only
    python manage.py run_ok_intraday_backtest --from 2026-04-28 --to 2026-04-30 --tf 5m

    # Custom risk grid
    python manage.py run_ok_intraday_backtest --from 2026-04-25 --to 2026-05-01 --sl 1.0,1.5 --rr 2.0,2.5

    # With Telegram
    python manage.py run_ok_intraday_backtest --from 2026-04-25 --to 2026-05-01 --telegram
"""
from datetime import date

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from logzero import logger

load_dotenv()


class Command(BaseCommand):
    help = "Run OK intraday backtest across 3m/5m/15m with risk optimization (v2 engine)"

    def add_arguments(self, parser):
        parser.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
        parser.add_argument("--to", dest="to_date", help="End date (default: today)")
        parser.add_argument("--symbols", help="Comma-separated symbols (default: NIFTY 50)")
        parser.add_argument("--tf", help="Timeframes: 3m,5m,15m (default: all three)")
        parser.add_argument("--sl", help="SL ATR multipliers: 1.0,1.5,2.0 (default)")
        parser.add_argument("--rr", help="R:R ratios: 1.5,2.0,2.5 (default)")
        parser.add_argument("--capital", type=float, help="Starting capital")
        parser.add_argument("--cooldown", type=int, default=5, help="Min bars between trades (default: 5)")
        parser.add_argument("--telegram", action="store_true", help="Send report to Telegram")

    def handle(self, *args, **options):
        from apps.market_data.constants import NIFTY_50_SYMBOLS
        from trading.backtester.compat import run_intraday_backtest
        from trading.backtester.report import ReportFormatter
        from trading.backtester.types import PnLMode

        # Resolve args
        if options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",")]
        else:
            symbols = list(NIFTY_50_SYMBOLS)

        from_date = options["from_date"]
        to_date = options.get("to_date") or date.today().strftime("%Y-%m-%d")

        tfs = options["tf"].split(",") if options.get("tf") else None
        sl_mults = [float(x) for x in options["sl"].split(",")] if options.get("sl") else None
        rr_ratios = [float(x) for x in options["rr"].split(",")] if options.get("rr") else None

        tf_str = ", ".join(tfs or ["3m", "5m", "15m"])
        sl_str = ", ".join(str(x) for x in (sl_mults or [1.0, 1.5, 2.0]))
        rr_str = ", ".join(str(x) for x in (rr_ratios or [1.5, 2.0, 2.5]))

        self.stdout.write(
            f"\n{'═' * 70}\n"
            f"  OK Intraday Multi-TF Backtest (v2 Engine)\n"
            f"  {from_date} → {to_date} | {len(symbols)} symbols\n"
            f"  TFs: {tf_str}\n"
            f"  SL ATR: {sl_str} | R:R: {rr_str}\n"
            f"{'═' * 70}\n"
        )

        results = run_intraday_backtest(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            timeframes=tfs,
            sl_atr_mults=sl_mults,
            rr_ratios=rr_ratios,
            capital=options.get("capital"),
            cooldown_bars=options["cooldown"],
        )

        # Display grid
        self.stdout.write(
            f"\n{'TF':>4} {'SL':>5} {'RR':>4} {'Trades':>7} {'Win%':>6} "
            f"{'PF':>6} {'P&L':>12} {'MaxDD':>10}"
        )
        self.stdout.write(f"{'─' * 65}")

        best = None
        for r in sorted(results, key=lambda x: -x["stats"].profit_factor):
            s = r["stats"]
            if s.total_trades == 0:
                continue
            if best is None or (s.total_trades >= 5 and s.profit_factor > (best["stats"].profit_factor if best["stats"].total_trades >= 5 else 0)):
                best = r
            self.stdout.write(
                f"{r['tf']:>4} {r['sl_atr']:>5.1f} {r['rr']:>4.1f} "
                f"{s.total_trades:>7} {s.win_rate:>5.0%} "
                f"{s.profit_factor:>6.2f} ₹{s.total_pnl:>+10,.0f} "
                f"₹{s.max_drawdown:>8,.0f}"
            )

        if best:
            b = best["stats"]
            self.stdout.write(f"{'─' * 65}")
            self.stdout.write(
                f"  BEST: {best['tf']} | SL {best['sl_atr']} ATR | RR {best['rr']} | "
                f"PF {b.profit_factor:.2f} | {b.win_rate:.0%} win | ₹{b.total_pnl:+,.0f}"
            )
        self.stdout.write(f"{'═' * 65}\n")

        if options["telegram"]:
            self._send_telegram(results, from_date, to_date, len(symbols))

    def _send_telegram(self, results, from_date, to_date, n_symbols):
        from plugins.strategy_swing.ok_alerts import OKAlertService
        import time

        svc = OKAlertService()
        if not svc.is_configured:
            self.stdout.write(self.style.WARNING(
                "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            ))
            return

        def r(n):
            sign = "+" if n >= 0 else "-"
            return f"{sign}₹{abs(n):,.0f}"

        lines = [
            f"<b>OK Intraday Multi-TF Backtest (v2)</b>",
            f"<i>{from_date} → {to_date}</i> | {n_symbols} symbols\n",
        ]

        sorted_results = sorted(
            [x for x in results if x["stats"].total_trades > 0],
            key=lambda x: -x["stats"].profit_factor,
        )

        for res in sorted_results[:10]:
            s = res["stats"]
            icon = "🟢" if s.profit_factor >= 1.5 else "🟡" if s.profit_factor >= 1.0 else "🔴"
            lines.append(
                f"{icon} <b>{res['tf']} SL:{res['sl_atr']} RR:{res['rr']}</b> "
                f"{s.total_trades}t {s.win_rate:.0%}W PF:{s.profit_factor:.2f} {r(s.total_pnl)}"
            )

        best = sorted_results[0] if sorted_results else None
        if best:
            b = best["stats"]
            lines.append(
                f"\n<b>Best: {best['tf']} SL:{best['sl_atr']} RR:{best['rr']} "
                f"PF:{b.profit_factor:.2f} {r(b.total_pnl)}</b>"
            )

        svc._send("\n".join(lines), parse_mode="HTML")
        time.sleep(2)
        self.stdout.write(self.style.SUCCESS("Telegram report sent."))
