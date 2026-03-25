"""
Backtest intraday structures + indicators on historical data.

Usage:
    # Quick test: last 3 trading days, top 8 stocks
    python manage.py run_backtest_intraday

    # Custom range and universe
    python manage.py run_backtest_intraday --from 2026-03-10 --to 2026-03-17 --universe nifty50

    # Parameter optimization
    python manage.py run_backtest_intraday --optimize
"""
from datetime import date, timedelta

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Backtest intraday structure detectors + indicators on real candle data"

    def add_arguments(self, parser):
        parser.add_argument("--from", dest="from_date", default=None, help="Start date YYYY-MM-DD")
        parser.add_argument("--to", dest="to_date", default=None, help="End date YYYY-MM-DD")
        parser.add_argument("--days", type=int, default=5, help="Lookback days (default: 5)")
        parser.add_argument("--universe", default="high_volume", help="Stock universe")
        parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
        parser.add_argument("--symbols", nargs="+", default=None, help="Specific symbols to test")

    def handle(self, *args, **options):
        from trading.services.intraday_backtester import (
            run_intraday_backtest, optimize_parameters, BacktestConfig,
        )
        from trading.intraday.universe import get_universe

        to_date = options["to_date"] or (date.today() - timedelta(days=1)).isoformat()
        from_date = options["from_date"] or (date.today() - timedelta(days=options["days"])).isoformat()
        symbols = options["symbols"] or get_universe(options["universe"])[:10]

        if options["optimize"]:
            self.stdout.write(self.style.SUCCESS(
                f"\nOptimizing parameters: {len(symbols)} symbols | {from_date} → {to_date}\n"
            ))
            results = optimize_parameters(symbols, from_date, to_date)

            self.stdout.write(f"\n{'Config':45s} {'Trades':>6} {'WR%':>5} {'P&L':>10} {'PF':>5} {'MaxDD':>8}")
            self.stdout.write("-" * 85)
            for r in results["runs"]:
                self.stdout.write(
                    f"{r['config']:45s} {r['trades']:>6} {r['win_rate']:>5.0f} "
                    f"{r['pnl']:>+10,.0f} {r['profit_factor']:>5.2f} {r['max_dd']:>8,.0f}"
                )
            if results["best"]:
                b = results["best"]
                self.stdout.write(self.style.SUCCESS(
                    f"\nBEST: {b['config']} | PF={b['profit_factor']:.2f} | P&L={b['pnl']:+,.0f}"
                ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"\nBacktest: {len(symbols)} symbols | {from_date} → {to_date}\n"
            ))
            results = run_intraday_backtest(symbols, from_date, to_date)
            s = results["summary"]

            self.stdout.write(f"\n{'='*70}")
            self.stdout.write(f"  Trades: {s['total_trades']} | W:{s['wins']} L:{s['losses']} | "
                              f"WR: {s['win_rate']:.0f}%")
            self.stdout.write(f"  P&L: {s['total_pnl']:+,.0f} INR | PF: {s['profit_factor']:.2f} | "
                              f"MaxDD: {s['max_drawdown']:+,.0f}")
            self.stdout.write(f"  Avg win: {s['avg_win']:+,.0f} | Avg loss: {s['avg_loss']:+,.0f}")
            self.stdout.write(f"  Targets: {s['outcomes'].get('TARGET', 0)} | "
                              f"SLs: {s['outcomes'].get('SL', 0)} | "
                              f"EOD: {s['outcomes'].get('EOD', 0)}")
            self.stdout.write(f"{'='*70}")

            self.stdout.write("\nTrades:")
            for t in results["trades"]:
                m = "+" if t.pnl_inr > 0 else "-"
                self.stdout.write(
                    f"  {m} {t.date} {t.symbol:10s} {t.setup_type:15s} {t.side:4s} "
                    f"entry={t.entry:>9.2f} exit={t.exit_price:>9.2f} {t.outcome:3s} "
                    f"P&L={t.pnl_inr:>+8,.0f} qty={t.quantity} {t.near_pivot}"
                )

            self.stdout.write("\nBy setup:")
            for stype, d in results["by_setup"].items():
                wr = (d["wins"] / d["trades"] * 100) if d["trades"] else 0
                self.stdout.write(
                    f"  {stype:15s} T={d['trades']} W={d['wins']} WR={wr:.0f}% "
                    f"P&L={d['pnl']:+,.0f} TGT={d['targets']} SL={d['sl_hits']}"
                )
