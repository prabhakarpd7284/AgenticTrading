"""
Django management command for backtesting.

Usage:
    python manage.py run_backtest --symbol MFSL --csv data/mfsl_candles.csv
    python manage.py run_backtest --symbol MFSL --csv data/mfsl_candles.csv --capital 1000000
    python manage.py run_backtest --symbol MFSL --sample  (uses synthetic sample data)
"""
from datetime import date

from django.core.management.base import BaseCommand, CommandError
from logzero import logger


class Command(BaseCommand):
    help = "Run backtest: replay historical candles through planner + risk engine"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            required=True,
            help="Stock symbol to backtest (e.g. MFSL)",
        )
        parser.add_argument(
            "--csv",
            type=str,
            default="",
            help="Path to CSV with columns: date,open,high,low,close,volume",
        )
        parser.add_argument(
            "--capital",
            type=float,
            default=500000.0,
            help="Initial capital in INR (default: 500000)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="",
            help="LLM model override (default: uses env LLM_MODEL)",
        )
        parser.add_argument(
            "--sample",
            action="store_true",
            help="Use synthetic sample candles for quick testing",
        )

    def handle(self, *args, **options):
        from trading.services.backtester import run_backtest, run_backtest_from_csv

        symbol = options["symbol"].upper()
        capital = options["capital"]
        model = options["model"] or None

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n{'='*60}\n"
            f"  BACKTEST ENGINE\n"
            f"  Symbol: {symbol}\n"
            f"  Capital: {capital:,.0f} INR\n"
            f"{'='*60}\n"
        ))

        if options["csv"]:
            result = run_backtest_from_csv(
                symbol=symbol,
                csv_path=options["csv"],
                initial_capital=capital,
                model=model,
            )
        elif options["sample"]:
            candles = self._generate_sample_candles(symbol)
            result = run_backtest(
                symbol=symbol,
                candles=candles,
                initial_capital=capital,
                model=model,
            )
        else:
            raise CommandError(
                "Provide --csv <path> or --sample.\n"
                "  Example: python manage.py run_backtest --symbol MFSL --csv data/mfsl.csv\n"
                "  Example: python manage.py run_backtest --symbol MFSL --sample"
            )

        if "error" in result:
            self.stdout.write(self.style.ERROR(f"Backtest failed: {result['error']}"))
            return

        self._display_results(result)

    def _generate_sample_candles(self, symbol: str) -> list:
        """Generate 5 synthetic candles for quick testing."""
        import random
        random.seed(42)

        base_price = 1500.0
        candles = []
        for i in range(5):
            o = base_price + random.uniform(-20, 20)
            h = o + random.uniform(5, 30)
            l = o - random.uniform(5, 30)
            c = random.uniform(l, h)
            vol = random.randint(50000, 200000)
            candles.append({
                "date": f"2026-02-{20+i:02d}",
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": vol,
            })
            base_price = c  # next candle starts near previous close

        self.stdout.write(f"Generated {len(candles)} sample candles for {symbol}")
        return candles

    def _display_results(self, result: dict):
        s = result["summary"]
        trades = result["trades"]

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("  BACKTEST RESULTS"))
        self.stdout.write("=" * 60)

        self.stdout.write(f"  Symbol:           {s['symbol']}")
        self.stdout.write(f"  Candles:          {s['candles_processed']}")
        self.stdout.write(f"  Trades Executed:  {s['trades_executed']}")
        self.stdout.write(f"  Trades Skipped:   {s['trades_skipped']}")
        self.stdout.write(f"  Wins / Losses:    {s['wins']}W / {s['losses']}L")
        self.stdout.write(f"  Win Rate:         {s['win_rate']}%")
        self.stdout.write(f"  Total P&L:        {s['total_pnl']:+,.0f} INR")
        self.stdout.write(f"  Return:           {s['return_pct']:+.1f}%")
        self.stdout.write(f"  Max Drawdown:     {s['max_drawdown_pct']:.1f}%")
        self.stdout.write(f"  Final Capital:    {s['final_capital']:,.0f} INR")

        self.stdout.write("\n" + "-" * 60)
        self.stdout.write("  TRADE LOG:")
        self.stdout.write("-" * 60)

        for t in trades:
            if t["action"] == "TRADED":
                sim = t["simulation"]
                pnl_style = self.style.SUCCESS if t["pnl"] >= 0 else self.style.ERROR
                self.stdout.write(
                    f"  {t['candle']} | {sim['side']} {sim['qty']}x @ {sim['entry']:.2f} "
                    f"→ {sim['outcome']} @ {sim['exit_price']:.2f} | "
                    + pnl_style(f"P&L: {t['pnl']:+,.0f}")
                    + f" | Capital: {t['capital_after']:,.0f}"
                )
            elif t["action"] == "REJECTED":
                self.stdout.write(
                    self.style.WARNING(f"  {t['candle']} | REJECTED: {t['reason']}")
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f"  {t['candle']} | {t['action']}: {t.get('reason', '')}")
                )

        self.stdout.write("\n" + "=" * 60 + "\n")
