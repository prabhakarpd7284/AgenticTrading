"""
Greedy Aggressive Pyramiding Strategy — options momentum capture.

Fetches real 5-min candle data for an option from Angel One and runs
the pyramiding simulation.

Usage:
    # Run on NIFTY 24200 CE, auto-detect nearest expiry, today's data
    python manage.py run_pyramid --strike 24200 --type CE

    # Specific expiry and date
    python manage.py run_pyramid --strike 24200 --type CE --expiry 08MAY26 --date 2026-05-05

    # BANKNIFTY
    python manage.py run_pyramid --underlying BANKNIFTY --strike 52000 --type CE

    # Custom capital and risk
    python manage.py run_pyramid --strike 24200 --type CE --capital 200000 --risk-pct 3.0

    # With 3-min candles instead of 5-min
    python manage.py run_pyramid --strike 24200 --type CE --interval THREE_MINUTE

    # Dry run with sample data (no broker needed)
    python manage.py run_pyramid --strike 24200 --type CE --dry-run
"""
from datetime import date, datetime

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from logzero import logger

load_dotenv()


class Command(BaseCommand):
    help = "Run pyramiding strategy on option candles (live data from Angel One)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--underlying", default="NIFTY",
            help="Underlying index: NIFTY or BANKNIFTY (default: NIFTY)",
        )
        parser.add_argument(
            "--strike", type=int, required=True,
            help="Strike price (e.g. 24200)",
        )
        parser.add_argument(
            "--type", dest="opt_type", default="CE", choices=["CE", "PE"],
            help="Option type: CE or PE (default: CE)",
        )
        parser.add_argument(
            "--expiry",
            help="Expiry in any format: 08MAY26, 2026-05-08 (default: nearest)",
        )
        parser.add_argument(
            "--date",
            help="Date to fetch candles for: YYYY-MM-DD (default: today or last trading day)",
        )
        parser.add_argument(
            "--interval", default="FIVE_MINUTE",
            help="Candle interval: THREE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE (default: FIVE_MINUTE)",
        )
        parser.add_argument(
            "--capital", type=float, default=100_000,
            help="Capital for position sizing (default: 100000)",
        )
        parser.add_argument(
            "--risk-pct", type=float, default=2.0,
            help="Initial risk %% of capital (default: 2.0)",
        )
        parser.add_argument(
            "--profit-risk", type=float, default=0.80,
            help="Fraction of unrealized profit to risk on pyramids (default: 0.80)",
        )
        parser.add_argument(
            "--max-pyramids", type=int, default=5,
            help="Max pyramid add-ons (default: 5)",
        )
        parser.add_argument(
            "--lot-size", type=int, default=25,
            help="Lot size (default: 25 for NIFTY)",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Generate sample candles (no broker login needed)",
        )
        parser.add_argument(
            "--telegram", action="store_true",
            help="Send result report to Telegram",
        )

    def handle(self, *args, **options):
        from plugins.strategy_pyramid.strategy import (
            Candle, PyramidConfig, run_pyramid, format_result,
        )

        underlying = options["underlying"]
        strike = options["strike"]
        opt_type = options["opt_type"]
        interval = options["interval"]

        config = PyramidConfig(
            lot_size=options["lot_size"],
            initial_capital=options["capital"],
            initial_risk_pct=options["risk_pct"],
            profit_risk_pct=options["profit_risk"],
            max_pyramids=options["max_pyramids"],
        )

        # ── Resolve expiry ──
        expiry_str = options.get("expiry")
        if not expiry_str:
            from trading.utils.expiry_utils import next_expiry_date
            exp_date = next_expiry_date(underlying)
            if exp_date:
                from trading.utils.expiry_utils import iso_to_angel
                expiry_str = iso_to_angel(exp_date.isoformat())
                self.stdout.write(f"Auto-detected expiry: {expiry_str} ({exp_date})")
            else:
                self.stderr.write("Cannot determine next expiry. Use --expiry.")
                return

        # ── Resolve date ──
        candle_date = options.get("date")
        if not candle_date:
            from trading.utils.time_utils import get_candle_date_range
            candle_date = get_candle_date_range()[0].isoformat()
        self.stdout.write(f"Candle date: {candle_date}")

        symbol_label = f"{underlying} {strike} {opt_type} (exp {expiry_str})"

        if options["dry_run"]:
            candles = self._generate_sample_candles()
            self.stdout.write(f"DRY RUN: {len(candles)} sample candles generated")
        else:
            candles = self._fetch_option_candles(
                underlying, strike, expiry_str, opt_type,
                candle_date, interval,
            )

        if not candles:
            self.stderr.write("No candle data. Market may be closed or token not found.")
            return

        self.stdout.write(f"Loaded {len(candles)} candles for {symbol_label}")
        self.stdout.write(
            f"Range: {candles[0].timestamp} → {candles[-1].timestamp} | "
            f"Open: {candles[0].open:.2f} → Close: {candles[-1].close:.2f}"
        )

        # ── Run strategy ──
        result = run_pyramid(candles, symbol=symbol_label, config=config)
        self.stdout.write(format_result(result))

        # ── Telegram ──
        if options.get("telegram"):
            from plugins.strategy_pyramid.strategy import run_pyramid_with_chart_data
            from plugins.strategy_pyramid.telegram import send_pyramid_report
            import time

            data = run_pyramid_with_chart_data(candles, symbol=symbol_label, config=config)
            data["config"] = {
                "strike": strike, "type": opt_type, "underlying": underlying,
                "expiry": expiry_str, "date": candle_date,
                "capital": config.initial_capital, "risk_pct": config.initial_risk_pct,
                "profit_risk": config.profit_risk_pct, "dry_run": options["dry_run"],
            }
            sent = send_pyramid_report(data)
            if sent:
                self.stdout.write("Telegram report sent.")
                time.sleep(2)  # Let the async thread finish
            else:
                self.stderr.write("Telegram not configured (check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")

    def _fetch_option_candles(
        self, underlying, strike, expiry_str, opt_type, candle_date, interval,
    ):
        """Fetch real option candles from Angel One."""
        from trading.options.data_service import find_option_token
        from trading.services.data_service import BrokerClient
        from plugins.strategy_pyramid.strategy import Candle
        from trading.utils.time_utils import cap_end_time

        # Find token
        result = find_option_token(underlying, strike, expiry_str, opt_type)
        if not result:
            self.stderr.write(
                f"Token not found for {underlying} {strike} {opt_type} exp {expiry_str}"
            )
            return []

        symbol, token = result
        self.stdout.write(f"Found: {symbol} (token {token})")

        # Fetch candles
        broker = BrokerClient.get_instance()
        broker.ensure_login()

        end_str = cap_end_time(candle_date)
        raw = broker.fetch_candles(
            symbol_token=token,
            start=f"{candle_date} 09:15",
            end=end_str,
            interval=interval,
            exchange="NFO",
        )

        if not raw:
            return []

        return [Candle.from_raw(r) for r in raw]

    @staticmethod
    def _generate_sample_candles():
        """Generate synthetic momentum candles for dry-run testing.

        Simulates a real-ish options day: flat open → momentum breakout →
        sustained trending with higher lows → late fade.
        """
        import random
        from plugins.strategy_pyramid.strategy import Candle

        candles = []
        price = 180.0
        base_time = datetime(2026, 5, 5, 9, 15)

        random.seed(77)
        phases = {
            # (start_bar, end_bar): (drift, volatility, description)
            (0, 15):  (0.1, 0.8, "opening chop"),
            (15, 25): (0.8, 1.0, "breakout"),
            (25, 45): (1.2, 0.7, "momentum run — higher lows"),
            (45, 55): (0.6, 0.5, "consolidation — tight range"),
            (55, 65): (1.5, 0.9, "second leg — acceleration"),
            (65, 75): (-0.3, 1.2, "late fade"),
        }

        for i in range(75):
            minutes = i * 5
            total_min = 15 + minutes
            ts = base_time.replace(hour=9 + total_min // 60, minute=total_min % 60)
            if ts.hour >= 15 and ts.minute > 30:
                break

            # Find phase
            drift, vol = 0.1, 0.8
            for (s, e), (d, v, _) in phases.items():
                if s <= i < e:
                    drift, vol = d, v
                    break

            open_p = price
            move = drift + random.gauss(0, vol)
            close_p = open_p + move

            # Wicks — realistic: high extends above max, low below min
            wick_up = abs(random.gauss(0, vol * 0.6))
            wick_dn = abs(random.gauss(0, vol * 0.5))
            high_p = max(open_p, close_p) + wick_up
            low_p = min(open_p, close_p) - wick_dn
            volume = random.randint(8000, 60000)

            candles.append(Candle(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                open=round(open_p, 2),
                high=round(high_p, 2),
                low=round(low_p, 2),
                close=round(close_p, 2),
                volume=volume,
            ))
            price = close_p

        return candles
