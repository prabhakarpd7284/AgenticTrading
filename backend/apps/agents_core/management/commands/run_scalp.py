"""Headless scalp simulator — the deterministic test/tuning harness.

Mirrors ``run_pyramid``: build a ScalpConfig from args, fetch (or synthesize)
seconds candles, replay them through the engine, print the result.

Examples
--------
    # Deterministic dry-run (no broker needed)
    ./manage.py run_scalp --strike 23800 --type CE --underlying NIFTY --dry-run

    # Real Fyers seconds data (requires a linked, re-logged-in Fyers BrokerLink)
    ./manage.py run_scalp --strike 23800 --type CE --expiry 07JUL26 \
        --date 2026-06-24 --resolution 5S
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run the scalping strategy over seconds candles (dry-run sample or live Fyers data)."

    def add_arguments(self, parser):
        parser.add_argument("--underlying", default="NIFTY", choices=["NIFTY", "BANKNIFTY", "SENSEX"])
        parser.add_argument("--strike", type=int, required=True)
        parser.add_argument("--type", dest="opt_type", default="CE", choices=["CE", "PE"])
        parser.add_argument("--expiry", default="", help="DDMMMYY (e.g. 07JUL26); empty = next weekly")
        parser.add_argument("--date", default="", help="Session date YYYY-MM-DD; empty = last trading day")
        parser.add_argument("--resolution", default="5S", choices=["5S", "10S", "15S", "30S", "45S", "1"])
        parser.add_argument("--decision-secs", type=int, default=600)
        parser.add_argument("--tick-synthesis", default="ohlc", choices=["ohlc", "close"])
        # engine knobs
        parser.add_argument("--bin-width", type=float, default=20.0)
        parser.add_argument("--window-secs", type=int, default=180)
        parser.add_argument("--value-area", type=float, default=0.70)
        parser.add_argument("--entry-threshold", type=float, default=0.30)
        parser.add_argument("--add-threshold", type=float, default=0.22)
        parser.add_argument("--reverse-threshold", type=float, default=0.55)
        parser.add_argument("--no-trend", action="store_true", help="disable with-trend-only gating")
        parser.add_argument("--trend-span", type=int, default=5)
        parser.add_argument("--trend-flat-band", type=float, default=0.012)
        parser.add_argument("--reentry-cooldown", type=float, default=60.0)
        parser.add_argument("--no-reversal-entry", action="store_true",
                            help="enter on pressure threshold instead of a bin-reversal")
        parser.add_argument("--pullback-bins", type=float, default=0.6)
        parser.add_argument("--reversal-bins", type=float, default=0.35)
        parser.add_argument("--capital", type=float, default=100_000)
        parser.add_argument("--risk-pct", type=float, default=1.0)
        parser.add_argument("--max-pyramids", type=int, default=4)
        parser.add_argument("--lot-size", type=int, default=65)
        parser.add_argument("--no-bias-gate", action="store_true", help="disable bias alignment")
        parser.add_argument("--no-reverse", action="store_true", help="disable reverse-on-flip")
        parser.add_argument("--dry-run", action="store_true", help="use the synthetic sample (no broker)")

    def handle(self, *args, **o):
        from plugins.strategy_scalp.engine import ScalpConfig
        from plugins.strategy_scalp.replay import (
            format_result, generate_scalp_sample, resolution_to_secs, run_scalp,
        )

        res_secs = resolution_to_secs(o["resolution"])
        config = ScalpConfig(
            lot_size=o["lot_size"], initial_capital=o["capital"], initial_risk_pct=o["risk_pct"],
            max_pyramids=o["max_pyramids"], bin_width=o["bin_width"], window_secs=o["window_secs"],
            value_area_pct=o["value_area"], entry_threshold=o["entry_threshold"],
            add_threshold=o["add_threshold"], reverse_threshold=o["reverse_threshold"],
            trend_only=not o["no_trend"], trend_ema_span=o["trend_span"],
            trend_flat_band=o["trend_flat_band"], reentry_cooldown_secs=o["reentry_cooldown"],
            bin_reversal_entry=not o["no_reversal_entry"], pullback_min_bins=o["pullback_bins"],
            reversal_bins=o["reversal_bins"],
            require_bias_alignment=not o["no_bias_gate"], allow_reverse=not o["no_reverse"],
        )
        symbol = f"{o['underlying']} {o['strike']} {o['opt_type']}"

        if o["dry_run"]:
            self.stdout.write(self.style.WARNING("DRY-RUN — synthetic sample candles (no broker call)"))
            candles = generate_scalp_sample(res_secs)
            symbol += " (sample)"
        else:
            candles = self._fetch_fyers_candles(o, res_secs)

        if not candles:
            raise CommandError("No candles to run.")

        self.stdout.write(f"Loaded {len(candles)} × {o['resolution']} candles "
                          f"({candles[0].timestamp} → {candles[-1].timestamp})")
        res = run_scalp(candles, symbol=symbol, config=config,
                        resolution_secs=res_secs, decision_secs=o["decision_secs"],
                        tick_synthesis=o["tick_synthesis"])
        self.stdout.write(format_result(res))

    # ── live data fetch ────────────────────────────────────────────────
    def _fetch_fyers_candles(self, o, res_secs):
        try:
            from plugins.strategy_scalp.data import fetch_scalp_candles
        except ImportError as e:  # pragma: no cover
            raise CommandError(f"Fyers data layer unavailable: {e}. Use --dry-run.")
        try:
            return fetch_scalp_candles(
                underlying=o["underlying"], strike=o["strike"], opt_type=o["opt_type"],
                expiry=o["expiry"], date=o["date"], resolution=o["resolution"],
            )
        except Exception as e:
            raise CommandError(
                f"Fyers fetch failed ({e}). Ensure a Fyers BrokerLink is linked and re-logged-in "
                f"today, or run with --dry-run."
            )
