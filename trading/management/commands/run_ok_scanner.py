"""
Oliver Kell Cycle Scanner — find swing/positional opportunities on daily/weekly charts.

Usage:
    # Full NIFTY 100 scan
    python manage.py run_ok_scanner

    # Specific symbols
    python manage.py run_ok_scanner --symbols RELIANCE,TCS,HDFCBANK

    # Only actionable setups (BUY/SHORT with aligned trends)
    python manage.py run_ok_scanner --actionable-only

    # With Telegram alerts
    python manage.py run_ok_scanner --telegram

    # JSON output
    python manage.py run_ok_scanner --json

    # Clear cache and rescan
    python manage.py run_ok_scanner --clear-cache

    # Custom extension threshold
    python manage.py run_ok_scanner --threshold 1.5
"""
import json as json_module
import sys
from datetime import date

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from logzero import logger

load_dotenv()


class Command(BaseCommand):
    help = "Run Oliver Kell Cycle of Price Action scanner on daily/weekly charts"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols", help="Comma-separated NSE symbols (default: NIFTY 100)"
        )
        parser.add_argument(
            "--universe",
            choices=["nifty50", "nifty100"],
            default="nifty100",
            help="Symbol universe (default: nifty100)",
        )
        parser.add_argument(
            "--date",
            help="Scan date YYYY-MM-DD (default: today)",
        )
        parser.add_argument(
            "--actionable-only",
            action="store_true",
            help="Show only BUY/SHORT signals with aligned trends",
        )
        parser.add_argument(
            "--telegram",
            action="store_true",
            help="Send Telegram alerts for actionable phases",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            help="Extension band threshold (std dev multiplier)",
        )
        parser.add_argument(
            "--clear-cache",
            action="store_true",
            help="Clear disk cache before scanning",
        )

    def handle(self, *args, **options):
        from dashboard_utils.market_scanner import (
            NIFTY_50_SYMBOLS,
            SCREENER_UNIVERSE,
        )
        from trading.config import OKCycleConfig, config as trading_config
        from trading.swing.ok_scanner import OKScanner, invalidate_cache

        # ── Clear cache if requested ──
        if options["clear_cache"]:
            invalidate_cache()
            self.stdout.write(self.style.SUCCESS("Cache cleared."))

        # ── Resolve symbols ──
        if options["symbols"]:
            symbols = [s.strip().upper() for s in options["symbols"].split(",")]
        elif options["universe"] == "nifty50":
            symbols = list(NIFTY_50_SYMBOLS)
        else:
            symbols = list(SCREENER_UNIVERSE)

        # ── Config override ──
        from dataclasses import replace
        cfg = trading_config.ok_cycle
        if options.get("threshold"):
            cfg = replace(cfg, ext_threshold=options["threshold"])

        # ── Scan date ──
        scan_date = options.get("date") or date.today().strftime("%Y-%m-%d")

        # ── Run scanner ──
        self.stdout.write(
            f"\n{'═' * 60}\n"
            f"  Oliver Kell Cycle Scanner\n"
            f"  Date: {scan_date} | Symbols: {len(symbols)} | "
            f"Extension: {cfg.ext_threshold}σ\n"
            f"{'═' * 60}\n"
        )

        scanner = OKScanner(cfg=cfg)
        results = scanner.scan(symbols, scan_date=scan_date)

        if options["actionable_only"]:
            results = scanner.get_actionable()

        # ── Output ──
        if options["json"]:
            self._output_json(results)
        else:
            self._output_table(results, scanner)

        # ── Persist actionable signals to SignalLog ──
        self._persist_signals(results, scan_date)

        # ── Telegram ──
        if options["telegram"]:
            self._send_telegram(results, scanner)

    def _persist_signals(self, results, scan_date):
        """Persist actionable cycle results to SignalLog for monthly feedback."""
        from trading.swing.ok_cycles import CyclePhase
        try:
            from trading.models import SignalLog
            from datetime import datetime

            scan_dt = datetime.strptime(scan_date, "%Y-%m-%d") if isinstance(scan_date, str) else datetime.combine(scan_date, datetime.min.time())
            actionable = [r for r in results if r.phase != CyclePhase.NONE and r.action in ("BUY", "SHORT")]
            count = 0
            for r in actionable:
                side = "BUY" if r.action == "BUY" else "SELL"
                # Use EMA levels as approximate entry/SL/target
                entry = r.last_close
                if side == "BUY":
                    stoploss = r.ema_slow if r.ema_slow > 0 else entry * 0.97
                    target = r.upper_ext if r.upper_ext > 0 else entry * 1.05
                else:
                    stoploss = r.ema_slow if r.ema_slow > 0 else entry * 1.03
                    target = r.lower_ext if r.lower_ext > 0 else entry * 0.95

                risk = abs(entry - stoploss)
                rr = abs(target - entry) / risk if risk > 0 else 0

                SignalLog.objects.get_or_create(
                    symbol=r.symbol,
                    signal_date=scan_dt.date(),
                    source=SignalLog.Source.OK_SCANNER,
                    strategy=r.phase.value,
                    defaults=dict(
                        signal_time=scan_dt,
                        side=side,
                        entry_price=entry,
                        stoploss=stoploss,
                        target=target,
                        confidence=r.confidence,
                        risk_reward=round(rr, 2),
                        reasons=[r.phase_label],
                        indicators={
                            "ema10": r.ema_fast,
                            "ema20": r.ema_mid,
                            "ema50": r.ema_slow,
                            "upper_ext": r.upper_ext,
                            "lower_ext": r.lower_ext,
                            "volume_ratio": r.volume_ratio,
                            "trend_daily": r.trend_daily.value,
                            "trend_weekly": r.trend_weekly.value,
                        },
                    ),
                )
                count += 1
            if count:
                self.stdout.write(self.style.SUCCESS(f"\n  Persisted {count} actionable signals to SignalLog"))
        except Exception as e:
            logger.warning(f"SignalLog persist failed (non-blocking): {e}")

    def _output_table(self, results, scanner):
        """Pretty CLI table output."""
        from trading.swing.ok_cycles import (
            BULLISH_ACTIONABLE,
            BEARISH_ACTIONABLE,
            CyclePhase,
            TrendState,
        )

        if not results:
            self.stdout.write(self.style.WARNING("No results."))
            return

        # Actionable results first
        actionable = [r for r in results if r.phase != CyclePhase.NONE]
        inactive = [r for r in results if r.phase == CyclePhase.NONE]

        if actionable:
            self.stdout.write(f"\n{'─' * 92}")
            self.stdout.write(
                f"{'Symbol':<14} {'Phase':<8} {'Action':<7} "
                f"{'Daily':>8} {'Weekly':>8} {'Aligned':>8} "
                f"{'Close':>10} {'Conf':>6} {'Vol':>5}"
            )
            self.stdout.write(f"{'─' * 92}")

            for r in actionable:
                # Color-code the action
                if r.phase in BULLISH_ACTIONABLE:
                    action_str = self.style.SUCCESS(f"{r.action:<7}")
                elif r.phase in BEARISH_ACTIONABLE:
                    action_str = self.style.ERROR(f"{r.action:<7}")
                elif r.phase == CyclePhase.EXHAUSTION_EXTENSION:
                    action_str = self.style.WARNING(f"{r.action:<7}")
                else:
                    action_str = f"{r.action:<7}"

                aligned_str = "  ✓" if r.aligned else "  ✗"

                self.stdout.write(
                    f"{r.symbol:<14} {r.phase.value:<8} {action_str} "
                    f"{r.trend_daily.value:>8} {r.trend_weekly.value:>8} {aligned_str:>8} "
                    f"{r.last_close:>10,.2f} {r.confidence:>5.0%} {r.volume_ratio:>5.1f}x"
                )

        # Summary
        summary = scanner.summary()
        self.stdout.write(f"\n{'─' * 92}")
        self.stdout.write(f"  Total: {len(results)} | Active phases: {len(actionable)} | No phase: {len(inactive)}")

        phase_str = " | ".join(f"{k}: {v}" for k, v in sorted(summary.items()) if k != "NONE")
        if phase_str:
            self.stdout.write(f"  Phases: {phase_str}")

        buy_count = len([r for r in results if r.phase in BULLISH_ACTIONABLE and r.aligned])
        short_count = len([r for r in results if r.phase in BEARISH_ACTIONABLE and r.aligned])
        if buy_count or short_count:
            self.stdout.write(
                self.style.SUCCESS(f"  Aligned BUY: {buy_count}") + " | " +
                self.style.ERROR(f"Aligned SHORT: {short_count}")
            )
        self.stdout.write(f"{'═' * 92}\n")

    def _output_json(self, results):
        """JSON output for piping to other tools."""
        data = []
        for r in results:
            data.append({
                "symbol": r.symbol,
                "phase": r.phase.value,
                "action": r.action,
                "trend_daily": r.trend_daily.value,
                "trend_weekly": r.trend_weekly.value,
                "aligned": r.aligned,
                "confidence": r.confidence,
                "close": r.last_close,
                "ema10": r.ema_fast,
                "ema20": r.ema_mid,
                "ema50": r.ema_slow,
                "volume_ratio": r.volume_ratio,
                "error": r.error,
            })
        self.stdout.write(json_module.dumps(data, indent=2))

    def _send_telegram(self, results, scanner):
        """Send Telegram alerts."""
        from trading.swing.ok_alerts import OKAlertService

        alerts = OKAlertService()
        if not alerts.is_configured:
            self.stdout.write(self.style.WARNING(
                "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            ))
            return

        # Send digest of all active phases
        active = [r for r in results if r.phase.value != "NONE"]
        if active:
            alerts.send_digest(active)
            self.stdout.write(self.style.SUCCESS(f"Telegram digest sent ({len(active)} active phases)"))

        # Send individual alerts for high-confidence actionable setups
        actionable = scanner.get_actionable()
        high_conf = [r for r in actionable if r.confidence >= 0.6]
        for r in high_conf:
            alerts.send_alert(r)

        if high_conf:
            self.stdout.write(self.style.SUCCESS(
                f"Telegram alerts sent for {len(high_conf)} high-confidence setups"
            ))
