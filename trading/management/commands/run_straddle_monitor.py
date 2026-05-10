"""
Straddle Monitor — automated babysitter for active short straddle positions.

Runs continuously during market hours. Triggers the full LangGraph workflow
(fetch → analyze → LLM → validate → execute → journal) on a schedule
and on significant NIFTY moves.

Usage:
    # Run with defaults (every 15 min, 0.5% NIFTY move trigger)
    python manage.py run_straddle_monitor

    # Custom interval and move threshold
    python manage.py run_straddle_monitor --interval 20 --move-pct 0.3

    # Monitor a specific position only
    python manage.py run_straddle_monitor --position 3

    # Dry run (analyze only, never execute)
    python manage.py run_straddle_monitor --dry-run
"""
import os
import signal
import time
from datetime import date, datetime

from django.core.management.base import BaseCommand
from logzero import logger


class Command(BaseCommand):
    help = "Automated straddle monitor — runs analysis cycles during market hours"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = True

    def add_arguments(self, parser):
        parser.add_argument(
            "--interval", type=int, default=15,
            help="Minutes between scheduled analysis cycles (default: 15)",
        )
        parser.add_argument(
            "--move-pct", type=float, default=0.5, dest="move_pct",
            help="NIFTY %% move to trigger immediate re-analysis (default: 0.5)",
        )
        parser.add_argument(
            "--position", type=int, default=None,
            help="Monitor specific position ID only (default: all active)",
        )
        parser.add_argument(
            "--dry-run", action="store_true", dest="dry_run",
            help="Analyze only — skip execution even if action is approved",
        )
        parser.add_argument(
            "--no-wait", action="store_true", dest="no_wait",
            help="Don't wait for market open — start immediately (for testing)",
        )

    def handle(self, *args, **options):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        interval = options["interval"]
        move_pct = options["move_pct"]
        position_id = options["position"]
        dry_run = options["dry_run"]
        no_wait = options["no_wait"]

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"STRADDLE MONITOR STARTED\n"
            f"{'='*60}\n"
            f"  Interval     : every {interval} min\n"
            f"  Move trigger : {move_pct}% NIFTY move\n"
            f"  Position     : {'all active' if not position_id else f'ID {position_id}'}\n"
            f"  Dry run      : {dry_run}\n"
            f"  Trading mode : {os.getenv('TRADING_MODE', 'paper').upper()}\n"
            f"  Planner mode : {os.getenv('PLANNER_MODE', 'cli').upper()}\n"
            f"{'='*60}\n"
        ))

        # ── Single shared broker session via singleton ──
        # BrokerClient.get_instance() is process-wide — all OptionsDataService
        # instances, DataService, market_scanner, and graph nodes share it.
        from trading.options.data_service import OptionsDataService
        from trading.services.data_service import BrokerClient

        svc = OptionsDataService()
        BrokerClient.get_instance().ensure_login()

        self._log(f"Broker session established (singleton). Stats: {BrokerClient.get_instance().get_stats()}")

        last_nifty = 0.0
        last_cycle_time = 0.0
        # Longer poll interval for move detection — use cached data (10s TTL)
        poll_interval = 30  # seconds between NIFTY spot checks

        while self._running:
            now = datetime.now()

            # ── Market hours check ──
            if not no_wait and not self._is_market_open(now):
                if now.hour < 9 or (now.hour == 9 and now.minute < 10):
                    self._log("Pre-market. Waiting for 09:15...")
                    self._sleep(60)
                elif now.hour >= 15 and now.minute > 35:
                    self._log("Market closed for the day. Shutting down.")
                    break
                else:
                    self._sleep(30)
                continue

            # ── AI pause check ──
            from dashboard_utils.data_layer import is_ai_paused
            if is_ai_paused():
                self._log("AI trading is PAUSED. Waiting...", style="WARNING")
                self._sleep(30)
                continue

            # ── Get active positions ──
            positions = self._get_positions(position_id)
            if not positions:
                self._log("No active straddle positions. Checking again in 60s...")
                self._sleep(60)
                continue

            # ── Fetch NIFTY spot for move detection (uses TTL cache) ──
            nifty_ltp = self._fetch_nifty_quick(svc)
            if nifty_ltp == 0:
                self._log("NIFTY spot unavailable. Retrying in 30s...")
                self._sleep(30)
                continue

            # ── Determine if we should run a cycle ──
            elapsed = time.monotonic() - last_cycle_time
            scheduled = elapsed >= (interval * 60)

            # NIFTY move trigger
            move_triggered = False
            if last_nifty > 0:
                move = abs(nifty_ltp - last_nifty) / last_nifty * 100
                if move >= move_pct:
                    move_triggered = True
                    direction = "UP" if nifty_ltp > last_nifty else "DOWN"
                    self._log(
                        f"NIFTY MOVE DETECTED: {direction} {move:.2f}% "
                        f"({last_nifty:.0f} → {nifty_ltp:.0f}) — triggering immediate analysis",
                        style="WARNING",
                    )

            # Expiry-day urgency: run every 5 min in last hour
            expiry_urgency = False
            for pos in positions:
                dte = (pos.expiry - date.today()).days
                if dte <= 0 and now.hour >= 14:
                    expiry_urgency = True
                    break

            if expiry_urgency and elapsed >= 300:  # 5 min
                scheduled = True

            if not (scheduled or move_triggered):
                self._sleep(poll_interval)
                continue

            # ── Run analysis cycle for each position ──
            self._log(
                f"{'─'*50}\n"
                f"  CYCLE at {now.strftime('%H:%M:%S')} | NIFTY: {nifty_ltp:.0f} | "
                f"Positions: {len(positions)}\n"
                f"  Trigger: {'MOVE' if move_triggered else 'EXPIRY_URGENCY' if expiry_urgency else 'SCHEDULED'}\n"
                f"{'─'*50}",
            )

            for pos in positions:
                self._run_cycle(pos, dry_run, svc)

            last_nifty = nifty_ltp
            last_cycle_time = time.monotonic()

            # Pause after cycle to stay well within rate limits
            self._sleep(5)

        self.stdout.write(self.style.SUCCESS("\nStraddle monitor stopped."))

    def _run_cycle(self, pos, dry_run: bool, svc):
        """Run one full straddle management cycle for a position."""
        self._log(f"  Analyzing: {pos.underlying} {pos.strike} [{pos.expiry}] (ID={pos.id})")

        try:
            if dry_run:
                from dashboard_utils.data_layer import run_straddle_analysis
                result = run_straddle_analysis(svc, pos.id, include_candles=False)
                if "error" in result:
                    self._log(f"  ERROR: {result['error']}", style="ERROR")
                else:
                    pnl = result.get("net_pnl_inr", 0)
                    phase = result.get("market_phase", "?")
                    delta = result.get("net_delta", 0)
                    underwater = result.get("is_underwater", False)
                    self._log(
                        f"  [DRY RUN] P&L: {pnl:+,.0f} INR | Delta: {delta:+.2f} | "
                        f"Phase: {phase} | Underwater: {underwater}"
                    )
                    if underwater:
                        self._log(
                            f"  UNDERWATER — would trigger CLOSE_BOTH in live mode",
                            style="WARNING",
                        )
                return

            # Full workflow: fetch → analyze → LLM → validate → execute → journal
            from trading.options.straddle.graph import run_straddle_workflow
            result = run_straddle_workflow(
                position_id=pos.id,
                underlying=pos.underlying,
                strike=pos.strike,
                expiry=pos.expiry.isoformat(),
                lot_size=pos.lot_size,
                lots=pos.lots,
                ce_symbol=pos.ce_symbol,
                ce_token=pos.ce_token,
                pe_symbol=pos.pe_symbol,
                pe_token=pos.pe_token,
                ce_sell_price=pos.ce_sell_price,
                pe_sell_price=pos.pe_sell_price,
            )

            action = (result.get("recommended_action") or {}).get("action", "N/A")
            urgency = (result.get("recommended_action") or {}).get("urgency", "?")
            pnl = (result.get("analysis") or {}).get("net_pnl_inr", 0)
            approved = result.get("action_approved", False)
            exec_result = result.get("execution_result") or {}

            self._log(
                f"  Result: {action}/{urgency} | P&L: {pnl:+,.0f} INR | "
                f"Approved: {approved} | Executed: {exec_result.get('actions_taken', [])}"
            )

            if action == "CLOSE_BOTH" and exec_result.get("success"):
                self._log(
                    f"  POSITION CLOSED: {pos.underlying} {pos.strike} | Final P&L: {pnl:+,.0f} INR",
                    style="SUCCESS",
                )

        except Exception as e:
            self._log(f"  Cycle failed for position {pos.id}: {e}", style="ERROR")
            logger.exception(f"Straddle monitor cycle failed: {e}")

    def _get_positions(self, position_id=None):
        """Get positions to monitor."""
        from trading.models import StraddlePosition
        if position_id:
            try:
                pos = StraddlePosition.objects.get(id=position_id)
                if pos.status in ("ACTIVE", "PARTIAL", "HEDGED"):
                    return [pos]
                self._log(f"Position {position_id} is {pos.status}, not active.")
                return []
            except StraddlePosition.DoesNotExist:
                self._log(f"Position {position_id} not found.", style="ERROR")
                return []
        return list(StraddlePosition.objects.filter(
            status__in=["ACTIVE", "PARTIAL", "HEDGED"]
        ))

    def _fetch_nifty_quick(self, svc) -> float:
        """Quick NIFTY spot fetch for move detection (uses TTL cache)."""
        try:
            data = svc.fetch_nifty_spot()
            return data.get("ltp", 0)
        except Exception:
            return 0

    def _is_market_open(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=35, second=0)  # 5 min buffer after close
        return market_open <= now <= market_close

    def _sleep(self, seconds: int):
        """Interruptible sleep."""
        end = time.monotonic() + seconds
        while self._running and time.monotonic() < end:
            time.sleep(min(1, end - time.monotonic()))

    def _shutdown(self, signum, frame):
        self._log("\nShutdown signal received. Finishing current cycle...")
        self._running = False

    def _log(self, msg: str, style: str = None):
        ts = datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {msg}"
        if style == "WARNING":
            self.stdout.write(self.style.WARNING(text))
        elif style == "ERROR":
            self.stdout.write(self.style.ERROR(text))
        elif style == "SUCCESS":
            self.stdout.write(self.style.SUCCESS(text))
        else:
            self.stdout.write(text)
