"""
Django management command: manage_straddle

Full e2e straddle lifecycle management via CLI.

Usage:
    # Register a new straddle position
    python manage.py manage_straddle --register \
        --underlying NIFTY --strike 24200 --expiry 2026-03-10 \
        --ce-symbol NIFTY10MAR2624200CE --ce-token 45482 --ce-sell 394.85 \
        --pe-symbol NIFTY10MAR2624200PE --pe-token 45483 --pe-sell 138.35 \
        --lots 1

    # Run full analysis + LLM recommendation (the main command)
    python manage.py manage_straddle --analyze --position 1

    # Show current status without LLM
    python manage.py manage_straddle --status --position 1

    # Execute a specific action directly (bypass LLM recommendation)
    python manage.py manage_straddle --execute CLOSE_BOTH --position 1

    # List all straddle positions
    python manage.py manage_straddle --list
"""
import os
import sys
import json
from datetime import date, datetime

from django.core.management.base import BaseCommand, CommandError
from logzero import logger


class Command(BaseCommand):
    help = "Manage short straddle positions — fetch data, analyze, recommend, execute"

    def add_arguments(self, parser):
        # ── Action modes ──
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--register", action="store_true", help="Register a new straddle position")
        mode.add_argument("--analyze",  action="store_true", help="Run full e2e analysis + LLM recommendation")
        mode.add_argument("--status",   action="store_true", help="Show current P&L + market snapshot (no LLM)")
        mode.add_argument("--execute",  metavar="ACTION",    help="Force-execute an action (CLOSE_BOTH, CLOSE_CE, CLOSE_PE, HOLD)")
        mode.add_argument("--list",     action="store_true", help="List all straddle positions")

        # ── Position selector ──
        parser.add_argument("--position", type=int, default=None,
                            help="StraddlePosition ID (required for --analyze/--status/--execute)")

        # ── Register options ──
        parser.add_argument("--underlying",  default="NIFTY")
        parser.add_argument("--strike",      type=int)
        parser.add_argument("--expiry",      help="YYYY-MM-DD format")
        parser.add_argument("--ce-symbol",   dest="ce_symbol")
        parser.add_argument("--ce-token",    dest="ce_token")
        parser.add_argument("--ce-sell",     dest="ce_sell_price", type=float)
        parser.add_argument("--pe-symbol",   dest="pe_symbol")
        parser.add_argument("--pe-token",    dest="pe_token")
        parser.add_argument("--pe-sell",     dest="pe_sell_price", type=float)
        parser.add_argument("--lots",        type=int, default=1)
        parser.add_argument("--lot-size",    dest="lot_size", type=int, default=65)

    def handle(self, *args, **options):
        if options["register"]:
            self._register(options)
        elif options["analyze"]:
            self._analyze(options)
        elif options["status"]:
            self._status(options)
        elif options["execute"]:
            self._execute(options["execute"], options)
        elif options["list"]:
            self._list()

    # ──────────────────────────────────────────────
    # Register a new straddle
    # ──────────────────────────────────────────────
    def _register(self, options):
        from trading.models import StraddlePosition

        required = ["strike", "expiry", "ce_symbol", "ce_token", "ce_sell_price",
                    "pe_symbol", "pe_token", "pe_sell_price"]
        for field in required:
            if not options.get(field):
                raise CommandError(f"--{field.replace('_', '-')} is required for --register")

        try:
            expiry_date = datetime.strptime(options["expiry"], "%Y-%m-%d").date()
        except ValueError:
            raise CommandError("--expiry must be in YYYY-MM-DD format (e.g. 2026-03-10)")

        pos = StraddlePosition.objects.create(
            underlying     = options["underlying"],
            strike         = options["strike"],
            expiry         = expiry_date,
            lot_size       = options["lot_size"],
            lots           = options["lots"],
            ce_symbol      = options["ce_symbol"],
            ce_token       = options["ce_token"],
            ce_sell_price  = options["ce_sell_price"],
            pe_symbol      = options["pe_symbol"],
            pe_token       = options["pe_token"],
            pe_sell_price  = options["pe_sell_price"],
            trade_date     = date.today(),
        )

        self.stdout.write(self.style.SUCCESS(
            f"\n✓ Straddle registered: ID={pos.id}\n"
            f"  {pos.underlying} {pos.strike} [{pos.expiry}]\n"
            f"  CE: {pos.ce_symbol} sold @ {pos.ce_sell_price}\n"
            f"  PE: {pos.pe_symbol} sold @ {pos.pe_sell_price}\n"
            f"  Combined premium: {pos.combined_sell_pts:.2f} pts = {pos.total_premium_sold:,.0f} INR\n"
            f"\nRun: python manage.py manage_straddle --analyze --position {pos.id}"
        ))

    # ──────────────────────────────────────────────
    # Full e2e analysis + LLM recommendation
    # ──────────────────────────────────────────────
    def _analyze(self, options):
        pos = self._get_position(options)

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"STRADDLE MANAGEMENT CYCLE")
        self.stdout.write(f"Position: {pos} | Running at {datetime.now().strftime('%H:%M:%S')}")
        self.stdout.write(f"{'='*60}\n")

        from trading.options.straddle.graph import run_straddle_workflow

        result = run_straddle_workflow(
            position_id   = pos.id,
            underlying    = pos.underlying,
            strike        = pos.strike,
            expiry        = pos.expiry.isoformat(),
            lot_size      = pos.lot_size,
            lots          = pos.lots,
            ce_symbol     = pos.ce_symbol,
            ce_token      = pos.ce_token,
            pe_symbol     = pos.pe_symbol,
            pe_token      = pos.pe_token,
            ce_sell_price = pos.ce_sell_price,
            pe_sell_price = pos.pe_sell_price,
        )

        self._print_result(result)

    # ──────────────────────────────────────────────
    # Status check — market data + P&L only (no LLM)
    # ──────────────────────────────────────────────
    def _status(self, options):
        pos = self._get_position(options)

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"POSITION STATUS (no LLM)")
        self.stdout.write(f"{'='*60}\n")

        from trading.options.data_service import OptionsDataService
        from trading.options.straddle.analyzer import analyze_straddle

        svc = OptionsDataService()
        snapshot = svc.fetch_straddle_snapshot(
            ce_symbol=pos.ce_symbol, ce_token=pos.ce_token,
            pe_symbol=pos.pe_symbol, pe_token=pos.pe_token,
            date_str=date.today().isoformat(),
        )

        nifty = snapshot.get("nifty", {})
        vix   = snapshot.get("vix", {})
        ce    = snapshot.get("ce", {})
        pe    = snapshot.get("pe", {})

        analysis = analyze_straddle(
            underlying     = pos.underlying,
            strike         = pos.strike,
            expiry         = pos.expiry.isoformat(),
            lot_size       = pos.lot_size,
            lots           = pos.lots,
            ce_sell_price  = pos.ce_sell_price,
            pe_sell_price  = pos.pe_sell_price,
            ce_ltp         = ce.get("ltp", 0),
            pe_ltp         = pe.get("ltp", 0),
            nifty_spot     = nifty.get("ltp", 0),
            nifty_prev_close = nifty.get("prev_close", 0),
            vix_current    = vix.get("ltp", 0),
            vix_prev_close = vix.get("prev_close", 0),
            candles        = snapshot.get("candles", []),
        )

        self.stdout.write(analysis.summary_text)
        self.stdout.write(f"\nStatus: {pos.status} | Action taken: {pos.action_taken}")
        if pos.management_log:
            self.stdout.write("\nManagement History:")
            for entry in pos.management_log[-5:]:
                self.stdout.write(
                    f"  {entry.get('time','?')} | {entry.get('action','?')} | "
                    f"NIFTY {entry.get('nifty','?'):.0f} | P&L {entry.get('pnl_inr',0):+,.0f} INR"
                )

    # ──────────────────────────────────────────────
    # Force-execute a specific action (bypass LLM)
    # ──────────────────────────────────────────────
    def _execute(self, action: str, options):
        valid_actions = ["CLOSE_BOTH", "CLOSE_CE", "CLOSE_PE", "HEDGE_FUTURES", "HOLD"]
        if action not in valid_actions:
            raise CommandError(f"Unknown action: {action}. Valid: {valid_actions}")

        pos = self._get_position(options)

        self.stdout.write(f"\nForce-executing: {action} on position {pos.id}")
        confirm = input(f"Confirm {action} for {pos}? [y/N]: ").strip().lower()
        if confirm != "y":
            self.stdout.write("Cancelled.")
            return

        from trading.options.straddle.graph import (
            fetch_market_data_node, analyze_position_node,
            execute_action_node, journal_action_node
        )

        state = {
            "position_id":    pos.id,
            "underlying":     pos.underlying,
            "strike":         pos.strike,
            "expiry":         pos.expiry.isoformat(),
            "lot_size":       pos.lot_size,
            "lots":           pos.lots,
            "ce_symbol":      pos.ce_symbol,
            "ce_token":       pos.ce_token,
            "pe_symbol":      pos.pe_symbol,
            "pe_token":       pos.pe_token,
            "ce_sell_price":  pos.ce_sell_price,
            "pe_sell_price":  pos.pe_sell_price,
            "recommended_action": {
                "action":     action,
                "urgency":    "IMMEDIATE",
                "ce_action":  "CLOSE" if action in ("CLOSE_BOTH", "CLOSE_CE") else "HOLD",
                "pe_action":  "CLOSE" if action in ("CLOSE_BOTH", "CLOSE_PE") else "HOLD",
                "hedge_side": "NONE",
                "hedge_lots": 0,
                "reasoning":  f"Manual force-execute: {action}",
                "confidence": 1.0,
                "key_risk":   "Manual override — no LLM validation",
            },
            "action_approved": True,
            "nifty_candles":  None,
            "market_snapshot": None,
            "analysis": None,
            "planner_raw": None,
            "validation_result": None,
            "execution_result": None,
            "journal_id": None,
            "error": None,
        }

        # Fetch market data first
        state.update(fetch_market_data_node(state))
        state.update(analyze_position_node(state))
        state.update(execute_action_node(state))
        state.update(journal_action_node(state))

        exec_result = state.get("execution_result", {})
        self.stdout.write(self.style.SUCCESS(
            f"\n✓ Executed: {action}\n"
            f"  Actions: {exec_result.get('actions_taken', [])}\n"
            f"  Mode: {exec_result.get('mode', 'paper')}\n"
            f"  P&L: {(state.get('analysis') or {}).get('net_pnl_inr', 0):+,.0f} INR"
        ))

    # ──────────────────────────────────────────────
    # List all positions
    # ──────────────────────────────────────────────
    def _list(self):
        from trading.models import StraddlePosition

        positions = StraddlePosition.objects.all()
        if not positions:
            self.stdout.write("No straddle positions found.")
            self.stdout.write("Register one: python manage.py manage_straddle --register ...")
            return

        self.stdout.write(f"\n{'ID':<5} {'Underlying':<12} {'Strike':<8} {'Expiry':<12} {'Status':<10} {'P&L (INR)':<14} {'Action'}")
        self.stdout.write("-" * 70)
        for pos in positions:
            self.stdout.write(
                f"{pos.id:<5} {pos.underlying:<12} {pos.strike:<8} "
                f"{str(pos.expiry):<12} {pos.status:<10} "
                f"{pos.current_pnl_inr:>+10,.0f}     {pos.action_taken}"
            )

    # ──────────────────────────────────────────────
    # Print full workflow result
    # ──────────────────────────────────────────────
    def _print_result(self, result: dict):
        analysis = result.get("analysis") or {}
        action   = result.get("recommended_action") or {}
        exec_res = result.get("execution_result") or {}
        validation = result.get("validation_result") or {}

        # ── Market + P&L ──
        self.stdout.write(analysis.get("summary_text", "No analysis available"))

        # ── LLM Recommendation ──
        self.stdout.write(f"\n{'─'*60}")
        self.stdout.write("MANAGEMENT RECOMMENDATION (LLM)")
        self.stdout.write(f"{'─'*60}")
        if action:
            self.stdout.write(f"  Action   : {action.get('action', '?')} [{action.get('urgency', '?')}]")
            self.stdout.write(f"  CE Leg   : {action.get('ce_action', '?')}")
            self.stdout.write(f"  PE Leg   : {action.get('pe_action', '?')}")
            if action.get("pe_stop_loss"):
                self.stdout.write(f"  PE Stop  : Close PE if > {action['pe_stop_loss']:.2f}")
            if action.get("pe_target"):
                self.stdout.write(f"  PE Target: Close PE if < {action['pe_target']:.2f}")
            if action.get("hedge_lots", 0) > 0:
                self.stdout.write(f"  Hedge    : {action.get('hedge_side')} {action['hedge_lots']}L NIFTY Futures")
            self.stdout.write(f"  Reasoning: {action.get('reasoning', '?')}")
            self.stdout.write(f"  Confidence: {action.get('confidence', 0):.2f}")
            self.stdout.write(f"  Key Risk : {action.get('key_risk', '?')}")

        # ── Validation ──
        self.stdout.write(f"\n{'─'*60}")
        self.stdout.write("VALIDATION")
        self.stdout.write(f"{'─'*60}")
        approved = validation.get("approved", False)
        status_str = self.style.SUCCESS("APPROVED") if approved else self.style.ERROR("REJECTED")
        self.stdout.write(f"  {status_str}: {validation.get('reason', '?')}")
        if validation.get("override_action"):
            self.stdout.write(f"  Overridden to: {validation['override_action']}")

        # ── Execution ──
        if exec_res and exec_res.get("actions_taken"):
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write("EXECUTION")
            self.stdout.write(f"{'─'*60}")
            mode = exec_res.get("mode", "paper")
            for act in exec_res.get("actions_taken", []):
                self.stdout.write(f"  [{mode.upper()}] {act}")

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"Journal ID: {result.get('journal_id', 'N/A')}")

    # ──────────────────────────────────────────────
    # Position loader helper
    # ──────────────────────────────────────────────
    def _get_position(self, options):
        from trading.models import StraddlePosition

        position_id = options.get("position")
        if not position_id:
            # Try the most recent active position
            pos = StraddlePosition.objects.filter(status=StraddlePosition.Status.ACTIVE).first()
            if not pos:
                raise CommandError(
                    "No active straddle position found. "
                    "Use --position <id> or --register a new one."
                )
            return pos

        try:
            return StraddlePosition.objects.get(id=position_id)
        except StraddlePosition.DoesNotExist:
            raise CommandError(f"StraddlePosition {position_id} not found.")
