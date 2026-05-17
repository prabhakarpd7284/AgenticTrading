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


EPILOG = """
Examples:
  # Register a new short straddle
  manage_straddle --register --underlying NIFTY --strike 24200 \\
      --expiry 2026-05-13 \\
      --ce-symbol NIFTY13MAY2624200CE --ce-token 41762 --ce-sell 394.85 \\
      --pe-symbol NIFTY13MAY2624200PE --pe-token 41763 --pe-sell 138.35

  # Auto-resolve legs (looks up CE/PE symbol + token from ticker_service)
  manage_straddle --register --underlying NIFTY --strike 24200 \\
      --expiry 2026-05-13 --ce-sell 394.85 --pe-sell 138.35

  # List positions (v2 ids are UUIDs — copy one for the commands below)
  manage_straddle --list

  # Snapshot only (P&L + market context, no LLM, no execution)
  manage_straddle --status --position <UUID>

  # Daily babysitting cycle — see deprecation notice; the v2 path lives in
  # the short_straddle plugin (apps.strategies.runtime.run_workflow).
  manage_straddle --analyze --position <UUID>

  # Force-close both legs immediately (skips LLM, asks for confirmation).
  # Emits an apps.events.Event row for the audit trail.
  manage_straddle --execute CLOSE_BOTH --position <UUID>

Mode banner is printed at startup. TRADING_MODE=live places real orders.
"""


def _print_mode_banner(stream):
    """Loud, unambiguous mode banner. Real money risk warrants real estate."""
    mode = (os.getenv("TRADING_MODE") or "paper").lower()
    if mode == "live":
        banner = (
            "================================================================\n"
            "[LIVE MODE - REAL MONEY] All orders will be sent to the broker.\n"
            "================================================================"
        )
    else:
        banner = (
            "----------------------------------------------------------------\n"
            "[PAPER MODE] Simulated fills only. No real orders sent.\n"
            "----------------------------------------------------------------"
        )
    stream.write(banner)


class Command(BaseCommand):
    help = "Manage short straddle positions — fetch data, analyze, recommend, execute"

    def create_parser(self, prog_name, subcommand, **kwargs):
        # Django builds the parser via this hook; override to set epilog +
        # the raw formatter that preserves our example indentation.
        import argparse
        parser = super().create_parser(prog_name, subcommand, **kwargs)
        parser.epilog = EPILOG
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        return parser

    def add_arguments(self, parser):
        # ── Action modes ──
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--register", action="store_true", help="Register a new straddle position")
        mode.add_argument("--analyze",  action="store_true", help="Run full e2e analysis + LLM recommendation")
        mode.add_argument("--status",   action="store_true", help="Show current P&L + market snapshot (no LLM)")
        mode.add_argument("--execute",  metavar="ACTION",    help="Force-execute an action (CLOSE_BOTH, CLOSE_CE, CLOSE_PE, HOLD)")
        mode.add_argument("--list",     action="store_true", help="List all straddle positions")

        # ── Position selector ──
        # v2 OptionsPosition ids are UUIDs; accept the raw string. The legacy
        # int-typed flag silently rejected every UUID via argparse before the
        # handler ever ran (see redesign-v2 audit). Validation happens in
        # _get_position by attempting the OptionsPosition.objects.get.
        parser.add_argument("--position", type=str, default=None,
                            help="OptionsPosition UUID (required for --analyze/--status/--execute)")

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
        _print_mode_banner(self.stdout)
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
        from django.db import transaction
        from apps.trading.models import OptionsPosition, OptionsLeg, Portfolio
        from apps.tenants.models import Membership

        # ── Auto-resolve legs from ticker_service when caller skipped them.
        # Common pattern: operator knows strike + expiry but doesn't want to
        # look up Angel-One tokens by hand. We resolve from the symbol master
        # and surface the result before persisting so they can sanity-check.
        if options.get("strike") and options.get("expiry"):
            for opt_type in ("CE", "PE"):
                sym_key = f"{opt_type.lower()}_symbol"
                tok_key = f"{opt_type.lower()}_token"
                if not options.get(sym_key) or not options.get(tok_key):
                    resolved = self._resolve_leg(
                        underlying=options["underlying"],
                        strike=options["strike"],
                        expiry=options["expiry"],
                        opt_type=opt_type,
                    )
                    if resolved:
                        sym, tok = resolved
                        if not options.get(sym_key):
                            options[sym_key] = sym
                        if not options.get(tok_key):
                            options[tok_key] = tok
                        self.stdout.write(self.style.WARNING(
                            f"  Auto-resolved {opt_type} leg: {sym} (token {tok})"
                        ))

        required = ["strike", "expiry", "ce_symbol", "ce_token", "ce_sell_price",
                    "pe_symbol", "pe_token", "pe_sell_price"]
        missing = [f for f in required if not options.get(f)]
        if missing:
            hint = self._registration_hint(options)
            raise CommandError(
                f"Missing required fields: {', '.join('--' + m.replace('_', '-') for m in missing)}"
                f"{hint}"
            )

        try:
            expiry_date = datetime.strptime(options["expiry"], "%Y-%m-%d").date()
        except ValueError:
            raise CommandError("--expiry must be in YYYY-MM-DD format (e.g. 2026-03-10)")

        mem = (
            Membership.objects.filter(is_active=True, role="owner")
            .select_related("tenant").first()
        )
        if mem is None:
            raise CommandError("No owner Membership found — cannot register against a tenant.")
        portfolio = (
            Portfolio.objects.filter(tenant=mem.tenant).order_by("created_at").first()
            or Portfolio.objects.create(tenant=mem.tenant, name="Default")
        )

        with transaction.atomic():
            pos = OptionsPosition.objects.create(
                tenant=mem.tenant,
                portfolio=portfolio,
                position_type=OptionsPosition.PositionType.SHORT_STRADDLE,
                underlying=options["underlying"],
                expiry=expiry_date,
                lot_size=options["lot_size"],
                lots=options["lots"],
                status=OptionsPosition.Status.ACTIVE,
                trade_date=date.today(),
            )
            # CE leg (we sold a call)
            leg_qty = int(options["lots"]) * int(options["lot_size"])
            ce = OptionsLeg.objects.create(
                position=pos,
                leg_role=OptionsLeg.LegRole.SHORT_CE,
                symbol=options["ce_symbol"],
                token=options["ce_token"],
                strike=int(options["strike"]),
                qty=leg_qty,
                open_price=options["ce_sell_price"],
            )
            # PE leg (we sold a put)
            pe = OptionsLeg.objects.create(
                position=pos,
                leg_role=OptionsLeg.LegRole.SHORT_PE,
                symbol=options["pe_symbol"],
                token=options["pe_token"],
                strike=int(options["strike"]),
                qty=leg_qty,
                open_price=options["pe_sell_price"],
            )

        combined_pts = float(ce.open_price) + float(pe.open_price)
        total_premium = combined_pts * leg_qty
        self.stdout.write(self.style.SUCCESS(
            f"\nStraddle registered: ID={pos.id}\n"
            f"  {pos.underlying} {options['strike']} [{pos.expiry}]\n"
            f"  CE: {ce.symbol} sold @ {ce.open_price}\n"
            f"  PE: {pe.symbol} sold @ {pe.open_price}\n"
            f"  Combined premium: {combined_pts:.2f} pts = {total_premium:,.0f} INR"
        ))
        self._print_next_steps([
            f"python manage.py manage_straddle --analyze --position {pos.id}",
            f"python manage.py manage_straddle --status  --position {pos.id}",
            "python manage.py run_trading_day  # auto-monitors all active straddles",
        ])

    def _resolve_leg(self, underlying, strike, expiry, opt_type):
        """Look up symbol + token for a strike/expiry via the symbol master.

        Returns (symbol, token) tuple if matched, else None. Failures are
        swallowed silently — the caller already prints the helpful error
        through _registration_hint if both legs end up unresolved.
        """
        try:
            from trading.utils.expiry_utils import iso_to_angel
            from trading.options.data_service import find_option_token
            expiry_angel = iso_to_angel(expiry)
            return find_option_token(underlying, int(strike), expiry_angel, opt_type)
        except Exception:
            return None

    def _registration_hint(self, options):
        """Surface a likely-matching strike/expiry to help the operator fix
        the command. Empty string if we can't help."""
        if not options.get("strike") or not options.get("expiry"):
            return (
                "\n\nTip: pass --strike and --expiry (YYYY-MM-DD) and we will "
                "auto-resolve the CE/PE symbol + token from the broker symbol master."
            )
        return ""

    def _print_next_steps(self, steps):
        """Print a suggested follow-up command block at the end of a flow."""
        if not steps:
            return
        self.stdout.write("\nSuggested next:")
        for s in steps:
            self.stdout.write(f"  > {s}")

    # ──────────────────────────────────────────────
    # Full e2e analysis + LLM recommendation
    # ──────────────────────────────────────────────
    def _analyze(self, options):
        pos = self._get_position(options)
        legs = self._unpack_legs(pos)
        if not (legs["ce_symbol"] and legs["pe_symbol"]):
            raise CommandError(
                f"Position {pos.id} is not a two-leg short straddle "
                f"(missing SHORT_CE or SHORT_PE leg). Use the short_straddle "
                "plugin for multi-leg strategies."
            )

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"STRADDLE MANAGEMENT CYCLE")
        self.stdout.write(f"Position: {pos} | Running at {datetime.now().strftime('%H:%M:%S')}")
        self.stdout.write(f"{'='*60}\n")

        from trading.options.straddle.graph import run_straddle_workflow

        result = run_straddle_workflow(
            position_id   = pos.id,
            underlying    = pos.underlying,
            strike        = legs["strike"] or 0,
            expiry        = pos.expiry.isoformat(),
            lot_size      = pos.lot_size,
            lots          = pos.lots,
            ce_symbol     = legs["ce_symbol"],
            ce_token      = legs["ce_token"],
            pe_symbol     = legs["pe_symbol"],
            pe_token      = legs["pe_token"],
            ce_sell_price = legs["ce_sell_price"],
            pe_sell_price = legs["pe_sell_price"],
        )

        self._print_result(result)

        # ── Surface the next likely command based on what just happened.
        action_dict = (result.get("recommended_action") or {})
        action = action_dict.get("action", "HOLD")
        validation = result.get("validation_result") or {}
        approved = validation.get("approved", False)
        steps = []
        if action == "HOLD" or not approved:
            steps.append(f"python manage.py manage_straddle --status   --position {pos.id}")
            steps.append(f"python manage.py manage_straddle --analyze  --position {pos.id}  # re-run later")
        else:
            steps.append(f"python manage.py manage_straddle --execute {action} --position {pos.id}")
            steps.append(f"python manage.py manage_straddle --status   --position {pos.id}")
        self._print_next_steps(steps)

    # ──────────────────────────────────────────────
    # Status check — market data + P&L only (no LLM)
    # ──────────────────────────────────────────────
    def _status(self, options):
        pos = self._get_position(options)
        legs = self._unpack_legs(pos)
        if not (legs["ce_symbol"] and legs["pe_symbol"]):
            raise CommandError(
                f"Position {pos.id} is not a two-leg short straddle "
                f"(missing SHORT_CE or SHORT_PE leg)."
            )

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"POSITION STATUS (no LLM)")
        self.stdout.write(f"{'='*60}\n")

        from trading.options.data_service import OptionsDataService
        from trading.options.straddle.analyzer import analyze_straddle

        svc = OptionsDataService()
        snapshot = svc.fetch_straddle_snapshot(
            ce_symbol=legs["ce_symbol"], ce_token=legs["ce_token"],
            pe_symbol=legs["pe_symbol"], pe_token=legs["pe_token"],
            date_str=date.today().isoformat(),
        )

        nifty = snapshot.get("nifty", {})
        vix   = snapshot.get("vix", {})
        ce    = snapshot.get("ce", {})
        pe    = snapshot.get("pe", {})

        analysis = analyze_straddle(
            underlying     = pos.underlying,
            strike         = legs["strike"] or 0,
            expiry         = pos.expiry.isoformat(),
            lot_size       = pos.lot_size,
            lots           = pos.lots,
            ce_sell_price  = legs["ce_sell_price"],
            pe_sell_price  = legs["pe_sell_price"],
            ce_ltp         = ce.get("ltp", 0),
            pe_ltp         = pe.get("ltp", 0),
            nifty_spot     = nifty.get("ltp", 0),
            nifty_prev_close = nifty.get("prev_close", 0),
            vix_current    = vix.get("ltp", 0),
            vix_prev_close = vix.get("prev_close", 0),
            candles        = snapshot.get("candles", []),
        )

        self.stdout.write(analysis.summary_text)
        self.stdout.write(f"\nStatus: {pos.status} | Current P&L: {float(pos.current_pnl_inr):+,.0f} INR")

        # Management history — read from the v2 Event firehose (the legacy
        # JSON management_log on the position row is gone). Quiet if no
        # events have been emitted for this position yet.
        events = self._recent_events(pos, limit=5)
        if events:
            self.stdout.write("\nRecent management events:")
            for ev in events:
                payload = ev.payload or {}
                self.stdout.write(
                    f"  {ev.ts.strftime('%H:%M')} | {ev.type} | "
                    f"{payload.get('action', payload.get('text', ev.text or '—'))}"
                )

        self._print_next_steps([
            f"python manage.py manage_straddle --analyze --position {pos.id}  # add LLM recommendation",
            f"python manage.py manage_straddle --execute CLOSE_BOTH --position {pos.id}  # square off",
        ])

    # ──────────────────────────────────────────────
    # Force-execute a specific action (bypass LLM)
    # ──────────────────────────────────────────────
    def _execute(self, action: str, options):
        valid_actions = ["CLOSE_BOTH", "CLOSE_CE", "CLOSE_PE", "HEDGE_FUTURES", "HOLD"]
        if action not in valid_actions:
            raise CommandError(f"Unknown action: {action}. Valid: {valid_actions}")

        pos = self._get_position(options)
        legs = self._unpack_legs(pos)
        if not (legs["ce_symbol"] and legs["pe_symbol"]):
            raise CommandError(
                f"Position {pos.id} is not a two-leg short straddle "
                f"(missing SHORT_CE or SHORT_PE leg)."
            )

        self.stdout.write(f"\nForce-executing: {action} on position {pos.id}")
        confirm = input(f"Confirm {action} for {pos}? [y/N]: ").strip().lower()
        if confirm != "y":
            self.stdout.write("Cancelled.")
            return

        # NOTE: deliberately do NOT import `journal_action_node` here. That
        # node still writes to legacy `StraddlePosition.ce_current_price /
        # management_log` fields that v2 `apps.trading.OptionsPosition` does
        # not have, and its try/except swallows the AttributeError silently
        # — leaving the operator to think the execute succeeded when the
        # state-update never landed. We emit a v2 Event row at the bottom
        # of this method instead. The full journal-update rewrite belongs
        # in the short_straddle plugin.
        from trading.options.straddle.graph import (
            fetch_market_data_node, analyze_position_node,
            execute_action_node,
        )

        state = {
            "position_id":    pos.id,
            "underlying":     pos.underlying,
            "strike":         legs["strike"] or 0,
            "expiry":         pos.expiry.isoformat(),
            "lot_size":       pos.lot_size,
            "lots":           pos.lots,
            "ce_symbol":      legs["ce_symbol"],
            "ce_token":       legs["ce_token"],
            "pe_symbol":      legs["pe_symbol"],
            "pe_token":       legs["pe_token"],
            "ce_sell_price":  legs["ce_sell_price"],
            "pe_sell_price":  legs["pe_sell_price"],
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

        # Persist the action to the v2 Event firehose so the audit trail
        # picks it up (replaces the legacy management_log JSON write).
        self._record_execute_event(pos, action, state)

        exec_result = state.get("execution_result", {})
        self.stdout.write(self.style.SUCCESS(
            f"\nExecuted: {action}\n"
            f"  Actions: {exec_result.get('actions_taken', [])}\n"
            f"  Mode: {exec_result.get('mode', 'paper')}\n"
            f"  P&L: {(state.get('analysis') or {}).get('net_pnl_inr', 0):+,.0f} INR"
        ))
        # CLOSE_BOTH typically ends the position lifecycle; surface the
        # right follow-up depending on what we just did.
        if action in ("CLOSE_BOTH", "CLOSE_CE", "CLOSE_PE"):
            self._print_next_steps([
                "python manage.py manage_straddle --list",
                f"python manage.py manage_straddle --status --position {pos.id}",
            ])
        else:
            self._print_next_steps([
                f"python manage.py manage_straddle --status --position {pos.id}",
                f"python manage.py manage_straddle --analyze --position {pos.id}  # re-run cycle",
            ])

    # ──────────────────────────────────────────────
    # List all positions
    # ──────────────────────────────────────────────
    def _list(self):
        from apps.trading.models import OptionsPosition

        positions = list(OptionsPosition.objects.all().prefetch_related("legs"))
        if not positions:
            self.stdout.write("No options positions found.")
            self.stdout.write("Register one: python manage.py manage_straddle --register ...")
            return

        self.stdout.write(
            f"\n{'ID':<38} {'Type':<18} {'Underlying':<12} {'Expiry':<12} "
            f"{'Status':<10} {'P&L (INR)':<14}"
        )
        self.stdout.write("-" * 110)
        active_ids = []
        for pos in positions:
            # Best-effort display strike — pick the first leg's strike if all
            # legs share one (true for straddles), else "various".
            strikes = sorted({leg.strike for leg in pos.legs.all() if leg.strike})
            strike_label = str(strikes[0]) if len(strikes) == 1 else "/".join(map(str, strikes)) or "—"
            self.stdout.write(
                f"{str(pos.id):<38} {pos.position_type:<18} "
                f"{pos.underlying} {strike_label:<6} "
                f"{str(pos.expiry):<12} {pos.status:<10} "
                f"{float(pos.current_pnl_inr):>+10,.0f}"
            )
            if pos.status == OptionsPosition.Status.ACTIVE:
                active_ids.append(pos.id)

        if active_ids:
            steps = [
                f"python manage.py manage_straddle --analyze --position {pid}"
                for pid in active_ids[:3]
            ]
            self._print_next_steps(steps)
        else:
            self._print_next_steps([
                "python manage.py manage_straddle --register --underlying NIFTY "
                "--strike 24200 --expiry 2026-05-13 --ce-sell 394.85 --pe-sell 138.35",
            ])

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
        from apps.trading.models import OptionsPosition

        position_id = options.get("position")
        if not position_id:
            # Most recent ACTIVE OptionsPosition across all tenants.
            pos = (
                OptionsPosition.objects
                .filter(status=OptionsPosition.Status.ACTIVE)
                .order_by("-created_at")
                .first()
            )
            if not pos:
                raise CommandError(
                    "No active options position found. "
                    "Use --position <id> or --register a new one."
                )
            return pos

        try:
            return OptionsPosition.objects.get(id=position_id)
        except OptionsPosition.DoesNotExist:
            raise CommandError(f"OptionsPosition {position_id} not found.")
        except (ValueError, Exception) as e:
            # UUID parse errors raise ValidationError under the hood; surface
            # a friendlier message instead of a stack trace.
            raise CommandError(
                f"Invalid position id {position_id!r}: {e}. "
                "v2 OptionsPosition ids are UUIDs — copy one from --list."
            )

    # ──────────────────────────────────────────────
    # Leg unpacker — bridges the legacy flat-shape CLI to the v2 multi-leg model
    # ──────────────────────────────────────────────
    def _unpack_legs(self, pos):
        """Return a flat dict of (strike, ce_symbol, ce_token, ce_sell_price,
        pe_symbol, pe_token, pe_sell_price) derived from `pos.legs.all()`.

        The v2 `apps.trading.OptionsPosition` keeps strike + entry price on
        per-leg `OptionsLeg` rows (so spreads/condors fit the same schema).
        The straddle CLI predates that change — it still uses the legacy
        flat-shape kwargs (`ce_symbol=`, `ce_token=`, `strike=`, ...) that
        the analyzer/data-service/graph accept. This helper is the bridge.

        Returns None for missing leg sides (e.g. one-legged custom position)
        so callers can detect "this isn't a 2-leg straddle" gracefully.
        """
        from apps.trading.models import OptionsLeg

        out = {
            "strike": None,
            "ce_symbol": "", "ce_token": "", "ce_sell_price": 0.0,
            "pe_symbol": "", "pe_token": "", "pe_sell_price": 0.0,
        }
        for leg in pos.legs.all():
            if leg.leg_role == OptionsLeg.LegRole.SHORT_CE:
                out["ce_symbol"] = leg.symbol
                out["ce_token"] = leg.token
                out["ce_sell_price"] = float(leg.open_price)
                if out["strike"] is None and leg.strike:
                    out["strike"] = leg.strike
            elif leg.leg_role == OptionsLeg.LegRole.SHORT_PE:
                out["pe_symbol"] = leg.symbol
                out["pe_token"] = leg.token
                out["pe_sell_price"] = float(leg.open_price)
                if out["strike"] is None and leg.strike:
                    out["strike"] = leg.strike
        return out

    def _record_execute_event(self, pos, action: str, state: dict) -> None:
        """Write the executed action to `apps.events.Event` so the Now feed
        and `--status` history surface it. Non-fatal — a failed audit
        write must not mask a successful (or failed) order placement.
        """
        try:
            from apps.events.models import Event
            from apps.events.services.event_writer import emit
            exec_result = state.get("execution_result") or {}
            analysis = state.get("analysis") or {}
            # Map operator action → semantic event type. CLOSE_* collapses
            # to STRADDLE_CLOSED (the position is no longer a 2-leg short);
            # HEDGE_FUTURES → STRADDLE_HEDGED; HOLD is a no-op so we skip
            # emitting to avoid polluting the firehose.
            if action == "HOLD":
                return
            ev_type = (
                Event.Type.STRADDLE_HEDGED
                if action == "HEDGE_FUTURES"
                else Event.Type.STRADDLE_CLOSED
            )
            emit(
                tenant=pos.tenant,
                type=ev_type,
                actor_kind=Event.ActorKind.SYSTEM,
                text=f"manage_straddle --execute {action} on {pos.underlying} {pos.expiry}",
                payload={
                    "source": "manage_straddle",
                    "position_id": str(pos.id),
                    "action": action,
                    "actions_taken": exec_result.get("actions_taken", []),
                    "success": exec_result.get("success"),
                    "mode": exec_result.get("mode", "paper"),
                    "net_pnl_inr": analysis.get("net_pnl_inr"),
                    "nifty_spot": analysis.get("nifty_spot"),
                },
            )
        except Exception as e:
            logger.warning(f"manage_straddle event emit failed (non-blocking): {e}")

    def _recent_events(self, pos, limit: int = 5) -> list:
        """Best-effort: pull recent `apps.events.Event` rows tied to this
        position so `--status` can show a management history.

        The legacy `pos.management_log` JSON field is gone in v2 — the per-leg
        action stream now lives in the unified Event firehose. Failures here
        are non-fatal; the CLI prints "no recent management events" instead.
        """
        try:
            from apps.events.models import Event
            qs = Event.objects.filter(
                tenant=pos.tenant,
                payload__contains={"position_id": str(pos.id)},
            ).order_by("-ts")[:limit]
            return list(qs)
        except Exception:
            return []
