"""
Django management command to run the agentic trading workflow.

Usage:
    python manage.py run_trading_agent "Plan a BUY trade for HDFCBANK"
    python manage.py run_trading_agent --symbol HDFCBANK "Plan a BUY trade"
    python manage.py run_trading_agent --show-journal

This is the command the Setup page's "Ask planner" OpButton fires, and
the one exposed in the Ops Console catalog.

The workflow:
    fetch_data → retrieve_context → planner (LLM) → risk → execute → journal
                                                            ↑
                              writes a v2 apps.trading.Trade row
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run the agentic trading workflow"

    def add_arguments(self, parser):
        parser.add_argument(
            "intent",
            nargs="?",
            default="",
            help="Trading intent, e.g. 'Plan a BUY trade for HDFCBANK'",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default="",
            help="Explicit stock symbol (e.g. HDFCBANK)",
        )
        parser.add_argument(
            "--show-journal",
            action="store_true",
            help="Show the most recent v2 Trade rows instead of running a workflow",
        )

    def handle(self, *args, **options):
        if options["show_journal"]:
            self._show_journal()
            return

        intent = options["intent"]
        if not intent:
            raise CommandError(
                "Provide a trading intent, e.g.:\n"
                "  python manage.py run_trading_agent 'Plan a BUY trade for HDFCBANK'\n"
                "  python manage.py run_trading_agent --show-journal"
            )

        from trading.graph.trading_graph import run_trading_workflow

        symbol = options["symbol"]

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n{'=' * 60}\n"
            f"  AGENTIC TRADING WORKFLOW\n"
            f"  Intent: {intent}\n"
            f"  Symbol: {symbol or '(auto-detect)'}\n"
            f"{'=' * 60}\n"
        ))

        result = run_trading_workflow(user_intent=intent, symbol=symbol)
        self._display_results(result)

    # -------------------------------------------------------------------- #
    # output                                                               #
    # -------------------------------------------------------------------- #
    def _display_results(self, result: dict) -> None:
        self.stdout.write("\n" + "─" * 60)

        plan = result.get("trade_plan")
        if plan:
            self.stdout.write(self.style.SUCCESS("\n📋 TRADE PLAN:"))
            self.stdout.write(f"  Symbol:     {plan['symbol']}")
            self.stdout.write(f"  Side:       {plan['side']}")
            self.stdout.write(f"  Entry:      {plan['entry_price']:.2f}")
            self.stdout.write(f"  SL:         {plan['stop_loss']:.2f}")
            self.stdout.write(f"  Target:     {plan['target']:.2f}")
            self.stdout.write(f"  Quantity:   {plan['quantity']}")
            self.stdout.write(f"  Confidence: {plan['confidence']:.2f}")
            self.stdout.write(f"  Reasoning:  {plan.get('reasoning', '(none)')}")
        else:
            self.stdout.write(self.style.WARNING("\n⚠ No trade plan generated"))
            if result.get("error"):
                self.stdout.write(f"  Error: {result['error']}")

        risk = result.get("risk_result", {})
        approved = result.get("risk_approved", False)
        if risk:
            style = self.style.SUCCESS if approved else self.style.ERROR
            self.stdout.write(style(
                f"\n🛡 RISK: {'APPROVED' if approved else 'REJECTED'}"
            ))
            self.stdout.write(f"  Reason: {risk.get('reason', 'N/A')}")
            if risk.get("risk_amount"):
                self.stdout.write(f"  Risk Amount:        {risk['risk_amount']:.0f} INR")
                self.stdout.write(f"  Risk % of Capital:  {risk.get('risk_pct_of_capital', 0):.1f}%")

        exec_r = result.get("execution_result", {}) or {}
        if exec_r.get("success"):
            self.stdout.write(self.style.SUCCESS(
                f"\n✅ EXECUTED [{exec_r.get('mode', 'paper').upper()}]"
            ))
            self.stdout.write(f"  Order ID: {exec_r.get('order_id', '?')}")
            if exec_r.get("fill_price") is not None:
                self.stdout.write(
                    f"  Fill: {exec_r.get('fill_quantity', '?')}x @ {exec_r['fill_price']:.2f}"
                )

        jid = result.get("journal_id")
        if jid:
            self.stdout.write(self.style.SUCCESS(f"\n📓 Trade row: {jid}"))

        self.stdout.write("\n" + "─" * 60 + "\n")

    # -------------------------------------------------------------------- #
    # --show-journal                                                       #
    # -------------------------------------------------------------------- #
    def _show_journal(self) -> None:
        from apps.trading.models import Trade

        entries = Trade.objects.order_by("-created_at")[:20]
        if not entries:
            self.stdout.write("No trades in the v2 ledger yet.")
            return

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n{'=' * 90}\n  TRADE LEDGER (last {len(entries)} entries — apps.trading.Trade)\n{'=' * 90}"
        ))
        for t in entries:
            pnl_str = f"{t.realized_pnl:+,.0f} INR" if t.realized_pnl is not None else "open"
            risk_str = "✅" if t.risk_approved else "❌"
            self.stdout.write(
                f"  {t.created_at.strftime('%Y-%m-%d %H:%M')} | "
                f"{t.side:4s} {t.quantity:>5d}x {t.symbol:<14s} @ {float(t.entry_price):>8.2f} | "
                f"SL {float(t.stop_loss):>8.2f}  T {float(t.target):>8.2f} | "
                f"{t.status:<10s} {risk_str} | P&L: {pnl_str}"
            )
        self.stdout.write("")
