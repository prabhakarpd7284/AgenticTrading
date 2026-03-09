"""
Django management command to run the trading agent.

Usage:
    python manage.py run_trading_agent "Plan a BUY trade for MFSL"
    python manage.py run_trading_agent --symbol MFSL "Analyze MFSL and create a trade plan"
    python manage.py run_trading_agent --seed-strategies
    python manage.py run_trading_agent --show-journal
    python manage.py run_trading_agent --init-portfolio 500000
"""
import json
from datetime import date

from django.core.management.base import BaseCommand, CommandError
from logzero import logger


class Command(BaseCommand):
    help = "Run the agentic trading workflow"

    def add_arguments(self, parser):
        parser.add_argument(
            "intent",
            nargs="?",
            default="",
            help="Trading intent, e.g. 'Plan a BUY trade for MFSL'",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default="",
            help="Explicit stock symbol (e.g. MFSL)",
        )
        parser.add_argument(
            "--seed-strategies",
            action="store_true",
            help="Seed default strategy documents into the database",
        )
        parser.add_argument(
            "--init-portfolio",
            type=float,
            default=0,
            help="Initialize a portfolio snapshot with given capital (INR)",
        )
        parser.add_argument(
            "--show-journal",
            action="store_true",
            help="Show recent trade journal entries",
        )

    def handle(self, *args, **options):
        # ── Seed strategies ──
        if options["seed_strategies"]:
            self._seed_strategies()
            return

        # ── Init portfolio ──
        if options["init_portfolio"] > 0:
            self._init_portfolio(options["init_portfolio"])
            return

        # ── Show journal ──
        if options["show_journal"]:
            self._show_journal()
            return

        # ── Run trading workflow ──
        intent = options["intent"]
        if not intent:
            raise CommandError(
                "Provide a trading intent, e.g.:\n"
                "  python manage.py run_trading_agent 'Plan a BUY trade for MFSL'\n"
                "  python manage.py run_trading_agent --seed-strategies\n"
                "  python manage.py run_trading_agent --init-portfolio 500000"
            )

        from trading.graph.trading_graph import run_trading_workflow

        symbol = options["symbol"]

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n{'='*60}\n"
            f"  AGENTIC TRADING WORKFLOW\n"
            f"  Intent: {intent}\n"
            f"  Symbol: {symbol or '(auto-detect)'}\n"
            f"{'='*60}\n"
        ))

        result = run_trading_workflow(user_intent=intent, symbol=symbol)

        # ── Display results ──
        self._display_results(result)

    def _display_results(self, result: dict):
        self.stdout.write("\n" + "─" * 60)

        # Trade plan
        plan = result.get("trade_plan")
        if plan:
            self.stdout.write(self.style.SUCCESS("\n📋 TRADE PLAN:"))
            self.stdout.write(f"  Symbol:   {plan['symbol']}")
            self.stdout.write(f"  Side:     {plan['side']}")
            self.stdout.write(f"  Entry:    {plan['entry_price']:.2f}")
            self.stdout.write(f"  SL:       {plan['stop_loss']:.2f}")
            self.stdout.write(f"  Target:   {plan['target']:.2f}")
            self.stdout.write(f"  Quantity: {plan['quantity']}")
            self.stdout.write(f"  Confidence: {plan['confidence']:.2f}")
            self.stdout.write(f"  Reasoning: {plan['reasoning']}")
        else:
            self.stdout.write(self.style.WARNING("\n⚠ No trade plan generated"))
            if result.get("error"):
                self.stdout.write(f"  Error: {result['error']}")

        # Risk result
        risk = result.get("risk_result", {})
        approved = result.get("risk_approved", False)
        if risk:
            style = self.style.SUCCESS if approved else self.style.ERROR
            self.stdout.write(style(
                f"\n🛡 RISK: {'APPROVED' if approved else 'REJECTED'}"
            ))
            self.stdout.write(f"  Reason: {risk.get('reason', 'N/A')}")
            if risk.get("risk_amount"):
                self.stdout.write(f"  Risk Amount: {risk['risk_amount']:.0f} INR")
                self.stdout.write(f"  Risk % of Capital: {risk.get('risk_pct_of_capital', 0):.1f}%")

        # Execution result
        exec_r = result.get("execution_result", {})
        if exec_r and exec_r.get("success"):
            self.stdout.write(self.style.SUCCESS(
                f"\n✅ EXECUTED [{exec_r.get('mode', 'paper').upper()}]"
            ))
            self.stdout.write(f"  Order ID: {exec_r['order_id']}")
            self.stdout.write(f"  Fill: {exec_r['fill_quantity']}x @ {exec_r['fill_price']:.2f}")

        # Journal
        jid = result.get("journal_id")
        if jid:
            self.stdout.write(self.style.SUCCESS(f"\n📓 Journal Entry: #{jid}"))

        self.stdout.write("\n" + "─" * 60 + "\n")

    def _seed_strategies(self):
        """Seed default Camarilla / price-action strategy docs."""
        from trading.models import StrategyDoc

        strategies = [
            {
                "title": "Core Risk Rule: 1% Max Risk Per Trade",
                "content": (
                    "Never risk more than 1% of total capital on a single trade. "
                    "Position size = (1% of capital) / (entry - stop_loss). "
                    "This is non-negotiable."
                ),
                "category": "RISK",
            },
            {
                "title": "Intraday Bias: First 15min Structure",
                "content": (
                    "Wait for the first 15 minutes of the session. "
                    "If the stock makes a new high after the first candle without making a new low, "
                    "bias is bullish. If it makes a new low without a new high, bias is bearish. "
                    "If both, wait for clarity."
                ),
                "category": "ENTRY",
            },
            {
                "title": "Open-High / Open-Low Pattern",
                "content": (
                    "Open=High (OH) candle is bearish — stock could not trade above open. "
                    "Open=Low (OL) candle is bullish — stock could not trade below open. "
                    "Use these as confirmation signals, not standalone entries."
                ),
                "category": "ENTRY",
            },
            {
                "title": "Avoid Low Volume Breakouts",
                "content": (
                    "Do not trade breakouts on low volume. "
                    "A genuine breakout should have volume at least 1.5x the average 5-candle volume. "
                    "Low volume breakouts are more likely to fail."
                ),
                "category": "FILTER",
            },
            {
                "title": "Daily Loss Limit",
                "content": (
                    "Stop trading for the day if cumulative losses exceed 3% of capital. "
                    "No revenge trading. Walk away. Tomorrow is another day."
                ),
                "category": "RISK",
            },
            {
                "title": "Doji at Extremes",
                "content": (
                    "A doji candle at the day's high or low often signals reversal. "
                    "Look for doji at cumulative_high or cumulative_low as potential reversal entries. "
                    "Confirm with the next candle before entering."
                ),
                "category": "ENTRY",
            },
            {
                "title": "Camarilla Pivot Strategy",
                "content": (
                    "Use pivot points as reference levels. "
                    "Buy near pivot support with stop below the day's low. "
                    "Sell near pivot resistance with stop above the day's high. "
                    "Pivot = (High + Low) / 2."
                ),
                "category": "EXIT",
            },
        ]

        created = 0
        for s in strategies:
            _, was_created = StrategyDoc.objects.get_or_create(
                title=s["title"],
                defaults={"content": s["content"], "category": s["category"]},
            )
            if was_created:
                created += 1

        self.stdout.write(self.style.SUCCESS(
            f"Seeded {created} new strategy docs ({len(strategies) - created} already existed)"
        ))

    def _init_portfolio(self, capital: float):
        """Create a portfolio snapshot."""
        from trading.models import PortfolioSnapshot

        snap = PortfolioSnapshot.objects.create(
            capital=capital,
            invested=0,
            available_cash=capital,
            daily_pnl=0,
            total_pnl=0,
            daily_loss=0,
            open_positions=0,
            open_positions_count=0,
            snapshot_date=date.today(),
        )
        self.stdout.write(self.style.SUCCESS(
            f"Portfolio initialized: {capital:,.0f} INR on {snap.snapshot_date}"
        ))

    def _show_journal(self):
        """Display recent journal entries."""
        from trading.models import TradeJournal

        entries = TradeJournal.objects.order_by("-created_at")[:20]

        if not entries:
            self.stdout.write("No journal entries yet.")
            return

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n{'='*80}\n  TRADE JOURNAL (last {len(entries)} entries)\n{'='*80}"
        ))

        for e in entries:
            pnl_str = f"{e.pnl:+.0f} INR" if e.pnl is not None else "open"
            risk_str = "✅" if e.risk_approved else "❌"
            self.stdout.write(
                f"  #{e.id:4d} | {e.created_at.strftime('%Y-%m-%d %H:%M')} | "
                f"{e.side:4s} {e.quantity:4d}x {e.symbol:10s} @ {e.entry_price:8.2f} | "
                f"SL: {e.stop_loss:8.2f} | T: {e.target:8.2f} | "
                f"{e.status:10s} | Risk: {risk_str} | P&L: {pnl_str}"
            )

        self.stdout.write("")
