"""
Derive swing (Oliver Kell) paper trades for a historical window.

The Monthly report's trade-driven sections should reflect *all* the strategies
AlphaDesk runs — not just intraday. Swing is a first-class source: the
backtester route (/backtester → ``build_ok_backtest``) already simulates Oliver
Kell cycle trades, but only for on-screen display — nothing persists them.

This reuses the same engine (``run_ok_backtest`` → ``BacktestResult.trades``)
and writes each *closed* swing trade as a cash ``Trade`` row stamped at its exit
date, so it lands in the month the P&L was realised. Idempotent: prior swing
rows in the window are cleared first, so re-running never double-counts.

Usage:
    python manage.py derive_swing_trades --from 2026-05-01 --to 2026-05-31
"""
from __future__ import annotations

from datetime import date, datetime

from django.core.management.base import BaseCommand, CommandError

# Swing rows are tagged with this reasoning prefix so a re-run can find and
# replace exactly its own output (idempotency) without touching intraday rows.
_SWING_TAG = "[Swing"

_CLOSE_REASON = {
    "Target": "TARGET_HIT",
    "SL": "SL_HIT",
    "Trail SL": "TRAIL",
    "Trailing Stop": "TRAIL",
    "Phase reversal": "MANUAL",
    "Max hold": "EOD",
    "End of test": "EOD",
}


class Command(BaseCommand):
    help = "Derive Oliver Kell swing paper trades into the Monthly report (cash, multi-day)"

    def add_arguments(self, parser):
        parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
        parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
        parser.add_argument("--capital", type=float, default=None)
        parser.add_argument("--symbols", help="Comma-separated NSE symbols (default: smart universe)")

    def handle(self, *args, **opts):
        from apps.trading.models import Trade
        from plugins.strategy_swing.ok_backtest import run_ok_backtest

        first = date.fromisoformat(opts["from_date"])
        last = date.fromisoformat(opts["to_date"])
        if last < first:
            raise CommandError("--to is before --from")

        tenant, portfolio = self._resolve_owner()
        if not tenant or not portfolio:
            raise CommandError("No owner tenant / portfolio found.")

        symbols = self._symbols(opts, opts["to_date"])
        self.stdout.write(
            f"Running OK swing backtest over {first} → {last} on {len(symbols)} symbols…"
        )
        result = run_ok_backtest(symbols, opts["from_date"], opts["to_date"], capital=opts["capital"])

        # Idempotency — drop this command's own prior output in the window.
        deleted, _ = Trade.objects.filter(
            tenant=tenant, reasoning__startswith=_SWING_TAG,
            trade_date__gte=first, trade_date__lte=last,
        ).delete()

        created = 0
        for t in result.trades:
            if not t.exit_date:
                continue  # still open — no realised P&L to book
            try:
                exit_d = datetime.strptime(t.exit_date[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if not (first <= exit_d <= last):
                continue
            self._persist(Trade, tenant, portfolio, t, exit_d)
            created += 1

        self.stdout.write(self.style.SUCCESS(
            f"Done — {created} swing trades booked ({deleted} stale rows replaced). "
            f"Backtest total: {result.total_trades} trades, "
            f"P&L {result.total_pnl:+,.0f}."
        ))

    # ── helpers ────────────────────────────────────────────────────────
    def _persist(self, Trade, tenant, portfolio, t, exit_d):
        from datetime import datetime, time as dtime

        from django.utils import timezone

        side = "BUY" if str(t.side).upper() == "BUY" else "SELL"
        reason = _CLOSE_REASON.get(t.exit_reason, "EOD")
        # Store the INITIAL stop (the risk you took), not the trailed stop.
        # The backtest mutates `stoploss` as the trade trails, so on a winner it
        # can end up above a long's entry — meaningless as a stop-loss. The exit
        # price already reflects where the trail actually closed the trade.
        risk = getattr(t, "risk_points", 0.0) or 0.0
        if risk > 0:
            init_stop = (t.entry_price - risk) if side == "BUY" else (t.entry_price + risk)
        else:
            init_stop = t.stoploss
        # Store the real entry + exit dates (swing is multi-day) so the chart
        # can mark the actual trade timeline instead of guessing from price.
        try:
            entry_d = datetime.strptime(t.entry_date[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            entry_d = exit_d
        filled_at = timezone.make_aware(datetime.combine(entry_d, dtime(9, 15)))
        closed_at = timezone.make_aware(datetime.combine(exit_d, dtime(15, 30)))
        Trade.objects.create(
            tenant=tenant,
            portfolio=portfolio,
            symbol=t.symbol,
            exchange="NSE",
            side=side,
            entry_price=t.entry_price,
            stop_loss=round(init_stop, 2),
            target=t.target,
            quantity=t.quantity,
            status=Trade.Status.CLOSED,
            fill_price=t.entry_price,
            fill_quantity=t.quantity,
            exit_price=t.exit_price,
            exit_quantity=t.quantity,
            realized_pnl=round(t.pnl, 2),
            pnl_percent=round(t.pnl_pct, 2),
            close_reason=reason,
            reasoning=f"{_SWING_TAG} {t.phase}] {t.exit_reason}"[:255],
            confidence=min(1.0, max(0.0, t.rr_achieved / 3.0)) if t.rr_achieved else 0.0,
            risk_approved=True,
            risk_reason="Swing backtest",
            origin=Trade.Origin.WORKFLOW,
            trade_date=exit_d,
            filled_at=filled_at,
            closed_at=closed_at,
        )

    def _resolve_owner(self):
        from apps.tenants.models import Membership
        from apps.trading.models import Portfolio

        mem = (
            Membership.objects.filter(is_active=True, role="owner")
            .select_related("tenant").first()
        )
        if not mem:
            return None, None
        portfolio = (
            Portfolio.objects.filter(tenant=mem.tenant).order_by("created_at").first()
            or Portfolio.objects.create(tenant=mem.tenant, name="Default")
        )
        return mem.tenant, portfolio

    def _symbols(self, opts, to_date) -> list[str]:
        if opts.get("symbols"):
            return [s.strip().upper() for s in opts["symbols"].split(",") if s.strip()]
        from apps.market_data.services.ok_backtest_service import _smart_universe
        return _smart_universe("daily", to_date)
