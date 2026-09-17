"""Reconcile INTRADAY trades that outlived their session.

An INTRADAY position squares off at 15:30 the same day. When a session dies
mid-flight — worker killed, process crash, machine slept — the Trade row keeps
its FILLED/APPROVED status indefinitely. Nothing ever cleans it up.

That is not merely untidy. `PortfolioSnapshot.open_positions` counts these
rows, and `RiskEngine` rejects any new order once that count reaches
MAX_OPEN_POSITIONS. Ten orphaned rows from March/May 2026 made order placement
impossible on this install until 2026-09-09 — silently, because the positions
API reads a different table and showed an empty book.

Deliberately conservative:

* Only INTRADAY. DELIVERY and CARRYFORWARD are *meant* to be held overnight.
* Only trade_date strictly before today, so a live session is never touched.
* No invented P&L. We do not know the real exit, so a FILLED trade closes at
  its own entry price for a realized P&L of exactly zero. A fabricated win or
  loss would silently corrupt the monthly report; a flat close is visibly
  neutral and carries close_reason=EOD plus an audit event saying why.
"""
from __future__ import annotations

import structlog
from django.utils import timezone

from apps.trading.models import Trade
from apps.trading.services import trade_lifecycle

log = structlog.get_logger(__name__)

_RECONCILE_REASON = "stale_intraday_auto_reconciled"

# Statuses that leave a Trade counting as live work. SENT/PARTIAL/FILLED are
# what snapshots.refresh_all counts into open_positions (and therefore what
# RiskEngine caps on); APPROVED is a pre-trade state that can also stick.
OPEN_STATUSES = (
    Trade.Status.APPROVED,
    Trade.Status.SENT,
    Trade.Status.PARTIAL,
    Trade.Status.FILLED,
)

# Where each stuck state legally terminates. SENT cannot go straight to CLOSED
# (see trade_lifecycle._TRANSITIONS), and it never confirmed a fill, so there is
# no position to close — it cancels. PARTIAL and FILLED did take a position.
_NEVER_FILLED = (Trade.Status.APPROVED, Trade.Status.SENT)


def reconcile_stale_intraday(*, tenant_id=None, today=None) -> dict:
    """Close/expire INTRADAY trades left open from a previous session.

    Returns counts: ``{"closed": n, "expired": n, "failed": n}``.
    """
    today = today or timezone.localtime().date()

    qs = Trade.objects.filter(
        product="INTRADAY",
        status__in=OPEN_STATUSES,
        trade_date__lt=today,
    )
    if tenant_id is not None:
        qs = qs.filter(tenant_id=tenant_id)

    closed = expired = cancelled = failed = 0
    for trade in qs.iterator():
        try:
            if trade.status == Trade.Status.APPROVED:
                # Approved but never queued — the plan simply lapsed.
                trade_lifecycle.expire(trade, reason=_RECONCILE_REASON)
                expired += 1
            elif trade.status == Trade.Status.SENT:
                # Broker took it but never confirmed a fill. No position exists,
                # and SENT → CLOSED is not a legal transition.
                trade_lifecycle.cancel(trade, reason=_RECONCILE_REASON)
                cancelled += 1
            else:
                # PARTIAL / FILLED — a position was taken. Close it flat.
                trade_lifecycle.close(
                    trade,
                    price=float(trade.entry_price),
                    qty=trade.quantity,
                    reason=Trade.CloseReason.EOD,
                    realized_pnl=0.0,
                )
                closed += 1
        except Exception:  # noqa: BLE001 — one bad row must not stop the sweep.
            failed += 1
            log.warning(
                "reconcile.stale_intraday.failed",
                trade_id=str(trade.id), symbol=trade.symbol, exc_info=True,
            )

    if closed or expired or cancelled or failed:
        log.info(
            "reconcile.stale_intraday.done",
            closed=closed, expired=expired, cancelled=cancelled, failed=failed,
        )
    return {
        "closed": closed, "expired": expired,
        "cancelled": cancelled, "failed": failed,
    }
