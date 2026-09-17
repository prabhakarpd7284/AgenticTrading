"""Drive paper orders through their whole life.

    sent ──fill──► Trade(FILLED) ──stop/target/EOD──► Trade(CLOSED) + P&L

Nothing in the v2 backend created a Trade from an Order, and the paper adapter's
`place()` only mints an id, so every paper order stopped at ``sent``. Paper mode
therefore produced no fills, no positions and no P&L — nothing to tune a
strategy against, which is the entire reason paper mode exists here.

Prices come from `reference_price()` (Redis ``ltp:`` ← live ticks, falling back
to the newest candle). The fill/exit decisions themselves live in
`paper_fills.py` and are pure; this module only supplies prices, persists
outcomes, and refuses to touch anything that is not paper.
"""
from __future__ import annotations

from decimal import Decimal

import structlog
from django.db import transaction
from django.utils import timezone

from apps.trading.models import Order, Trade
from apps.trading.services import trade_lifecycle
from apps.trading.services.paper_fills import exit_for, fill_price_for
from apps.trading.services.place_order import reference_price

log = structlog.get_logger(__name__)

# NSE/BSE square-off. Intraday positions do not survive it.
MARKET_CLOSE = (15, 30)


def _is_after_close(now) -> bool:
    return (now.hour, now.minute) >= MARKET_CLOSE


def _pnl(trade, exit_price: Decimal) -> Decimal:
    """Realized P&L in rupees, signed by direction."""
    units = trade.quantity * (trade.lot_size or 1)
    direction = 1 if trade.side == "BUY" else -1
    return (exit_price - trade.entry_price) * units * direction


def process_paper_fills(*, tenant_id=None, now=None) -> dict:
    """One sweep: fill what should fill, close what should close.

    Returns ``{"filled": n, "closed": n, "squared_off": n, "failed": n}``.
    """
    now = now or timezone.localtime()
    counts = {
        "filled": 0, "closed": 0, "squared_off": 0, "cancelled": 0, "failed": 0,
    }

    _fill_resting_orders(tenant_id, now, counts)
    _close_open_trades(tenant_id, now, counts)

    if any(counts.values()):
        log.info("paper_fills.sweep", **counts)
    return counts


def _paper_scope(qs, tenant_id):
    qs = qs.filter(portfolio__mode="paper")
    return qs.filter(tenant_id=tenant_id) if tenant_id is not None else qs


def _fill_resting_orders(tenant_id, now, counts) -> None:
    orders = _paper_scope(
        Order.objects.filter(status=Order.Status.SENT), tenant_id,
    ).select_related("portfolio")

    after_close = _is_after_close(now)

    for order in orders.iterator():
        try:
            # An INTRADAY order that never filled dies with its session. Left
            # resting it survives the night and fills the next morning at a
            # stale limit as though it were fresh — observed 2026-09-10, when a
            # 09-Sep PAYTM order filled at 09:16 the following day.
            if after_close and order.product == "INTRADAY":
                order.status = Order.Status.CANCELLED
                order.error = "unfilled at session close"
                order.save(update_fields=["status", "error", "updated_at"])
                counts["cancelled"] += 1
                continue

            ltp = reference_price(order.symbol, order.tenant_id)
            fill = fill_price_for(order, ltp)
            if fill is None:
                continue
            with transaction.atomic():
                _record_fill(order, Decimal(str(fill)))
            counts["filled"] += 1
        except Exception:  # noqa: BLE001 — one bad order must not stop the sweep
            counts["failed"] += 1
            log.warning(
                "paper_fills.fill_failed",
                order_id=str(order.id), symbol=order.symbol, exc_info=True,
            )


def _record_fill(order: Order, fill_price: Decimal) -> Trade:
    order.status = Order.Status.FILLED
    order.save(update_fields=["status", "updated_at"])
    return Trade.objects.create(
        tenant_id=order.tenant_id,
        portfolio=order.portfolio,
        primary_order=order,
        symbol=order.symbol,
        side=order.side,
        quantity=order.qty,
        entry_price=fill_price,
        stop_loss=order.sl,
        target=order.tp,
        status=Trade.Status.FILLED,
        product=order.product,
        origin=order.origin,
        trade_date=timezone.localtime().date(),
        fill_price=fill_price,
        fill_quantity=order.qty,
        filled_at=timezone.now(),
        risk_approved=True,
    )


def _close_open_trades(tenant_id, now, counts) -> None:
    trades = _paper_scope(
        Trade.objects.filter(
            status__in=[Trade.Status.FILLED, Trade.Status.PARTIAL],
        ),
        tenant_id,
    ).select_related("portfolio")

    after_close = _is_after_close(now)

    for trade in trades.iterator():
        try:
            ltp = reference_price(trade.symbol, trade.tenant_id)

            if ltp and ltp > 0:
                # Carry a durable mark while the feed is live. The tick feed
                # stops at 15:30 and the ltp cache expires 120s later, with an
                # empty Candle table behind it — so without this, every price
                # reads 0.0 exactly when square-off needs one, and positions
                # survive the close. (Also the only thing that ever populates
                # Trade.last_ltp, which the positions UI and RAG both read.)
                Trade.objects.filter(pk=trade.pk).update(
                    last_ltp=Decimal(str(ltp)),
                )
                trade.last_ltp = Decimal(str(ltp))

            hit = exit_for(trade, ltp) if (ltp and ltp > 0) else None
            if hit is not None:
                price, reason = hit
                bucket = "closed"
            elif after_close and trade.product == "INTRADAY":
                # Never reached a level, but intraday cannot be carried. Fall
                # back to the last mark when the feed has already stopped.
                mark = ltp if (ltp and ltp > 0) else trade.last_ltp
                if not mark or float(mark) <= 0:
                    # Never marked and no feed — closing would mean inventing
                    # a price. Leave it for the reconciliation sweep to flag.
                    continue
                price, reason = float(mark), Trade.CloseReason.EOD
                bucket = "squared_off"
            else:
                continue

            exit_price = Decimal(str(price))
            trade_lifecycle.close(
                trade,
                price=float(exit_price),
                qty=trade.quantity,
                reason=reason,
                realized_pnl=float(_pnl(trade, exit_price)),
            )
            counts[bucket] += 1
        except Exception:  # noqa: BLE001
            counts["failed"] += 1
            log.warning(
                "paper_fills.close_failed",
                trade_id=str(trade.id), symbol=trade.symbol, exc_info=True,
            )
