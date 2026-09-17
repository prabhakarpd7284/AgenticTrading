from decimal import Decimal

from celery import shared_task


@shared_task(ignore_result=True)
def refresh_all() -> int:
    # Open positions live in trading.Trade (SENT/PARTIAL/FILLED) — the legacy
    # trading.Position table is never populated in v2, so the old count was
    # always 0 and its unrealized_pnl field is never marked to market. We mark
    # to market here from a non-blocking server LTP (cache→candle; this runs in
    # a Celery beat task, off the ASGI thread).
    from apps.trading.models import Portfolio, PortfolioSnapshot, Trade
    from apps.trading.services.place_order import reference_price

    open_statuses = [Trade.Status.SENT, Trade.Status.PARTIAL, Trade.Status.FILLED]
    n = 0
    for p in Portfolio.objects.all():
        opens = list(
            Trade.objects.filter(portfolio=p, status__in=open_statuses)
            .values("symbol", "side", "quantity", "lot_size", "entry_price")
        )
        unrealized = Decimal("0")
        for t in opens:
            ltp = reference_price(t["symbol"], p.tenant_id)
            if ltp > 0:
                direction = 1 if t["side"] == "BUY" else -1
                units = t["quantity"] * (t["lot_size"] or 1)
                unrealized += (Decimal(str(ltp)) - t["entry_price"]) * units * direction
        PortfolioSnapshot.objects.create(
            tenant_id=p.tenant_id,
            portfolio=p,
            # Equity reflects realized day P&L + open-position mark-to-market.
            equity=p.capital + p.day_pnl + unrealized,
            day_pnl=p.day_pnl,
            unrealized_pnl=unrealized,
            open_positions=len(opens),
        )
        n += 1
    return n
