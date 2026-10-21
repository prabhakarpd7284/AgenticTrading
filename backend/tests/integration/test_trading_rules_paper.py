"""Trailing stop + 15:14 automated square-off in the paper sweep (TRADING_RULES.md §2)."""
from __future__ import annotations

from decimal import Decimal

import pytest
from django.utils import timezone

from apps.events.models import Event
from apps.trading.models import Order, Trade
from tests.factories import PortfolioFactory, UserFactory

pytestmark = pytest.mark.django_db


def _order(portfolio, **kw):
    defaults = dict(
        tenant=portfolio.tenant, portfolio=portfolio, created_by=UserFactory(),
        symbol="PAYTM", side="BUY", qty=10, order_type="LIMIT", price=Decimal("100.00"),
        sl=Decimal("95.00"), tp=Decimal("130.00"), status=Order.Status.SENT,
        product="INTRADAY", origin="strategy",
    )
    defaults.update(kw)
    return Order.objects.create(**defaults)


def _at(h, m):
    return timezone.localtime().replace(hour=h, minute=m, second=0, microsecond=0)


@pytest.fixture
def drv(monkeypatch):
    import apps.trading.services.paper_fill_driver as drv
    import apps.trading.services.trailing_stop as ts

    monkeypatch.setattr(ts, "atr_for_symbol", lambda sym: 2.0)
    book = {}
    monkeypatch.setattr(drv, "reference_price", lambda s, t: book.get(s, 0.0))
    drv._book = book
    return drv


def test_automated_trade_trails_and_closes_as_trail(drv):
    p = PortfolioFactory(mode="paper")
    o = _order(p)
    drv._book["PAYTM"] = 99.0
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 0))
    t = Trade.objects.get(primary_order=o)

    drv._book["PAYTM"] = 105.0          # +1R → breakeven (105-3=102 > 100 → trails to 102)
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 5))
    t.refresh_from_db()
    assert t.trail_stop == Decimal("102.0000") and t.best_price == Decimal("105.0000")
    assert t.stop_loss == Decimal("95.00")          # initial stop untouched (defines 1R)

    drv._book["PAYTM"] = 110.0          # trail → 107
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 10))
    t.refresh_from_db()
    assert t.trail_stop == Decimal("107.0000")

    drv._book["PAYTM"] = 108.0          # pullback: stop must not loosen
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 15))
    t.refresh_from_db()
    assert t.trail_stop == Decimal("107.0000") and t.status == Trade.Status.FILLED

    drv._book["PAYTM"] = 106.5          # through the trail → closed TRAIL at 107
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 20))
    t.refresh_from_db()
    assert t.status == Trade.Status.CLOSED and t.close_reason == "TRAIL"
    assert t.exit_price == Decimal("107.0000") and t.realized_pnl == Decimal("70.00")
    assert Event.objects.filter(type="trade.trailed", trade_id=t.id).count() == 2


def test_manual_trade_keeps_fixed_stop(drv):
    p = PortfolioFactory(mode="paper")
    o = _order(p, origin="ui")
    drv._book["PAYTM"] = 99.0
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 0))
    drv._book["PAYTM"] = 115.0
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 5))
    t = Trade.objects.get(primary_order=o)
    assert t.trail_stop is None


def test_automated_intraday_squared_off_at_1514_manual_waits_for_1530(drv):
    p = PortfolioFactory(mode="paper")
    auto = _order(p)
    manual = _order(p, origin="ui")
    drv._book["PAYTM"] = 99.0
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(11, 0))

    drv._book["PAYTM"] = 101.0
    res = drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(15, 14))
    assert res["squared_off"] == 1
    t_auto = Trade.objects.get(primary_order=auto)
    t_man = Trade.objects.get(primary_order=manual)
    assert t_auto.status == Trade.Status.CLOSED and t_auto.close_reason == "EOD"
    assert t_man.status == Trade.Status.FILLED

    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(15, 31))
    t_man.refresh_from_db()
    assert t_man.status == Trade.Status.CLOSED


def test_unfilled_automated_intraday_order_cancelled_at_1514(drv):
    p = PortfolioFactory(mode="paper")
    o = _order(p)
    drv._book["PAYTM"] = 105.0          # limit 100 not reached
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_at(15, 14))
    o.refresh_from_db()
    assert o.status == Order.Status.CANCELLED
