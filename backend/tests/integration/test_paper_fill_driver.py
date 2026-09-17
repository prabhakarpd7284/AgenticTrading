"""Paper fill driver — orders become trades, trades close on their levels.

Nothing in the v2 backend ever created a Trade from an Order, so a paper order
reached `sent` and died there. This drives the full lifecycle:

    sent ──fill──► Trade(FILLED) ──SL/target/EOD──► Trade(CLOSED) + realized P&L

Only paper portfolios are touched. Live orders are the broker's business.
"""
from __future__ import annotations

from datetime import time
from decimal import Decimal

import pytest
from django.utils import timezone

from apps.trading.models import Order, Trade
from tests.factories import PortfolioFactory, UserFactory

pytestmark = pytest.mark.django_db


def _order(portfolio, **kw):
    defaults = dict(
        tenant=portfolio.tenant,
        portfolio=portfolio,
        created_by=UserFactory(),
        symbol="PAYTM",
        side="BUY",
        qty=10,
        order_type="LIMIT",
        price=Decimal("100.00"),
        sl=Decimal("95.00"),
        tp=Decimal("110.00"),
        status=Order.Status.SENT,
        product="INTRADAY",
    )
    defaults.update(kw)
    return Order.objects.create(**defaults)


def _prices(mapping):
    """Patch reference_price to a fixed book."""
    return lambda symbol, tenant_id: mapping.get(symbol, 0.0)


# Tests must not depend on when they run. Every call passes an explicit
# clock: IN_SESSION for normal behaviour, AFTER_CLOSE for the EOD rules.
# Defaulting to wall-clock made the whole module pass only between 09:15
# and 15:30 — it went red a week later purely because it was evening.
def _in_session():
    return timezone.localtime().replace(hour=11, minute=0, second=0, microsecond=0)


def _after_close():
    return timezone.localtime().replace(hour=15, minute=31, second=0, microsecond=0)


def test_resting_order_fills_and_becomes_a_trade(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))

    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    o.refresh_from_db()
    assert o.status == Order.Status.FILLED
    assert result["filled"] == 1

    t = Trade.objects.get(primary_order=o)
    assert t.status == Trade.Status.FILLED
    assert t.entry_price == Decimal("100.00")   # at the limit, not the touch
    assert t.stop_loss == Decimal("95.00")
    assert t.target == Decimal("110.00")


def test_order_rests_when_price_is_not_reached(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 105.0}))

    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    o.refresh_from_db()
    assert o.status == Order.Status.SENT
    assert result["filled"] == 0
    assert not Trade.objects.filter(primary_order=o).exists()


def test_live_portfolios_are_never_simulated(monkeypatch):
    """Filling a live order in software would invent a position that does not
    exist at the broker."""
    p = PortfolioFactory(mode="live")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))

    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    o.refresh_from_db()
    assert o.status == Order.Status.SENT


def test_open_trade_closes_on_target_with_correct_pnl(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    # Now the target trades.
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 112.0}))
    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    t = Trade.objects.get(primary_order=o)
    assert t.status == Trade.Status.CLOSED
    assert t.close_reason == "TARGET_HIT"
    assert t.exit_price == Decimal("110.00")
    # (110 - 100) * 10 units
    assert t.realized_pnl == Decimal("100.00")
    assert result["closed"] == 1


def test_short_trade_pnl_is_signed_correctly(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p, side="SELL", price=Decimal("100.00"),
               sl=Decimal("110.00"), tp=Decimal("90.00"))

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 101.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 89.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    t = Trade.objects.get(primary_order=o)
    assert t.close_reason == "TARGET_HIT"
    # Short: sold 100, covered 90 → +10 a unit.
    assert t.realized_pnl == Decimal("100.00")


def test_stop_loss_books_a_loss(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 94.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    t = Trade.objects.get(primary_order=o)
    assert t.close_reason == "SL_HIT"
    assert t.realized_pnl == Decimal("-50.00")   # (95 - 100) * 10


def test_intraday_is_squared_off_after_the_close(monkeypatch):
    """A position that never hit either level must not survive the session."""
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    # Mid-range price, but the session is over.
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 103.0}))
    after_close = _after_close()
    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=after_close)

    t = Trade.objects.get(primary_order=o)
    assert t.status == Trade.Status.CLOSED
    assert t.close_reason == "EOD"
    assert t.exit_price == Decimal("103.00")     # squared off at the market
    assert result["squared_off"] == 1


def test_unknown_price_leaves_everything_untouched(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({}))   # 0.0 = unknown

    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    o.refresh_from_db()
    assert o.status == Order.Status.SENT
    assert result == {
        "filled": 0, "closed": 0, "squared_off": 0, "cancelled": 0, "failed": 0,
    }


# ---------------------------------------------------------------------------
# EOD square-off must not depend on a live feed.
#
# Confirmed on 2026-09-10: `squared_off` had never once been non-zero. The tick
# feed stops at 15:30, the ltp cache expires 120s later, the Candle fallback
# table is empty (0 rows), so reference_price() returns 0.0 for everything —
# exactly when square-off needs a price. The open PAGEIND position survived the
# close and was closed next morning by the reconciliation fallback at a
# fabricated-flat 0.00 P&L.
#
# Fix: carry a durable mark on the Trade itself, written while the feed IS live.
# ---------------------------------------------------------------------------
def test_sweep_records_a_durable_mark_while_the_feed_is_live(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 103.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    t = Trade.objects.get(primary_order=o)
    assert t.last_ltp == Decimal("103.00"), "open trades must carry a live mark"


def test_square_off_uses_the_last_mark_when_the_feed_has_stopped(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 104.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())   # records the mark

    # Feed is now dead — every price reads 0.0, as it does after 15:30.
    monkeypatch.setattr(drv, "reference_price", _prices({}))
    after_close = _after_close()
    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=after_close)

    t = Trade.objects.get(primary_order=o)
    assert result["squared_off"] == 1, "a dead feed must not block square-off"
    assert t.status == Trade.Status.CLOSED
    assert t.exit_price == Decimal("104.00")
    assert t.realized_pnl == Decimal("40.00")     # (104 - 100) * 10


def test_square_off_prefers_the_live_price_over_the_stale_mark(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 99.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 104.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 106.0}))
    after_close = _after_close()
    drv.process_paper_fills(tenant_id=p.tenant_id, now=after_close)

    t = Trade.objects.get(primary_order=o)
    assert t.exit_price == Decimal("106.00")


def test_no_price_at_all_does_not_invent_one(monkeypatch):
    """Never marked, feed dead — closing would require making a price up."""
    p = PortfolioFactory(mode="paper")
    o = _order(p, order_type="MARKET", price=None)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 100.0}))
    drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    t = Trade.objects.get(primary_order=o)
    Trade.objects.filter(pk=t.pk).update(last_ltp=None)

    monkeypatch.setattr(drv, "reference_price", _prices({}))
    after_close = _after_close()
    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=after_close)

    t.refresh_from_db()
    assert t.status == Trade.Status.FILLED
    assert result["squared_off"] == 0


# ---------------------------------------------------------------------------
# Resting INTRADAY orders must die with their session.
#
# PAYTM was placed 2026-09-09 as INTRADAY, never filled, and was left at `sent`
# overnight — then filled at 09:16 the next morning at its original limit, as
# though it were a fresh order.
# ---------------------------------------------------------------------------
def test_resting_intraday_orders_are_cancelled_after_the_close(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)          # LIMIT 100, market never reaches it

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 120.0}))
    after_close = _after_close()
    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=after_close)

    o.refresh_from_db()
    assert o.status == Order.Status.CANCELLED
    assert result["cancelled"] == 1
    assert not Trade.objects.filter(primary_order=o).exists()


def test_resting_orders_survive_during_the_session(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p)

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 120.0}))
    result = drv.process_paper_fills(tenant_id=p.tenant_id, now=_in_session())

    o.refresh_from_db()
    assert o.status == Order.Status.SENT
    assert result["cancelled"] == 0


def test_positional_orders_are_not_cancelled_at_the_close(monkeypatch):
    p = PortfolioFactory(mode="paper")
    o = _order(p, product="DELIVERY")

    import apps.trading.services.paper_fill_driver as drv
    monkeypatch.setattr(drv, "reference_price", _prices({"PAYTM": 120.0}))
    after_close = _after_close()
    drv.process_paper_fills(tenant_id=p.tenant_id, now=after_close)

    o.refresh_from_db()
    assert o.status == Order.Status.SENT
