"""INTRADAY trades must not stay open past their session.

An INTRADAY position squares off at 15:30 the same day. When a session dies
mid-flight the Trade row keeps its FILLED/APPROVED status forever, and because
`open_positions` counts those rows, the portfolio reads permanently at the
MAX_OPEN_POSITIONS cap — every subsequent order is rejected with
"Max open positions reached".

Found live on 2026-09-09: ten INTRADAY trades from March and May 2026 (121-176
days old) were still open, making order placement impossible.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from django.utils import timezone

from apps.trading.models import Trade
from tests.factories import PortfolioFactory

pytestmark = pytest.mark.django_db


def _trade(portfolio, *, status, days_ago, product="INTRADAY", symbol="RELIANCE"):
    today = timezone.localtime().date()
    return Trade.objects.create(
        tenant=portfolio.tenant,
        portfolio=portfolio,
        symbol=symbol,
        side="BUY",
        quantity=10,
        entry_price=Decimal("100.00"),
        stop_loss=Decimal("95.00"),
        target=Decimal("110.00"),
        status=status,
        product=product,
        trade_date=today - timedelta(days=days_ago),
    )


def test_stale_filled_intraday_is_closed():
    p = PortfolioFactory()
    t = _trade(p, status=Trade.Status.FILLED, days_ago=176)

    from apps.trading.services.reconcile import reconcile_stale_intraday

    result = reconcile_stale_intraday(tenant_id=p.tenant_id)

    t.refresh_from_db()
    assert t.status == Trade.Status.CLOSED
    assert t.close_reason == Trade.CloseReason.EOD
    assert result["closed"] == 1


def test_stale_approved_intraday_is_expired():
    """APPROVED never filled — there is no position to close, so it expires."""
    p = PortfolioFactory()
    t = _trade(p, status=Trade.Status.APPROVED, days_ago=121)

    from apps.trading.services.reconcile import reconcile_stale_intraday

    result = reconcile_stale_intraday(tenant_id=p.tenant_id)

    t.refresh_from_db()
    assert t.status == Trade.Status.EXPIRED
    assert result["expired"] == 1


def test_reconcile_invents_no_pnl():
    """We do not know the real exit. Closing must not fabricate a result that
    would pollute the monthly report with imaginary wins or losses."""
    p = PortfolioFactory()
    t = _trade(p, status=Trade.Status.FILLED, days_ago=170)

    from apps.trading.services.reconcile import reconcile_stale_intraday

    reconcile_stale_intraday(tenant_id=p.tenant_id)

    t.refresh_from_db()
    assert t.realized_pnl == Decimal("0.00")
    assert t.exit_price == t.entry_price


def test_todays_intraday_is_left_alone():
    """The session may still be live — only prior days are stale."""
    p = PortfolioFactory()
    t = _trade(p, status=Trade.Status.FILLED, days_ago=0)

    from apps.trading.services.reconcile import reconcile_stale_intraday

    result = reconcile_stale_intraday(tenant_id=p.tenant_id)

    t.refresh_from_db()
    assert t.status == Trade.Status.FILLED
    assert result["closed"] == 0


@pytest.mark.parametrize("product", ["DELIVERY", "CARRYFORWARD"])
def test_positional_products_are_left_alone(product):
    """DELIVERY and CARRYFORWARD are *supposed* to be held overnight."""
    p = PortfolioFactory()
    t = _trade(p, status=Trade.Status.FILLED, days_ago=176, product=product)

    from apps.trading.services.reconcile import reconcile_stale_intraday

    reconcile_stale_intraday(tenant_id=p.tenant_id)

    t.refresh_from_db()
    assert t.status == Trade.Status.FILLED


def test_reconcile_unblocks_order_placement():
    """The behaviour that actually matters: open_positions drops back to 0."""
    p = PortfolioFactory()
    for i in range(10):
        _trade(p, status=Trade.Status.FILLED, days_ago=170, symbol=f"SYM{i}")

    open_statuses = [Trade.Status.FILLED, Trade.Status.APPROVED]
    assert Trade.objects.filter(portfolio=p, status__in=open_statuses).count() == 10

    from apps.trading.services.reconcile import reconcile_stale_intraday

    reconcile_stale_intraday(tenant_id=p.tenant_id)

    assert Trade.objects.filter(portfolio=p, status__in=open_statuses).count() == 0


# ---------------------------------------------------------------------------
# Every status that counts toward the open-positions cap must be reconcilable.
# snapshots.refresh_all counts SENT/PARTIAL/FILLED, so a stuck SENT row wedges
# the cap exactly like a stuck FILLED one. Legal transitions differ per state
# (SENT cannot go straight to CLOSED), so each needs its own terminal.
# ---------------------------------------------------------------------------
def test_stale_sent_intraday_is_cancelled():
    """SENT means the broker took the order but never confirmed a fill —
    there is no position to close, so it cancels."""
    p = PortfolioFactory()
    t = _trade(p, status=Trade.Status.SENT, days_ago=176)

    from apps.trading.services.reconcile import reconcile_stale_intraday

    result = reconcile_stale_intraday(tenant_id=p.tenant_id)

    t.refresh_from_db()
    assert t.status == Trade.Status.CANCELLED
    assert result["cancelled"] == 1


def test_stale_partial_intraday_is_closed():
    p = PortfolioFactory()
    t = _trade(p, status=Trade.Status.PARTIAL, days_ago=176)

    from apps.trading.services.reconcile import reconcile_stale_intraday

    reconcile_stale_intraday(tenant_id=p.tenant_id)

    t.refresh_from_db()
    assert t.status == Trade.Status.CLOSED


def test_snapshot_open_count_drops_to_zero():
    """The number RiskEngine actually reads is the snapshot's open_positions,
    which counts SENT/PARTIAL/FILLED — that must reach 0."""
    p = PortfolioFactory()
    for i, st in enumerate(
        [Trade.Status.SENT, Trade.Status.PARTIAL, Trade.Status.FILLED]
    ):
        _trade(p, status=st, days_ago=170, symbol=f"S{i}")

    counted = [Trade.Status.SENT, Trade.Status.PARTIAL, Trade.Status.FILLED]
    assert Trade.objects.filter(portfolio=p, status__in=counted).count() == 3

    from apps.trading.services.reconcile import reconcile_stale_intraday

    reconcile_stale_intraday(tenant_id=p.tenant_id)

    assert Trade.objects.filter(portfolio=p, status__in=counted).count() == 0
