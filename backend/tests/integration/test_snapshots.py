"""Portfolio snapshot mark-to-market over the live Trade model (#48)."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pytest

from apps.trading.models import PortfolioSnapshot, Trade
from apps.trading.tasks import snapshots
from tests.factories import PortfolioFactory, TenantFactory

pytestmark = pytest.mark.django_db


def _trade(tenant, portfolio, symbol, qty, entry, status):
    return Trade.objects.create(
        tenant=tenant, portfolio=portfolio, symbol=symbol, side="BUY",
        quantity=qty, lot_size=1, entry_price=Decimal(str(entry)),
        stop_loss=Decimal("1"), target=Decimal("99999"),
        status=status, trade_date=date(2026, 6, 28),
    )


def test_snapshot_marks_open_trades_to_market():
    tenant = TenantFactory()
    p = PortfolioFactory(tenant=tenant, capital=Decimal("500000"), day_pnl=Decimal("1200"))
    _trade(tenant, p, "HDFCBANK", 10, 1600, Trade.Status.FILLED)   # open → MTM'd
    _trade(tenant, p, "TCS", 5, 3000, Trade.Status.CLOSED)         # closed → excluded

    with patch("apps.trading.services.place_order.reference_price", return_value=1650.0):
        snapshots.refresh_all()

    snap = PortfolioSnapshot.objects.filter(portfolio=p).latest("captured_at")
    assert snap.open_positions == 1               # counts the live Trade, not the dead Position table
    assert snap.unrealized_pnl == Decimal("500")  # (1650-1600) * 10
    assert snap.equity == Decimal("501700")       # capital + day_pnl + unrealized


def test_snapshot_zero_unrealized_when_ltp_unknown():
    # Cold cache → reference_price returns 0 → equity falls back to capital+day_pnl, no crash.
    tenant = TenantFactory()
    p = PortfolioFactory(tenant=tenant, capital=Decimal("500000"), day_pnl=Decimal("0"))
    _trade(tenant, p, "HDFCBANK", 10, 1600, Trade.Status.FILLED)

    with patch("apps.trading.services.place_order.reference_price", return_value=0.0):
        snapshots.refresh_all()

    snap = PortfolioSnapshot.objects.filter(portfolio=p).latest("captured_at")
    assert snap.open_positions == 1
    assert snap.unrealized_pnl == Decimal("0")
    assert snap.equity == Decimal("500000")
