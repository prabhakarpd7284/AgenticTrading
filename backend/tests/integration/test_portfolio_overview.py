"""Portfolio overview + attribution (UX redesign R2) and the per-portfolio
snapshot fix (R0 bug: the tenant's newest snapshot could be the MF book's)."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest import mock

import pytest
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from apps.accounts.api.jwt import _attach_tenant_claims
from apps.mutual_funds.models import MFHolding, MFNav, MFScheme
from apps.trading.models import OptionsLeg, OptionsPosition, PortfolioSnapshot, Trade
from apps.trading.services.portfolio_roles import (
    book_portfolios,
    latest_trading_snapshot,
    mf_portfolio_ids,
    trading_portfolio,
)
from tests.factories import PortfolioFactory

pytestmark = pytest.mark.django_db

OVERVIEW = "/api/v1/portfolio/overview/"
ATTRIBUTION = "/api/v1/portfolio/attribution/"
MARK = "apps.trading.services.portfolio_overview.mark_price"


def _client(user) -> APIClient:
    c = APIClient()
    access = RefreshToken.for_user(user).access_token
    _attach_tenant_claims(access, user)
    c.force_authenticate(user=user)
    c.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    return c


def _mf_book(tenant):
    return PortfolioFactory(tenant=tenant, name="Mutual Funds", capital=Decimal("0"))


def _snap(portfolio, equity, day_pnl="0", unrealized="0", open_positions=0, at=None):
    s = PortfolioSnapshot.objects.create(
        tenant=portfolio.tenant, portfolio=portfolio, equity=Decimal(str(equity)),
        day_pnl=Decimal(str(day_pnl)), unrealized_pnl=Decimal(str(unrealized)),
        open_positions=open_positions)
    if at is not None:
        PortfolioSnapshot.objects.filter(id=s.id).update(captured_at=at)
        s.refresh_from_db()
    return s


def _trade(portfolio, symbol, *, status, entry, qty=10, side="BUY", stop="95", realized=None,
           closed_at=None, run=None):
    return Trade.objects.create(
        tenant=portfolio.tenant, portfolio=portfolio, symbol=symbol, side=side, quantity=qty, lot_size=1,
        entry_price=Decimal(str(entry)), stop_loss=Decimal(stop), target=Decimal("999"),
        status=status, trade_date=timezone.localdate(), fill_price=Decimal(str(entry)),
        realized_pnl=None if realized is None else Decimal(str(realized)), closed_at=closed_at,
        strategy_run=run, origin="manual" if run is None else "workflow",
    )


def _mf_holding(tenant, portfolio, *, code="100001", units="10", cost="20", nav="25"):
    scheme = MFScheme.objects.create(
        scheme_code=code, amc_name="Acme Mutual Fund", scheme_name=f"Acme Fund {code} - Direct Plan - Growth",
        plan="Direct Plan", option="Growth Option", is_direct=True,
        category="Open Ended Schemes(Equity Scheme - Large Cap Fund)")
    MFNav.objects.create(scheme=scheme, nav_date=timezone.localdate(), nav=Decimal(nav))
    return MFHolding.objects.create(tenant=tenant, portfolio=portfolio, scheme=scheme, folio_number="F1",
                                    units=Decimal(units), avg_cost_per_unit=Decimal(cost), source="cas")


# ─── Snapshot regression (R0 bug) ────────────────────────────────────────

def test_summary_ignores_newer_mf_snapshot(two_tenants):
    t = two_tenants
    trading = t.portfolio_a
    mf = _mf_book(t.a)
    now = timezone.now()
    _snap(trading, 499000, day_pnl="-1000", at=now - timedelta(minutes=2))  # capital + day P&L
    _snap(mf, 0, at=now)          # newest snapshot of the tenant belongs to the MF book

    body = _client(t.owner_a).get("/api/v1/portfolios/summary/").json()
    assert body["capital"] == 500000.0
    assert body["daily_pnl"] == -1000.0
    # Before the fix: invested = 500000 - 0 + 0 = 500000 and available 0.
    assert body["invested"] == 0.0
    assert body["available_cash"] == 500000.0


def test_risk_view_reads_trading_snapshot(two_tenants):
    t = two_tenants
    mf = _mf_book(t.a)
    now = timezone.now()
    _snap(t.portfolio_a, 499000, day_pnl="-1000", open_positions=2, at=now - timedelta(minutes=1))
    _snap(mf, 0, at=now)
    body = _client(t.owner_a).get("/api/v1/risk/").json()
    util = body.get("utilization", body)
    assert util["daily_loss"] == 1000.0
    assert util["open_positions"] == 2


def test_latest_trading_snapshot_helpers(two_tenants):
    t = two_tenants
    mf = _mf_book(t.a)
    now = timezone.now()
    trading_snap = _snap(t.portfolio_a, 500000, at=now - timedelta(minutes=5))
    _snap(mf, 0, at=now)
    assert trading_portfolio(t.a) == t.portfolio_a
    assert latest_trading_snapshot(t.a) == trading_snap
    assert mf.id in mf_portfolio_ids(t.a)
    assert [p.id for p in book_portfolios(t.a)] == [t.portfolio_a.id]
    # Legacy (tenant-less) path: newest snapshot among *trading* portfolios only.
    b_snap = _snap(t.portfolio_b, 400000, at=now - timedelta(minutes=10))
    assert latest_trading_snapshot() == trading_snap
    PortfolioSnapshot.objects.filter(id=trading_snap.id).delete()
    assert latest_trading_snapshot() == b_snap


# ─── Overview ────────────────────────────────────────────────────────────

def test_overview_requires_auth():
    assert APIClient().get(OVERVIEW).status_code in (401, 403)
    assert APIClient().get(ATTRIBUTION).status_code in (401, 403)


def test_overview_empty_book_is_not_zero(two_tenants):
    t = two_tenants
    _mf_book(t.a)
    _snap(t.portfolio_a, 500000)
    body = _client(t.owner_a).get(OVERVIEW).json()

    assert body["total"]["cash"] == 500000.0
    assert body["total"]["mutual_funds_value"] is None
    assert body["total"]["components_missing"] == ["mutual_funds"]
    mf = body["components"]["mutual_funds"]
    assert mf["status"] == "not_connected"
    assert mf["current_value"] is None
    eq = body["components"]["equity"]
    assert eq["status"] == "empty"
    assert eq["market_value"] is None
    assert body["pnl"]["status"] == "empty"
    assert body["pnl"]["day"]["total"] is None
    assert body["risk"]["open_risk"] is None
    assert body["broker_reconciliation"]["status"] == "not_connected"   # only a paper link
    roles = {p["name"]: p["role"] for p in body["portfolios"]}
    assert roles == {"Default": "trading", "Mutual Funds": "mutual_funds"}


def test_overview_values_positions_mf_and_history(two_tenants):
    t = two_tenants
    trading = t.portfolio_a
    mf = _mf_book(t.a)
    now = timezone.now()
    _trade(trading, "HDFCBANK", status="FILLED", entry=100, qty=10, stop="95")
    _trade(trading, "TCS", status="CLOSED", entry=50, qty=10, realized=200, closed_at=now)
    _mf_holding(t.a, mf, units="10", cost="20", nav="25")
    _snap(trading, 500000, unrealized="0", at=now - timedelta(days=1))
    _snap(trading, 500100, unrealized="100", at=now)
    _snap(mf, 0, at=now)

    with mock.patch(MARK, return_value=(110.0, "live tick cache", now)):
        body = _client(t.owner_a).get(OVERVIEW).json()

    eq = body["components"]["equity"]
    assert eq["open_count"] == 1
    assert eq["unrealized_pnl"] == 100.0           # (110-100) × 10
    assert eq["open_risk"] == 150.0                # (110-95) × 10
    tot = body["total"]
    assert tot["trading_value"] == 500300.0        # capital + realized 200 + open 100
    assert tot["cash"] == 499200.0                 # capital + realized - cost 1000
    assert tot["mutual_funds_value"] == 250.0      # 10 units × NAV 25
    assert tot["value"] == 500550.0
    assert tot["invested"] == 1200.0               # equity cost 1000 + MF cost 200
    assert body["components"]["mutual_funds"]["status"] == "ok"
    assert body["components"]["mutual_funds"]["gain"] == 50.0
    assert body["pnl"]["day"]["realized"] == 200.0
    assert body["pnl"]["since_start"]["total"] == 300.0
    assert body["risk"]["daily_loss"]["used"] == 0.0
    classes = {r["key"]: r["value"] for r in body["allocation"]["by_asset_class"]}
    assert classes["equity"] == 1100.0
    assert classes["mutual_funds"] == 250.0
    assert {r["key"] for r in body["allocation"]["by_strategy"]} == {"Manual", "Mutual funds"}
    assert body["allocation"]["by_sector"][0]["key"] == "Banking"
    hist = body["history"]
    assert hist["status"] == "ok"
    assert hist["points"][-1]["equity"] == 500300.0   # capital + realized 200 + unrealized 100
    # The MF book's snapshots never enter the trading curve.
    assert all(p["equity"] >= 500000 for p in hist["points"])


def test_overview_options_component(two_tenants):
    t = two_tenants
    pos = OptionsPosition.objects.create(
        tenant=t.a, portfolio=t.portfolio_a, position_type="SHORT_STRADDLE", underlying="NIFTY",
        expiry=timezone.localdate() + timedelta(days=5), lot_size=65, lots=1, status="ACTIVE",
        current_pnl_inr=Decimal("-500"), trade_date=timezone.localdate())
    OptionsLeg.objects.create(position=pos, leg_role="SHORT_CE", symbol="NIFTYCE", strike=24200, qty=65,
                              open_price=Decimal("100"), current_price=Decimal("105"))
    OptionsLeg.objects.create(position=pos, leg_role="SHORT_PE", symbol="NIFTYPE", strike=24200, qty=65,
                              open_price=Decimal("80"), current_price=Decimal("80"))
    body = _client(t.owner_a).get(OVERVIEW).json()
    op = body["components"]["options"]
    assert op["status"] == "ok"
    assert op["unrealized_pnl"] == -500.0
    assert op["short_premium_sold"] == 11700.0      # (100 + 80) × 65
    assert op["positions_without_stop"] == 1
    assert body["total"]["trading_value"] == 499500.0
    assert body["allocation"]["by_sector"][0]["key"] == "Index options"


def test_overview_tenant_isolation(two_tenants):
    t = two_tenants
    mf_b = _mf_book(t.b)
    now = timezone.now()
    _trade(t.portfolio_b, "INFY", status="FILLED", entry=1000, qty=5)
    _trade(t.portfolio_b, "TCS", status="CLOSED", entry=50, realized=999, closed_at=now)
    _mf_holding(t.b, mf_b)
    _snap(t.portfolio_b, 123456, at=now)

    with mock.patch(MARK, return_value=(1010.0, "live tick cache", now)):
        a = _client(t.owner_a).get(OVERVIEW).json()
        b = _client(t.owner_b).get(OVERVIEW).json()

    assert a["components"]["equity"]["open_count"] == 0
    assert a["components"]["mutual_funds"]["status"] == "not_connected"
    assert a["pnl"]["status"] == "empty"
    assert a["history"]["status"] == "not_connected"
    assert {p["id"] for p in a["portfolios"]} == {str(t.portfolio_a.id)}
    assert b["components"]["equity"]["open_count"] == 1
    assert b["pnl"]["since_start"]["realized"] == 999.0
    assert b["components"]["mutual_funds"]["status"] == "ok"

    att_a = _client(t.owner_a).get(ATTRIBUTION).json()
    assert att_a["status"] == "empty"
    assert att_a["total_pnl"] is None


def test_attribution_groups_by_strategy_symbol_month(two_tenants):
    from apps.agents_core.models import AgentRun

    t = two_tenants
    now = timezone.now()
    run = AgentRun.objects.create(tenant=t.a, strategy_name="intraday_screener", strategy_version="1",
                                  portfolio=t.portfolio_a, triggered_by=t.owner_a, status="succeeded")
    _trade(t.portfolio_a, "TCS", status="CLOSED", entry=50, realized=300, closed_at=now, run=run)
    _trade(t.portfolio_a, "TCS", status="CLOSED", entry=50, realized=-100, closed_at=now, run=run)
    _trade(t.portfolio_a, "INFY", status="CLOSED", entry=50, realized=50, closed_at=now)
    body = _client(t.owner_a).get(ATTRIBUTION + "?months=3").json()
    assert body["status"] == "ok"
    assert body["total_pnl"] == 250.0
    strat = {g["key"]: g for g in body["by_strategy"]}
    assert strat["intraday_screener"]["pnl"] == 200.0
    assert strat["intraday_screener"]["win_rate"] == 50.0
    assert strat["Manual"]["pnl"] == 50.0
    assert [g["key"] for g in body["by_symbol"]] == ["TCS", "INFY"]
    assert body["by_month"][0]["key"] == timezone.localtime(now).strftime("%Y-%m")
