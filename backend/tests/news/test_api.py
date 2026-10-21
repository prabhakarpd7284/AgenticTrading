"""Nightly-scan + trading-rules API: auth, payloads, tenant isolation."""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from django.utils import timezone
from rest_framework.test import APIClient

from apps.news.models import NewsItem, NewsItemSymbol, NightlyScan, NightlyScanRow
from apps.trading.models import Trade
from tests.factories import PortfolioFactory

pytestmark = pytest.mark.django_db


def _client_for(user) -> APIClient:
    from rest_framework_simplejwt.tokens import RefreshToken

    from apps.accounts.api.jwt import _attach_tenant_claims

    c = APIClient()
    access = RefreshToken.for_user(user).access_token
    _attach_tenant_claims(access, user)
    c.force_authenticate(user=user)
    c.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    return c


@pytest.fixture
def scan():
    s = NightlyScan.objects.create(for_date=date(2026, 10, 21), session_date=date(2026, 10, 20),
                                   status="success", started_at=timezone.now(),
                                   finished_at=timezone.now(),
                                   summary={"skip": [{"symbol": "HDFCBANK"}],
                                            "market": {"label": "neutral"}},
                                   timings={"total": 12.5})
    NightlyScanRow.objects.create(scan=s, symbol="INFY", sentiment_label="strong_positive",
                                  sentiment_score=Decimal("0.7"), allow_short=False,
                                  allowed="long", cpr_width_pct=Decimal("0.21"),
                                  headlines=[{"title": "Infosys beats", "url": "https://x"}])
    it = NewsItem.objects.create(url="https://x", url_hash="h1", title="Infosys beats",
                                 title_hash="t1", source="google_news",
                                 published_at=timezone.now() - timedelta(hours=2))
    NewsItemSymbol.objects.create(item=it, symbol="INFY")
    return s


def test_requires_auth(api_client, scan):
    assert api_client.get("/api/v1/nightly-scan/latest/").status_code in (401, 403)
    assert api_client.get("/api/v1/trading-rules/cooldown/").status_code in (401, 403)


def test_latest_and_by_date(two_tenants, scan):
    c = _client_for(two_tenants.owner_a)
    r = c.get("/api/v1/nightly-scan/latest/")
    assert r.status_code == 200
    body = r.json()
    assert body["scan"]["for_date"] == "2026-10-21"
    assert body["scan"]["rows"][0]["allowed"] == "long"
    assert body["scan"]["rows"][0]["cpr_width_pct"] == 0.21
    assert body["thresholds"]["STRONG_THRESHOLD"] == 0.5

    assert c.get("/api/v1/nightly-scan/2026-10-21/").json()["scan"]["id"] == scan.id
    assert c.get("/api/v1/nightly-scan/2026-10-22/").status_code == 404
    assert c.get("/api/v1/nightly-scan/not-a-date/").status_code == 400
    lst = c.get("/api/v1/nightly-scan/").json()["results"]
    assert lst[0]["restricted"] == 1 and lst[0]["total_secs"] == 12.5


def test_row_detail_with_headlines(two_tenants, scan):
    c = _client_for(two_tenants.owner_a)
    r = c.get("/api/v1/nightly-scan/2026-10-21/infy/")
    assert r.status_code == 200
    assert r.json()["row"]["symbol"] == "INFY"
    assert r.json()["headlines"][0]["title"] == "Infosys beats"
    assert c.get("/api/v1/nightly-scan/2026-10-21/WIPRO/").status_code == 404


def test_latest_empty_state(two_tenants):
    r = _client_for(two_tenants.owner_a).get("/api/v1/nightly-scan/latest/")
    assert r.status_code == 200 and r.json()["scan"] is None


def test_cooldown_endpoint_is_tenant_isolated(two_tenants):
    from apps.market_data.services.market_calendar import is_trading_day

    pa = two_tenants.portfolio_a
    d, n = timezone.localdate(), 0
    while n < 3:
        d -= timedelta(days=1)
        if is_trading_day(d):
            n += 1
            Trade.objects.create(tenant=two_tenants.a, portfolio=pa, symbol="TCS", side="BUY",
                                 entry_price=Decimal("100"), stop_loss=Decimal("95"),
                                 target=Decimal("110"), quantity=1, status="CLOSED",
                                 exit_price=Decimal("95"), realized_pnl=Decimal("-5"),
                                 trade_date=d)
    PortfolioFactory(tenant=two_tenants.a, name="Mutual Funds", capital=Decimal("0"))

    a = _client_for(two_tenants.owner_a).get("/api/v1/trading-rules/cooldown/").json()
    b = _client_for(two_tenants.owner_b).get("/api/v1/trading-rules/cooldown/").json()
    assert [r["portfolio_id"] for r in a["results"]] == [str(pa.id)]      # MF book excluded
    assert a["results"][0]["status"] == "cooldown"
    assert [r["portfolio_id"] for r in b["results"]] == [str(two_tenants.portfolio_b.id)]
    assert b["results"][0]["status"] == "normal"
    assert a["rules"]["loss_days"] == 3
