"""Integration tests for watchlist plumbing into other AlphaDesk surfaces:

  * TradingViewLink.watchlist FK as an autofire allowlist by symbol
  * Cross-owner watchlist binding is rejected at serializer level
  * /watchlists/by-symbol/?symbol=X lookup for the Setup page badges
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from apps.notifications.models import (
    TradingViewLink, TradingViewWatchlist,
)
from tests.factories import (
    MembershipFactory, TenantFactory, UserFactory,
)


pytestmark = pytest.mark.django_db


# ── Autofire allowlist via watchlist ────────────────────────────────────

class TestAutofireWatchlistGate:
    @pytest.fixture
    def setup(self, owner, paper_portfolio):
        t = owner.memberships.first().tenant
        wl = TradingViewWatchlist.objects.create(
            tenant=t, owner=owner,
            name="Allowed",
            kind=TradingViewWatchlist.Kind.MANUAL,
            symbols=["RELIANCE", "HDFCBANK"],
        )
        link = TradingViewLink.objects.create(
            tenant=t, owner=owner,
            display_name="autofire test",
            autofire_enabled=True,
            default_strategy_name="directional",
            portfolio=paper_portfolio,
            watchlist=wl,
            allowed_actions=["BUY"],
        )
        return link, wl

    def test_symbol_in_watchlist_fires(self, api_client, setup, monkeypatch):
        link, _wl = setup
        fired = []
        from apps.agents_core.tasks import run as run_task
        monkeypatch.setattr(run_task.execute_run, "delay",
                            lambda run_id: fired.append(run_id))

        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data=json.dumps({"symbol": "RELIANCE", "action": "BUY", "price": 100}),
            content_type="application/json",
        )
        assert resp.status_code == 200, resp.content
        assert resp.json()["run_id"] is not None
        assert len(fired) == 1

    def test_symbol_outside_watchlist_does_not_fire(self, api_client, setup, monkeypatch):
        link, _wl = setup
        fired = []
        from apps.agents_core.tasks import run as run_task
        monkeypatch.setattr(run_task.execute_run, "delay",
                            lambda run_id: fired.append(run_id))

        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data=json.dumps({"symbol": "ITC", "action": "BUY", "price": 100}),
            content_type="application/json",
        )
        # Persistence + Signal-ledger writes still happen; only autofire is gated.
        assert resp.status_code == 200
        assert resp.json()["run_id"] is None
        assert fired == []

    def test_no_watchlist_means_no_symbol_gate(self, api_client, owner, paper_portfolio, monkeypatch):
        """Sanity: a link without a watchlist FK fires for every symbol
        (assuming other gates pass). Confirms the gate is purely additive."""
        link = TradingViewLink.objects.create(
            tenant=owner.memberships.first().tenant, owner=owner,
            display_name="no wl",
            autofire_enabled=True,
            default_strategy_name="directional",
            portfolio=paper_portfolio,
        )
        fired = []
        from apps.agents_core.tasks import run as run_task
        monkeypatch.setattr(run_task.execute_run, "delay",
                            lambda run_id: fired.append(run_id))

        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data=json.dumps({"symbol": "ITC", "action": "BUY", "price": 100}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.json()["run_id"] is not None
        assert len(fired) == 1


class TestCrossOwnerBindGuard:
    def test_cannot_bind_to_another_users_watchlist(self, auth_client, owner):
        # Owner has a link
        t = owner.memberships.first().tenant
        link = TradingViewLink.objects.create(
            tenant=t, owner=owner, display_name="mine",
        )
        # Another user's watchlist
        other_t = TenantFactory()
        other_u = UserFactory()
        MembershipFactory(user=other_u, tenant=other_t, role="owner")
        foreign = TradingViewWatchlist.objects.create(
            tenant=other_t, owner=other_u, name="not mine",
            kind=TradingViewWatchlist.Kind.MANUAL, symbols=["RELIANCE"],
        )

        resp = auth_client.patch(
            f"/api/v1/notifications/tradingview/{link.id}/",
            {"watchlist": str(foreign.id)},
            format="json",
        )
        # Reachable IDs filter happens at queryset → DRF returns 400 saying
        # the FK doesn't point at a row in scope. Either 400 or "watchlist"
        # in the error body is acceptable.
        assert resp.status_code == 400


# ── /by-symbol/ lookup ──────────────────────────────────────────────────

class TestByLookup:
    def test_returns_watchlists_containing_symbol(self, auth_client, owner):
        t = owner.memberships.first().tenant
        TradingViewWatchlist.objects.create(
            tenant=t, owner=owner, name="core", symbols=["RELIANCE", "TCS"],
            kind=TradingViewWatchlist.Kind.MANUAL,
        )
        TradingViewWatchlist.objects.create(
            tenant=t, owner=owner, name="momentum", symbols=["RELIANCE", "INFY"],
            kind=TradingViewWatchlist.Kind.MANUAL,
        )
        TradingViewWatchlist.objects.create(
            tenant=t, owner=owner, name="other", symbols=["HDFCBANK"],
            kind=TradingViewWatchlist.Kind.MANUAL,
        )

        resp = auth_client.get(
            "/api/v1/notifications/tradingview/watchlists/by-symbol/?symbol=reliance",
        )
        assert resp.status_code == 200
        # The api response interceptor strips pagination envelope client-side,
        # but the raw DRF response here returns a bare list (no envelope on
        # custom actions). Compare directly.
        names = {row["name"] for row in resp.json()}
        assert names == {"core", "momentum"}

    def test_no_symbol_returns_empty(self, auth_client):
        resp = auth_client.get(
            "/api/v1/notifications/tradingview/watchlists/by-symbol/",
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_does_not_leak_other_owners_lists(self, auth_client, owner):
        # Symbol is in another owner's list — should NOT appear.
        other_t = TenantFactory()
        other_u = UserFactory()
        MembershipFactory(user=other_u, tenant=other_t, role="owner")
        TradingViewWatchlist.objects.create(
            tenant=other_t, owner=other_u, name="leaked", symbols=["RELIANCE"],
            kind=TradingViewWatchlist.Kind.MANUAL,
        )

        resp = auth_client.get(
            "/api/v1/notifications/tradingview/watchlists/by-symbol/?symbol=RELIANCE",
        )
        assert resp.status_code == 200
        assert resp.json() == []
