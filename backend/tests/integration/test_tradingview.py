"""Integration tests for the TradingView webhook integration.

Three flows that need to be tight:

  1. Operator creates a link via POST /notifications/tradingview/ → gets back
     a webhook URL embedding the secret.
  2. TradingView (public) POSTs an alert to /webhooks/tradingview/<secret>/ →
     a TradingViewSignal row is created, a strategies.Signal ledger row is
     created (so Now feed + monthly report pick it up), the link's
     receive_count is bumped.
  3. With autofire enabled + a valid strategy + portfolio, the same POST also
     enqueues an AgentRun.

Plus the security path: a webhook URL with an unknown secret returns 404.
"""
from __future__ import annotations

import json

import pytest
from django.urls import reverse

from apps.notifications.models import TradingViewLink, TradingViewSignal
from apps.notifications.services.tradingview import (
    ParsedAlert, parse_payload,
)
from apps.strategies.models import Signal


pytestmark = pytest.mark.django_db


# ── Parser unit tests ───────────────────────────────────────────────────

class TestParsePayload:
    def test_json_with_full_fields(self):
        body = json.dumps({
            "symbol": "RELIANCE",
            "action": "BUY",
            "price": 1234.56,
            "strategy": "vwap_breakout",
            "comment": "rsi cross above 60",
        })
        p = parse_payload(body, "application/json")
        assert p.symbol == "RELIANCE"
        assert p.action == "BUY"
        assert p.price == 1234.56
        assert p.strategy == "vwap_breakout"

    def test_json_ticker_alias_and_extra_fields(self):
        # TradingView templates often use `ticker` + extra fields like
        # `{{plot_0}}`; we keep extras around without losing them.
        body = json.dumps({
            "ticker": "tcs",
            "side": "sell",
            "close": 3500,
            "exchange": "NSE",
        })
        p = parse_payload(body, "application/json")
        assert p.symbol == "TCS"
        assert p.action == "SELL"
        assert p.price == 3500.0
        assert p.extra == {"exchange": "NSE"}

    def test_plaintext_fallback(self):
        p = parse_payload("BUY HDFCBANK @ 1500.25", "text/plain")
        assert p.symbol == "HDFCBANK"
        assert p.action == "BUY"
        assert p.price == 1500.25

    def test_plaintext_without_price(self):
        p = parse_payload("SELL ICICIBANK", "text/plain")
        assert p.symbol == "ICICIBANK"
        assert p.action == "SELL"
        assert p.price is None

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            parse_payload("hello world nothing here", "text/plain")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_payload("", "text/plain")


# ── CRUD viewset ────────────────────────────────────────────────────────

class TestTradingViewLinkCRUD:
    def test_create_returns_webhook_url_with_secret(self, auth_client):
        resp = auth_client.post(
            "/api/v1/notifications/tradingview/",
            {"display_name": "Test alert"},
            format="json",
        )
        assert resp.status_code == 201, resp.content
        data = resp.json()
        assert data["display_name"] == "Test alert"
        assert data["webhook_secret"]   # non-empty
        assert data["webhook_url"].endswith(f"/api/v1/webhooks/tradingview/{data['webhook_secret']}/")

    def test_rotate_secret_changes_url(self, auth_client):
        link = TradingViewLink.objects.create(
            tenant=auth_client.handler._force_user.memberships.first().tenant,
            owner=auth_client.handler._force_user,
            display_name="rotate me",
        )
        old = link.webhook_secret
        resp = auth_client.post(
            f"/api/v1/notifications/tradingview/{link.id}/rotate-secret/",
        )
        assert resp.status_code == 200
        link.refresh_from_db()
        assert link.webhook_secret != old

    def test_list_filters_by_owner(self, auth_client, owner):
        # One link for this owner; another for a freshly-minted user/tenant.
        TradingViewLink.objects.create(
            tenant=owner.memberships.first().tenant,
            owner=owner,
            display_name="mine",
        )
        from tests.factories import UserFactory, MembershipFactory, TenantFactory
        other_t = TenantFactory()
        other_u = UserFactory()
        MembershipFactory(user=other_u, tenant=other_t, role="owner")
        TradingViewLink.objects.create(
            tenant=other_t,
            owner=other_u,
            display_name="not mine",
        )

        resp = auth_client.get("/api/v1/notifications/tradingview/")
        assert resp.status_code == 200
        names = [row["display_name"] for row in resp.json()["results"]]
        assert names == ["mine"]

    def test_can_bind_same_tenant_portfolio(self, auth_client, owner, paper_portfolio):
        link = TradingViewLink.objects.create(
            tenant=owner.memberships.first().tenant,
            owner=owner,
            display_name="bind ok",
        )
        resp = auth_client.patch(
            f"/api/v1/notifications/tradingview/{link.id}/",
            {"portfolio": str(paper_portfolio.id)},
            format="json",
        )
        assert resp.status_code == 200, resp.content
        link.refresh_from_db()
        assert link.portfolio_id == paper_portfolio.id

    def test_cannot_bind_cross_tenant_portfolio(self, auth_client, owner, two_tenants):
        """Autofire routes trades into the bound portfolio — binding one from
        another tenant would cross the book boundary. Must 400."""
        link = TradingViewLink.objects.create(
            tenant=owner.memberships.first().tenant,
            owner=owner,
            display_name="bind cross",
        )
        resp = auth_client.patch(
            f"/api/v1/notifications/tradingview/{link.id}/",
            {"portfolio": str(two_tenants.portfolio_b.id)},
            format="json",
        )
        assert resp.status_code == 400
        # Errors are wrapped in an RFC 7807 problem doc → field lives under `detail`.
        assert "portfolio" in resp.json()["detail"]


# ── Webhook receiver ────────────────────────────────────────────────────

class TestWebhookReceiver:
    @pytest.fixture
    def link(self, owner):
        return TradingViewLink.objects.create(
            tenant=owner.memberships.first().tenant,
            owner=owner,
            display_name="webhook test",
        )

    def test_unknown_secret_returns_404(self, api_client):
        resp = api_client.post(
            "/api/v1/webhooks/tradingview/this-is-not-a-real-secret/",
            data="BUY HDFCBANK",
            content_type="text/plain",
        )
        assert resp.status_code == 404

    def test_valid_json_persists_signal_and_bumps_counter(self, api_client, link):
        body = json.dumps({"symbol": "RELIANCE", "action": "BUY", "price": 1500})
        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data=body, content_type="application/json",
        )
        assert resp.status_code == 200, resp.content
        data = resp.json()
        assert data["received"] is True
        assert data["parsed"]["symbol"] == "RELIANCE"
        assert data["parse_error"] is None
        assert data["run_id"] is None   # autofire is OFF by default

        link.refresh_from_db()
        assert link.receive_count == 1
        assert link.last_received_at is not None

        # The audit row + the strategies.Signal ledger row both exist.
        assert TradingViewSignal.objects.filter(link=link).count() == 1
        ledger = Signal.objects.filter(symbol="RELIANCE", source=Signal.Source.TRADINGVIEW)
        assert ledger.exists()

    def test_garbage_payload_returns_200_with_parse_error(self, api_client, link):
        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data="lorem ipsum nothing here",
            content_type="text/plain",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["parse_error"]
        # Audit row exists even though parsing failed — operator can inspect.
        assert TradingViewSignal.objects.filter(
            link=link, parse_error__icontains="parse",
        ).count() == 1

    def test_autofire_enqueues_agent_run(
        self, api_client, link, paper_portfolio, monkeypatch,
    ):
        link.autofire_enabled = True
        link.default_strategy_name = "directional"
        link.portfolio = paper_portfolio
        link.allowed_actions = ["BUY"]
        link.save()

        # Stub the celery task so we don't need a running broker for the test.
        captured = {}
        from apps.agents_core.tasks import run as run_task
        monkeypatch.setattr(run_task.execute_run, "delay",
                            lambda run_id: captured.setdefault("run_id", run_id))

        body = json.dumps({"symbol": "HDFCBANK", "action": "BUY", "price": 1500})
        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data=body, content_type="application/json",
        )
        assert resp.status_code == 200, resp.content
        data = resp.json()
        assert data["run_id"]
        assert captured.get("run_id") == data["run_id"]

    def test_autofire_action_outside_allowlist_does_not_fire(
        self, api_client, link, paper_portfolio, monkeypatch,
    ):
        link.autofire_enabled = True
        link.default_strategy_name = "directional"
        link.portfolio = paper_portfolio
        link.allowed_actions = ["BUY"]   # SELL not allowed
        link.save()

        fired = []
        from apps.agents_core.tasks import run as run_task
        monkeypatch.setattr(run_task.execute_run, "delay",
                            lambda run_id: fired.append(run_id))

        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data=json.dumps({"symbol": "HDFCBANK", "action": "SELL", "price": 1500}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.json()["run_id"] is None
        assert fired == []
        # Signal is still recorded — only the workflow fire was gated.
        assert TradingViewSignal.objects.filter(link=link).count() == 1

    def test_inactive_link_returns_404(self, api_client, link):
        link.is_active = False
        link.save()
        resp = api_client.post(
            f"/api/v1/webhooks/tradingview/{link.webhook_secret}/",
            data=json.dumps({"symbol": "X", "action": "BUY"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
