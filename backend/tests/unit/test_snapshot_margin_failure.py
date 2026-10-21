"""_take_snapshot tolerates a margin-only failure.

Regression: Fyers' funds endpoint hangs off-hours while positions/holdings and
the token are fine; the whole link was flagged ERRORED.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from django.utils import timezone

from apps.market_data.adapters.base import Margin
from apps.market_data.api import broker_views
from apps.market_data.models import BrokerLink, BrokerPositionSnapshot
from tests.factories import BrokerLinkFactory

pytestmark = pytest.mark.django_db


class _Adapter:
    margin_exc: Exception | None = None

    def fetch_positions(self):
        return []

    def fetch_holdings(self):
        return []

    def fetch_margin(self):
        if self.margin_exc:
            raise self.margin_exc
        return Margin(available_cash=1000.0, used=0.0, total=1000.0, raw={})

    serialise_positions = staticmethod(lambda xs: [])
    serialise_holdings = staticmethod(lambda xs: [])
    serialise_margin = staticmethod(
        lambda m: {"available_cash": m.available_cash, "used": m.used, "total": m.total})


@pytest.fixture
def adapter(monkeypatch):
    a = _Adapter()
    monkeypatch.setattr(broker_views, "build_adapter", lambda link: a)
    return a


def test_margin_timeout_keeps_link_active_and_carries_last_margin(adapter):
    link = BrokerLinkFactory(broker_name="fyers", status="active")
    BrokerPositionSnapshot.objects.create(
        tenant=link.tenant, link=link, fetched_at=timezone.now() - timedelta(minutes=5),
        ok=True, margin={"available_cash": 750.0, "used": 0.0, "total": 750.0},
    )
    adapter.margin_exc = RuntimeError("Fyers funds timed out after 15s")

    snap = broker_views._take_snapshot(link)

    link.refresh_from_db()
    assert snap.ok is True
    assert snap.margin["available_cash"] == 750.0
    assert "margin unavailable" in snap.error
    assert link.status == BrokerLink.Status.ACTIVE
    assert "funds timed out" in link.last_error


def test_margin_failure_without_history_stores_empty_margin(adapter):
    link = BrokerLinkFactory(broker_name="fyers", status="active")
    adapter.margin_exc = RuntimeError("Fyers funds: Bad request")

    snap = broker_views._take_snapshot(link)

    assert snap.ok is True and snap.margin == {}
    link.refresh_from_db()
    assert link.status == BrokerLink.Status.ACTIVE


def test_margin_auth_failure_still_expires_link(adapter):
    link = BrokerLinkFactory(broker_name="fyers", status="active")
    adapter.margin_exc = RuntimeError("Fyers authentication failed — daily re-login required")

    snap = broker_views._take_snapshot(link)

    link.refresh_from_db()
    assert snap.ok is False
    assert link.status == BrokerLink.Status.EXPIRED


def test_clean_snapshot_clears_previous_error(adapter):
    link = BrokerLinkFactory(broker_name="fyers", status="active", last_error="old")
    snap = broker_views._take_snapshot(link)
    link.refresh_from_db()
    assert snap.ok and snap.error == "" and link.last_error == ""
    assert snap.margin["available_cash"] == 1000.0
