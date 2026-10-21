"""MF-S S9 screener API: auth, shared-catalog reads, tenant-isolated audit events."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from rest_framework.test import APIClient

from apps.events.models import Event
from apps.mutual_funds.models import MFDataIssue
from apps.mutual_funds.services.metrics_runner import compute_snapshots
from tests.mutual_funds.test_api import _client_for
from tests.mutual_funds.test_screener_pipeline import flexi_world  # noqa: F401 — fixture

pytestmark = pytest.mark.django_db
BASE = "/api/v1/mutual-funds"
FLEXI = "Equity Scheme - Flexi Cap Fund"


@pytest.fixture
def world(flexi_world, two_tenants):  # noqa: F811
    compute_snapshots(flexi_world)
    MFDataIssue.objects.create(scheme_id="F1", kind="jump", start_date=date(2024, 1, 2), end_date=date(2024, 1, 3),
                               detail={"move_pct": 30})
    return two_tenants


@pytest.mark.parametrize("path", ["screener/", "schemes/F1/metrics/", "schemes/F1/nav-history/"])
def test_requires_auth(world, path):
    assert APIClient().get(f"{BASE}/{path}").status_code == 401


def test_screener_lists_categories_without_category(world):
    r = _client_for(world.owner_a).get(f"{BASE}/screener/")
    assert r.status_code == 200
    body = r.json()
    assert body["result"] is None and body["metrics_as_of"] == "2026-10-16"
    cats = {c["category"]: c["count"] for c in body["available_categories"]}
    assert cats[FLEXI] == 10
    assert "Personal-use" in body["personal_use_note"]


def test_screener_ranks_with_components_and_index_check(world):
    r = _client_for(world.owner_a).get(f"{BASE}/screener/", {"category": FLEXI.lower(), "top": 3})
    assert r.status_code == 200
    res = r.json()["result"]
    assert res["category"] == FLEXI and [x["scheme_code"] for x in res["rows"]] == ["F5", "F4", "F8"]
    assert res["rows"][0]["components"]["cost"] is None and res["rows"][0]["expense_ratio_pct"] is None
    assert res["index_check"]["basis"] == "index_fund"
    r2 = _client_for(world.owner_a).get(f"{BASE}/screener/",
                                        {"category": FLEXI, "min_years": "1", "index_check": "0"})
    assert r2.json()["result"]["eligible_count"] == 10 and r2.json()["result"]["index_check"] is None


def test_screener_validation(world):
    c = _client_for(world.owner_a)
    assert c.get(f"{BASE}/screener/", {"category": "Nope"}).status_code == 400
    assert c.get(f"{BASE}/screener/", {"category": FLEXI, "min_sharpe": "x"}).status_code == 400
    assert c.get(f"{BASE}/screener/", {"category": FLEXI, "top": "x"}).status_code == 400


def test_screener_audit_event_is_tenant_isolated(world):
    _client_for(world.owner_a).get(f"{BASE}/screener/", {"category": FLEXI, "index_check": "0"})
    a = Event.objects.filter(tenant=world.a, step_name="mf_screener.run")
    assert a.count() == 1
    assert not Event.objects.filter(tenant=world.b, step_name="mf_screener.run").exists()
    # Tenant B sees the same shared ranking, and its own run is logged to B only.
    rb = _client_for(world.owner_b).get(f"{BASE}/screener/", {"category": FLEXI, "index_check": "0"})
    ra = _client_for(world.owner_a).get(f"{BASE}/screener/", {"category": FLEXI, "index_check": "0"})
    assert rb.json()["result"]["rows"] == ra.json()["result"]["rows"]
    assert Event.objects.filter(tenant=world.b, step_name="mf_screener.run").count() == 1
    assert Event.objects.filter(tenant=world.a, step_name="mf_screener.run").count() == 2


def test_scheme_metrics(world):
    c = _client_for(world.owner_a)
    body = c.get(f"{BASE}/schemes/F1/metrics/").json()
    assert body["category"] == FLEXI and body["data_missing"] is False
    assert body["metrics"]["as_of"] == "2026-10-16" and body["metrics"]["cagr_3y"] is not None
    assert body["reference_data"]["expense_ratio_pct"] is None
    assert body["data_issues"][0]["kind"] == "jump"
    assert body["risk_free_rate"]["source"]
    dead = c.get(f"{BASE}/schemes/DEAD/metrics/").json()
    assert dead["is_active"] is False and dead["data_missing"] is True and dead["metrics"] is None
    assert c.get(f"{BASE}/schemes/000000/metrics/").status_code == 404


def test_scheme_nav_history_with_benchmark(world):
    c = _client_for(world.owner_a)
    body = c.get(f"{BASE}/schemes/F1/nav-history/", {"days": 365, "benchmark": "1"}).json()
    assert 240 <= body["count"] <= 270 and body["last_date"] == "2026-10-16"
    b = body["benchmark"]
    assert b["index_name"] == "NIFTY 500" and b["points"][0]["value"] == pytest.approx(body["navs"][0]["nav"], rel=1e-3)
    plain = c.get(f"{BASE}/schemes/F1/nav-history/").json()
    assert plain["benchmark"] is None and plain["count"] > body["count"]
    assert c.get(f"{BASE}/schemes/F1/nav-history/", {"days": "x"}).status_code == 400
    assert Decimal(str(plain["navs"][-1]["nav"])) > 0
