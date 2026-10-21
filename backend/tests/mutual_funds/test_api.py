"""MF-UI REST API tests — fixtures only (no real holdings exist)."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from django.utils import timezone
from rest_framework.test import APIClient

from apps.events.models import Event
from apps.events.services.event_writer import emit
from apps.mutual_funds.models import MFHolding, MFNav, MFScheme
from tests.factories import MembershipFactory, UserFactory

pytestmark = pytest.mark.django_db

LARGE = "Open Ended Schemes(Equity Scheme - Large Cap Fund)"
LIQUID = "Open Ended Schemes(Debt Scheme - Liquid Fund)"
BASE = "/api/v1/mutual-funds"


def _client_for(user) -> APIClient:
    from rest_framework_simplejwt.tokens import RefreshToken

    from apps.accounts.api.jwt import _attach_tenant_claims

    c = APIClient()
    access = RefreshToken.for_user(user).access_token
    _attach_tenant_claims(access, user)
    c.force_authenticate(user=user)
    c.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    return c


def _scheme(code, name, plan="Regular Plan", category=LARGE, amc="Acme Mutual Fund"):
    return MFScheme.objects.create(
        scheme_code=code, amc_name=amc, scheme_name=name, plan=plan, option="Growth Option",
        category=category, is_direct="direct" in plan.lower(),
    )


def _nav(s, d, v):
    return MFNav.objects.create(scheme=s, nav_date=d, nav=Decimal(str(v)))


def _holding(tenant, pf, s, units, cost=None, folio="F1"):
    return MFHolding.objects.create(
        tenant=tenant, portfolio=pf, scheme=s, folio_number=folio,
        units=Decimal(str(units)), avg_cost_per_unit=None if cost is None else Decimal(str(cost)),
        source=MFHolding.SOURCE_CAS,
    )


@pytest.fixture
def world(two_tenants):
    today = timezone.localdate()
    large = _scheme("100001", "Acme Bluechip Fund - Regular Plan - Growth")
    liquid = _scheme("100002", "Acme Liquid Fund - Direct Plan - Growth", plan="Direct Plan", category=LIQUID)
    other = _scheme("100003", "Zeta Midcap Fund - Direct Plan - Growth", plan="Direct Plan", amc="Zeta MF")
    _nav(large, today - timedelta(days=400), 80)
    _nav(large, today - timedelta(days=30), 95)
    _nav(large, today, 100)
    _nav(liquid, today, 2000)
    _nav(other, today, 50)

    t = two_tenants
    _holding(t.a, t.portfolio_a, large, 10, cost=50)        # value 1000, invested 500
    _holding(t.a, t.portfolio_a, liquid, 1, cost=None)      # value 2000, no cost
    _holding(t.a, t.portfolio_a, other, 0, cost=40, folio="CLOSED")  # closed folio
    _holding(t.b, t.portfolio_b, other, 999, cost=1)        # tenant B only
    return t


# ── holdings ─────────────────────────────────────────────────────────────

def test_holdings_values_and_summary(world):
    r = _client_for(world.owner_a).get(f"{BASE}/holdings/")
    assert r.status_code == 200
    body = r.json()
    codes = [h["scheme_code"] for h in body["results"]]
    assert sorted(codes) == ["100001", "100002"]  # closed folio hidden, B's hidden

    large = next(h for h in body["results"] if h["scheme_code"] == "100001")
    assert large["current_value"] == 1000.0
    assert large["invested"] == 500.0
    assert large["gain"] == 500.0
    assert large["gain_pct"] == 100.0
    assert large["category"] == "Equity Scheme - Large Cap Fund"
    assert large["nav_is_stale"] is False

    s = body["summary"]
    assert s["holding_count"] == 2
    assert s["current_value"] == 3000.0
    assert s["invested"] == 500.0
    assert s["gain"] == 500.0             # only over holdings with a cost basis
    assert s["gain_pct"] == 100.0
    assert s["missing_cost_count"] == 1
    alloc = {a["category"]: a["pct"] for a in s["allocation"]}
    assert alloc == {"Debt Scheme - Liquid Fund": 66.67, "Equity Scheme - Large Cap Fund": 33.33}


def test_holdings_include_closed_flag(world):
    r = _client_for(world.owner_a).get(f"{BASE}/holdings/?include_closed=1")
    assert {h["folio_number"] for h in r.json()["results"]} == {"F1", "CLOSED"}


def test_holdings_cross_tenant_isolation(world):
    rb = _client_for(world.owner_b).get(f"{BASE}/holdings/").json()
    assert [h["units"] for h in rb["results"]] == [999.0]
    # A cannot read B's portfolio even by id.
    ra = _client_for(world.owner_a).get(f"{BASE}/holdings/?portfolio={world.portfolio_b.id}")
    assert ra.status_code == 400


def test_holdings_bad_portfolio_id_is_400(world):
    assert _client_for(world.owner_a).get(f"{BASE}/holdings/?portfolio=nope").status_code == 400


def test_holdings_empty_state(two_tenants):
    body = _client_for(two_tenants.owner_a).get(f"{BASE}/holdings/").json()
    assert body["results"] == []
    assert body["summary"]["holding_count"] == 0
    assert body["summary"]["current_value"] == 0
    assert body["summary"]["gain_pct"] is None
    assert body["summary"]["allocation"] == []


def test_requires_auth(db):
    assert APIClient().get(f"{BASE}/holdings/").status_code == 401


# ── schemes ──────────────────────────────────────────────────────────────

def test_scheme_search_terms_and_pagination(world):
    c = _client_for(world.owner_a)
    body = c.get(f"{BASE}/schemes/?q=acme growth").json()
    assert body["count"] == 2
    assert {s["scheme_code"] for s in body["results"]} == {"100001", "100002"}
    assert c.get(f"{BASE}/schemes/?q=acme&direct=1").json()["count"] == 1
    assert c.get(f"{BASE}/schemes/?q=100003").json()["results"][0]["amc_name"] == "Zeta MF"
    assert c.get(f"{BASE}/schemes/?q=a").json()["count"] == 0
    page = c.get(f"{BASE}/schemes/?q=fund&limit=1").json()
    assert page["count"] == 3 and len(page["results"]) == 1 and page["next"]


def test_scheme_detail_nav_history_window(world):
    c = _client_for(world.owner_a)
    body = c.get(f"{BASE}/schemes/100001/?days=90").json()
    assert body["scheme"]["latest_nav"] == "100.0000"
    assert [n["nav"] for n in body["navs"]] == [95.0, 100.0]
    assert body["held_units"] == 10.0
    assert len(c.get(f"{BASE}/schemes/100001/?days=1000").json()["navs"]) == 3


def test_scheme_detail_held_units_tenant_scoped(world):
    assert _client_for(world.owner_a).get(f"{BASE}/schemes/100003/").json()["held_units"] == 0.0
    assert _client_for(world.owner_b).get(f"{BASE}/schemes/100003/").json()["held_units"] == 999.0


def test_scheme_detail_404(world):
    assert _client_for(world.owner_a).get(f"{BASE}/schemes/nope/").status_code == 404


# ── advisor ──────────────────────────────────────────────────────────────

def _finding(tenant, kind, text, ts, **payload):
    return emit(
        tenant=tenant, type=Event.Type.WORKFLOW_STEP_COMPLETED, text=text, severity="warn",
        actor_kind="workflow", step_name=f"mf_advisor.{kind}", ts=ts,
        payload={"advisor": "mf", "kind": kind, **payload},
    )


def test_advisor_findings_latest_run_only_and_isolated(world):
    now = timezone.now()
    _finding(world.a, "overlap", "old run", now - timedelta(days=2))
    _finding(world.a, "direct_vs_regular", "drag est", now - timedelta(seconds=5), is_estimate=True)
    _finding(world.a, "overlap", "new run", now)
    _finding(world.b, "overlap", "B secret", now)
    emit(tenant=world.a, type=Event.Type.WORKFLOW_STEP_COMPLETED, text="unrelated",
         step_name="screener.scan", ts=now)

    body = _client_for(world.owner_a).get(f"{BASE}/advisor/findings/").json()
    assert body["count"] == 2
    assert [f["text"] for f in body["results"]] == ["drag est", "new run"]
    assert body["results"][0]["is_estimate"] is True
    assert body["results"][0]["kind"] == "direct_vs_regular"


def test_advisor_findings_empty_state(two_tenants):
    body = _client_for(two_tenants.owner_a).get(f"{BASE}/advisor/findings/").json()
    assert body == {"latest_run_at": None, "count": 0, "results": []}


def test_advisor_run_emits_for_own_tenant_only(world):
    r = _client_for(world.owner_a).post(f"{BASE}/advisor/run/", {}, format="json")
    assert r.status_code == 200
    body = r.json()
    kinds = {f["kind"] for f in body["findings"]}
    assert "direct_vs_regular" in kinds  # 100001 is a regular plan
    assert all(f["is_estimate"] for f in body["findings"] if f["kind"] == "direct_vs_regular")
    assert Event.objects.filter(tenant=world.a, step_name__startswith="mf_advisor.").count() == body["count"]
    assert not Event.objects.filter(tenant=world.b, step_name__startswith="mf_advisor.").exists()


def test_advisor_run_owner_only(world):
    trader = UserFactory()
    MembershipFactory(user=trader, tenant=world.a, role="trader")
    assert _client_for(trader).post(f"{BASE}/advisor/run/", {}, format="json").status_code == 403


def test_advisor_run_rejects_foreign_portfolio(world):
    r = _client_for(world.owner_a).post(
        f"{BASE}/advisor/run/", {"portfolio": str(world.portfolio_b.id)}, format="json")
    assert r.status_code == 400


# ── status ───────────────────────────────────────────────────────────────

def test_status_counts_only_held_stale_schemes(world):
    stale_held = _scheme("100009", "Old Fund - Regular Plan - Growth")
    _nav(stale_held, timezone.localdate() - timedelta(days=60), 10)
    _holding(world.a, world.portfolio_a, stale_held, 5, cost=10)
    _scheme("100010", "Never Priced Fund")  # stale globally, not held -> ignored

    body = _client_for(world.owner_a).get(f"{BASE}/status/").json()
    assert body["latest_nav_date"] == timezone.localdate().isoformat()
    assert body["catalog_is_stale"] is False
    assert body["held_scheme_count"] == 3
    assert body["stale_holdings_count"] == 1
    assert body["stale_holdings"][0]["scheme_code"] == "100009"
    assert body["last_pull_at"] is not None
    # B holds only a fresh scheme
    assert _client_for(world.owner_b).get(f"{BASE}/status/").json()["stale_holdings_count"] == 0


def test_status_catalog_date_ignores_forward_dated_liquid_navs(world):
    liquid = MFScheme.objects.get(scheme_code="100002")
    _nav(liquid, timezone.localdate() + timedelta(days=2), 2001)  # AMFI weekend NAV
    body = _client_for(world.owner_a).get(f"{BASE}/status/").json()
    assert body["latest_nav_date"] == timezone.localdate().isoformat()
    assert body["max_nav_date"] == (timezone.localdate() + timedelta(days=2)).isoformat()


def test_catalog_date_not_pulled_back_by_backfilled_history(world):
    """A history backfill can give an older day more rows than today's full
    pull; the catalog date must still be today."""
    from apps.mutual_funds.services.nav_backfill import catalog_date

    today = timezone.localdate()
    older = today - timedelta(days=3)
    for i in range(10):
        s = _scheme(f"2000{i:02d}", f"Backfilled Fund {i}")
        _nav(s, older, 10 + i)
        if i < 6:
            _nav(s, today, 11 + i)
    _nav(MFScheme.objects.get(scheme_code="100002"), today + timedelta(days=2), 2001)
    assert catalog_date() == today


def test_status_empty_db(two_tenants):
    body = _client_for(two_tenants.owner_a).get(f"{BASE}/status/").json()
    assert body["latest_nav_date"] is None
    assert body["catalog_is_stale"] is True
    assert body["held_scheme_count"] == 0
    assert body["stale_holdings"] == []
