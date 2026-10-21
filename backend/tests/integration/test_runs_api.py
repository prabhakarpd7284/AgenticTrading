"""Runs API (UX redesign R1): receipt list/detail, schedule, status, rerun — incl. tenant isolation."""
from __future__ import annotations

from unittest import mock

import pytest
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from apps.accounts.api.jwt import _attach_tenant_claims
from apps.system.models import RunReceipt
from tests.factories import MembershipFactory, UserFactory


def _client(user) -> APIClient:
    c = APIClient()
    access = RefreshToken.for_user(user).access_token
    _attach_tenant_claims(access, user)
    c.force_authenticate(user=user)
    c.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    return c


@pytest.fixture
def receipts(two_tenants):
    now = timezone.now()
    mk = lambda **kw: RunReceipt.objects.create(kind="job", started_at=now, **kw)  # noqa: E731
    return {
        "system": mk(name="pull_amfi_nav", status="success", task_name="apps.mutual_funds.tasks.pull_amfi_nav"),
        "a": mk(name="backtest", status="failed", tenant=two_tenants.a, error="boom"),
        "b": mk(name="backtest", status="success", tenant=two_tenants.b, outputs={"secret": "b-only"}),
    }


@pytest.mark.django_db
def test_list_shows_own_and_system_receipts_only(two_tenants, receipts):
    res = _client(two_tenants.owner_a).get("/api/v1/runs/receipts/")
    assert res.status_code == 200
    ids = {r["id"] for r in res.json()["results"]}
    assert ids == {str(receipts["system"].id), str(receipts["a"].id)}
    scopes = {r["id"]: r["scope"] for r in res.json()["results"]}
    assert scopes[str(receipts["system"].id)] == "system"
    assert scopes[str(receipts["a"].id)] == "tenant"


@pytest.mark.django_db
def test_list_filters(two_tenants, receipts):
    c = _client(two_tenants.owner_a)
    assert [r["name"] for r in c.get("/api/v1/runs/receipts/?status=failed").json()["results"]] == ["backtest"]
    assert c.get("/api/v1/runs/receipts/?name=pull_amfi_nav").json()["count"] == 1
    assert c.get("/api/v1/runs/receipts/?kind=nope").json()["count"] == 0


@pytest.mark.django_db
def test_detail_isolation(two_tenants, receipts):
    c = _client(two_tenants.owner_a)
    assert c.get(f"/api/v1/runs/receipts/{receipts['b'].id}/").status_code == 404
    ok = c.get(f"/api/v1/runs/receipts/{receipts['a'].id}/")
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == "failed"
    assert "checks" in body
    assert body["previous"] is None


@pytest.mark.django_db
def test_detail_previous_run_is_from_visible_set(two_tenants, receipts):
    later = RunReceipt.objects.create(kind="job", name="backtest", status="success", tenant=two_tenants.b)
    body = _client(two_tenants.owner_b).get(f"/api/v1/runs/receipts/{later.id}/").json()
    assert body["previous"]["id"] == str(receipts["b"].id)


@pytest.mark.django_db
def test_unauthenticated_is_rejected():
    assert APIClient().get("/api/v1/runs/receipts/").status_code in (401, 403)
    assert APIClient().get("/api/v1/runs/status/").status_code in (401, 403)


@pytest.mark.django_db
def test_schedule_and_status_shapes(auth_client):
    s = auth_client.get("/api/v1/runs/schedule/")
    assert s.status_code == 200
    body = s.json()
    assert {"date", "slots", "summary", "jobs"} <= body.keys()
    assert any(j["key"] == "nightly-scan" and j["tracked"] for j in body["jobs"])

    st = auth_client.get("/api/v1/runs/status/")
    assert st.status_code == 200
    data = st.json()
    assert data["mode"]["mode"] == "PAPER"
    assert data["data"]["state"] == "synthetic"   # no real broker link in the test DB
    assert "label" in data["jobs"]
    assert "problems" in data["jobs"]


@pytest.mark.django_db
def test_status_data_state_uses_only_own_tenant_broker(two_tenants):
    from tests.factories import BrokerLinkFactory

    BrokerLinkFactory(tenant=two_tenants.b, owner=two_tenants.owner_b, broker_name="angel_one",
                      status="active")
    a = _client(two_tenants.owner_a).get("/api/v1/runs/status/").json()
    b = _client(two_tenants.owner_b).get("/api/v1/runs/status/").json()
    assert a["data"]["state"] == "synthetic"
    assert b["data"]["state"] != "synthetic"
    assert a["data"]["brokers"] == []


@pytest.mark.django_db
def test_rerun_owner_only_and_known_jobs_only(two_tenants):
    viewer = UserFactory()
    MembershipFactory(user=viewer, tenant=two_tenants.a, role="viewer")
    assert _client(viewer).post("/api/v1/runs/rerun/pull_amfi_nav/").status_code == 403

    owner = _client(two_tenants.owner_a)
    assert owner.post("/api/v1/runs/rerun/pipeline.screener_session/").status_code == 404
    fake = mock.Mock(id="celery-123")
    with mock.patch("apps.mutual_funds.tasks.pull_amfi_nav.delay", return_value=fake) as delay:
        res = owner.post("/api/v1/runs/rerun/pull_amfi_nav/")
    assert res.status_code == 202
    assert res.json()["celery_task_id"] == "celery-123"
    delay.assert_called_once_with(trigger="manual")
