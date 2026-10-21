"""zerodha_login command + shared connect_broker() path, with kiteconnect faked.

No network: a stub ``kiteconnect`` module is injected into sys.modules, which
both the command and ZerodhaAdapter.authenticate() import from.
"""
from __future__ import annotations

import sys
import types
from io import StringIO

import pytest
from django.core.management import CommandError, call_command
from rest_framework.test import APIClient

from apps.market_data.models import BrokerLink
from apps.market_data.services.broker_connect import BrokerConnectError, connect_broker
from apps.market_data.services.crypto import decrypt_credentials
from tests.factories import MembershipFactory, TenantFactory, UserFactory

SECRET = "sekret-api-secret"
ACCESS = "acc-token-XYZ"


class FakeKite:
    instances: list["FakeKite"] = []
    profile_user_id = "AB1234"
    reject_session = False

    def __init__(self, api_key):
        self.api_key = api_key
        self.access_token = None
        FakeKite.instances.append(self)

    def login_url(self):
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"

    def generate_session(self, request_token, api_secret):
        if FakeKite.reject_session or request_token != "good-req":
            raise RuntimeError("Token is invalid or has expired.")
        assert api_secret == SECRET
        return {"access_token": ACCESS, "user_id": "AB1234", "user_name": "Owner"}

    def set_access_token(self, tok):
        self.access_token = tok

    def profile(self):
        if self.access_token != ACCESS:
            raise RuntimeError("TokenException: invalid token")
        return {"user_id": FakeKite.profile_user_id, "user_name": "Owner"}

    # used by the best-effort snapshot after connect
    def positions(self):
        return {"net": []}

    def holdings(self):
        return []

    def margins(self):
        return {"equity": {"available": {"cash": 1000}, "utilised": {"debits": 0}}}


@pytest.fixture(autouse=True)
def fake_kite(monkeypatch, settings):
    mod = types.ModuleType("kiteconnect")
    mod.KiteConnect = FakeKite
    monkeypatch.setitem(sys.modules, "kiteconnect", mod)
    FakeKite.instances = []
    FakeKite.reject_session = False
    settings.KITE_API_KEY = "kite-key"
    settings.KITE_API_SECRET = SECRET
    return FakeKite


@pytest.fixture
def tenant_owner(db):
    t = TenantFactory(slug="mf-owner")
    u = UserFactory(email="owner@alphadesk.test")
    MembershipFactory(user=u, tenant=t, role="owner")
    return t, u


def _run(*args, **kw):
    out = StringIO()
    call_command("zerodha_login", *args, stdout=out, **kw)
    return out.getvalue()


def test_login_creates_active_encrypted_link_without_printing_secrets(tenant_owner):
    t, u = tenant_owner
    out = _run("--tenant", "mf-owner", "--request-token", "good-req")

    link = BrokerLink.objects.get(tenant=t, broker_name="zerodha")
    assert link.owner == u
    assert link.status == BrokerLink.Status.ACTIVE
    assert link.credential_meta == {"user_id": "AB1234", "user_name": "Owner"}
    creds = decrypt_credentials(link.credential_blob)
    assert creds == {"api_key": "kite-key", "api_secret": SECRET, "access_token": ACCESS}
    assert "kite.zerodha.com/connect/login" in out
    assert SECRET not in out and ACCESS not in out


def test_rerun_updates_same_link(tenant_owner):
    t, _ = tenant_owner
    _run("--tenant", "mf-owner", "--request-token", "good-req")
    _run("--tenant", str(t.id), "--request-token", "good-req")
    assert BrokerLink.objects.filter(tenant=t, broker_name="zerodha").count() == 1


def test_accepts_full_redirect_url_and_prompt(tenant_owner, monkeypatch):
    t, _ = tenant_owner
    monkeypatch.setattr("builtins.input",
                        lambda _p: "http://127.0.0.1/?action=login&status=success&request_token=good-req")
    _run("--tenant", "mf-owner")
    assert BrokerLink.objects.filter(tenant=t, broker_name="zerodha").exists()


def test_rejected_request_token_saves_nothing(tenant_owner):
    t, _ = tenant_owner
    with pytest.raises(CommandError, match="rejected the request_token"):
        _run("--tenant", "mf-owner", "--request-token", "bad-req")
    assert not BrokerLink.objects.filter(tenant=t).exists()


def test_auth_probe_failure_saves_nothing(tenant_owner, fake_kite):
    t, _ = tenant_owner
    fake_kite.profile_user_id = ""  # profile() returns no user_id -> authenticate() False
    try:
        with pytest.raises(CommandError, match="Broker rejected"):
            _run("--tenant", "mf-owner", "--request-token", "good-req")
    finally:
        fake_kite.profile_user_id = "AB1234"
    assert not BrokerLink.objects.filter(tenant=t).exists()


def test_dry_run_does_not_save(tenant_owner):
    t, _ = tenant_owner
    out = _run("--tenant", "mf-owner", "--request-token", "good-req", "--dry-run")
    assert "authenticate()=True" in out
    assert not BrokerLink.objects.filter(tenant=t).exists()


def test_missing_settings_errors(tenant_owner, settings):
    settings.KITE_API_SECRET = ""
    with pytest.raises(CommandError, match="KITE_API_KEY"):
        _run("--tenant", "mf-owner", "--request-token", "good-req")


def test_unknown_tenant_errors(db):
    with pytest.raises(CommandError, match="No tenant"):
        _run("--tenant", "nope", "--request-token", "good-req")


def test_connect_broker_rejects_empty_and_unknown(tenant_owner):
    t, u = tenant_owner
    with pytest.raises(BrokerConnectError, match="Unknown broker"):
        connect_broker(tenant=t, owner=u, broker_name="nosuch", credentials={"a": 1})
    with pytest.raises(BrokerConnectError, match="non-empty"):
        connect_broker(tenant=t, owner=u, broker_name="zerodha", credentials={})


def test_connect_endpoint_still_validates_via_shared_path(tenant_owner):
    t, u = tenant_owner
    from rest_framework_simplejwt.tokens import RefreshToken

    from apps.accounts.api.jwt import _attach_tenant_claims

    access = RefreshToken.for_user(u).access_token
    _attach_tenant_claims(access, u)
    client = APIClient()
    client.force_authenticate(user=u)
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {access}")
    url = "/api/v1/brokers/zerodha/connect/"
    hdr = {}
    bad = client.post(url, {"credentials": {"api_key": "k", "access_token": "wrong"}},
                      format="json", **hdr)
    assert bad.status_code == 400
    assert not BrokerLink.objects.filter(tenant=t).exists()
    good = client.post(url, {"credentials": {"api_key": "kite-key", "access_token": ACCESS}},
                       format="json", **hdr)
    assert good.status_code == 201, good.content
    assert good.json()["status"] == "active"
