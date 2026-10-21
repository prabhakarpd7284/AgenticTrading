"""fyers_login command, with the Fyers SDK calls faked.

No network: the command's login-URL builder and token exchange, plus
FyersAdapter.authenticate(), are monkeypatched.
"""
from __future__ import annotations

from io import StringIO

import pytest
from django.core.management import CommandError, call_command

from apps.market_data.models import BrokerLink
from apps.market_data.services.crypto import decrypt_credentials
from tests.factories import MembershipFactory, TenantFactory, UserFactory

SECRET = "fyers-secret-key"
ACCESS = "fyers-access-XYZ"
REDIRECT = "https://trade.fyers.in/api-login/redirect-uri/index.html"


@pytest.fixture(autouse=True)
def fake_fyers(monkeypatch, settings):
    settings.FYERS_APP_ID = "ABCD1234-100"
    settings.FYERS_SECRET_KEY = SECRET
    settings.FYERS_REDIRECT_URI = REDIRECT
    calls = {"exchange": [], "auth_ok": True}

    def login_url(handshake, state, redirect_uri):
        return f"https://api-t1.fyers.in/api/v3/generate-authcode?client_id={handshake['app_id']}"

    def exchange(handshake, params, redirect_uri):
        calls["exchange"].append((params["auth_code"], redirect_uri))
        if params["auth_code"] != "good-code":
            raise ValueError("fyers token exchange failed")
        return {**handshake, "access_token": ACCESS}, {}

    cmd = "apps.market_data.management.commands.fyers_login"
    monkeypatch.setattr(f"{cmd}._fyers_login_url", login_url)
    monkeypatch.setattr(f"{cmd}._fyers_exchange", exchange)
    monkeypatch.setattr("plugins.broker_fyers.adapter.FyersAdapter.authenticate",
                        lambda self: calls["auth_ok"])
    monkeypatch.setattr("apps.market_data.api.broker_views._take_snapshot", lambda link: None)
    return calls


@pytest.fixture
def tenant_owner(db):
    t = TenantFactory(slug="fy-owner")
    u = UserFactory(email="owner@alphadesk.test")
    MembershipFactory(user=u, tenant=t, role="owner")
    return t, u


def _run(*args, **kw):
    out = StringIO()
    call_command("fyers_login", *args, stdout=out, **kw)
    return out.getvalue()


def test_login_creates_active_encrypted_link_without_printing_secrets(tenant_owner, fake_fyers):
    t, u = tenant_owner
    out = _run("--tenant", "fy-owner", "--auth-code", "good-code")

    link = BrokerLink.objects.get(tenant=t, broker_name="fyers")
    assert link.owner == u
    assert link.status == BrokerLink.Status.ACTIVE
    assert decrypt_credentials(link.credential_blob) == {
        "app_id": "ABCD1234-100", "secret_key": SECRET, "access_token": ACCESS,
    }
    assert fake_fyers["exchange"] == [("good-code", REDIRECT)]
    assert "generate-authcode" in out
    assert SECRET not in out and ACCESS not in out


def test_accepts_full_redirect_url_from_prompt(tenant_owner, monkeypatch, fake_fyers):
    monkeypatch.setattr("builtins.input",
                        lambda _: f"{REDIRECT}?s=ok&code=200&auth_code=good-code&state=abc")
    _run("--tenant", "fy-owner")
    assert fake_fyers["exchange"] == [("good-code", REDIRECT)]
    assert BrokerLink.objects.filter(broker_name="fyers").count() == 1


def test_rejected_auth_code_saves_nothing(tenant_owner):
    with pytest.raises(CommandError, match="rejected the auth_code"):
        _run("--tenant", "fy-owner", "--auth-code", "bad-code")
    assert not BrokerLink.objects.filter(broker_name="fyers").exists()


def test_auth_probe_failure_saves_nothing(tenant_owner, fake_fyers):
    fake_fyers["auth_ok"] = False
    with pytest.raises(CommandError, match="not saved"):
        _run("--tenant", "fy-owner", "--auth-code", "good-code")
    assert not BrokerLink.objects.filter(broker_name="fyers").exists()


def test_dry_run_does_not_save(tenant_owner):
    out = _run("--tenant", "fy-owner", "--auth-code", "good-code", "--dry-run")
    assert "NOT saved" in out
    assert not BrokerLink.objects.filter(broker_name="fyers").exists()


def test_missing_settings_errors(tenant_owner, settings):
    settings.FYERS_APP_ID = ""
    with pytest.raises(CommandError, match="FYERS_APP_ID"):
        _run("--tenant", "fy-owner", "--auth-code", "good-code")
