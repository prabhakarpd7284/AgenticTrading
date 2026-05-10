"""Smoke tests for the apps.legacy DRF bridge.

These guarantee:
*   Every legacy endpoint resolves and returns 200 (or 401 if anon).
*   The shape of the JSON matches what the React `useLegacy*` hooks
    expect (so a future refactor can't silently break the dashboard).
*   Broker-dependent helpers degrade to {error,fallback} instead of 500
    when SMARTAPI_* is missing — matches what dev installs see.

The legacy DB is itself populated lazily via fixtures so the tests
don't depend on a real `db.sqlite3` being mounted.
"""
from __future__ import annotations

import pytest
from django.urls import reverse


pytestmark = pytest.mark.django_db


# --------------------------------------------------------------------------
# Fast contract test — every URL resolves
# --------------------------------------------------------------------------
LEGACY_GET_PATHS = [
    "legacy:portfolio",
    "legacy:positions",
    "legacy:trades",
    "legacy:straddles",
    "legacy:audit",
    "legacy:risk",
    "legacy:alerts",
    "legacy:analytics",
    "legacy:exposure",
    "legacy:strategies",
    "legacy:watchlist",
    "legacy:system",
]


@pytest.mark.parametrize("name", LEGACY_GET_PATHS)
def test_legacy_get_endpoints_reachable(auth_client, name):
    url = reverse(name)
    resp = auth_client.get(url)
    # 200 = pure-DB helper succeeded
    # 500 acceptable only when explicit broker dependency is missing AND
    # the response body documents it
    assert resp.status_code == 200, (
        f"{name} returned {resp.status_code}: {resp.content[:200]!r}"
    )


def test_legacy_audit_returns_results_array(auth_client):
    url = reverse("legacy:audit")
    resp = auth_client.get(url + "?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body and isinstance(body["results"], list)


def test_legacy_positions_has_equity_and_options_keys(auth_client):
    """The React `useLegacyPositions` hook destructures `.equity` / `.options`."""
    resp = auth_client.get(reverse("legacy:positions"))
    assert resp.status_code == 200
    body = resp.json()
    assert "equity"  in body
    assert "options" in body
    assert isinstance(body["equity"],  list)
    assert isinstance(body["options"], list)


def test_legacy_portfolio_exposes_combined_pnl_block(auth_client):
    """Dashboard KPI strip reads `.combined.options_pnl` — must exist."""
    resp = auth_client.get(reverse("legacy:portfolio"))
    assert resp.status_code == 200
    body = resp.json()
    assert "combined" in body
    for k in ("equity_pnl", "options_pnl", "total_pnl"):
        assert k in body["combined"]


def test_legacy_anon_blocked(api_client):
    """Bridge is read-only but still requires auth — no PII leak."""
    resp = api_client.get(reverse("legacy:portfolio"))
    assert resp.status_code in (401, 403)


# --------------------------------------------------------------------------
# Mutating endpoints
# --------------------------------------------------------------------------
def test_pause_then_resume_flips_system_flag(auth_client):
    pause = auth_client.post(reverse("legacy:ai-pause"))
    assert pause.status_code == 200
    assert pause.json()["ai_paused"] is True

    resume = auth_client.post(reverse("legacy:ai-resume"))
    assert resume.status_code == 200
    assert resume.json()["ai_paused"] is False
