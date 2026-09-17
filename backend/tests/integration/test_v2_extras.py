"""Contract smoke tests for the v2-native endpoints that absorbed the old
/api/v1/legacy/* surface in the v1→v2 migration.

These guarantee:
*   Every endpoint resolves and returns 200 (or 401 if anon).
*   The JSON shape matches what the React hooks in lib/v2.ts destructure
    — so a future refactor of legacy_compat_views can't silently break
    the dashboard.
"""
from __future__ import annotations

import pytest
from django.urls import reverse


pytestmark = pytest.mark.django_db


V2_GET_NAMES = [
    "portfolio-summary",        # /portfolios/summary/
    "trading-positions",        # /positions/
    "trading-trades",           # /trades/
    "trading-options-positions",# /options-positions/
    "trading-risk",             # /risk/
    "trading-risk-alerts",      # /risk/alerts/
    "audit-feed",               # /events/audit/
    "rag-knowledge",            # /rag/knowledge/
    "system-status",            # /system/
]


@pytest.mark.parametrize("name", V2_GET_NAMES)
def test_v2_get_endpoints_reachable(auth_client, name):
    url = reverse(name)
    resp = auth_client.get(url)
    assert resp.status_code == 200, (
        f"{name} ({url}) returned {resp.status_code}: {resp.content[:200]!r}"
    )


def test_audit_feed_returns_results_array(auth_client):
    """The React `useAuditFeed` hook unwraps `.results` — must exist as a list."""
    resp = auth_client.get(reverse("audit-feed") + "?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body and isinstance(body["results"], list)


def test_positions_has_equity_and_options_keys(auth_client):
    """The React `usePositions` hook destructures `.equity` / `.options`."""
    resp = auth_client.get(reverse("trading-positions"))
    assert resp.status_code == 200
    body = resp.json()
    assert "equity"  in body
    assert "options" in body
    assert isinstance(body["equity"],  list)
    assert isinstance(body["options"], list)


def test_portfolio_summary_exposes_combined_pnl_block(auth_client):
    """Dashboard KPI strip reads `.combined.options_pnl` — must exist."""
    resp = auth_client.get(reverse("portfolio-summary"))
    assert resp.status_code == 200
    body = resp.json()
    assert "combined" in body
    for k in ("equity_pnl", "options_pnl", "total_pnl"):
        assert k in body["combined"]


def test_anon_blocked(api_client):
    """Read-only but still requires auth — no PII leak."""
    resp = api_client.get(reverse("portfolio-summary"))
    assert resp.status_code in (401, 403)


# --------------------------------------------------------------------------
# Mutating endpoints
# --------------------------------------------------------------------------
def test_pause_then_resume_flips_system_flag(auth_client):
    pause = auth_client.post(reverse("system-pause"))
    assert pause.status_code == 200
    assert pause.json()["ai_paused"] is True

    resume = auth_client.post(reverse("system-resume"))
    assert resume.status_code == 200
    assert resume.json()["ai_paused"] is False
