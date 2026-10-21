"""Broker OAuth callback must bounce the browser back to the SPA's /brokers page.

Regression: it used to redirect to /broker (no such route — the SPA rendered its
error boundary) and took the origin from the Referer, which on a broker
callback is the broker's own login page (kite.zerodha.com).
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from rest_framework.test import APIClient


@pytest.mark.django_db
def test_callback_error_redirects_to_spa_brokers_page_not_referer(settings):
    settings.FRONTEND_ORIGIN = "http://localhost:5173"
    r = APIClient().get(
        "/api/v1/brokers/zerodha/oauth/callback/",
        {"request_token": "tok", "state": "unknown-state"},
        HTTP_REFERER="https://kite.zerodha.com/connect/login?api_key=x",
    )
    assert r.status_code == 302
    loc = urlparse(r["Location"])
    assert f"{loc.scheme}://{loc.netloc}" == "http://localhost:5173"
    assert loc.path == "/brokers"
    qs = parse_qs(loc.query)
    assert qs["error"] == ["zerodha"]
    assert qs["reason"] == ["invalid or expired state"]
