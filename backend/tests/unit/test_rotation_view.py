"""Contract test for /api/v1/market-data/rotation/.

Checks that the URL resolves, renders through DRF, and returns the shape
the frontend expects.  Uses a stub provider so no yfinance required.

Reuses the project-wide `auth_client` fixture from tests/conftest.py, which
logs in a user with membership on a tenant — matching how the real app
authenticates DRF requests.
"""
from __future__ import annotations

import pytest
from django.core.cache import cache
from django.urls import reverse

from apps.market_data.services import rotation_service
from apps.market_data.services.pulse_service import SECTOR_TICKERS


class _StubProvider:
    def fetch(self, symbols):  # noqa: ARG002
        return {
            SECTOR_TICKERS["NIFTY_IT"]: {
                "last": 42000.0, "prev_close": 41500.0,
                "change": 500.0, "change_pct": 1.2, "as_of": "t",
            },
            "TCS.NS": {
                "last": 4150.0, "prev_close": 4000.0,
                "change": 150.0, "change_pct": 3.75, "as_of": "t",
            },
            "INFY.NS": {
                "last": 1450.0, "prev_close": 1470.0,
                "change": -20.0, "change_pct": -1.36, "as_of": "t",
            },
        }


@pytest.mark.django_db
def test_rotation_endpoint_returns_expected_shape(auth_client, monkeypatch):
    monkeypatch.setattr(rotation_service, "YFinanceProvider", lambda: _StubProvider())

    # Bust the rotation cache so the monkeypatch actually runs.
    cache.delete(rotation_service.CACHE_KEY)

    url = reverse("sector-rotation")
    resp = auth_client.get(url, {"force": "1"})
    assert resp.status_code == 200, resp.content
    body = resp.json()

    # Top-level envelope
    assert "as_of" in body
    assert "sectors" in body
    assert "errors" in body
    assert isinstance(body["sectors"], list)

    # IT sector should have TCS as top leader, INFY as laggard
    it = next(s for s in body["sectors"] if s["key"] == "NIFTY_IT")
    assert it["rank"] == 1  # only sector with a move in our stub
    assert it["leaders"][0]["symbol"] == "TCS"
    assert any(l["symbol"] == "INFY" for l in it["laggards"])

    # Breadth keys match the frontend contract
    assert set(it["breadth"].keys()) == {"up", "down", "flat"}
