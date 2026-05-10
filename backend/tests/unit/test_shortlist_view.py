"""Contract test for /api/v1/market-data/shortlist/.

Resolves the URL, hits the view through DRF auth, asserts the response
shape the frontend consumes.  Stubs both ``build_rotation`` and the
history provider — no network, no yfinance.
"""
from __future__ import annotations

import pytest
from django.core.cache import cache
from django.urls import reverse

from apps.market_data.services import shortlist_service
from apps.market_data.services.shortlist_service import Fundamentals


class _StubHistoryProvider:
    def fetch(self, symbols):  # noqa: ARG002
        return {
            "TCS": Fundamentals(
                atr14=120.0, avg_volume_20=3_000_000, avg_close_20=4150.0,
                high_52w=4200, low_52w=3100, last_volume=4_500_000,
            ),
            "INFY": Fundamentals(
                atr14=25.0, avg_volume_20=5_000_000, avg_close_20=1450.0,
                high_52w=1800, low_52w=1300, last_volume=5_000_000,
            ),
        }


def _fake_rotation():
    class _RP:
        as_of = "2026-04-20T04:00:00+00:00"
        errors: list = []
        sectors = [
            {
                "key": "NIFTY_IT", "label": "IT", "rank": 1,
                "change_pct": 1.8, "last": 42000.0,
                "leaders":  [{"symbol": "TCS",  "last": 4150.0, "change_pct": 2.5,
                              "change": 2.5, "prev_close": 4000.0}],
                "laggards": [{"symbol": "INFY", "last": 1450.0, "change_pct": 1.9,
                              "change": 1.9, "prev_close": 1430.0}],
                "breadth": {"up": 3, "down": 1, "flat": 1},
            },
        ]
    return _RP()


@pytest.mark.django_db
def test_shortlist_endpoint_returns_expected_shape(auth_client, monkeypatch):
    monkeypatch.setattr(shortlist_service, "build_rotation", lambda force=False: _fake_rotation())
    monkeypatch.setattr(
        shortlist_service, "YFinanceHistoryProvider", lambda: _StubHistoryProvider(),
    )
    # Bust caches so the stubs actually run.
    cache.delete(shortlist_service.CACHE_KEY)

    url = reverse("shortlist")
    resp = auth_client.get(url, {"force": "1"})
    assert resp.status_code == 200, resp.content
    body = resp.json()

    # Top-level envelope
    assert set(body.keys()) >= {"as_of", "hot_sectors", "candidates", "filtered_out", "errors"}
    assert "NIFTY_IT" in body["hot_sectors"]
    assert isinstance(body["candidates"], list)

    # TCS should be first and clearly shaped
    assert body["candidates"], "expected at least one candidate"
    top = body["candidates"][0]
    assert top["symbol"] == "TCS"
    # Score is a float 0..100 with the sector & leader bonuses visible.
    assert 0 <= top["score"] <= 100
    assert any("sector #1" in r for r in top["reasons"])
    assert top["is_leader"] is True
