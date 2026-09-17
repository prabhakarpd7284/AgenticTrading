"""Contract test for /api/v1/market-data/setup/.

Auth through DRF + URL resolved via reverse(); stubs the data port
so there's no yfinance/Redis dependency.  Asserts the JSON shape the
frontend consumes.
"""
from __future__ import annotations

import pytest
from django.core.cache import cache
from django.urls import reverse

from apps.market_data.services import setup_service
from trading.services.risk_engine import PULSE_CACHE_KEY


class _StubDataPort:
    def __init__(self, last, closes, bar=20.0):
        self._last = last
        self._closes = closes
        self._bar = bar

    def ltp(self, _s):
        return self._last

    def candles(self, _s, _i, n):
        return [
            {"o": c, "h": c + self._bar, "l": c - self._bar, "c": c,
             "v": 100_000, "t": "2026-04-21T09:15:00+00:00"}
            for c in self._closes[-n:]
        ]


@pytest.fixture(autouse=True)
def _clear_pulse_cache():
    cache.delete(PULSE_CACHE_KEY)
    yield
    cache.delete(PULSE_CACHE_KEY)


def _seed_tradeable_regime():
    cache.set(
        PULSE_CACHE_KEY,
        {"regime": {"vol": "moderate", "trend": "up", "global_tone": "risk-on",
                    "tradeable": True, "summary": "Tradeable."}},
        60,
    )


@pytest.mark.django_db
def test_setup_endpoint_returns_expected_shape(auth_client, monkeypatch):
    _seed_tradeable_regime()
    closes = [4100 + i * 2 for i in range(30)]

    # Patch the DefaultMarketData import site — build_setup imports lazily,
    # so monkeypatching the module attribute the orchestrator resolves.
    monkeypatch.setattr(
        "apps.market_data.services.data_port.DefaultMarketData",
        lambda _tid: _StubDataPort(last=4150.0, closes=closes),
    )

    url = reverse("setup-preview")
    resp = auth_client.get(url, {"symbol": "TCS", "side": "BUY"})
    assert resp.status_code == 200, resp.content
    body = resp.json()

    # Top-level envelope
    assert set(body.keys()) >= {
        "as_of", "symbol", "side", "market", "plan", "regime", "risk", "errors"
    }
    assert body["symbol"] == "TCS"
    assert body["side"] == "BUY"

    # Market snapshot
    assert body["market"]["last"] == pytest.approx(4150.0, rel=0.01)
    assert body["market"]["atr"] is not None
    assert body["market"]["candle_count"] >= 2

    # Plan
    plan = body["plan"]
    assert plan is not None
    assert plan["side"] == "BUY"
    assert plan["stop_loss"] < plan["entry_price"] < plan["target"]
    assert plan["quantity"] > 0

    # Risk breakdown
    assert isinstance(body["risk"]["criteria"], list)
    assert len(body["risk"]["criteria"]) == 10
    # Verdict object is well-shaped
    assert isinstance(body["risk"]["approved"], bool)
    assert isinstance(body["risk"]["reason"], str)


@pytest.mark.django_db
def test_setup_endpoint_missing_symbol_returns_400(auth_client):
    url = reverse("setup-preview")
    resp = auth_client.get(url)
    assert resp.status_code == 400


@pytest.mark.django_db
def test_setup_endpoint_invalid_side_returns_400(auth_client):
    url = reverse("setup-preview")
    resp = auth_client.get(url, {"symbol": "TCS", "side": "LONG"})
    assert resp.status_code == 400


@pytest.mark.django_db
def test_setup_endpoint_respects_capital_override(auth_client, monkeypatch):
    _seed_tradeable_regime()
    closes = [4100 + i * 2 for i in range(30)]
    monkeypatch.setattr(
        "apps.market_data.services.data_port.DefaultMarketData",
        lambda _tid: _StubDataPort(last=4150.0, closes=closes),
    )

    url = reverse("setup-preview")
    # 1L capital — position notional 41,500 at 10 qty is over the 10% cap,
    # so the quantity must be sized down relative to 5L baseline.
    resp_small = auth_client.get(url, {"symbol": "TCS", "side": "BUY",
                                       "capital": "100000"})
    resp_big = auth_client.get(url, {"symbol": "TCS", "side": "BUY",
                                     "capital": "1000000"})
    assert resp_small.status_code == 200
    assert resp_big.status_code == 200
    qty_small = resp_small.json()["plan"]["quantity"]
    qty_big = resp_big.json()["plan"]["quantity"]
    assert qty_big > qty_small, \
        f"Larger capital must allow larger position: {qty_small=} {qty_big=}"
