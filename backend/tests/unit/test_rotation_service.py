"""Stage 3 — sector rotation service.

Covers the pure-Python logic of ``rotation_service.build_rotation``:
ranking, leader/laggard selection, breadth classification, and cache
behaviour.  We stub the yfinance provider so tests don't need network.
"""
from __future__ import annotations

from typing import Any

import pytest
from django.core.cache import cache

from apps.market_data.services import rotation_service
from apps.market_data.services.rotation_service import (
    CACHE_KEY,
    SECTOR_CONSTITUENTS,
    build_rotation,
)
from apps.market_data.services.pulse_service import SECTOR_TICKERS


@pytest.fixture(autouse=True)
def _clear_cache():
    cache.delete(CACHE_KEY)
    yield
    cache.delete(CACHE_KEY)


def _make_fake_raw(*, sector_pct: dict[str, float], stock_pct: dict[str, float]
                   ) -> dict[str, dict[str, Any]]:
    """Build a fake yfinance ``raw`` map keyed by yfinance symbols.

    Mirrors the shape ``YFinanceProvider.fetch`` produces: each entry has
    ``last``, ``prev_close``, ``change``, ``change_pct``, ``as_of``.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, pct in sector_pct.items():
        yf_sym = SECTOR_TICKERS[key]
        out[yf_sym] = {
            "last": 1000.0 + pct * 10,
            "prev_close": 1000.0,
            "change": pct * 10,
            "change_pct": pct,
            "as_of": "2026-04-20T04:00:00+00:00",
        }
    for symbol, pct in stock_pct.items():
        out[f"{symbol}.NS"] = {
            "last": 100.0 * (1 + pct / 100),
            "prev_close": 100.0,
            "change": pct,
            "change_pct": pct,
            "as_of": "2026-04-20T04:00:00+00:00",
        }
    return out


class _StubProvider:
    def __init__(self, raw: dict[str, dict[str, Any]]):
        self._raw = raw

    def fetch(self, symbols):  # noqa: ARG002 — symbols ignored by the stub
        return self._raw


# ---------------------------------------------------------------------------
# Core ranking + leader/laggard contract
# ---------------------------------------------------------------------------
def test_sectors_ranked_by_change_pct_desc(monkeypatch):
    raw = _make_fake_raw(
        sector_pct={"NIFTY_IT": 1.8, "NIFTY_BANK": -0.4, "NIFTY_AUTO": 0.9},
        stock_pct={"TCS": 2.5, "INFY": 1.9, "HCLTECH": -0.2, "WIPRO": -1.4, "TECHM": 0.7},
    )
    monkeypatch.setattr(rotation_service, "YFinanceProvider", lambda: _StubProvider(raw))

    payload = build_rotation(force=True)
    keys = [s["key"] for s in payload.sectors]

    # NIFTY_IT had the best move → rank 1, should be first in the list.
    assert keys[0] == "NIFTY_IT"
    assert payload.sectors[0]["rank"] == 1
    # Every sector gets a rank, sorted strictly desc (missing → bottom).
    ranks = [s["rank"] for s in payload.sectors]
    assert ranks == sorted(ranks)


def test_leaders_and_laggards_computed_for_it(monkeypatch):
    raw = _make_fake_raw(
        sector_pct={"NIFTY_IT": 1.8},
        stock_pct={"TCS": 2.5, "INFY": 1.9, "HCLTECH": -0.2, "WIPRO": -1.4, "TECHM": 0.7},
    )
    monkeypatch.setattr(rotation_service, "YFinanceProvider", lambda: _StubProvider(raw))

    payload = build_rotation(force=True)
    it = next(s for s in payload.sectors if s["key"] == "NIFTY_IT")

    # Leaders: top 3 by change_pct (desc). Best first.
    assert [l["symbol"] for l in it["leaders"]] == ["TCS", "INFY", "TECHM"]
    # Laggards: worst 3, worst first.
    assert [l["symbol"] for l in it["laggards"][:2]] == ["WIPRO", "HCLTECH"]


def test_breadth_classification(monkeypatch):
    """1 up, 1 down, 1 flat (|pct| ≤ 0.1) — sanity-check the thresholds."""
    raw = _make_fake_raw(
        sector_pct={"NIFTY_IT": 0.0},
        stock_pct={"TCS": 1.2, "INFY": -0.9, "HCLTECH": 0.05},
    )
    monkeypatch.setattr(rotation_service, "YFinanceProvider", lambda: _StubProvider(raw))

    payload = build_rotation(force=True)
    it = next(s for s in payload.sectors if s["key"] == "NIFTY_IT")
    # Note: SECTOR_CONSTITUENTS["NIFTY_IT"] has 5 names, but only 3 have data;
    # the other 2 are skipped (None change_pct → flat doesn't count them).
    assert it["breadth"]["up"] == 1
    assert it["breadth"]["down"] == 1
    # Flat counts: len(moves) - up - down — moves only include constituents
    # that produced a StockMove (all of them, since we always build one).
    assert it["breadth"]["up"] + it["breadth"]["down"] + it["breadth"]["flat"] == len(
        SECTOR_CONSTITUENTS["NIFTY_IT"]
    )


# ---------------------------------------------------------------------------
# Empty / missing data behaviour
# ---------------------------------------------------------------------------
def test_empty_sector_list_no_leaders(monkeypatch):
    """NIFTY_REALTY intentionally has no constituents — endpoint must not
    crash and should return empty leader/laggard arrays."""
    raw = _make_fake_raw(sector_pct={"NIFTY_REALTY": 1.0}, stock_pct={})
    monkeypatch.setattr(rotation_service, "YFinanceProvider", lambda: _StubProvider(raw))

    payload = build_rotation(force=True)
    realty = next(s for s in payload.sectors if s["key"] == "NIFTY_REALTY")
    assert realty["leaders"] == []
    assert realty["laggards"] == []
    assert realty["breadth"] == {"up": 0, "down": 0, "flat": 0}


def test_provider_failure_degrades_gracefully(monkeypatch):
    class _Broken:
        def fetch(self, symbols):  # noqa: ARG002
            raise RuntimeError("yfinance exploded")

    monkeypatch.setattr(rotation_service, "YFinanceProvider", lambda: _Broken())

    payload = build_rotation(force=True)
    # All sector entries present but with None change_pct — UI renders "—".
    assert len(payload.sectors) == len(SECTOR_TICKERS)
    assert all(s["change_pct"] is None for s in payload.sectors)
    assert any("yfinance exploded" in e for e in payload.errors)


# ---------------------------------------------------------------------------
# Cache behaviour — second call without force must not hit the provider.
# ---------------------------------------------------------------------------
def test_cache_hit_second_call(monkeypatch):
    calls = {"n": 0}

    class _Counting(_StubProvider):
        def fetch(self, symbols):  # noqa: ARG002
            calls["n"] += 1
            return self._raw

    raw = _make_fake_raw(sector_pct={"NIFTY_IT": 0.5}, stock_pct={"TCS": 0.4})
    monkeypatch.setattr(rotation_service, "YFinanceProvider", lambda: _Counting(raw))

    build_rotation(force=True)   # populates cache
    build_rotation(force=False)  # should be served from cache
    assert calls["n"] == 1
