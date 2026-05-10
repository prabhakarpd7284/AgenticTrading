"""Stage 4 — shortlist service.

Covers the pure-Python logic of ``shortlist_service.build_shortlist``:
  * hard filters (ATR-%, turnover) actually reject,
  * the scorer rewards the things TRADING_FRAMEWORK.md says it should,
  * rotation output is reused (no duplicate fetch),
  * cache behaviour,
  * provider failure degrades gracefully.

We stub two things, never yfinance:
  1. ``build_rotation`` — so the upstream Stage 3 output is deterministic.
  2. A ``_StubHistoryProvider`` that returns canned ``Fundamentals``.
"""
from __future__ import annotations

import pytest
from django.core.cache import cache

from apps.market_data.services import shortlist_service
from apps.market_data.services.shortlist_service import (
    CACHE_KEY,
    Fundamentals,
    build_shortlist,
)
from apps.market_data.services.rotation_service import (
    CACHE_KEY as ROTATION_CACHE_KEY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_caches():
    """Every test starts with both Stage 3 + Stage 4 caches empty so the
    monkeypatches actually run on the next call.
    """
    cache.delete(CACHE_KEY)
    cache.delete(ROTATION_CACHE_KEY)
    yield
    cache.delete(CACHE_KEY)
    cache.delete(ROTATION_CACHE_KEY)


def _fake_rotation_payload(sectors):
    """Build a minimal RotationPayload-shaped stand-in.  The shortlist
    service only reads ``.sectors`` as a list of dicts."""
    class _RP:
        def __init__(self, s):
            self.sectors = s
            self.as_of = "2026-04-20T04:00:00+00:00"
            self.errors = []
    return _RP(sectors)


def _sector(key, label, rank, change_pct, leaders=None, laggards=None):
    """One sector dict in the shape rotation_service.build_rotation returns."""
    return {
        "key": key,
        "label": label,
        "rank": rank,
        "change_pct": change_pct,
        "last": 42000.0,
        "leaders": leaders or [],
        "laggards": laggards or [],
        "breadth": {"up": 3, "down": 1, "flat": 1},
    }


def _stock(symbol, change_pct, last):
    return {"symbol": symbol, "last": last, "change_pct": change_pct,
            "change": change_pct, "prev_close": last - change_pct}


class _StubHistoryProvider:
    """Return canned Fundamentals per symbol — no pandas, no network."""
    def __init__(self, data: dict[str, Fundamentals]):
        self._data = data
        self.called_with: list[list[str]] | None = None

    def fetch(self, symbols):
        self.called_with = list(symbols)
        return {s: self._data.get(s, Fundamentals()) for s in self._data}


# ---------------------------------------------------------------------------
# Hard filters — ATR-% and turnover rejections
# ---------------------------------------------------------------------------
def test_hard_filter_rejects_low_atr(monkeypatch):
    """A candidate with ATR/price < MIN_ATR_PCT must land in filtered_out,
    not in candidates."""
    rot = _fake_rotation_payload([
        _sector("NIFTY_IT", "IT", 1, 1.8,
                leaders=[_stock("TCS", 2.5, 4150.0),
                         _stock("INFY", 1.9, 1450.0)]),
    ])
    monkeypatch.setattr(shortlist_service, "build_rotation", lambda force=False: rot)

    # TCS: ATR 10 on price 4150 → 0.24% — below MIN_ATR_PCT (1.0%). Reject.
    # INFY: ATR 25 on price 1450 → 1.72% — passes.
    funds = {
        "TCS":  Fundamentals(atr14=10.0, avg_volume_20=3e6, avg_close_20=4150.0,
                             high_52w=4500, low_52w=3500, last_volume=3e6),
        "INFY": Fundamentals(atr14=25.0, avg_volume_20=5e6, avg_close_20=1450.0,
                             high_52w=1700, low_52w=1200, last_volume=5e6),
    }
    provider = _StubHistoryProvider(funds)
    payload = build_shortlist(force=True, history_provider=provider)

    syms = [c["symbol"] for c in payload.candidates]
    rejected = [r["symbol"] for r in payload.filtered_out]
    assert "TCS" in rejected
    assert "INFY" in syms
    tcs_reject = next(r for r in payload.filtered_out if r["symbol"] == "TCS")
    assert any("ATR" in reason for reason in tcs_reject["reject_reasons"])


def test_hard_filter_rejects_low_turnover(monkeypatch):
    """Turnover = avg_vol_20 * avg_close_20 / 1e7 (crore). Below 10cr → out."""
    rot = _fake_rotation_payload([
        _sector("NIFTY_IT", "IT", 1, 1.8,
                leaders=[_stock("INFY", 1.9, 1450.0)]),
    ])
    monkeypatch.setattr(shortlist_service, "build_rotation", lambda force=False: rot)

    # 50,000 shares × 1450 / 1e7 = 7.25 cr — below default 10cr threshold.
    funds = {
        "INFY": Fundamentals(atr14=25.0, avg_volume_20=50_000, avg_close_20=1450.0,
                             high_52w=1700, low_52w=1200, last_volume=50_000),
    }
    provider = _StubHistoryProvider(funds)
    payload = build_shortlist(force=True, history_provider=provider)

    assert all(c["symbol"] != "INFY" for c in payload.candidates)
    assert any(r["symbol"] == "INFY" for r in payload.filtered_out)
    infy = next(r for r in payload.filtered_out if r["symbol"] == "INFY")
    assert any("turnover" in reason for reason in infy["reject_reasons"])


# ---------------------------------------------------------------------------
# Scoring — a known-good leader should score well above a middling laggard
# ---------------------------------------------------------------------------
def test_scorer_ranks_aligned_sector_leader_highest(monkeypatch):
    rot = _fake_rotation_payload([
        _sector("NIFTY_IT", "IT", 1, 1.8,
                leaders=[_stock("TCS", 2.5, 4150.0),
                         _stock("INFY", 1.9, 1450.0)],
                laggards=[_stock("WIPRO", -1.4, 450.0)]),
    ])
    monkeypatch.setattr(shortlist_service, "build_rotation", lambda force=False: rot)

    funds = {
        # TCS — leader, aligned with sector, high ATR, rel-vol spike, near 52w high.
        "TCS":   Fundamentals(atr14=120.0, avg_volume_20=3_000_000, avg_close_20=4150.0,
                              high_52w=4200, low_52w=3100, last_volume=4_800_000),
        # INFY — leader but smaller move, moderate ATR, normal vol.
        "INFY":  Fundamentals(atr14=25.0, avg_volume_20=5_000_000, avg_close_20=1450.0,
                              high_52w=1800, low_52w=1300, last_volume=5_000_000),
        # WIPRO — laggard, against sector (−1.4% while sector up). Should be lowest or rejected.
        "WIPRO": Fundamentals(atr14=8.0, avg_volume_20=6_000_000, avg_close_20=450.0,
                              high_52w=550, low_52w=400, last_volume=6_000_000),
        # HCLTECH/TECHM — extras that pass filters but rank lower.
        "HCLTECH": Fundamentals(atr14=30.0, avg_volume_20=4_000_000, avg_close_20=1650.0,
                                high_52w=1900, low_52w=1400, last_volume=4_000_000),
        "TECHM":   Fundamentals(atr14=20.0, avg_volume_20=4_000_000, avg_close_20=1250.0,
                                high_52w=1500, low_52w=1000, last_volume=4_000_000),
    }
    provider = _StubHistoryProvider(funds)
    payload = build_shortlist(force=True, history_provider=provider)

    # TCS should be the #1 candidate and carry an "aligned"-style reason.
    symbols = [c["symbol"] for c in payload.candidates]
    assert symbols[0] == "TCS", f"expected TCS first, got {symbols}"

    tcs = payload.candidates[0]
    assert tcs["is_leader"] is True
    assert tcs["score"] > 0
    # Reasons include at least the sector bonus + leader.
    assert any("leader" in r for r in tcs["reasons"])
    assert any("sector #1" in r for r in tcs["reasons"])


# ---------------------------------------------------------------------------
# Reuse-first: build_rotation must be invoked (no duplicate data pipe)
# ---------------------------------------------------------------------------
def test_reuses_rotation_output(monkeypatch):
    calls = {"n": 0}

    def _fake_build_rotation(force=False):  # noqa: ARG001
        calls["n"] += 1
        return _fake_rotation_payload([
            _sector("NIFTY_IT", "IT", 1, 1.8,
                    leaders=[_stock("INFY", 1.9, 1450.0)]),
        ])

    monkeypatch.setattr(shortlist_service, "build_rotation", _fake_build_rotation)

    provider = _StubHistoryProvider({
        "INFY": Fundamentals(atr14=25.0, avg_volume_20=5_000_000, avg_close_20=1450.0,
                             high_52w=1800, low_52w=1300, last_volume=5_000_000),
    })
    build_shortlist(force=True, history_provider=provider)
    assert calls["n"] == 1, "Stage 4 must call Stage 3 exactly once on force"


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------
def test_cache_hit_on_second_call(monkeypatch):
    rot_calls = {"n": 0}

    def _fake_build_rotation(force=False):  # noqa: ARG001
        rot_calls["n"] += 1
        return _fake_rotation_payload([
            _sector("NIFTY_IT", "IT", 1, 1.8,
                    leaders=[_stock("INFY", 1.9, 1450.0)]),
        ])

    monkeypatch.setattr(shortlist_service, "build_rotation", _fake_build_rotation)

    provider = _StubHistoryProvider({
        "INFY": Fundamentals(atr14=25.0, avg_volume_20=5_000_000, avg_close_20=1450.0,
                             high_52w=1800, low_52w=1300, last_volume=5_000_000),
    })
    build_shortlist(force=True, history_provider=provider)
    build_shortlist(force=False, history_provider=provider)

    # With force=False on the second call, rotation must NOT be re-fetched —
    # the entire shortlist payload is served from cache.
    assert rot_calls["n"] == 1


# ---------------------------------------------------------------------------
# Provider failure — don't crash, surface the error on the payload.
# ---------------------------------------------------------------------------
def test_history_provider_failure_degrades_gracefully(monkeypatch):
    rot = _fake_rotation_payload([
        _sector("NIFTY_IT", "IT", 1, 1.8,
                leaders=[_stock("INFY", 1.9, 1450.0)]),
    ])
    monkeypatch.setattr(shortlist_service, "build_rotation", lambda force=False: rot)

    class _Broken:
        def fetch(self, symbols):  # noqa: ARG002
            raise RuntimeError("yfinance history exploded")

    payload = build_shortlist(force=True, history_provider=_Broken())

    # Endpoint survives — candidates list may be empty (no ATR data to pass
    # the hard filter) but the error surfaces on the payload, and we don't
    # 500 the frontend.
    assert any("exploded" in e for e in payload.errors)


# ---------------------------------------------------------------------------
# Fnp: non F&O names silently excluded from the pool
# ---------------------------------------------------------------------------
def test_non_fno_names_not_in_pool(monkeypatch):
    """Sanity — if SECTOR_CONSTITUENTS happened to include a non-F&O name,
    shortlist must exclude it.  Uses a synthetic sector where we pretend
    the constituents list contains a non-eligible ticker."""
    # Swap SECTOR_CONSTITUENTS temporarily for the test.
    monkeypatch.setattr(
        shortlist_service, "SECTOR_CONSTITUENTS",
        {"NIFTY_IT": ["TCS", "NONEXISTENT_SMALLCAP"]},
    )
    rot = _fake_rotation_payload([
        _sector("NIFTY_IT", "IT", 1, 1.8,
                leaders=[_stock("TCS", 2.5, 4150.0)]),
    ])
    monkeypatch.setattr(shortlist_service, "build_rotation", lambda force=False: rot)

    provider = _StubHistoryProvider({
        "TCS": Fundamentals(atr14=120.0, avg_volume_20=3_000_000, avg_close_20=4150.0,
                            high_52w=4200, low_52w=3100, last_volume=3_000_000),
    })
    payload = build_shortlist(force=True, history_provider=provider)
    all_symbols = ([c["symbol"] for c in payload.candidates]
                   + [r["symbol"] for r in payload.filtered_out])
    assert "NONEXISTENT_SMALLCAP" not in all_symbols
