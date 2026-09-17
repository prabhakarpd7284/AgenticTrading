"""Momentum analytics over a StockEdge score universe — service + command."""
from __future__ import annotations

from datetime import date
from io import StringIO

import pytest
from django.core.management import call_command

from apps.market_data.integrations.stockedge import momentum as M
from apps.market_data.models import StockEdgeScanRow, StockEdgeSnapshot

pytestmark = pytest.mark.django_db


def _attrs(s1, z1, s3, z3, s6, z6):
    return {
        "1m_score": s1, "1m_score_zone": z1,
        "3m_score": s3, "3m_score_zone": z3,
        "6m_score": s6, "6m_score_zone": z6,
    }


@pytest.fixture
def snapshot():
    snap = StockEdgeSnapshot.objects.create(
        dataset="momentum_scores_nifty_500", as_of_date=date(2026, 6, 25),
        exchange="NSE", raw={}, meta={"kind": "momentum_scores"},
    )
    rows = [
        # symbol, sector, mcap, attrs
        ("SUSTAIN", "Banking", 90000, _attrs(85, "Bullish", 80, "Bullish", 75, "Bullish")),
        ("EMERGE", "Realty", 40000, _attrs(80, "Bullish", 60, "Neutral", 45, "Bearish")),
        ("FADE", "IT", 60000, _attrs(35, "Bearish", 70, "Bullish", 72, "Bullish")),
        ("MEH", "Pharma", 10000, _attrs(50, "Neutral", 50, "Neutral", 50, "Neutral")),
    ]
    for sym, sector, mcap, attrs in rows:
        StockEdgeScanRow.objects.create(
            snapshot=snap, symbol=sym, name=sym, sector=sector, industry="",
            ltp=100.0, change_pct=1.0, market_cap_cr=mcap, attrs=attrs,
            as_of_date=date(2026, 6, 25), exchange="NSE",
        )
    return snap


# ── pure analytics ────────────────────────────────────────────────────
def test_composite_recency_weighted():
    # 0.5*80 + 0.3*60 + 0.2*50 = 40+18+10 = 68.0
    assert M.composite_score(_attrs(80, "x", 60, "x", 50, "x")) == 68.0


def test_composite_renormalizes_missing_horizon():
    # only 1M present -> equals the 1M score (weights renormalized)
    a = {"1m_score": 80, "1m_score_zone": "Bullish"}
    assert M.composite_score(a) == 80.0


def test_acceleration_sign():
    assert M.acceleration(_attrs(80, "x", 60, "x", 50, "x")) == 20.0   # accelerating
    assert M.acceleration(_attrs(40, "x", 70, "x", 60, "x")) == -30.0  # decelerating


@pytest.mark.parametrize("attrs,expected", [
    (_attrs(85, "Bullish", 80, "Bullish", 75, "Bullish"), "sustained"),
    (_attrs(80, "Bullish", 60, "Neutral", 45, "Bearish"), "emerging"),
    (_attrs(35, "Bearish", 70, "Bullish", 72, "Bullish"), "fading"),
    (_attrs(50, "Neutral", 50, "Neutral", 50, "Neutral"), "neutral"),
])
def test_classify(attrs, expected):
    assert M.classify(attrs) == expected


# ── universe / filters ────────────────────────────────────────────────
def test_universe_ranked_by_composite(snapshot):
    uni = M.momentum_universe(snapshot)
    comps = [r["composite"] for r in uni]
    assert comps == sorted(comps, reverse=True)
    assert uni[0]["symbol"] == "SUSTAIN"  # highest composite


def test_filter_by_classification_and_mcap(snapshot):
    uni = M.momentum_universe(snapshot)
    emerging = M.filter_universe(uni, classification="emerging")
    assert [r["symbol"] for r in emerging] == ["EMERGE"]
    big = M.filter_universe(uni, min_mcap_cr=50000)
    assert set(r["symbol"] for r in big) == {"SUSTAIN", "FADE"}


def test_universe_summary(snapshot):
    s = M.universe_summary(M.momentum_universe(snapshot))
    assert s["count"] == 4
    assert s["classes"] == {"sustained": 1, "emerging": 1, "fading": 1, "neutral": 1}
    assert s["bullish_1m"] == 2  # SUSTAIN + EMERGE


# ── command ───────────────────────────────────────────────────────────
def test_command_runs(snapshot):
    out = StringIO()
    call_command("stockedge_momentum", "--top", "10", stdout=out)
    text = out.getvalue()
    assert "nifty_500" in text
    assert "SUSTAIN" in text
    assert "sustained 1" in text


def test_command_no_snapshot_errors():
    from django.core.management.base import CommandError
    with pytest.raises(CommandError):
        call_command("stockedge_momentum", "--index", "does_not_exist")
