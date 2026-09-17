"""StockEdge composite-momentum → first-class Signals + signal.fired events.

Verifies the bridge from the momentum shortlist into the strategies/events
pipeline: selection + entry/stop/target shaping, idempotent firing, and that the
Signals carry the source/strategy needed for the monthly report.
"""
from __future__ import annotations

from datetime import date

import pytest

from apps.events.models import Event
from apps.market_data.integrations.stockedge import momentum_signals as MS
from apps.market_data.models import StockEdgeScanRow, StockEdgeSnapshot
from apps.strategies.models import Signal

pytestmark = pytest.mark.django_db


def _attrs(s1, z1, s3, z3, s6, z6):
    return {
        "1m_score": s1, "1m_score_zone": z1,
        "3m_score": s3, "3m_score_zone": z3,
        "6m_score": s6, "6m_score_zone": z6,
    }


@pytest.fixture
def snapshot(db):
    snap = StockEdgeSnapshot.objects.create(
        dataset="momentum_scores_nifty_500", as_of_date=date(2026, 6, 25),
        exchange="NSE", raw={}, meta={},
    )
    # (symbol, sector, mcap_cr, attrs, ltp)
    data = [
        ("STRONG", "Banking", 90000, _attrs(90, "Bullish", 88, "Bullish", 85, "Bullish"), 1000.0),
        ("FRESH", "Realty", 40000, _attrs(85, "Bullish", 75, "Neutral", 55, "Bearish"), 500.0),
        ("SMALL", "IT", 2000, _attrs(95, "Bullish", 95, "Bullish", 95, "Bullish"), 100.0),   # mcap < 5000 → filtered
        ("WEAK", "Pharma", 60000, _attrs(40, "Bearish", 45, "Neutral", 50, "Neutral"), 200.0),  # neutral → filtered
    ]
    for sym, sector, mcap, attrs, ltp in data:
        StockEdgeScanRow.objects.create(
            snapshot=snap, symbol=sym, name=sym, sector=sector, industry="",
            ltp=ltp, change_pct=1.0, market_cap_cr=mcap, attrs=attrs,
            as_of_date=date(2026, 6, 25), exchange="NSE",
        )
    return snap


# ── selection + shaping ───────────────────────────────────────────────
def test_build_signal_rows_filters_and_shapes(snapshot):
    rows = MS.build_signal_rows(snapshot)
    syms = [r["symbol"] for r in rows]
    # SMALL excluded (mcap), WEAK excluded (classification); ranked by composite
    assert syms == ["STRONG", "FRESH"]

    strong = rows[0]
    assert strong["side"] == "BUY"
    assert strong["entry"] == 1000.0
    assert strong["stop"] == 960.0                      # 4% swing stop
    assert strong["target"] == 1080.0                   # entry + risk*2 (2:1)
    assert 0.88 <= strong["confidence"] <= 0.89         # composite/100


def test_build_signal_rows_respects_filters(snapshot):
    assert len(MS.build_signal_rows(snapshot, top=1)) == 1
    # raise the bar past FRESH's composite (~76) but below STRONG's (~88)
    only_strong = MS.build_signal_rows(snapshot, min_composite=80)
    assert [r["symbol"] for r in only_strong] == ["STRONG"]
    # restrict to sustained only → FRESH (emerging) drops out
    assert [r["symbol"] for r in MS.build_signal_rows(snapshot, classes=("sustained",))] == ["STRONG"]


# ── firing ────────────────────────────────────────────────────────────
def test_dry_run_writes_nothing(snapshot, tenant):
    res = MS.fire_momentum_signals(snapshot, dry_run=True)
    assert res["dry_run"] and res["would_fire"] == 2
    assert Signal.objects.count() == 0


def test_fire_creates_signals_and_events(snapshot, tenant):
    res = MS.fire_momentum_signals(snapshot)
    assert res["fired"] == 2

    sigs = Signal.objects.filter(source=Signal.Source.STOCKEDGE)
    assert sigs.count() == 2
    s = sigs.get(symbol="STRONG")
    assert s.strategy == MS.STRATEGY_NAME
    assert s.side == "BUY"
    assert s.entry_price == 1000.0 and s.stoploss == 960.0 and s.target == 1080.0
    assert s.outcome == Signal.Outcome.PENDING
    assert s.signal_date == date(2026, 6, 25)
    assert s.indicators["classification"] == "sustained"

    # one signal.fired event per signal, hard-linked + tagged
    events = Event.objects.filter(type=Event.Type.SIGNAL_FIRED,
                                  signal_id__in=list(sigs.values_list("id", flat=True)))
    assert events.count() == 2
    assert events.first().payload["source"] == "STOCKEDGE"


def test_firing_is_idempotent(snapshot, tenant):
    MS.fire_momentum_signals(snapshot)
    again = MS.fire_momentum_signals(snapshot)
    assert again.get("already") and again["fired"] == 0
    assert Signal.objects.filter(source=Signal.Source.STOCKEDGE).count() == 2

    forced = MS.fire_momentum_signals(snapshot, force=True)
    assert forced["fired"] == 2
    assert Signal.objects.filter(source=Signal.Source.STOCKEDGE).count() == 2  # replaced, not doubled


def test_command_dry_run(snapshot, tenant):
    from io import StringIO
    from django.core.management import call_command
    out = StringIO()
    call_command("stockedge_momentum_signals", "--dry-run", stdout=out)
    text = out.getvalue()
    assert "STRONG" in text and "DRY RUN" in text
    assert Signal.objects.count() == 0
