"""StockEdge Market-Breadth ingestion — parser, summary, and command.

Covers the vertical slice end to end:
  - pure parsing/normalization of the captured sample payload
  - the derived regime summary
  - the management command in --no-persist mode (no DB writes)
  - the management command default path persisting a snapshot + 11 rows
"""
from __future__ import annotations

import json
from datetime import date
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command

from apps.market_data.integrations.stockedge.parser import (
    parse_breadth_payload,
    summarize_breadth,
)
from apps.market_data.models import StockEdgeBreadthRow, StockEdgeSnapshot

pytestmark = pytest.mark.django_db

SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "apps" / "market_data" / "integrations" / "stockedge" / "sample_breadth.json"
)


def _load_sample() -> dict:
    with SAMPLE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


# ── parser ────────────────────────────────────────────────────────────
def test_parse_breadth_payload_sample_shape():
    parsed = parse_breadth_payload(_load_sample())

    assert parsed["dataset"] == "market_breadth"
    assert parsed["exchange"] == "NSE"
    assert parsed["as_of_date"] == "2026-06-25"
    rows = parsed["rows"]
    assert len(rows) == 11

    by_name = {r["index_name"]: r for r in rows}
    bank = by_name["Nifty Bank"]
    # Coerced to float, value preserved.
    assert bank["sma20"] == 93
    assert isinstance(bank["sma20"], float)
    assert isinstance(bank["rs_pos"], float)
    assert isinstance(bank["constituent_count"], int)
    assert bank["constituent_count"] == 14


def test_parse_breadth_payload_coerces_and_tolerates_missing():
    payload = {
        "dataset": "market_breadth",
        "as_of_date": "2026-06-25",
        "exchange": "nse",
        "rows": [
            {"index_name": "Nifty 50", "rs_pos": "54%", "sma20": "64",
             "sma50": None, "sma100": "", "constituent_count": "50"},
        ],
    }
    parsed = parse_breadth_payload(payload)
    row = parsed["rows"][0]
    assert parsed["exchange"] == "NSE"  # upcased
    assert row["rs_pos"] == 54.0
    assert row["sma20"] == 64.0
    assert row["sma50"] is None
    assert row["sma100"] is None
    assert row["sma200"] is None
    assert row["constituent_count"] == 50


@pytest.mark.parametrize("bad", [
    [],                                              # not a dict
    {},                                              # no rows
    {"as_of_date": "2026-06-25", "rows": []},        # empty rows
    {"as_of_date": "2026-06-25", "rows": [{"rs_pos": 50}]},   # row missing index_name
    {"rows": [{"index_name": "Nifty 50"}]},          # missing as_of_date
])
def test_parse_breadth_payload_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_breadth_payload(bad)


@pytest.mark.parametrize("given,expected", [
    ("2026-06-25", "2026-06-25"),                    # already ISO
    ("2026-06-25T00:00:00Z", "2026-06-25"),          # ISO datetime
    ("25-06-2026", "2026-06-25"),                    # DD-MM-YYYY (Indian display)
    ("25/06/2026", "2026-06-25"),                    # DD/MM/YYYY
])
def test_parse_breadth_payload_normalizes_as_of_date(given, expected):
    payload = {"as_of_date": given, "rows": [{"index_name": "Nifty 50", "sma20": 64}]}
    assert parse_breadth_payload(payload)["as_of_date"] == expected


def test_parse_breadth_payload_rejects_unparseable_date():
    # A bad date must surface as a clean ValueError (-> CommandError), not crash
    # downstream on strptime during persistence.
    payload = {"as_of_date": "not-a-date", "rows": [{"index_name": "Nifty 50"}]}
    with pytest.raises(ValueError):
        parse_breadth_payload(payload)


# ── summary ───────────────────────────────────────────────────────────
def test_summarize_breadth_returns_regime():
    parsed = parse_breadth_payload(_load_sample())
    summary = summarize_breadth(parsed["rows"])

    assert summary["broad_regime"] in {"risk-on", "neutral", "risk-off"}
    # Sample avg SMA50 sits in the high-50s/low-60s band.
    assert summary["avg_sma50"] is not None
    assert summary["strongest_index"] is not None
    assert summary["weakest_index"] is not None
    # Nifty Bank is the strongest universe in the sample (93/93 short-term).
    assert summary["strongest_index"] == "Nifty Bank"


def test_summarize_breadth_thresholds():
    def rows(sma50):
        return [{"index_name": "X", "rs_pos": None, "sma20": None,
                 "sma50": sma50, "sma100": None, "sma200": None}]

    assert summarize_breadth(rows(75))["broad_regime"] == "risk-on"
    assert summarize_breadth(rows(50))["broad_regime"] == "neutral"
    assert summarize_breadth(rows(30))["broad_regime"] == "risk-off"


# ── model property ────────────────────────────────────────────────────
def test_breadth_score_property_ignores_none():
    snap = StockEdgeSnapshot.objects.create(
        dataset="market_breadth", as_of_date=date(2026, 6, 25), exchange="NSE", raw={},
    )
    row = StockEdgeBreadthRow.objects.create(
        snapshot=snap, index_name="Nifty 50", as_of_date=date(2026, 6, 25),
        exchange="NSE", rs_pos=54.0, sma20=64.0, sma50=50.0, sma100=None, sma200=None,
    )
    # mean of (54, 64, 50) = 56.0
    assert row.breadth_score == 56.0


# ── management command ────────────────────────────────────────────────
def test_command_no_persist_runs():
    out = StringIO()
    call_command(
        "pull_stockedge_breadth", "--from-json", str(SAMPLE_PATH),
        "--no-persist", stdout=out,
    )
    output = out.getvalue()
    assert "Nifty Bank" in output
    assert "Regime:" in output
    # Nothing written.
    assert StockEdgeSnapshot.objects.count() == 0
    assert StockEdgeBreadthRow.objects.count() == 0


def test_command_persists_snapshot_and_rows():
    out = StringIO()
    # Default source = bundled sample.
    call_command("pull_stockedge_breadth", stdout=out)

    snap = StockEdgeSnapshot.objects.get(
        dataset="market_breadth", as_of_date=date(2026, 6, 25), exchange="NSE",
    )
    assert snap.breadth_rows.count() == 11
    assert snap.raw["rows"][2]["index_name"] == "Nifty Bank"
    assert StockEdgeBreadthRow.objects.filter(as_of_date=date(2026, 6, 25)).count() == 11


def test_command_upserts_on_repull():
    call_command("pull_stockedge_breadth", stdout=StringIO())
    call_command("pull_stockedge_breadth", stdout=StringIO())

    # Upsert — exactly one snapshot, children replaced (not duplicated).
    assert StockEdgeSnapshot.objects.filter(dataset="market_breadth").count() == 1
    snap = StockEdgeSnapshot.objects.get(dataset="market_breadth")
    assert snap.breadth_rows.count() == 11
