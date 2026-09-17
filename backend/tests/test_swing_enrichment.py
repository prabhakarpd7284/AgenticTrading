"""Swing-aware forward enrichment — forward MFE/MAE over daily candles."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from apps.strategies.models import Signal
from apps.strategies.services import swing_enrichment as SE

pytestmark = pytest.mark.django_db


def _c(d: date, h: float, lo: float, cl: float) -> dict:
    return {"ts": datetime(d.year, d.month, d.day, 15, 30), "h": h, "l": lo, "c": cl}


# ── pure excursion math ───────────────────────────────────────────────
def test_forward_excursion_buy():
    candles = [_c(date(2026, 6, 26), 110, 98, 108), _c(date(2026, 6, 29), 115, 105, 112)]
    fx = SE.forward_excursion(100.0, "BUY", candles)
    assert fx["mfe"] == 15.0   # 115 high − 100 entry
    assert fx["mae"] == 2.0    # 100 entry − 98 low
    assert fx["eod_price"] == 112.0
    assert fx["days"] == 2


def test_forward_excursion_sell_is_mirrored():
    fx = SE.forward_excursion(100.0, "SELL", [_c(date(2026, 6, 26), 110, 90, 95)])
    assert fx["mfe"] == 10.0   # short: favorable = entry − low
    assert fx["mae"] == 10.0   # adverse = high − entry


def test_forward_excursion_empty_is_none():
    assert SE.forward_excursion(100.0, "BUY", []) is None


def test_horizon_snapshots():
    candles = [_c(date(2026, 6, 20 + i), 100 + i, 99, 100 + i) for i in range(6)]
    fx = SE.forward_excursion(100.0, "BUY", candles, horizons=(2, 5))
    assert fx["horizons"]["fwd_2d_mfe"] == 1.0    # max high in first 2 = 101
    assert fx["horizons"]["fwd_5d_mfe"] == 4.0    # max high in first 5 = 104


# ── orchestrator ──────────────────────────────────────────────────────
@pytest.fixture
def momentum_signal(db, tenant):
    return Signal.objects.create(
        tenant=tenant, symbol="STRONG", signal_date=date(2026, 6, 25),
        signal_time=datetime(2026, 6, 25, 15, 30),
        source=Signal.Source.STOCKEDGE, strategy="StockEdge Composite Momentum",
        side="BUY", entry_price=100.0, stoploss=96.0, target=108.0,
        confidence=0.9, outcome=Signal.Outcome.PENDING,
    )


def _fetcher(by_symbol):
    return lambda symbol, start, end: by_symbol.get(symbol, [])


def test_complete_window_finalizes(momentum_signal, tenant):
    candles = [_c(date(2026, 6, 26), 105, 99, 104),
               _c(date(2026, 6, 29), 110, 103, 108),
               _c(date(2026, 6, 30), 112, 106, 111)]
    res = SE.enrich_swing_signals(
        tenant, source="STOCKEDGE", horizon=3, as_of=date(2026, 7, 15),
        candle_fetcher=_fetcher({"STRONG": candles}),
    )
    assert res["enriched"] == 1
    momentum_signal.refresh_from_db()
    assert momentum_signal.max_favorable_move == 12.0   # 112 − 100
    assert momentum_signal.max_adverse_move == 1.0      # 100 − 99
    assert momentum_signal.eod_price == 111.0
    assert momentum_signal.outcome == Signal.Outcome.EXPIRED
    assert momentum_signal.indicators["fwd_days"] == 3


def test_incomplete_window_skipped(momentum_signal, tenant):
    candles = [_c(date(2026, 6, 26), 105, 99, 104)]
    res = SE.enrich_swing_signals(
        tenant, source="STOCKEDGE", horizon=20, as_of=date(2026, 6, 27),
        candle_fetcher=_fetcher({"STRONG": candles}),
    )
    assert res == {"enriched": 0, "skipped": 1, "source": "STOCKEDGE", "as_of": date(2026, 6, 27)}
    momentum_signal.refresh_from_db()
    assert momentum_signal.eod_price is None
    assert momentum_signal.outcome == Signal.Outcome.PENDING


def test_partial_enriches_but_stays_pending(momentum_signal, tenant):
    candles = [_c(date(2026, 6, 26), 105, 99, 104)]
    res = SE.enrich_swing_signals(
        tenant, source="STOCKEDGE", horizon=20, as_of=date(2026, 6, 27),
        candle_fetcher=_fetcher({"STRONG": candles}), partial=True,
    )
    assert res["enriched"] == 1
    momentum_signal.refresh_from_db()
    assert momentum_signal.max_favorable_move == 5.0   # 105 − 100
    assert momentum_signal.eod_price is None           # not finalized
    assert momentum_signal.outcome == Signal.Outcome.PENDING


def test_signals_before_their_date_are_untouched(momentum_signal, tenant):
    # as_of == signal_date → nothing has elapsed, signal excluded by signal_date__lt
    res = SE.enrich_swing_signals(
        tenant, source="STOCKEDGE", horizon=5, as_of=date(2026, 6, 25),
        candle_fetcher=_fetcher({"STRONG": [_c(date(2026, 6, 26), 110, 99, 108)]}),
    )
    assert res["enriched"] == 0


def test_intraday_enricher_excludes_swing_sources():
    # the single source of truth the legacy intraday command excludes
    assert "STOCKEDGE" in SE.SWING_SOURCES
