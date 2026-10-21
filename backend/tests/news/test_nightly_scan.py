"""Nightly scan orchestration — every external dependency mocked."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from django.utils import timezone

from apps.events.models import Event
from apps.market_data.models import Candle, Symbol
from apps.news.models import NightlyScan, NightlyScanRow
from apps.news.services import nightly_scan as ns
from tests.factories import PortfolioFactory
from tests.news.fixtures import fake_runner, make_fetch

pytestmark = pytest.mark.django_db

SESSION = date(2026, 10, 21)       # Wed (20 Oct is Dussehra)
FOR = date(2026, 10, 22)           # Thu
NOW = timezone.make_aware(datetime(2026, 10, 21, 21, 0))
SYMS = ["INFY", "HDFCBANK", "TCS"]


def _daily(sym, d, h, l, c):  # noqa: E741
    s, _ = Symbol.objects.get_or_create(exchange="NSE", tradingsymbol=sym,
                                        defaults={"token": sym, "name": sym})
    Candle.objects.create(symbol=s, interval="1d",
                          t=timezone.make_aware(datetime.combine(d, datetime.min.time())),
                          o=Decimal(c), h=Decimal(h), l=Decimal(l), c=Decimal(c), v=1)


@pytest.fixture
def world():
    for sym in SYMS:
        _daily(sym, date(2026, 10, 19), 110, 100, 105)
        _daily(sym, SESSION, 120, 108, 118)
    p = PortfolioFactory(mode="paper")
    return p


def _run(**kw):
    kw.setdefault("telegram", False)
    return ns.run_nightly_scan(session_date=SESSION, refresh=False, symbols=SYMS, now=NOW,
                               fetch=make_fetch(NOW, calendar_date=FOR), runner=fake_runner(), **kw)


def test_full_scan_rows_verdicts_and_event(world):
    scan = _run()
    assert scan.for_date == FOR and scan.session_date == SESSION
    assert scan.status == NightlyScan.Status.SUCCESS, scan.summary.get("degraded")
    rows = {r.symbol: r for r in scan.rows.all()}
    assert set(rows) == set(SYMS)

    infy = rows["INFY"]
    assert infy.sentiment_label == "strong_positive" and infy.allowed == "long"
    assert not infy.allow_short and infy.headlines and infy.headlines[0]["url"]
    assert infy.levels["cpr"]["class"] in ("narrow", "normal", "wide")
    assert infy.two_day_relationship == "higher" and infy.bias == "bullish"

    hdfc = rows["HDFCBANK"]                       # strong negative + results tomorrow
    assert hdfc.event_tomorrow and hdfc.allowed == "none"
    assert any("Financial Results" in r for r in hdfc.reasons)

    tcs = rows["TCS"]                             # "Other business matters" isn't major
    assert not tcs.event_tomorrow and tcs.allowed == "both" and tcs.sentiment_label == "neutral"

    assert scan.summary["market"]["scored"] >= 1
    assert {s["symbol"] for s in scan.summary["skip"]} == {"INFY", "HDFCBANK"}
    assert scan.summary["cooldowns"][0]["status"] == "normal"
    assert set(scan.timings) >= {"levels", "news", "sentiment", "rollup", "cooldown", "total"}
    assert Event.objects.filter(step_name="nightly_scan", tenant=world.tenant).exists()


def test_scan_is_idempotent_per_date(world):
    first = _run()
    second = _run()
    assert first.id == second.id
    assert NightlyScan.objects.count() == 1
    assert NightlyScanRow.objects.count() == len(SYMS)


def test_sentiment_failure_degrades_but_never_blocks(world):
    def broken(prompt, *, timeout, model=""):
        raise RuntimeError("not logged in")

    scan = ns.run_nightly_scan(session_date=SESSION, refresh=False, symbols=SYMS, now=NOW,
                               telegram=False, fetch=make_fetch(NOW, calendar_date=FOR),
                               runner=broken)
    assert scan.status == NightlyScan.Status.PARTIAL
    infy = scan.rows.get(symbol="INFY")
    assert infy.sentiment_label == "unscored" and infy.allow_long and infy.allow_short
    assert scan.summary["sentiment_available"] is False
    assert "INFY" in scan.summary["unscored"]
    # HDFC still skipped — events are deterministic, not LLM-derived.
    assert scan.rows.get(symbol="HDFCBANK").allowed == "none"
    assert "could NOT be computed" in ns.format_telegram(scan)


def test_source_failure_is_partial_not_fatal(world):
    scan = ns.run_nightly_scan(session_date=SESSION, refresh=False, symbols=SYMS, now=NOW,
                               telegram=False, runner=fake_runner(),
                               fetch=make_fetch(NOW, calendar_date=FOR, fail={"livemint"}))
    assert scan.status == NightlyScan.Status.PARTIAL
    assert scan.sources["livemint_markets"]["ok"] is False
    assert scan.rows.count() == len(SYMS)


def test_refresh_failure_tolerated(world):
    with patch("django.core.management.call_command", side_effect=SystemExit(1)):
        scan = ns.run_nightly_scan(session_date=SESSION, refresh=True, symbols=SYMS, now=NOW,
                                   telegram=False, runner=fake_runner(),
                                   fetch=make_fetch(NOW, calendar_date=FOR))
    assert scan.sources["refresh_bhavcopy"]["ok"] is False
    assert scan.status == NightlyScan.Status.PARTIAL and scan.rows.count() == 3


def test_missing_levels_flagged(world):
    scan = ns.run_nightly_scan(session_date=SESSION, refresh=False, symbols=SYMS + ["WIPRO"],
                               now=NOW, telegram=False, runner=fake_runner(),
                               fetch=make_fetch(NOW, calendar_date=FOR))
    wipro = scan.rows.get(symbol="WIPRO")
    assert wipro.levels == {} and any("levels unavailable" in r for r in wipro.reasons)


def test_telegram_summary_sent_synchronously(world):
    sent = []

    class FakeTG:
        is_configured = True
        total_errors = 0

        def _do_send(self, text, parse_mode):
            sent.append((text, parse_mode))

    with patch("plugins.strategy_screener.telegram.TelegramAlertService", FakeTG):
        scan = _run(telegram=True)
    assert scan.telegram_sent is True
    text, mode = sent[0]
    assert mode == "HTML" and "Nightly scan" in text and "HDFCBANK" in text
    assert "Cooldown" in text and len(text) <= 4000


def test_resolve_dates():
    from apps.news.services.nightly_scan import resolve_dates

    # Friday evening → Monday.
    assert resolve_dates(date(2026, 10, 23)) == (date(2026, 10, 23), date(2026, 10, 26))
    # Sunday given → session Friday, for Monday.
    assert resolve_dates(date(2026, 10, 25)) == (date(2026, 10, 23), date(2026, 10, 26))
    # 01:30 on a trading Wednesday → the scan is for today; the last session
    # is Monday because Tuesday 20 Oct is the Dussehra holiday.
    late = timezone.make_aware(datetime(2026, 10, 21, 1, 30))
    assert resolve_dates(None, late) == (date(2026, 10, 19), date(2026, 10, 21))
    evening = timezone.make_aware(datetime(2026, 10, 21, 21, 0))
    assert resolve_dates(None, evening) == (date(2026, 10, 21), date(2026, 10, 22))


def test_beat_task_skips_when_tomorrow_is_not_a_trading_day(monkeypatch):
    from apps.news import tasks

    called = []
    monkeypatch.setattr("apps.news.services.nightly_scan.run_nightly_scan",
                        lambda **kw: called.append(kw))
    with patch("django.utils.timezone.localdate", return_value=date(2026, 10, 23)):  # Friday
        assert tasks.run_nightly_scan() == {"skipped": "next day is not a trading day"}
    assert called == []


def test_news_retriever_returns_real_headlines(world):
    from apps.agents_core.domain.contracts import RetrievalQuery
    from apps.rag.retrievers.news import NewsRetriever

    _run()
    # The retriever looks back from the wall clock; pin it to the fixture's NOW.
    with patch("django.utils.timezone.now", return_value=NOW):
        docs = NewsRetriever().retrieve(RetrievalQuery(text="Plan a BUY trade for Infosys"), k=3)
        assert docs and all("news/" in d.source for d in docs)
        assert any("Infosys" in d.text for d in docs)
        docs = NewsRetriever().retrieve(RetrievalQuery(text="x", filters={"symbol": "NSE:INFY-EQ"}))
    assert docs[0].metadata["symbol"] == "INFY"
