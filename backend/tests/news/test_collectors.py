"""News collectors with mocked HTTP."""
from __future__ import annotations

import pytest
from django.utils import timezone

from apps.news import constants
from apps.news.models import CorporateEvent, NewsItem, NewsItemSymbol
from apps.news.services import collectors
from tests.news.fixtures import make_fetch

pytestmark = pytest.mark.django_db


def test_collect_all_stores_dedupes_and_tags():
    now = timezone.now()
    st = collectors.collect_all(["INFY", "HDFCBANK", "TCS"], now=now, fetch=make_fetch(now))

    assert st["google_news"]["ok"] and st["google_news"]["requests"] == 3
    # 2 fresh INFY (old + non-mentioning dropped) + 1 HDFC
    assert st["google_news"]["seen"] == 3
    infy = set(NewsItemSymbol.objects.filter(symbol="INFY").values_list("item__title", flat=True))
    assert "Infosys Q2 profit beats estimates, raises guidance" in infy   # publisher suffix stripped
    assert not NewsItem.objects.filter(title__icontains="Old Infosys").exists()

    # ET carried the same Infosys headline → title-dedupe, no second row
    assert NewsItem.objects.filter(title__startswith="Infosys Q2 profit").count() == 1
    macro = NewsItem.objects.get(url="https://et/m1")
    assert macro.is_market_wide and not macro.mentions.exists()

    # NSE filings: boilerplate pre-classified as noise without the LLM
    np = NewsItem.objects.get(url="https://nse/infy-np.pdf")
    assert np.materiality == "noise" and np.sentiment_model == "rule" and np.kind == "filing"
    res = NewsItem.objects.get(url="https://nse/infy.pdf")
    assert res.scored_at is None

    assert CorporateEvent.objects.filter(symbol="HDFCBANK", purpose="Financial Results").exists()
    assert st["nse_event_calendar"]["ok"] and st["nse_board_meetings"]["ok"]


def test_failing_source_never_aborts_the_rest():
    now = timezone.now()
    st = collectors.collect_all(["INFY"], now=now,
                                fetch=make_fetch(now, fail={"livemint", "event-calendar",
                                                             "news.google.com"}))
    assert st["livemint_markets"]["ok"] is False and st["livemint_markets"]["errors"]
    assert st["nse_event_calendar"]["ok"] is False
    assert st["google_news"]["ok"] is False          # every request failed
    assert st["et_markets"]["ok"] is True
    assert NewsItem.objects.filter(source="et_markets").exists()


def test_rerun_is_idempotent():
    now = timezone.now()
    collectors.collect_all(["INFY", "HDFCBANK"], now=now, fetch=make_fetch(now))
    n = NewsItem.objects.count()
    st = collectors.collect_all(["INFY", "HDFCBANK"], now=now, fetch=make_fetch(now))
    assert NewsItem.objects.count() == n
    assert st["google_news"]["new"] == 0


def test_rss_parser_rejects_entities():
    with pytest.raises(ValueError):
        collectors.parse_rss(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><rss/>')


@pytest.mark.parametrize("text,sym,expected", [
    ("SBI Life shares rally", "SBIN", False),
    ("SBI, SBI Life in focus", "SBIN", True),
    ("SBI Life shares rally", "SBILIFE", True),
    ("L&T bags mega order", "LT", True),
    ("L&T Finance results", "LT", False),
    ("an eternal question for markets", "ETERNAL", False),
    ("Eternal shares jump on Blinkit growth", "ETERNAL", True),
    ("Tech Mahindra and M&M lead gains", "M&M", True),
    ("Kotak Mahindra Bank Q2", "TECHM", False),
])
def test_alias_matching(text, sym, expected):
    assert constants.mentions(text, sym) is expected


def test_indexing_batches_into_news_namespace(monkeypatch):
    import importlib

    from apps.news.services import indexing as idx_mod

    idx = importlib.reload(idx_mod)        # undo the autouse stub for this test
    from apps.rag.models import Embedding

    now = timezone.now()
    collectors.collect_all(["INFY"], now=now, fetch=make_fetch(now))
    monkeypatch.setenv("VOYAGE_API_KEY", "test")

    class FakeEmb:
        def embed(self, texts):
            return [[0.0] * 3 for _ in texts]

    monkeypatch.setattr("apps.rag.embedders.voyage.VoyageEmbedder", lambda: FakeEmb())
    items = list(NewsItem.objects.all())
    st = idx.index_items(items)
    assert st["ok"] and st["indexed"] == len(items)
    assert Embedding.objects.filter(namespace="news").count() == len(items)
    assert idx.index_items(list(NewsItem.objects.all()))["candidates"] == 0
    monkeypatch.delenv("VOYAGE_API_KEY")
    importlib.reload(idx_mod)
