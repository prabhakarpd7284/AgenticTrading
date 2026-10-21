"""Sentiment parsing, CLI batching (mocked) and per-stock rollup thresholds."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from django.test import override_settings
from django.utils import timezone

from apps.news.models import NewsItem
from apps.news.services import sentiment
from apps.news.services.store import upsert_item

pytestmark = pytest.mark.django_db


def test_parse_scores_handles_fences_prose_and_clamping():
    raw = ('Sure! Here you go:\n```json\n[{"id": 1, "score": 1.7, "materiality": "RESULTS", '
           '"confidence": 2}, {"id": "2", "score": -0.4, "materiality": "weird"}, '
           '{"id": 3, "score": "nan?"}, "junk"]\n```')
    out = sentiment.parse_scores(raw)
    assert out[1] == {"score": 1.0, "materiality": "results", "confidence": 1.0}
    assert out[2] == {"score": -0.4, "materiality": "other", "confidence": 0.5}
    assert 3 not in out


def test_parse_scores_object_wrapper_and_garbage():
    assert sentiment.parse_scores('{"results": [{"id": 1, "score": 0.2}]}')[1]["score"] == 0.2
    with pytest.raises(ValueError):
        sentiment.parse_scores("I cannot help with that.")


def _item(title, *, hours_ago=1, sym="INFY", score=None, mat="", conf=None):
    now = timezone.now()
    it, _ = upsert_item(title=title, url=f"https://x/{title}", source="google_news",
                        published_at=now - timedelta(hours=hours_ago), symbols=[sym])
    if score is not None:
        it.sentiment_score = Decimal(str(score))
        it.materiality = mat
        it.confidence = Decimal(str(conf if conf is not None else 0.9))
        it.scored_at = now
        it.save()
    return it


def test_score_items_batches_and_persists():
    items = [_item(f"Infosys headline {i} beats") for i in range(5)]

    calls = []

    def runner(prompt, *, timeout, model=""):
        calls.append(prompt)
        n = prompt.count(" | INFY | ")
        return str([{"id": i, "score": 0.6, "materiality": "results", "confidence": 0.8}
                    for i in range(1, n + 1)]).replace("'", '"')

    with override_settings(NEWS_SENTIMENT={"BATCH_SIZE": 2}):
        st = sentiment.score_items(items, runner=runner)
    assert st.batches == 3 and st.scored == 5 and len(calls) == 3
    assert NewsItem.objects.filter(scored_at__isnull=False).count() == 5
    # Already-scored items are skipped on a re-run.
    assert sentiment.score_items(list(NewsItem.objects.all()), runner=runner).candidates == 0


def test_score_items_aborts_after_consecutive_failures_and_leaves_unscored():
    items = [_item(f"h{i}") for i in range(10)]

    def broken(prompt, *, timeout, model=""):
        raise RuntimeError("Claude CLI timed out")

    with override_settings(NEWS_SENTIMENT={"BATCH_SIZE": 2}):
        st = sentiment.score_items(items, runner=broken)
    assert st.aborted and st.failed_batches == 2 and st.scored == 0
    assert not NewsItem.objects.filter(scored_at__isnull=False).exists()


def test_rollup_labels_and_thresholds():
    strong = [_item("a", score=0.8, mat="results")]
    assert sentiment.rollup(strong).label == "strong_positive"
    assert sentiment.rollup([_item("b", score=-0.8, mat="regulatory")]).label == "strong_negative"
    # A single weak-evidence item (old, low materiality) can't be "strong".
    weak = [_item("c", score=0.9, mat="macro", hours_ago=48, conf=0.5)]
    r = sentiment.rollup(weak)
    assert r.label == "positive" and r.weight < 0.5
    assert sentiment.rollup([_item("d", score=0.1, mat="results")]).label == "neutral"
    assert sentiment.rollup([_item("e", score=-0.3, mat="order")]).label == "negative"


def test_rollup_recency_weighting():
    old_bad = _item("old", score=-0.9, mat="results", hours_ago=72)
    new_good = _item("new", score=0.6, mat="results", hours_ago=1)
    r = sentiment.rollup([old_bad, new_good])
    assert r.score > 0.3


def test_rollup_unscored_and_no_news():
    assert sentiment.rollup([_item("pending")]).label == "unscored"
    assert sentiment.rollup([]).label == "neutral"
    noise = _item("noise", score=0.0, mat="noise")
    assert sentiment.rollup([noise]).label == "neutral"


@override_settings(NEWS_SENTIMENT={"STRONG_THRESHOLD": 0.9})
def test_thresholds_are_configurable():
    assert sentiment.rollup([_item("z", score=0.8, mat="results")]).label == "positive"
