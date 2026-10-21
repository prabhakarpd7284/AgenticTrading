"""NewsItem persistence: URL-hash dedupe, then same-title dedupe (one story
syndicated by several outlets becomes one row with merged symbol mentions)."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta

from django.db import IntegrityError, transaction
from django.utils import timezone

from apps.news.models import NewsItem, NewsItemSymbol

TITLE_DEDUPE_WINDOW = timedelta(days=4)
MARKET = "MARKET"   # pseudo-symbol used in prompts for market-wide items


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def url_hash(url: str) -> str:
    return _sha1(url.strip())


def normalize_title(title: str, publisher: str = "") -> str:
    t = (title or "").strip()
    if publisher and t.endswith(f" - {publisher}"):
        t = t[: -len(publisher) - 3]
    t = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    return t


def title_hash(title: str, publisher: str = "") -> str:
    return _sha1(normalize_title(title, publisher))


def clean_title(title: str, publisher: str = "") -> str:
    t = re.sub(r"\s+", " ", (title or "")).strip()
    if publisher and t.endswith(f" - {publisher}"):
        t = t[: -len(publisher) - 3].strip()
    return t[:500]


def upsert_item(*, title: str, url: str, source: str, published_at: datetime,
                symbols: list[str] | None = None, publisher: str = "",
                kind: str = NewsItem.Kind.NEWS, raw: dict | None = None,
                is_market_wide: bool = False, materiality: str = "",
                sentiment_model: str = "") -> tuple[NewsItem | None, bool]:
    """Store one headline. Returns ``(item, created)``; ``(None, False)`` when
    the title is empty. Existing rows only gain symbol mentions."""
    title = clean_title(title, publisher)
    if not title or not url:
        return None, False
    uh, th = url_hash(url), title_hash(title)
    symbols = sorted(set(symbols or []))

    item = NewsItem.objects.filter(url_hash=uh).first()
    if item is None:
        item = NewsItem.objects.filter(
            title_hash=th, published_at__gte=published_at - TITLE_DEDUPE_WINDOW,
            published_at__lte=published_at + TITLE_DEDUPE_WINDOW,
        ).first()
    created = False
    if item is None:
        fields = dict(
            url=url[:1000], url_hash=uh, title=title, title_hash=th, source=source,
            publisher=publisher[:120], kind=kind, published_at=published_at,
            raw=raw or {}, is_market_wide=is_market_wide and not symbols,
        )
        if materiality:
            # Pre-classified (e.g. boilerplate NSE filings): no LLM call needed.
            fields.update(materiality=materiality, sentiment_score=0, confidence=1,
                          sentiment_model=sentiment_model or "rule", scored_at=timezone.now())
        try:
            with transaction.atomic():
                item = NewsItem.objects.create(**fields)
            created = True
        except IntegrityError:
            item = NewsItem.objects.get(url_hash=uh)
    elif symbols and item.is_market_wide:
        NewsItem.objects.filter(pk=item.pk).update(is_market_wide=False)
        item.is_market_wide = False

    for sym in symbols:
        NewsItemSymbol.objects.get_or_create(item=item, symbol=sym)
    return item, created


def recent_headlines(*, symbol: str | None = None, days: int = 3, limit: int = 10,
                     include_noise: bool = False, now: datetime | None = None):
    now = now or timezone.now()
    qs = NewsItem.objects.filter(published_at__gte=now - timedelta(days=days),
                                 published_at__lte=now + timedelta(hours=1))
    if symbol:
        qs = qs.filter(mentions__symbol=symbol)
    else:
        qs = qs.filter(is_market_wide=True)
    if not include_noise:
        qs = qs.exclude(materiality=NewsItem.Materiality.NOISE)
    return list(qs.order_by("-published_at").distinct()[:limit])
