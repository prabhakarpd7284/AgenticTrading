"""Index headlines into the RAG "news" namespace (Voyage embeddings).

Best effort: batched, one retry on HTTP 429, skipped entirely when no
VOYAGE_API_KEY is set (the embedder's hash fallback would make the vectors
meaningless). Failures never affect the scan's verdicts.
"""
from __future__ import annotations

import os
import time

import structlog
from django.utils import timezone

from apps.news.models import NewsItem

log = structlog.get_logger(__name__)

NAMESPACE = "news"
CHUNK = 64


def _text(it: NewsItem) -> str:
    syms = ",".join(sorted(m.symbol for m in it.mentions.all())) or "MARKET"
    when = timezone.localtime(it.published_at).strftime("%Y-%m-%d")
    return f"{syms}: {it.title} ({it.publisher or it.source}, {when})"


def index_items(items: list[NewsItem]) -> dict:
    todo = [it for it in items if it.indexed_at is None]
    st = {"candidates": len(todo), "indexed": 0, "ok": True, "error": ""}
    if not todo:
        return st
    if not os.environ.get("VOYAGE_API_KEY"):
        st.update(ok=False, error="VOYAGE_API_KEY not set — skipped")
        return st
    from apps.rag.embedders.voyage import VoyageEmbedder
    from apps.rag.vectorstores.pgvector_store import PgVectorStore

    emb, store = VoyageEmbedder(), PgVectorStore(NAMESPACE)
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        texts = [_text(it) for it in chunk]
        vectors = None
        for attempt in range(2):
            try:
                vectors = emb.embed(texts)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if attempt == 0 and "429" in msg:
                    time.sleep(25)
                    continue
                st.update(ok=False, error=msg[:200])
                break
        if vectors is None:
            break
        payloads = [{
            "tenant_id": None, "news_id": it.id, "url": it.url, "source": it.source,
            "publisher": it.publisher, "published_at": it.published_at.isoformat(),
            "symbols": sorted(m.symbol for m in it.mentions.all()),
            "score": float(it.sentiment_score) if it.sentiment_score is not None else 0.0,
            "materiality": it.materiality,
        } for it in chunk]
        try:
            store.upsert(ids=[f"news:{it.id}" for it in chunk], vectors=vectors,
                         payloads=[{**p, "text": t} for p, t in zip(payloads, texts, strict=True)])
        except Exception as e:  # noqa: BLE001
            st.update(ok=False, error=str(e)[:200])
            break
        NewsItem.objects.filter(pk__in=[it.pk for it in chunk]).update(indexed_at=timezone.now())
        st["indexed"] += len(chunk)
    return st
