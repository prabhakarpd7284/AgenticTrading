"""News retriever — recent headlines (+ Claude sentiment) from apps.news.

Symbol resolution: ``q.filters["symbol"]`` if given, else the first NIFTY 50
symbol / company name found in ``q.text``; with no symbol it returns the
newest market-wide headlines. Headlines are stored by the nightly scan's
collectors (Google News, ET, Livemint, Business Standard, NSE filings) and
also embedded into the RAG "news" namespace.
"""
from __future__ import annotations

import structlog

from apps.agents_core.domain.contracts import RetrievalQuery, RetrievedDoc

log = structlog.get_logger(__name__)


def _symbol_from_text(text: str) -> str | None:
    from apps.market_data.constants import NIFTY_50_SYMBOLS
    from apps.news.constants import match_symbols

    hits = match_symbols(text or "", list(NIFTY_50_SYMBOLS))
    return hits[0] if hits else None


class NewsRetriever:
    name = "news"

    def retrieve(self, q: RetrievalQuery, k: int = 5) -> list[RetrievedDoc]:
        try:
            from django.utils import timezone

            from apps.news.services.store import recent_headlines

            symbol = (q.filters or {}).get("symbol") or _symbol_from_text(q.text)
            if symbol:
                from apps.trading.services.trailing_stop import normalize_symbol
                symbol = normalize_symbol(symbol)
            items = recent_headlines(symbol=symbol, days=3, limit=k)
            if not items and symbol:
                items = recent_headlines(symbol=None, days=3, limit=k)
        except Exception as e:  # noqa: BLE001 — retrieval must never break a workflow
            log.warning("rag.news_retrieve_failed", error=str(e))
            return []
        docs = []
        for it in items:
            when = timezone.localtime(it.published_at).strftime("%Y-%m-%d %H:%M")
            sent = ""
            if it.sentiment_score is not None:
                sent = f" [sentiment {float(it.sentiment_score):+.2f}, {it.materiality}]"
            docs.append(RetrievedDoc(
                source=f"news/{it.source}",
                score=float(it.sentiment_score or 0),
                text=f"[{when}] {it.title} ({it.publisher or it.source}){sent}",
                metadata={"url": it.url, "symbol": symbol, "news_id": it.id,
                          "published_at": it.published_at.isoformat(),
                          "materiality": it.materiality, "kind": it.kind},
            ))
        return docs
