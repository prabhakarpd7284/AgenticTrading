"""News retriever — pulls from the news index (external API cached locally)."""
from __future__ import annotations

from apps.agents_core.domain.contracts import RetrievalQuery, RetrievedDoc


class NewsRetriever:
    name = "news"

    def retrieve(self, q: RetrievalQuery, k: int = 5) -> list[RetrievedDoc]:
        # Stub — to be wired to a news provider (e.g., Google News, Benzinga, Tavily).
        return [
            RetrievedDoc(
                source="news.stub",
                score=0.0,
                text="(news retriever not yet configured)",
                metadata={"query": q.text},
            )
        ]
