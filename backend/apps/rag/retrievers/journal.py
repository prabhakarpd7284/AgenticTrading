"""Journal retriever — vector search over prior trade journal entries."""
from __future__ import annotations

from apps.agents_core.domain.contracts import RetrievalQuery, RetrievedDoc


class JournalRetriever:
    name = "journal"

    def retrieve(self, q: RetrievalQuery, k: int = 5) -> list[RetrievedDoc]:
        # Placeholder: real impl calls the vector store.
        from apps.rag.services.vector_search import search_journal
        return search_journal(tenant_id=q.filters.get("tenant_id"), text=q.text, k=k)
