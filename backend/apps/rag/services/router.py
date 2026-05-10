"""Multi-retriever fan-out + rerank. This is the `RAGRouter` implementation.

Plugins ask for named retrievers by key; if the retriever isn't registered, it is
skipped with a warning (fail-open). Results are deduped and reranked before returning.
"""
from __future__ import annotations

from typing import Iterable

import structlog

from apps.agents_core.domain.contracts import RetrievalQuery, RetrievedDoc
from apps.agents_core.registry import retriever_registry

log = structlog.get_logger()


class DefaultRAGRouter:
    def __init__(self, tenant_id, reranker=None):
        self.tenant_id = tenant_id
        from apps.rag.rerankers.noop import NoopReranker
        self.reranker = reranker or NoopReranker()

    def retrieve(
        self,
        query: RetrievalQuery,
        retrievers: Iterable[str] | None = None,
        k: int = 5,
    ) -> list[RetrievedDoc]:
        query.filters.setdefault("tenant_id", self.tenant_id)
        names = list(retrievers) if retrievers else retriever_registry.names()
        out: list[RetrievedDoc] = []
        for name in names:
            try:
                retriever = retriever_registry.get(name)
                out.extend(retriever.retrieve(query, k=k))
            except Exception:  # noqa: BLE001
                log.exception("rag.retrieve_failed", retriever=name)
        # Dedup by (source, text) keeping max score
        best: dict[tuple[str, str], RetrievedDoc] = {}
        for d in out:
            key = (d.source, d.text[:200])
            if key not in best or d.score > best[key].score:
                best[key] = d
        docs = list(best.values())
        docs = self.reranker.rerank(query, docs)
        return docs[:k]
