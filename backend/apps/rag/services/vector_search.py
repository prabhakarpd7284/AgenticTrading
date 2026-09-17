"""High-level helpers that combine embedder + vector store."""
from __future__ import annotations

from apps.agents_core.domain.contracts import RetrievedDoc


def _pipeline(namespace: str):
    from apps.rag.embedders.voyage import VoyageEmbedder
    from apps.rag.vectorstores.pgvector_store import PgVectorStore
    return VoyageEmbedder(), PgVectorStore(namespace)


def index_document(namespace: str, external_id: str, text: str, payload: dict) -> None:
    emb, store = _pipeline(namespace)
    vectors = emb.embed([text])
    payload = {**payload, "text": text}
    store.upsert(ids=[external_id], vectors=vectors, payloads=[payload])


def search_journal(tenant_id, text: str, k: int = 5) -> list[RetrievedDoc]:
    emb, store = _pipeline("journal")
    vec = emb.embed([text])[0]
    return store.search(vec, k=k, filters={"tenant_id": tenant_id})
