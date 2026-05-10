"""Pluggable RAG interfaces. Retrievers, Embedders, VectorStores, Rerankers.

Designed for dependency-injection. Concrete implementations live under
`apps/rag/{retrievers, embedders, vectorstores, rerankers}/`.
Third parties can ship their own via the `alphadesk.retrievers` entry-point.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from apps.agents_core.domain.contracts import RetrievalQuery, RetrievedDoc


@runtime_checkable
class Retriever(Protocol):
    name: str
    def retrieve(self, q: RetrievalQuery, k: int = 5) -> list[RetrievedDoc]: ...


@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None: ...
    def search(self, vector: list[float], k: int, filters: dict | None = None) -> list[RetrievedDoc]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, q: RetrievalQuery, docs: list[RetrievedDoc]) -> list[RetrievedDoc]: ...
