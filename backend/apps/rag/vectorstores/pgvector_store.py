"""pgvector-backed VectorStore. Uses the `rag.Embedding` Django model."""
from __future__ import annotations

from apps.agents_core.domain.contracts import RetrievedDoc


class PgVectorStore:
    def __init__(self, namespace: str):
        self.namespace = namespace  # usually "journal" or "news"

    def upsert(self, ids, vectors, payloads):
        from apps.rag.models import Embedding

        rows = [
            Embedding(
                external_id=i, namespace=self.namespace, vector=v, payload=p,
                tenant_id=p.get("tenant_id"),
            )
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]
        Embedding.objects.bulk_create(
            rows,
            update_conflicts=True,
            unique_fields=["namespace", "external_id"],
            update_fields=["vector", "payload"],
        )

    def search(self, vector, k=5, filters=None):
        from apps.rag.models import Embedding
        qs = Embedding.objects.filter(namespace=self.namespace)
        if filters:
            if "tenant_id" in filters:
                qs = qs.filter(tenant_id=filters["tenant_id"])
        # Assumes the model has a `vector` pgvector field and supports `l2_distance`.
        qs = qs.order_by(Embedding.vector.l2_distance(vector))[:k]
        return [
            RetrievedDoc(
                source=f"{self.namespace}/{row.external_id}",
                score=float(row.payload.get("score", 0.0)),
                text=row.payload.get("text", ""),
                metadata=row.payload,
            )
            for row in qs
        ]
