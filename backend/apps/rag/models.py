from __future__ import annotations

from django.db import models


class Embedding(models.Model):
    """Unified embeddings table (namespace column separates journal / news / etc).

    Uses pgvector's Vector field in prod; stored as JSONB fallback in dev on SQLite.
    """
    id = models.BigAutoField(primary_key=True)
    tenant_id = models.UUIDField(null=True, blank=True, db_index=True)
    namespace = models.CharField(max_length=64, db_index=True)
    external_id = models.CharField(max_length=128)
    # Placeholder — swap to pgvector.VectorField(1024) once pgvector extension is enabled.
    vector = models.JSONField()
    payload = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("namespace", "external_id")]
        indexes = [models.Index(fields=["namespace", "tenant_id"])]
