from __future__ import annotations

from django.db import models

from apps.common.tenancy import TenantModel


class KnowledgeDoc(TenantModel):
    """RAG knowledge document (rules, patterns, strategy notes).

    Replaces the legacy `trading.StrategyDoc`. Despite the legacy name,
    these aren't strategies — they're knowledge injected into LLM prompt
    contexts via the RAG retriever. Hence the move to apps.rag.
    """

    class Category(models.TextChoices):
        ENTRY    = "ENTRY"
        EXIT     = "EXIT"
        RISK     = "RISK"
        SIZING   = "SIZING"
        FILTER   = "FILTER"
        GENERAL  = "GENERAL"
        RULE     = "RULE"
        PATTERN  = "PATTERN"
        STRATEGY = "STRATEGY"

    id = models.BigAutoField(primary_key=True)
    title = models.CharField(max_length=200)
    content = models.TextField()
    category = models.CharField(max_length=16, choices=Category.choices, default=Category.GENERAL)
    is_active = models.BooleanField(default=True)

    # Legacy lift shim
    legacy_strategy_doc_id = models.IntegerField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [models.Index(fields=["tenant", "category", "is_active"])]

    def __str__(self) -> str:
        return f"[{self.category}] {self.title}"


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
