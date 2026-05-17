from django.urls import path

from apps.rag.api.views import RetrieversListView, IndexDocumentView
from apps.trading.api import legacy_compat_views as _compat

urlpatterns = [
    path("retrievers/", RetrieversListView.as_view(), name="rag-retrievers"),
    path("index/", IndexDocumentView.as_view(), name="rag-index"),
    # /rag/knowledge/ — list KnowledgeDoc rows (was /legacy/strategies/, which
    # was misleadingly named: it returns RAG knowledge entries, not strategies).
    path("knowledge/", _compat.strategies, name="rag-knowledge"),
]
