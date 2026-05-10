from django.urls import path
from apps.rag.api.views import RetrieversListView, IndexDocumentView

urlpatterns = [
    path("retrievers/", RetrieversListView.as_view(), name="rag-retrievers"),
    path("index/", IndexDocumentView.as_view(), name="rag-index"),
]
