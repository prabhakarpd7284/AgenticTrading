from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.agents_core.registry import retriever_registry
from apps.rag.services.vector_search import index_document


class RetrieversListView(APIView):
    def get(self, request):
        return Response([{"name": n} for n in retriever_registry.names()])


class IndexDocumentView(APIView):
    def post(self, request):
        d = request.data
        index_document(
            namespace=d["namespace"],
            external_id=d["external_id"],
            text=d["text"],
            payload={**d.get("payload", {}), "tenant_id": str(request.tenant.id)},
        )
        return Response(status=status.HTTP_204_NO_CONTENT)
