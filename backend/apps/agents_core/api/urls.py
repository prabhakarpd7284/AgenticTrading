from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.agents_core.api.views import AgentRunViewSet, StrategyCatalogViewSet

router = DefaultRouter()
router.register("runs", AgentRunViewSet, basename="agent-run")
router.register("catalog", StrategyCatalogViewSet, basename="strategy-catalog")

urlpatterns = [path("", include(router.urls))]
