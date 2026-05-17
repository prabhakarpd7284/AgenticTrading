from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.agents_core.api.views import (
    AgentRunViewSet, PlanStockView, StrategyCatalogViewSet,
)
from apps.agents_core.api.views_palace import (
    PalaceTasksView, team_run_view, team_status_view,
)

router = DefaultRouter()
router.register("runs", AgentRunViewSet, basename="agent-run")
router.register("catalog", StrategyCatalogViewSet, basename="strategy-catalog")

urlpatterns = [
    path("plan-stock/", PlanStockView.as_view(), name="plan-stock"),
    # Read-only snapshot of the mind palace tasks, consumed by /board/.
    path("palace/tasks/", PalaceTasksView.as_view(), name="palace-tasks"),
    # Trigger + monitor an AI team run from the board UI.
    path("palace/run/",    team_run_view,    name="palace-team-run"),
    path("palace/status/", team_status_view, name="palace-team-status"),
    path("", include(router.urls)),
]
