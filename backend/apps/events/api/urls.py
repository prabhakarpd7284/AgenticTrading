from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.events.api.views import EventViewSet
from apps.trading.api import legacy_compat_views as _compat

router = DefaultRouter()
router.register("", EventViewSet, basename="event")

urlpatterns = [
    # /events/audit/ — human-readable audit feed (legacy-bridge replacement).
    # Listed BEFORE the router so its specific path wins over the router's
    # `<id>/` detail regex.
    path("audit/", _compat.audit, name="audit-feed"),
    path("", include(router.urls)),
]
