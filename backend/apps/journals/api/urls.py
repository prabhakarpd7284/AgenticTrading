from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.journals.api.views import JournalEntryViewSet

router = DefaultRouter()
router.register("", JournalEntryViewSet, basename="journal")
urlpatterns = [path("", include(router.urls))]
