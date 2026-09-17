from django.urls import path

from apps.system.api.views import OpsCommandHelpView, OpsCommandListView

urlpatterns = [
    path("commands/", OpsCommandListView.as_view(), name="ops-commands"),
    path("commands/<str:name>/help/", OpsCommandHelpView.as_view(), name="ops-command-help"),
]
