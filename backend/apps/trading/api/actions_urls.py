from django.urls import path

from apps.trading.api.actions_views import ActionDecisionView, TodayActionsView, TrackRecordView

urlpatterns = [
    path("today/", TodayActionsView.as_view(), name="actions-today"),
    path("track-record/", TrackRecordView.as_view(), name="actions-track-record"),
    path("<uuid:action_id>/<str:decision>/", ActionDecisionView.as_view(), name="actions-decide"),
]
