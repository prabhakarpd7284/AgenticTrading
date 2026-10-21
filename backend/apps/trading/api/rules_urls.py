from django.urls import path

from apps.trading.api.rules_views import CooldownStatusView

urlpatterns = [
    path("cooldown/", CooldownStatusView.as_view(), name="trading-rules-cooldown"),
]
