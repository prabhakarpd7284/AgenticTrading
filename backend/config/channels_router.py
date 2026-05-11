"""Collects WebSocket routes from every app that exposes them."""
from django.urls import path

from apps.market_data.consumers import TickConsumer
from apps.portfolio.consumers import PnLConsumer
from apps.agents_core.consumers import AgentRunConsumer
from apps.notifications.consumers import AlertsConsumer
from apps.events.consumers import EventsFirehoseConsumer, RunTimelineConsumer

websocket_urlpatterns = [
    path("ws/ticks/", TickConsumer.as_asgi()),
    path("ws/pnl/", PnLConsumer.as_asgi()),
    path("ws/agents/<uuid:run_id>/", AgentRunConsumer.as_asgi()),  # legacy alias
    path("ws/runs/<uuid:run_id>/", RunTimelineConsumer.as_asgi()),  # workflow-vocab name
    path("ws/events/", EventsFirehoseConsumer.as_asgi()),           # system-wide firehose
    path("ws/alerts/", AlertsConsumer.as_asgi()),
]
