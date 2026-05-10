"""Collects WebSocket routes from every app that exposes them."""
from django.urls import path

from apps.market_data.consumers import TickConsumer
from apps.portfolio.consumers import PnLConsumer
from apps.agents_core.consumers import AgentRunConsumer
from apps.notifications.consumers import AlertsConsumer

websocket_urlpatterns = [
    path("ws/ticks/", TickConsumer.as_asgi()),
    path("ws/pnl/", PnLConsumer.as_asgi()),
    path("ws/agents/<uuid:run_id>/", AgentRunConsumer.as_asgi()),
    path("ws/alerts/", AlertsConsumer.as_asgi()),
]
