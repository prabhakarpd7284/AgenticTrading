"""WebSocket consumers for the unified Event log.

Two channels:

  /ws/events/                 — tenant-wide firehose (drives Now activity feed).
                                 Each Event row is published to group
                                 `events.{tenant_id}` by event_writer.emit().

  /ws/runs/<run_id>/          — per-workflow-run timeline. Subset of the
                                 firehose filtered to events whose
                                 workflow_run_id matches the URL param.
                                 Published to group `runs.{run_id}`.

Both use JWT auth via the `jwt` subprotocol (set by lib/ws.ts in the frontend).
The middleware that resolves `scope["tenant"]` from the JWT lives in
apps.core (formerly apps.common/middleware).
"""
from __future__ import annotations

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer


class EventsFirehoseConsumer(AsyncJsonWebsocketConsumer):
    """Tenant-wide event firehose. One subscription per Now-page session."""

    async def connect(self) -> None:
        user = self.scope.get("user")
        if user is None or user.is_anonymous:
            await self.close(code=4401)
            return
        tenant = self.scope.get("tenant")
        if tenant is None:
            await self.close(code=4403)
            return

        self.group = f"events.{tenant.id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)

    async def disconnect(self, code: int) -> None:
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    # Channel layer dispatch — name MUST match the `type` field used
    # in event_writer._broadcast (`event.message`, dots → underscores).
    async def event_message(self, message: dict) -> None:
        # Strip the channel-layer "type" key; forward the rest as the WS payload
        body = {k: v for k, v in message.items() if k != "type"}
        await self.send_json(body)


class RunTimelineConsumer(AsyncJsonWebsocketConsumer):
    """Per-workflow-run timeline. Subscribes to `runs.<run_id>` group."""

    async def connect(self) -> None:
        user = self.scope.get("user")
        if user is None or user.is_anonymous:
            await self.close(code=4401)
            return
        tenant = self.scope.get("tenant")
        if tenant is None:
            await self.close(code=4403)
            return

        self.run_id = str(self.scope["url_route"]["kwargs"]["run_id"])
        self.tenant_id = str(tenant.id)
        # Authorization: the run MUST belong to the authed tenant. Without this,
        # any user could subscribe to runs.<victim_run_id> and read another
        # tenant's full event stream (text, payload, trade_id, order_id).
        if not await self._run_belongs_to_tenant(tenant.id, self.run_id):
            await self.close(code=4403)
            return
        # Tenant-namespaced group — defence in depth so a run_id can never cross
        # tenants even if the broadcaster published to the wrong namespace.
        self.group = f"runs.{self.tenant_id}.{self.run_id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)

    @staticmethod
    @database_sync_to_async
    def _run_belongs_to_tenant(tenant_id, run_id) -> bool:
        from django.core.exceptions import ValidationError

        from apps.agents_core.models import AgentRun

        try:
            return AgentRun.objects.filter(tenant_id=tenant_id, pk=run_id).exists()
        except (ValueError, ValidationError):  # run_id not a valid UUID
            return False

    async def disconnect(self, code: int) -> None:
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    async def event_message(self, message: dict) -> None:
        body = {k: v for k, v in message.items() if k != "type"}
        await self.send_json(body)
