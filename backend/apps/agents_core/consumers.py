"""WebSocket consumer that streams AgentEvent rows to the browser."""
from __future__ import annotations

from channels.generic.websocket import AsyncJsonWebsocketConsumer


class AgentRunConsumer(AsyncJsonWebsocketConsumer):
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
        self.group = f"agent.{self.tenant_id}.{self.run_id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)

    async def disconnect(self, code: int) -> None:
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    async def agent_event(self, message: dict) -> None:
        await self.send_json(message["event"])
