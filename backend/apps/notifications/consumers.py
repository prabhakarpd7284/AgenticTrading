from channels.generic.websocket import AsyncJsonWebsocketConsumer


class AlertsConsumer(AsyncJsonWebsocketConsumer):
    """Streams in-app notifications (alerts, fills, agent events) to the SPA."""

    async def connect(self):
        user = self.scope.get("user")
        if user is None or user.is_anonymous:
            await self.close(code=4401)
            return
        tenant = self.scope.get("tenant")
        if tenant is None:
            await self.close(code=4403)
            return
        self.group = f"alerts.{tenant.id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)

    async def disconnect(self, code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    async def alert(self, message):
        await self.send_json(message["payload"])
