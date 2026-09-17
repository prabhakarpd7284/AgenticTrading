from channels.generic.websocket import AsyncJsonWebsocketConsumer


class PnLConsumer(AsyncJsonWebsocketConsumer):
    """Streams live portfolio P&L to every logged-in client in the tenant."""

    async def connect(self):
        user = self.scope.get("user")
        if user is None or user.is_anonymous:
            await self.close(code=4401)
            return
        tenant = self.scope.get("tenant")
        if tenant is None:
            await self.close(code=4403)
            return
        self.group = f"pnl.{tenant.id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        # Echo the `jwt` subprotocol back — the browser rejects the handshake
        # if the server doesn't pick one of the offered protocols.
        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)

    async def disconnect(self, code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    async def pnl_update(self, message):
        await self.send_json(message["payload"])
