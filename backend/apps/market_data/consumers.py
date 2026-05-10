from channels.generic.websocket import AsyncJsonWebsocketConsumer


class TickConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        user = self.scope.get("user")
        if user is None or user.is_anonymous:
            await self.close(code=4401)
            return
        tenant = self.scope.get("tenant")
        if tenant is None:
            await self.close(code=4403)
            return
        self.group = f"ticks.{tenant.id}"
        self.subscriptions: set[str] = set()
        await self.channel_layer.group_add(self.group, self.channel_name)
        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)

    async def disconnect(self, code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)

    async def receive_json(self, content, **kwargs):
        op = content.get("op")
        if op == "subscribe":
            for tok in content.get("tokens", []):
                self.subscriptions.add(tok)
            await self.send_json({"op": "subscribed", "tokens": list(self.subscriptions)})
        elif op == "unsubscribe":
            for tok in content.get("tokens", []):
                self.subscriptions.discard(tok)
            await self.send_json({"op": "unsubscribed", "tokens": list(self.subscriptions)})

    async def tick(self, message):
        tick = message["tick"]
        if tick["token"] in self.subscriptions:
            await self.send_json(tick)
