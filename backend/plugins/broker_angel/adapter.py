"""Angel One SmartAPI adapter skeleton. Secrets loaded by ARN, cached via TokenRefresher."""
from __future__ import annotations

from apps.agents_core.domain.contracts import BrokerOrderId


class AngelOneAdapter:
    name = "angel_one"

    def place(self, order: dict) -> BrokerOrderId:
        # TODO: implement against SmartAPI placeOrder
        raise NotImplementedError

    def cancel(self, order_id: BrokerOrderId) -> None:
        raise NotImplementedError

    async def stream_ticks(self, tokens):
        raise NotImplementedError
