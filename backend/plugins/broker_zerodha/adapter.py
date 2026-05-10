"""Zerodha Kite Connect adapter skeleton."""
from __future__ import annotations

from apps.agents_core.domain.contracts import BrokerOrderId


class ZerodhaAdapter:
    name = "zerodha"

    def place(self, order: dict) -> BrokerOrderId:
        raise NotImplementedError

    def cancel(self, order_id: BrokerOrderId) -> None:
        raise NotImplementedError

    async def stream_ticks(self, tokens):
        raise NotImplementedError
