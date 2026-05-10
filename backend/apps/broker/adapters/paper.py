"""Paper broker adapter — simulates fills at draft price (market) or limit, records to DB."""
from __future__ import annotations

import uuid

from apps.agents_core.domain.contracts import BrokerOrderId


class PaperBrokerAdapter:
    name = "paper"

    def place(self, order: dict) -> BrokerOrderId:
        return BrokerOrderId(broker="paper", id=f"PAPER-{uuid.uuid4().hex[:12]}")

    def cancel(self, order_id: BrokerOrderId) -> None:
        return None

    async def stream_ticks(self, tokens):
        # yields nothing — paper broker has no live feed
        return
        yield  # pragma: no cover
