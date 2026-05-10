from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OrderDraft(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int = Field(gt=0)
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET"
    product: Literal["INTRADAY", "DELIVERY", "CARRYFORWARD"] = "INTRADAY"
    price: float | None = None
    sl: float | None = None
    tp: float | None = None
    origin: Literal["ui", "agent", "strategy"] = "ui"


class OrderResult(BaseModel):
    id: str
    status: str
