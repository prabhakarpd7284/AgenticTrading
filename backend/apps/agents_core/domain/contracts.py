"""Contracts every agent-plugin must satisfy.

Kept at the top of the dependency graph — no imports from Django models.
The Strategy plugin receives an `AgentContext` containing ports; it never imports
Django models directly. This is the agentic-RAG plug boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel


# ---------- Events a plugin can emit during a run ----------
class AgentEvent(BaseModel):
    seq: int
    node: str
    type: Literal["token", "state", "result", "error", "info"]
    payload: dict[str, Any]


class EventPublisher(Protocol):
    def emit(self, event: AgentEvent) -> None: ...


# ---------- What a plugin gets handed at build-time ----------
@dataclass
class AgentContext:
    run_id: UUID
    tenant_id: UUID
    user_id: UUID
    portfolio_id: UUID
    market_data: "MarketDataPort"
    rag: "RAGRouter"
    risk: "RiskPort"
    journal: "JournalPort"
    publisher: EventPublisher
    config: dict[str, Any] = field(default_factory=dict)


# ---------- Ports the context exposes ----------
@runtime_checkable
class MarketDataPort(Protocol):
    def ltp(self, symbol: str) -> float: ...
    def candles(self, symbol: str, interval: str, n: int) -> list[dict]: ...
    def options_chain(self, underlying: str, expiry: str) -> dict: ...


@runtime_checkable
class RAGRouter(Protocol):
    def retrieve(self, query: "RetrievalQuery", retrievers: Iterable[str] | None = None,
                 k: int = 5) -> list["RetrievedDoc"]: ...


@runtime_checkable
class RiskPort(Protocol):
    def validate(self, draft: dict) -> "RiskDecision": ...


@runtime_checkable
class JournalPort(Protocol):
    def record(self, entry: dict) -> None: ...


# ---------- Simple value objects ----------
class RetrievalQuery(BaseModel):
    text: str
    filters: dict[str, Any] = {}


class RetrievedDoc(BaseModel):
    source: str
    score: float
    text: str
    metadata: dict[str, Any] = {}


class RiskDecision(BaseModel):
    approved: bool
    reason: str = ""
    adjustments: dict[str, Any] = {}


# ---------- Strategy plugin contract ----------
class StrategySchema(BaseModel):
    """JSON-schema-ish description of the strategy's configurable params.
    The frontend builder reads this to render the form.
    """
    name: str
    version: str
    asset_class: Literal["equity", "options", "futures"]
    params: dict[str, Any]   # JSONSchema object
    required_retrievers: list[str] = []


@runtime_checkable
class Strategy(Protocol):
    name: str
    version: str
    asset_class: Literal["equity", "options", "futures"]

    def schema(self) -> StrategySchema: ...

    def build_graph(self, ctx: AgentContext):
        """Return a compiled LangGraph runnable. The returned object must have
        an `ainvoke(state: dict)` method conforming to LangGraph's API."""
        ...


# ---------- Broker adapter contract ----------
class BrokerOrderId(BaseModel):
    broker: str
    id: str


class Tick(BaseModel):
    ts: float
    exchange: str
    token: str
    ltp: float
    oi: int | None = None
    vol: int | None = None


@runtime_checkable
class BrokerAdapter(Protocol):
    name: str
    def place(self, order: dict) -> BrokerOrderId: ...
    def cancel(self, order_id: BrokerOrderId) -> None: ...
    async def stream_ticks(self, tokens: list[str]): ...
