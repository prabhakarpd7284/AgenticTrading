"""Directional equity strategy — ported from trading/agents/planner.py.

Builds a LangGraph with 6 nodes: fetch_data → retrieve_context → planner
→ risk → execute → journal. Every node emits an AgentEvent for WS streaming.
"""
from __future__ import annotations

from apps.agents_core.domain.contracts import (
    AgentContext,
    AgentEvent,
    RetrievalQuery,
    Strategy,
    StrategySchema,
)


class DirectionalStrategy:
    name = "directional"
    version = "1.0.0"
    asset_class = "equity"

    def schema(self) -> StrategySchema:
        return StrategySchema(
            name=self.name,
            version=self.version,
            asset_class=self.asset_class,
            required_retrievers=["portfolio", "journal", "news"],
            params={
                "type": "object",
                "properties": {
                    "universe": {"type": "string", "enum": ["NIFTY50", "NIFTY100"], "default": "NIFTY50"},
                    "side_bias": {"type": "string", "enum": ["auto", "long", "short"], "default": "auto"},
                    "risk_per_trade_pct": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.0},
                    "max_positions": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                },
                "required": ["universe"],
            },
        )

    def build_graph(self, ctx: AgentContext):
        from langgraph.graph import StateGraph, END

        async def fetch(state: dict) -> dict:
            seq = _next(state)
            ctx.publisher.emit(AgentEvent(seq=seq, node="fetch_data", type="info",
                                          payload={"universe": state["config"].get("universe")}))
            state["ltp"] = ctx.market_data.ltp("NIFTY")
            return state

        async def retrieve(state: dict) -> dict:
            seq = _next(state)
            docs = ctx.rag.retrieve(RetrievalQuery(text="recent drawdowns"), k=5)
            state["context"] = [d.model_dump() for d in docs]
            ctx.publisher.emit(AgentEvent(seq=seq, node="retrieve_context", type="state",
                                          payload={"docs": len(docs)}))
            return state

        async def planner(state: dict) -> dict:
            seq = _next(state)
            # Placeholder: call the LLM here. Shipping without a network call in dev.
            state["plan"] = {
                "symbol": "HDFCBANK", "side": "BUY", "qty": 10,
                "sl": 1460.0, "tp": 1520.0, "rr": 2.5,
                "thesis": "VWAP reclaim on rising volume; 5d RS > NIFTY.",
            }
            ctx.publisher.emit(AgentEvent(seq=seq, node="planner", type="result",
                                          payload=state["plan"]))
            return state

        async def risk(state: dict) -> dict:
            seq = _next(state)
            d = ctx.risk.validate({"portfolio_id": ctx.portfolio_id, **state["plan"]})
            state["risk"] = d.model_dump()
            ctx.publisher.emit(AgentEvent(seq=seq, node="risk",
                                          type="result" if d.approved else "error",
                                          payload=d.model_dump()))
            return state

        async def journal_step(state: dict) -> dict:
            seq = _next(state)
            ctx.journal.record({
                "kind": "plan",
                "title": "Directional plan",
                "body": state["plan"]["thesis"],
                "meta": state,
                "portfolio_id": ctx.portfolio_id,
                "agent_run_id": ctx.run_id,
            })
            ctx.publisher.emit(AgentEvent(seq=seq, node="journal", type="info",
                                          payload={"ok": True}))
            return state

        g = StateGraph(dict)
        g.add_node("fetch_data", fetch)
        g.add_node("retrieve_context", retrieve)
        g.add_node("planner", planner)
        g.add_node("risk", risk)
        g.add_node("journal", journal_step)
        g.add_edge("fetch_data", "retrieve_context")
        g.add_edge("retrieve_context", "planner")
        g.add_edge("planner", "risk")
        g.add_edge("risk", "journal")
        g.add_edge("journal", END)
        g.set_entry_point("fetch_data")
        return g.compile()


def _next(state: dict) -> int:
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


# Sanity: make sure we satisfy the Strategy Protocol at import time.
assert isinstance(DirectionalStrategy(), Strategy)
