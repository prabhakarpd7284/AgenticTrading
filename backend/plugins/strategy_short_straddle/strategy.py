"""Short-straddle lifecycle strategy — ported from trading/options/straddle/graph.py.

Nodes: fetch_market_data → analyze_position → generate_action → validate_action
       → execute_action → journal_action.
"""
from __future__ import annotations

from apps.agents_core.domain.contracts import (
    AgentContext,
    AgentEvent,
    Strategy,
    StrategySchema,
)


class ShortStraddleStrategy:
    name = "short_straddle"
    version = "1.0.0"
    asset_class = "options"

    def schema(self) -> StrategySchema:
        return StrategySchema(
            name=self.name,
            version=self.version,
            asset_class=self.asset_class,
            required_retrievers=["portfolio", "journal"],
            params={
                "type": "object",
                "properties": {
                    "underlying": {"type": "string", "enum": ["NIFTY", "BANKNIFTY"]},
                    "position_id": {"type": "string", "format": "uuid"},
                    "hard_stop_pct": {"type": "number", "default": 40.0},
                    "profit_take_pct": {"type": "number", "default": 50.0},
                },
                "required": ["underlying", "position_id"],
            },
        )

    def build_graph(self, ctx: AgentContext):
        from langgraph.graph import StateGraph, END

        async def fetch_market(state: dict) -> dict:
            seq = _next(state, ctx.publisher)
            state["vix"] = 14.5  # placeholder
            ctx.publisher.emit(AgentEvent(seq=seq, node="fetch_market_data", type="info",
                                          payload={"vix": state["vix"]}))
            return state

        async def analyze(state: dict) -> dict:
            seq = _next(state, ctx.publisher)
            state["analysis"] = {
                "pnl": 0, "delta": 0.05, "phase": "theta_decay",
                "scenarios": {"up_1pct": 0, "down_1pct": 0},
            }
            ctx.publisher.emit(AgentEvent(seq=seq, node="analyze_position", type="state",
                                          payload=state["analysis"]))
            return state

        async def generate(state: dict) -> dict:
            seq = _next(state, ctx.publisher)
            state["action"] = {"action": "HOLD", "reason": "Delta flat, IV crushing."}
            ctx.publisher.emit(AgentEvent(seq=seq, node="generate_action", type="result",
                                          payload=state["action"]))
            return state

        async def validate(state: dict) -> dict:
            seq = _next(state, ctx.publisher)
            d = ctx.risk.validate({"portfolio_id": ctx.portfolio_id,
                                    "symbol": "STRADDLE", "side": state["action"]["action"],
                                    "qty": 1})
            state["validated"] = d.model_dump()
            ctx.publisher.emit(AgentEvent(seq=seq, node="validate_action",
                                          type="result" if d.approved else "error",
                                          payload=d.model_dump()))
            return state

        async def execute(state: dict) -> dict:
            seq = _next(state, ctx.publisher)
            state["executed"] = True
            ctx.publisher.emit(AgentEvent(seq=seq, node="execute_action", type="result",
                                          payload={"executed": True}))
            return state

        async def journal_step(state: dict) -> dict:
            seq = _next(state, ctx.publisher)
            ctx.journal.record({
                "kind": "adjustment",
                "title": f"Straddle {state['action']['action']}",
                "body": state["action"]["reason"],
                "meta": state,
                "portfolio_id": ctx.portfolio_id,
                "agent_run_id": ctx.run_id,
            })
            ctx.publisher.emit(AgentEvent(seq=seq, node="journal_action",
                                          type="info", payload={"ok": True}))
            return state

        g = StateGraph(dict)
        for n, fn in [
            ("fetch_market_data", fetch_market),
            ("analyze_position", analyze),
            ("generate_action", generate),
            ("validate_action", validate),
            ("execute_action", execute),
            ("journal_action", journal_step),
        ]:
            g.add_node(n, fn)
        g.add_edge("fetch_market_data", "analyze_position")
        g.add_edge("analyze_position", "generate_action")
        g.add_edge("generate_action", "validate_action")
        g.add_edge("validate_action", "execute_action")
        g.add_edge("execute_action", "journal_action")
        g.add_edge("journal_action", END)
        g.set_entry_point("fetch_market_data")
        return g.compile()


def _next(state: dict, publisher=None) -> int:
    # Prefer the run-level publisher's monotonic counter so plugin seqs
    # never collide with the run's bookend events on the (run, seq) unique
    # constraint. See plugins/strategy_vertical_spread/strategy.py for the
    # postmortem on this bug — same one-line fix applied here defensively.
    if publisher is not None and hasattr(publisher, "next_seq"):
        return publisher.next_seq()
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


assert isinstance(ShortStraddleStrategy(), Strategy)
