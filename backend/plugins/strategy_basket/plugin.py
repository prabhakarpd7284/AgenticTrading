"""PremarketBasketStrategy plugin contract wrapper.

Wraps the premarket basket pipeline (mood → signal selection → executor)
in the agents_core Strategy protocol so it can be discovered by the
plugin registry. The build_graph runs a single deterministic node — the
basket isn't an LLM workflow, but it still emits AgentEvents.
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import AgentContext, StrategySchema


class PremarketBasketStrategy:
    name = "premarket_basket"
    version = "1.0.0"
    asset_class = "equity"

    def schema(self) -> StrategySchema:
        return StrategySchema(
            name=self.name,
            version=self.version,
            asset_class=self.asset_class,
            params={
                "type": "object",
                "properties": {
                    "capital": {"type": "number", "minimum": 10_000, "default": 500_000},
                    "max_legs": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                    "risk_pct": {"type": "number", "minimum": 0.1, "maximum": 5.0, "default": 1.0},
                    "dry_run": {"type": "boolean", "default": True},
                },
            },
        )

    def build_graph(self, ctx: AgentContext) -> Any:
        from langgraph.graph import END, StateGraph

        def run_node(state: dict) -> dict:
            from plugins.strategy_basket.config import BasketConfig
            from plugins.strategy_basket.mood import MarketMoodAssessor
            from plugins.strategy_basket.signals import BasketSignalGenerator
            from plugins.strategy_basket.executor import BasketExecutor

            cfg = BasketConfig()
            if "capital" in state:
                cfg.capital = state["capital"]
            if "max_legs" in state:
                cfg.max_legs = state["max_legs"]
            if "risk_pct" in state:
                cfg.risk_pct_per_leg = state["risk_pct"]

            mood = MarketMoodAssessor(cfg).assess()
            signals = BasketSignalGenerator(cfg).build(mood)

            if state.get("dry_run", True):
                return {
                    **state,
                    "mood": mood.to_dict() if hasattr(mood, "to_dict") else mood.__dict__,
                    "signals": [s.__dict__ for s in signals],
                    "placed": False,
                }

            executor = BasketExecutor(cfg)
            legs = executor.place_basket(signals)
            return {
                **state,
                "mood": mood.to_dict() if hasattr(mood, "to_dict") else mood.__dict__,
                "signals": [s.__dict__ for s in signals],
                "legs": [l.__dict__ for l in legs],
                "placed": True,
            }

        g = StateGraph(dict)
        g.add_node("build", run_node)
        g.set_entry_point("build")
        g.add_edge("build", END)
        return g.compile()
