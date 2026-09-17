"""BacktestStrategy plugin contract wrapper.

The backtester is generic infrastructure, not a tradable strategy. The
plugin contract makes it discoverable by the registry, and build_graph
exposes a single node that delegates to one of the compat helpers based
on the `engine` param: ok | basket | screener | intraday.
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import AgentContext, StrategySchema


class BacktestStrategy:
    name = "backtest"
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
                    "engine": {
                        "type": "string",
                        "enum": ["ok", "basket", "screener", "intraday"],
                        "description": "Which strategy adapter to drive the backtest with",
                    },
                    "from_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "to_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "pnl_mode": {
                        "type": "string",
                        "enum": ["RUPEES", "PERCENT"],
                        "default": "RUPEES",
                    },
                },
                "required": ["engine", "from_date", "to_date"],
            },
        )

    def build_graph(self, ctx: AgentContext) -> Any:
        from langgraph.graph import END, StateGraph

        def run_node(state: dict) -> dict:
            from plugins.strategy_backtest import compat
            from plugins.strategy_backtest.types import PnLMode

            engine = state["engine"]
            kwargs = {
                "from_date": state["from_date"],
                "to_date": state["to_date"],
                "symbols": state.get("symbols"),
                "pnl_mode": PnLMode[state.get("pnl_mode", "RUPEES")],
            }

            if engine == "ok":
                result = compat.run_ok_backtest(**kwargs)
            elif engine == "basket":
                result = compat.run_basket_backtest(**kwargs)
            elif engine == "screener":
                result = compat.run_screener_backtest(**kwargs)
            elif engine == "intraday":
                result = compat.run_intraday_backtest(**kwargs)
            else:
                raise ValueError(f"unknown backtest engine: {engine!r}")

            return {**state, "backtest_result": result}

        g = StateGraph(dict)
        g.add_node("run", run_node)
        g.set_entry_point("run")
        g.add_edge("run", END)
        return g.compile()
