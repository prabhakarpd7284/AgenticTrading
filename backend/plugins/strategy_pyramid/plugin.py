"""PyramidStrategy plugin contract wrapper.

Wraps the pure-Python pyramid engine (strategy.py) in the agents_core
Strategy protocol so the plugin registry can load it via entry-point.

The engine itself runs deterministically — no LLM — so build_graph()
returns a thin LangGraph with a single node that delegates to
run_pyramid_with_chart_data().
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import AgentContext, StrategySchema


class PyramidStrategy:
    name = "pyramid"
    version = "1.0.0"
    asset_class = "options"

    def schema(self) -> StrategySchema:
        return StrategySchema(
            name=self.name,
            version=self.version,
            asset_class=self.asset_class,
            params={
                "type": "object",
                "properties": {
                    "underlying": {"type": "string",
                                    "enum": ["NIFTY", "BANKNIFTY", "SENSEX"],
                                    "default": "NIFTY"},
                    "strike": {"type": "integer"},
                    "type": {"type": "string", "enum": ["CE", "PE"], "default": "CE"},
                    "expiry": {"type": "string",
                                "description": "DDMMMYY (e.g. 13MAY26) or empty for next"},
                    "capital": {"type": "number", "minimum": 10_000, "default": 100_000},
                    "risk_pct": {"type": "number", "minimum": 0.1, "maximum": 5.0, "default": 2.0},
                    "profit_risk": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.80},
                    "max_pyramids": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "lot_size": {"type": "integer", "default": 65},
                    "max_risk_pct_of_price": {"type": "number",
                                                "minimum": 0.05, "maximum": 1.0, "default": 0.50,
                                                "description": "SL distance cap"},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["strike"],
            },
        )

    def build_graph(self, ctx: AgentContext) -> Any:
        """Return a LangGraph-compatible runnable. The pyramid is
        deterministic, so the graph is one node — but it still emits
        AgentEvents for the timeline UI.
        """
        # Lazy LangGraph import — only loaded when a run actually starts
        from langgraph.graph import END, StateGraph

        def run_node(state: dict) -> dict:
            from .strategy import (
                Candle, PyramidConfig, run_pyramid_with_chart_data,
                _generate_pyramid_sample,
            )
            config = PyramidConfig(
                lot_size=state.get("lot_size", 65),
                initial_capital=state.get("capital", 100_000),
                initial_risk_pct=state.get("risk_pct", 2.0),
                profit_risk_pct=state.get("profit_risk", 0.80),
                max_pyramids=state.get("max_pyramids", 5),
                max_risk_pct_of_price=state.get("max_risk_pct_of_price", 0.50),
            )

            if state.get("dry_run"):
                candles = _generate_pyramid_sample()
                symbol = f"{state.get('underlying','NIFTY')} {state.get('strike')} {state.get('type','CE')} (sample)"
            else:
                # Caller is expected to have hydrated `candles` in state
                # (e.g. from the legacy bridge view fetching live data).
                candles = state.get("candles") or []
                symbol = state.get("symbol", "PYRAMID")

            result = run_pyramid_with_chart_data(candles, symbol=symbol, config=config)
            return {**state, "pyramid_result": result}

        g = StateGraph(dict)
        g.add_node("run", run_node)
        g.set_entry_point("run")
        g.add_edge("run", END)
        return g.compile()
