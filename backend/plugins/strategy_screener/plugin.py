"""ScreenerStrategy plugin contract wrapper.

The intraday screener is a long-running loop (TickStream → CandleStore →
strategies) rather than a one-shot trade plan, so the LangGraph returned
by build_graph() runs a *bounded* batch (e.g. one bootstrap + N bars from
state['bars']) and emits any signals it produced. The live CLI variant
remains the screener's primary entry point.
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import AgentContext, StrategySchema


class ScreenerStrategy:
    name = "intraday_screener"
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
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Override symbol universe (default: NIFTY 50 + Next 50)",
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["nifty50", "nifty100", "all"],
                        "default": "all",
                    },
                    "capital": {"type": "number", "minimum": 10_000, "default": 500_000},
                    "max_risk_pct": {
                        "type": "number", "minimum": 0.1, "maximum": 5.0, "default": 1.0
                    },
                    "disabled_strategies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "only_strategy": {"type": "string", "description": "Run only this strategy"},
                    "max_bars": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 100,
                        "description": "Bar batch budget when invoked via build_graph",
                    },
                },
            },
        )

    def build_graph(self, ctx: AgentContext) -> Any:
        """Run a bounded screener batch and return all signals fired.

        The live screener has its own long-running loop (managed by the
        run_screener CLI) — this graph is what wires the screener into the
        agents framework for short-lived programmatic invocations.
        """
        from langgraph.graph import END, StateGraph

        def run_node(state: dict) -> dict:
            from plugins.strategy_screener.engine import ScreenerEngine
            from plugins.strategy_screener.strategies import STRATEGIES

            symbols = state.get("symbols") or []
            if not symbols:
                from apps.market_data.constants import (
                    NIFTY_50_SYMBOLS, SCREENER_UNIVERSE,
                )
                universe = state.get("universe", "all")
                symbols = NIFTY_50_SYMBOLS if universe == "nifty50" else SCREENER_UNIVERSE

            strategies = [s for s in STRATEGIES if s.name not in (state.get("disabled_strategies") or [])]
            only = state.get("only_strategy")
            if only:
                strategies = [s for s in strategies if s.name == only]

            engine = ScreenerEngine(
                symbols=symbols,
                strategies=strategies,
                capital=state.get("capital", 500_000),
                max_risk_pct=state.get("max_risk_pct", 1.0),
            )

            collected: list[dict] = []
            engine.add_output_handler(lambda sig: collected.append(sig.__dict__))

            # Caller is expected to seed engine via bootstrap (live REST hydrate
            # done by the CLI) and then feed ticks from state['ticks'].
            ticks = state.get("ticks") or []
            for t in ticks[: state.get("max_bars", 100)]:
                engine.on_tick(t["symbol"], t["ltp"], t.get("volume", 0), t.get("timestamp"))

            return {**state, "screener_signals": collected, "bars_processed": engine.bars_processed}

        g = StateGraph(dict)
        g.add_node("run", run_node)
        g.set_entry_point("run")
        g.add_edge("run", END)
        return g.compile()
