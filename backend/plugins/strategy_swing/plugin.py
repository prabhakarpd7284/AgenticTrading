"""SwingScannerStrategy plugin contract wrapper.

Wraps the OKScanner (Oliver Kell weekly cycle detector) in the agents_core
Strategy protocol so it can be discovered by the plugin registry. The
build_graph() returns a thin LangGraph with one node that runs the scan
across a configurable universe.
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import AgentContext, StrategySchema


class SwingScannerStrategy:
    name = "swing_scanner"
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
                        "description": "Override symbol universe (default: NIFTY 100)",
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["nifty50", "nifty100", "all"],
                        "default": "all",
                    },
                    "scan_date": {
                        "type": "string",
                        "description": "YYYY-MM-DD (default: today)",
                    },
                    "actionable_only": {"type": "boolean", "default": False},
                },
            },
        )

    def build_graph(self, ctx: AgentContext) -> Any:
        from langgraph.graph import END, StateGraph

        def run_node(state: dict) -> dict:
            from plugins.strategy_swing.ok_scanner import OKScanner
            from plugins.strategy_swing.ok_cycles import (
                BULLISH_ACTIONABLE, BEARISH_ACTIONABLE,
            )

            symbols = state.get("symbols") or []
            if not symbols:
                from apps.market_data.constants import (
                    NIFTY_50_SYMBOLS, SCREENER_UNIVERSE,
                )
                universe = state.get("universe", "all")
                symbols = NIFTY_50_SYMBOLS if universe == "nifty50" else SCREENER_UNIVERSE

            scanner = OKScanner()
            results = scanner.scan(symbols, scan_date=state.get("scan_date"))

            payload = []
            actionable = BULLISH_ACTIONABLE | BEARISH_ACTIONABLE
            for r in results:
                if state.get("actionable_only") and r.phase not in actionable:
                    continue
                payload.append({
                    "symbol": r.symbol,
                    "phase": getattr(r.phase, "name", str(r.phase)),
                    "trend": getattr(r.trend, "name", str(r.trend)),
                    "confidence": getattr(r, "confidence", None),
                    "close": getattr(r, "close", None),
                })
            return {**state, "swing_results": payload, "total_scanned": len(results)}

        g = StateGraph(dict)
        g.add_node("scan", run_node)
        g.set_entry_point("scan")
        g.add_edge("scan", END)
        return g.compile()
