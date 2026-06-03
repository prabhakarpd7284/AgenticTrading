"""SwingScannerV2Strategy — agents_core contract wrapper for the v2 engine.

Registered as `swing_scanner_v2` so it lives alongside the untouched v1
`swing_scanner`. A thin LangGraph with one node that runs the tier scan.
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import AgentContext, StrategySchema


class SwingScannerV2Strategy:
    name = "swing_scanner_v2"
    version = "2.0.0"
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
                        "description": "Override symbol universe (default: SCREENER_UNIVERSE)",
                    },
                    "universe": {
                        "type": "string",
                        "enum": ["nifty50", "nifty100", "all"],
                        "default": "all",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["small", "medium", "long"],
                        "default": "medium",
                        "description": "small=15m, medium=1h, long=daily",
                    },
                    "scan_date": {"type": "string", "description": "YYYY-MM-DD (default: today)"},
                    "actionable_only": {"type": "boolean", "default": True},
                    "capital": {"type": "number", "default": 500000},
                },
            },
        )

    def build_graph(self, ctx: AgentContext) -> Any:
        from langgraph.graph import END, StateGraph

        def run_node(state: dict) -> dict:
            from plugins.strategy_swing.v2.scanner import SwingScannerV2

            symbols = state.get("symbols") or []
            if not symbols:
                from apps.market_data.constants import NIFTY_50_SYMBOLS, SCREENER_UNIVERSE
                symbols = NIFTY_50_SYMBOLS if state.get("universe") == "nifty50" else SCREENER_UNIVERSE

            scanner = SwingScannerV2(
                tier=state.get("tier", "medium"),
                capital=float(state.get("capital", 500_000)),
            )
            results = scanner.scan(symbols, scan_date=state.get("scan_date"))

            only = state.get("actionable_only", True)
            payload = [r.to_dict() for r in results if (r.actionable or not only)]
            return {
                **state,
                "swing_v2_results": payload,
                "total_scanned": len(results),
                "actionable_count": len(scanner.actionable()),
            }

        g = StateGraph(dict)
        g.add_node("scan", run_node)
        g.set_entry_point("scan")
        g.add_edge("scan", END)
        return g.compile()
