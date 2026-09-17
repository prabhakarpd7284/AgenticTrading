"""ScalpStrategy plugin contract wrapper.

Wraps the pure-Python scalp engine (engine.py / profile.py / replay.py) in the
agents_core Strategy protocol so the plugin registry can load it via entry-point
and the frontend can read its config schema.

The *interactive* simulation runs in the ScalpSim WebSocket consumer (controllable
speed + manual override), not here. ``build_graph`` provides a deterministic,
non-streaming fallback so the strategy still works through the catalog/AgentRun
path (e.g. dry-run rendering).
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import AgentContext, StrategySchema


class ScalpStrategy:
    name = "scalp"
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
                    "underlying": {"type": "string", "enum": ["NIFTY", "BANKNIFTY", "SENSEX"], "default": "NIFTY"},
                    "strike": {"type": "integer"},
                    "type": {"type": "string", "enum": ["CE", "PE"], "default": "CE"},
                    "expiry": {"type": "string", "description": "DDMMMYY (e.g. 07JUL26) or empty for next"},
                    "date": {"type": "string", "description": "Session date YYYY-MM-DD (empty = last trading day)"},
                    "resolution": {"type": "string", "enum": ["5S", "10S", "15S", "30S", "45S", "1"], "default": "5S"},
                    "decision_secs": {"type": "integer", "minimum": 60, "maximum": 1800, "default": 600},
                    "bin_width": {"type": "number", "minimum": 1, "default": 20.0},
                    "window_secs": {"type": "integer", "minimum": 30, "maximum": 1800, "default": 180},
                    "value_area_pct": {"type": "number", "minimum": 0.5, "maximum": 0.95, "default": 0.70},
                    "entry_threshold": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.30},
                    "add_threshold": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.22},
                    "reverse_threshold": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.55},
                    "trend_only": {"type": "boolean", "default": True,
                                   "description": "Only take with-trend entries; sit out chop"},
                    "trend_ema_span": {"type": "integer", "minimum": 2, "maximum": 30, "default": 5},
                    "trend_flat_band": {"type": "number", "minimum": 0.0, "maximum": 0.1, "default": 0.012},
                    "reentry_cooldown_secs": {"type": "number", "minimum": 0, "maximum": 600, "default": 60},
                    "bin_reversal_entry": {"type": "boolean", "default": True,
                                           "description": "Enter on a pullback-to-bin reversal (vs chasing pressure)"},
                    "pullback_min_bins": {"type": "number", "minimum": 0.1, "maximum": 5, "default": 0.6},
                    "reversal_bins": {"type": "number", "minimum": 0.1, "maximum": 5, "default": 0.35},
                    "speed": {"type": "number", "minimum": 0.25, "maximum": 64, "default": 4.0},
                    "capital": {"type": "number", "minimum": 10_000, "default": 100_000},
                    "risk_pct": {"type": "number", "minimum": 0.1, "maximum": 5.0, "default": 1.0},
                    "max_pyramids": {"type": "integer", "minimum": 0, "maximum": 10, "default": 4},
                    "lot_size": {"type": "integer", "default": 65},
                    "require_bias_alignment": {"type": "boolean", "default": True},
                    "allow_reverse": {"type": "boolean", "default": True},
                    "tick_synthesis": {"type": "string", "enum": ["ohlc", "close"], "default": "ohlc"},
                    "mode": {"type": "string", "enum": ["sim", "live"], "default": "sim"},
                    "place_orders": {"type": "boolean", "default": False,
                                     "description": "Live mode: route opening decisions through PlaceOrder (paper only)"},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["strike"],
            },
        )

    def build_graph(self, ctx: AgentContext) -> Any:
        """Deterministic, non-streaming fallback graph (one node). The live
        controllable simulation is the WebSocket consumer, not this."""
        from langgraph.graph import END, StateGraph

        def run_node(state: dict) -> dict:
            from .engine import Candle, ScalpConfig
            from .replay import generate_scalp_sample, resolution_to_secs, run_scalp_with_chart_data

            res_secs = resolution_to_secs(state.get("resolution", "5S"))
            config = ScalpConfig(
                lot_size=state.get("lot_size", 65),
                initial_capital=state.get("capital", 100_000),
                initial_risk_pct=state.get("risk_pct", 1.0),
                max_pyramids=state.get("max_pyramids", 4),
                bin_width=state.get("bin_width", 20.0),
                window_secs=state.get("window_secs", 180),
                value_area_pct=state.get("value_area_pct", 0.70),
                entry_threshold=state.get("entry_threshold", 0.55),
                add_threshold=state.get("add_threshold", 0.45),
                reverse_threshold=state.get("reverse_threshold", 0.60),
                require_bias_alignment=state.get("require_bias_alignment", True),
                allow_reverse=state.get("allow_reverse", True),
            )
            if state.get("dry_run") or not state.get("candles"):
                candles = generate_scalp_sample(res_secs)
                symbol = f"{state.get('underlying','NIFTY')} {state.get('strike')} {state.get('type','CE')} (sample)"
            else:
                candles = [c if isinstance(c, Candle) else Candle.from_fyers(c) for c in state["candles"]]
                symbol = state.get("symbol", "SCALP")

            result = run_scalp_with_chart_data(
                candles, symbol=symbol, config=config,
                resolution_secs=res_secs, decision_secs=state.get("decision_secs", 600),
                tick_synthesis=state.get("tick_synthesis", "ohlc"),
            )
            return {**state, "scalp_result": result}

        g = StateGraph(dict)
        g.add_node("run", run_node)
        g.set_entry_point("run")
        g.add_edge("run", END)
        return g.compile()


# Enforce the protocol at import (mirrors strategy_directional).
from apps.agents_core.domain.contracts import Strategy as _Strategy  # noqa: E402

assert isinstance(ScalpStrategy(), _Strategy)
