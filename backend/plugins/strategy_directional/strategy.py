"""Directional equity strategy plugin — thin adapter over the legacy LangGraph nodes.

This plugin is registered with the v2 `agents_core` runner via the
`alphadesk.strategies` entry-point group. Each node here mirrors the legacy
6-step workflow in `trading/graph/trading_graph.py` but emits `AgentEvent`s
to the v2 WebSocket stream (`/ws/agents/{run_id}/`) and writes to both
ledgers (legacy `TradeJournal` SQLite + v2 `JournalEntry` Postgres).

Design choice (option 1 / bridge period):
    Each node reaches into the legacy `trading.*` package directly rather
    than going through fully-ported v2 ports. Legacy code stays the source
    of truth for data, planner, risk, and broker execution — the v2 stack
    just wraps and streams. The WebSocket tick pipeline stays in legacy and
    scales on its own port; this plugin only does one-shot synchronous
    calls from inside async LangGraph nodes via `asyncio.to_thread`.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from apps.agents_core.domain.contracts import (
    AgentContext,
    AgentEvent,
    Strategy,
    StrategySchema,
)


class DirectionalStrategy:
    name = "directional"
    version = "1.1.0"
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
                    "symbol": {"type": "string", "default": "HDFCBANK"},
                    "intent": {
                        "type": "string",
                        "default": "Plan an intraday trade based on the latest structure.",
                    },
                    "capital": {"type": "number", "minimum": 10000, "default": 500000},
                    "daily_loss": {"type": "number", "minimum": 0, "default": 0},
                    "open_positions": {"type": "integer", "minimum": 0, "default": 0},
                    "dry_run": {"type": "boolean", "default": False},
                    "model": {"type": "string"},
                },
                "required": ["symbol"],
            },
        )

    def build_graph(self, ctx: AgentContext):
        from langgraph.graph import StateGraph, END

        async def fetch_data(state: dict) -> dict:
            cfg = state["config"]
            symbol = cfg.get("symbol", "HDFCBANK")
            today = date.today().strftime("%Y-%m-%d")

            from trading.services.data_service import DataService

            seq = _next(state)
            ctx.publisher.emit(AgentEvent(
                seq=seq, node="fetch_data", type="info",
                payload={"symbol": symbol, "date": today},
            ))

            ds = DataService()
            md = await asyncio.to_thread(ds.fetch_intraday, symbol, today, "FIVE_MINUTE")

            # Pull a multi-day window of 5-min bars so the UI chart has shape.
            # Legacy fetch_historical loops day-by-day and collapses single-day
            # windows to a daily bar, so we need >= 2 days for intraday output.
            from datetime import timedelta
            window_start = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
            try:
                candles = await asyncio.to_thread(
                    ds.fetch_historical, symbol, window_start, today, "FIVE_MINUTE",
                )
            except Exception:  # noqa: BLE001
                candles = []
            md["candles"] = candles

            # For the indicator panel we need ≥ 50 bars (SMA-50). The 5-min
            # intraday window collapses to ~5 daily bars after-hours, so
            # also pull 60 days of ONE_DAY candles and compute indicators
            # on those — sma_20, sma_50, rsi_14, atr_14 will populate.
            try:
                daily_start = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
                daily = await asyncio.to_thread(
                    ds.fetch_historical, symbol, daily_start, today, "ONE_DAY",
                )
            except Exception:  # noqa: BLE001
                daily = []
            md["daily_candles"] = daily
            state["indicators"] = _compute_indicators(daily if daily else candles)

            state["symbol"] = symbol
            state["market_data"] = md

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="fetch_data", type="state",
                payload={
                    "candle_count": md.get("candle_count"),
                    "last_close": md.get("last_close"),
                    "range_pct": md.get("range_pct"),
                    "indicators": {k: v for k, v in (state["indicators"] or {}).items() if k in ("sma_5", "sma_20", "rsi_14", "atr_14", "vwap")},
                    "error": md.get("error"),
                },
            ))
            return state

        async def retrieve_context_node(state: dict) -> dict:
            from trading.rag.retriever import retrieve_context as legacy_retrieve

            symbol = state["symbol"]
            seq = _next(state)
            ctx.publisher.emit(AgentEvent(
                seq=seq, node="retrieve_context", type="info",
                payload={"symbol": symbol},
            ))

            rag = await asyncio.to_thread(legacy_retrieve, symbol)
            state["rag_context"] = rag

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="retrieve_context", type="state",
                payload={"chars": len(rag)},
            ))
            return state

        async def planner(state: dict) -> dict:
            from trading.agents.planner import run_planner

            cfg = state["config"]
            intent = cfg.get("intent") or "Plan an intraday trade based on the latest structure."
            model = cfg.get("model")

            seq = _next(state)
            ctx.publisher.emit(AgentEvent(
                seq=seq, node="planner", type="info",
                payload={"intent": intent, "model": model},
            ))

            md_summary = (state.get("market_data") or {}).get("summary", "")
            rag = state.get("rag_context", "")

            plan = await asyncio.to_thread(run_planner, intent, md_summary, rag, model)
            state["plan"] = plan

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="planner",
                type="error" if plan.get("error") else "result",
                payload=plan,
            ))
            return state

        async def risk(state: dict) -> dict:
            from trading.services.risk_engine import validate_trade

            plan = state.get("plan") or {}
            if plan.get("error") or not plan.get("symbol"):
                state["risk"] = {
                    "approved": False,
                    "reason": plan.get("error", "Planner returned empty plan"),
                    "details": {},
                }
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="risk", type="error",
                    payload=state["risk"],
                ))
                return state

            cfg = state["config"]
            capital = float(cfg.get("capital", 500_000))
            daily_loss = float(cfg.get("daily_loss", 0))
            open_positions = int(cfg.get("open_positions", 0))

            approved, reason, details = await asyncio.to_thread(
                validate_trade, plan, capital, daily_loss, open_positions,
            )
            details = dict(details or {})

            # Compute margin + leverage and attach so the UI can show what
            # buying-power this trade would consume. Also enforces a
            # secondary leverage gate in addition to legacy validate_trade.
            try:
                from apps.common.margin_calc import equity_margin
                product = "MIS" if (cfg.get("product") or "MIS").upper() == "MIS" else "CNC"
                m = equity_margin(plan["side"], int(plan["quantity"]), float(plan["entry_price"]), product=product)
                details["margin"] = m.as_dict()
                details["margin_pct_of_capital"] = round((m.total_margin / capital) * 100, 2) if capital else 0
                # Hard leverage cap — 10× for equity intraday is already extreme.
                max_leverage = float(cfg.get("max_leverage", 10))
                if m.leverage > max_leverage and approved:
                    approved = False
                    reason = f"Leverage {m.leverage:.2f}× exceeds cap {max_leverage:.0f}×"
            except Exception as e:  # noqa: BLE001
                details["margin_error"] = str(e)

            state["risk"] = {"approved": approved, "reason": reason, "details": details}

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="risk",
                type="result" if approved else "error",
                payload=state["risk"],
            ))
            return state

        async def execute(state: dict) -> dict:
            risk_ok = state.get("risk", {}).get("approved", False)
            cfg = state["config"]

            if not risk_ok or cfg.get("dry_run"):
                state["execution"] = {
                    "success": False,
                    "skipped": True,
                    "reason": "risk_rejected" if not risk_ok else "dry_run",
                }
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="execute", type="info",
                    payload=state["execution"],
                ))
                return state

            from trading.services.broker_service import BrokerService

            plan = state["plan"]
            result = await asyncio.to_thread(
                BrokerService().place_order,
                plan["symbol"], plan["side"], int(plan["quantity"]),
                float(plan["entry_price"]), "LIMIT", "INTRADAY", "NSE",
            )
            state["execution"] = result

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="execute",
                type="result" if result.get("success") else "error",
                payload=result,
            ))
            return state

        async def journal_step(state: dict) -> dict:
            await asyncio.to_thread(_write_journals, state, ctx)
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="journal", type="info",
                payload={"ok": True},
            ))
            return state

        g = StateGraph(dict)
        g.add_node("fetch_data", fetch_data)
        g.add_node("retrieve_context", retrieve_context_node)
        g.add_node("planner", planner)
        g.add_node("risk", risk)
        g.add_node("execute", execute)
        g.add_node("journal", journal_step)
        g.add_edge("fetch_data", "retrieve_context")
        g.add_edge("retrieve_context", "planner")
        g.add_edge("planner", "risk")
        g.add_edge("risk", "execute")
        g.add_edge("execute", "journal")
        g.add_edge("journal", END)
        g.set_entry_point("fetch_data")
        return g.compile()


def _next(state: dict) -> int:
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


def _compute_indicators(candles: list) -> dict:
    """Snapshot indicator values from a candle list.

    Returns the values at the *latest* bar. Empty if not enough bars.
    Candle shape (from legacy fetch_historical): dict with open/high/low/close/volume,
    OR list[ts,o,h,l,c,v].
    """
    if not candles:
        return {}

    def _close(c):
        return float(c["close"] if isinstance(c, dict) else c[4])

    def _high(c):
        return float(c["high"] if isinstance(c, dict) else c[2])

    def _low(c):
        return float(c["low"] if isinstance(c, dict) else c[3])

    def _vol(c):
        try:
            return float(c.get("volume", 0) if isinstance(c, dict) else (c[5] if len(c) > 5 else 0))
        except (ValueError, TypeError):
            return 0.0

    closes = [_close(c) for c in candles]
    highs = [_high(c) for c in candles]
    lows = [_low(c) for c in candles]
    vols = [_vol(c) for c in candles]
    n = len(closes)

    def sma(period: int) -> float | None:
        return sum(closes[-period:]) / period if n >= period else None

    def rsi(period: int = 14) -> float | None:
        if n < period + 1:
            return None
        gains, losses = [], []
        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            (gains if delta > 0 else losses).append(abs(delta))
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    def atr(period: int = 14) -> float | None:
        if n < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return round(sum(trs) / period, 2)

    def vwap() -> float | None:
        tot_pv = sum((highs[i] + lows[i] + closes[i]) / 3 * vols[i] for i in range(n))
        tot_v = sum(vols)
        return round(tot_pv / tot_v, 2) if tot_v else None

    return {
        "last_close": round(closes[-1], 2),
        "sma_5":   round(sma(5), 2) if sma(5) is not None else None,
        "sma_20":  round(sma(20), 2) if sma(20) is not None else None,
        "sma_50":  round(sma(50), 2) if sma(50) is not None else None,
        "rsi_14":  rsi(14),
        "atr_14":  atr(14),
        "vwap":    vwap(),
        "prev_close": round(closes[-2], 2) if n >= 2 else None,
        "day_high": round(max(highs), 2),
        "day_low":  round(min(lows), 2),
        "candles_used": n,
    }


def _write_journals(state: dict, ctx: AgentContext) -> None:
    """Write to both ledgers: legacy TradeJournal (SQLite) + v2 JournalEntry (Postgres)."""
    plan: dict[str, Any] = state.get("plan") or {}
    risk: dict[str, Any] = state.get("risk") or {}
    execution: dict[str, Any] = state.get("execution") or {}

    # 1. Legacy TradeJournal — keeps the legacy Streamlit + bridge endpoints in sync.
    if plan.get("symbol") and not plan.get("error"):
        from trading.models import TradeJournal

        if not risk.get("approved"):
            status = TradeJournal.Status.REJECTED
        elif execution.get("success"):
            status = (
                TradeJournal.Status.PAPER
                if execution.get("mode") == "paper"
                else TradeJournal.Status.EXECUTED
            )
        else:
            status = TradeJournal.Status.APPROVED

        try:
            TradeJournal.objects.create(
                symbol=plan["symbol"],
                side=plan["side"],
                entry_price=plan["entry_price"],
                stop_loss=plan["stop_loss"],
                target=plan["target"],
                quantity=plan["quantity"],
                reasoning=plan.get("reasoning", ""),
                confidence=plan.get("confidence", 0),
                status=status,
                order_id=execution.get("order_id", ""),
                fill_price=execution.get("fill_price"),
                fill_quantity=execution.get("fill_quantity"),
                risk_approved=risk.get("approved", False),
                risk_reason=risk.get("reason", ""),
                trade_date=date.today(),
            )
        except Exception:  # noqa: BLE001
            # Legacy write must never block the v2 run.
            pass

    # 2. v2 JournalEntry — feeds the React monthly view + agent console.
    ctx.journal.record({
        "portfolio_id": ctx.portfolio_id,
        "agent_run_id": ctx.run_id,
        "kind": "plan",
        "title": f"{plan.get('side', '?')} {plan.get('symbol', '?')} x{plan.get('quantity', 0)}",
        "body": plan.get("reasoning", "") or risk.get("reason", ""),
        "meta": {
            "plan": plan,
            "risk": risk,
            "execution": execution,
            "market_data": {
                "last_close": (state.get("market_data") or {}).get("last_close"),
                "range_pct": (state.get("market_data") or {}).get("range_pct"),
            },
        },
    })


# Sanity check at import time — assert the Protocol is satisfied.
assert isinstance(DirectionalStrategy(), Strategy)
