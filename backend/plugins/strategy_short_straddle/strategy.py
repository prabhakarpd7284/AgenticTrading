"""Short-straddle lifecycle plugin — thin adapter over the legacy straddle stack.

Mirrors the legacy 6-node workflow in `trading/options/straddle/graph.py` but
drives the deterministic `lifecycle.decide(...)` rule set instead of calling
Claude. The LLM-driven path can be added later; this bridge port keeps the
v2 surface working with paper-mode straddles without needing API credits.

Inputs (in `state["config"]`):
    position_id — UUID of a `trading.models.StraddlePosition` row
    dry_run     — if true, execute_action records intent only
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

from apps.agents_core.domain.contracts import (
    AgentContext,
    AgentEvent,
    Strategy,
    StrategySchema,
)


class ShortStraddleStrategy:
    name = "short_straddle"
    version = "1.1.0"
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
                    "position_id": {"type": "integer"},
                    "dry_run": {"type": "boolean", "default": True},
                },
                "required": ["position_id"],
            },
        )

    def build_graph(self, ctx: AgentContext):
        from langgraph.graph import StateGraph, END

        async def fetch_market_data(state: dict) -> dict:
            cfg = state["config"]
            position_id = cfg.get("position_id")

            seq = _next(state)
            ctx.publisher.emit(AgentEvent(
                seq=seq, node="fetch_market_data", type="info",
                payload={"position_id": position_id},
            ))

            position, snapshot = await asyncio.to_thread(_load_position_and_snapshot, position_id)
            if position is None:
                state["error"] = f"StraddlePosition {position_id} not found"
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="fetch_market_data", type="error",
                    payload={"detail": state["error"]},
                ))
                return state

            state["position"] = _position_to_dict(position)
            state["snapshot"] = snapshot

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="fetch_market_data", type="state",
                payload={
                    "nifty": (snapshot.get("nifty") or {}).get("ltp"),
                    "vix": (snapshot.get("vix") or {}).get("ltp"),
                    "ce_ltp": (snapshot.get("ce") or {}).get("ltp"),
                    "pe_ltp": (snapshot.get("pe") or {}).get("ltp"),
                },
            ))
            return state

        async def analyze_position(state: dict) -> dict:
            if state.get("error"):
                return state

            from trading.options.straddle.analyzer import analyze_straddle

            p = state["position"]
            snap = state["snapshot"]
            nifty = snap.get("nifty") or {}
            vix = snap.get("vix") or {}
            ce = snap.get("ce") or {}
            pe = snap.get("pe") or {}

            analysis = await asyncio.to_thread(
                analyze_straddle,
                p["underlying"], p["strike"], p["expiry"], p["lot_size"], p["lots"],
                p["ce_sell"], p["pe_sell"],
                ce.get("ltp", 0.0), pe.get("ltp", 0.0),
                nifty.get("ltp", 0.0), nifty.get("prev_close", 0.0),
                vix.get("ltp", 0.0), vix.get("prev_close", 0.0),
                snap.get("candles") or [],
            )
            # `analysis` is a Pydantic model — keep the dict form for the LangGraph state.
            state["analysis"] = analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis)

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="analyze_position", type="state",
                payload={
                    "net_pnl_inr": state["analysis"].get("net_pnl_inr"),
                    "premium_decayed_pct": state["analysis"].get("premium_decayed_pct"),
                    "vix_phase": state["analysis"].get("vix_phase"),
                },
            ))
            return state

        async def generate_action(state: dict) -> dict:
            if state.get("error"):
                return state

            import os

            from trading.options.straddle.graph import (
                _generate_action_cli,
                _generate_action_api,
                _get_position_history,
            )

            analysis = state.get("analysis") or {}
            analysis_text = analysis.get("summary_text") or ""
            position_id = state["position"]["id"]
            history = await asyncio.to_thread(_get_position_history, position_id)

            mode = os.getenv("PLANNER_MODE", "cli").lower()
            llm_fn = _generate_action_api if mode == "api" else _generate_action_cli

            seq = _next(state)
            ctx.publisher.emit(AgentEvent(
                seq=seq, node="generate_action", type="info",
                payload={"mode": mode, "analysis_chars": len(analysis_text)},
            ))

            llm = await asyncio.to_thread(llm_fn, analysis_text, history)

            if llm.get("error") or not llm.get("action"):
                # Fall back to the deterministic rules so the run still completes.
                from trading.options.straddle.lifecycle import decide, count_shifts_today

                p = state["position"]
                snap = state["snapshot"]
                shifts_today = count_shifts_today(p.get("management_log") or [])
                is_expiry_day = p["expiry"] == date.today().isoformat()
                d = await asyncio.to_thread(
                    decide,
                    p["ce_sell"], p["pe_sell"],
                    (snap.get("ce") or {}).get("ltp", 0.0),
                    (snap.get("pe") or {}).get("ltp", 0.0),
                    p["strike"], p["strike"],
                    (snap.get("nifty") or {}).get("ltp", 0.0),
                    is_expiry_day, shifts_today,
                )
                state["action"] = {
                    "action": d.action,
                    "urgency": d.urgency,
                    "confidence": 1.0,
                    "reasoning": d.reason,
                    "ce_action": "CLOSE" if d.action in ("CLOSE_BOTH", "SHIFT_TO_ATM") else "HOLD",
                    "pe_action": "CLOSE" if d.action in ("CLOSE_BOTH", "SHIFT_TO_ATM") else "HOLD",
                    "fallback_reason": llm.get("error") or "llm returned empty action",
                    "source": "lifecycle.decide (fallback)",
                }
            else:
                # Coerce LLM payload into a stable dict.
                state["action"] = {
                    "action": llm["action"],
                    "urgency": llm.get("urgency", "MONITOR"),
                    "ce_action": llm.get("ce_action", "HOLD"),
                    "pe_action": llm.get("pe_action", "HOLD"),
                    "reasoning": llm.get("reasoning", ""),
                    "confidence": float(llm.get("confidence", 0.0)),
                    "key_risk": llm.get("key_risk", ""),
                    "hedge_side": llm.get("hedge_side"),
                    "hedge_lots": int(llm.get("hedge_lots", 0) or 0),
                    "pe_stop_loss": llm.get("pe_stop_loss"),
                    "pe_target": llm.get("pe_target"),
                    "roll_to_strike": llm.get("roll_to_strike"),
                    "source": f"claude ({mode})",
                }

            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="generate_action", type="result",
                payload={
                    "action": state["action"]["action"],
                    "urgency": state["action"].get("urgency"),
                    "confidence": state["action"].get("confidence"),
                    "ce_action": state["action"].get("ce_action"),
                    "pe_action": state["action"].get("pe_action"),
                    "source": state["action"].get("source"),
                    "reasoning": state["action"].get("reasoning", "")[:300],
                },
            ))
            return state

        async def validate_action(state: dict) -> dict:
            # Safety net on top of the LLM: never trade against the hard-stop
            # multiplier or below the minimum confidence bar. Mirrors the
            # legacy `validate_action_node` overrides but only the must-haves.
            if state.get("error"):
                return state

            from trading.options.straddle.lifecycle import HARD_STOP_MULTIPLIER

            analysis = state.get("analysis") or {}
            action = state["action"]
            combined_sold = analysis.get("combined_sold", 0)
            combined_current = analysis.get("combined_current", 0)
            severity = (combined_current / combined_sold) if combined_sold else 0
            confidence = float(action.get("confidence", 0))

            override = None
            reason = "LLM action approved"

            if severity >= HARD_STOP_MULTIPLIER and action["action"] != "CLOSE_BOTH":
                override = "CLOSE_BOTH"
                reason = (
                    f"Hard stop override: severity {severity:.2f}x >= "
                    f"{HARD_STOP_MULTIPLIER}x. Forcing CLOSE_BOTH."
                )
                action.update({
                    "action": "CLOSE_BOTH",
                    "urgency": "IMMEDIATE",
                    "ce_action": "CLOSE",
                    "pe_action": "CLOSE",
                })
            elif confidence < 0.6 and action["action"] != "HOLD":
                override = "MONITOR"
                reason = f"Confidence {confidence:.2f} < 0.6 — downgrading to MONITOR."
                action.update({"action": "MONITOR", "urgency": "MONITOR"})

            state["validated"] = {"approved": True, "override": override, "reason": reason}
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="validate_action",
                type="result" if not override else "error",
                payload=state["validated"],
            ))
            return state

        async def execute_action(state: dict) -> dict:
            if state.get("error"):
                return state

            action = (state.get("action") or {}).get("action", "HOLD")
            dry_run = bool(state["config"].get("dry_run", True))

            # Real order execution lives in the legacy CLI for now
            # (`python manage.py manage_straddle --execute ...`). The v2
            # bridge records intent so the React UI can show the decision.
            state["execution"] = {
                "executed": False,
                "intent": action,
                "dry_run": dry_run,
                "note": "Straddle execution stays in legacy CLI for paper safety.",
            }
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="execute_action", type="info",
                payload=state["execution"],
            ))
            return state

        async def journal_action(state: dict) -> dict:
            await asyncio.to_thread(_write_straddle_journal, state, ctx)
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="journal_action", type="info",
                payload={"ok": True, "error": state.get("error")},
            ))
            return state

        g = StateGraph(dict)
        for n, fn in [
            ("fetch_market_data", fetch_market_data),
            ("analyze_position", analyze_position),
            ("generate_action", generate_action),
            ("validate_action", validate_action),
            ("execute_action", execute_action),
            ("journal_action", journal_action),
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


def _next(state: dict) -> int:
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


def _load_position_and_snapshot(position_id: int):
    """Sync helper — runs inside asyncio.to_thread.

    Builds the snapshot via individual `ltpData` calls rather than
    `market_data_batch`. The batch endpoint returns an empty list outside
    market hours (Angel One quirk) which would cascade into the analyzer
    seeing zero LTPs. Individual `ltpData` calls return the last-close
    OHLC even after-hours, which is what we want for review runs.

    Also pulls 5-min candles for NIFTY, CE and PE over the last few
    trading days so the React UI can render intraday close charts for
    each leg + the underlying.
    """
    from datetime import datetime, timedelta
    from trading.models import StraddlePosition
    from trading.options.data_service import OptionsDataService

    try:
        position = StraddlePosition.objects.get(id=position_id)
    except StraddlePosition.DoesNotExist:
        return None, {}

    ods = OptionsDataService()
    snapshot: dict = {"nifty": {}, "vix": {}, "ce": {}, "pe": {},
                       "candles": [], "ce_candles": [], "pe_candles": []}

    # ── 1. LTPs via individual ltpData calls (robust after-hours) ──
    try:
        snapshot["nifty"] = ods.fetch_nifty_spot()
    except Exception as e:  # noqa: BLE001
        snapshot["nifty_error"] = str(e)
    try:
        snapshot["vix"] = ods.fetch_vix()
    except Exception as e:  # noqa: BLE001
        snapshot["vix_error"] = str(e)
    try:
        snapshot["ce"] = ods.fetch_option_ltp(position.ce_symbol, position.ce_token)
    except Exception as e:  # noqa: BLE001
        snapshot["ce_error"] = str(e)
    try:
        snapshot["pe"] = ods.fetch_option_ltp(position.pe_symbol, position.pe_token)
    except Exception as e:  # noqa: BLE001
        snapshot["pe_error"] = str(e)

    # ── 2. 5-min intraday close candles for the UI charts ──
    # Window: last 5 days (handles weekends/holidays cleanly — broker
    # collapses to daily bars after-hours but we still get something
    # plottable). Pulls NIFTY (NSE) + CE/PE (NFO) directly via
    # BrokerClient.fetch_candles since the OptionsDataService helper
    # only knows NIFTY and doesn't take a date-range.
    try:
        from trading.services.data_service import BrokerClient
        from trading.options.data_service import NIFTY_SPOT_TOKEN

        today = date.today()
        broker = BrokerClient.get_instance()
        broker.ensure_login()
        start = (today - timedelta(days=5)).strftime("%Y-%m-%d %H:%M")
        end = today.strftime("%Y-%m-%d 15:30")

        def _fetch(token: str, exchange: str) -> list:
            try:
                return broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange=exchange) or []
            except TypeError:
                return broker.fetch_candles(token, start, end, "FIVE_MINUTE") or []

        snapshot["candles"]    = _fetch(NIFTY_SPOT_TOKEN, "NSE")
        snapshot["ce_candles"] = _fetch(position.ce_token, "NFO")
        snapshot["pe_candles"] = _fetch(position.pe_token, "NFO")
    except Exception as e:  # noqa: BLE001
        snapshot["option_candles_error"] = str(e)

    return position, snapshot


def _position_to_dict(position) -> dict[str, Any]:
    """Flatten StraddlePosition into a JSON-safe dict for the LangGraph state."""
    return {
        "id": position.id,
        "underlying": position.underlying,
        "strike": position.strike,
        "expiry": position.expiry.isoformat() if position.expiry else "",
        "lot_size": position.lot_size,
        "lots": position.lots,
        "ce_symbol": position.ce_symbol,
        "ce_token": position.ce_token,
        "ce_sell": position.ce_sell_price,
        "pe_symbol": position.pe_symbol,
        "pe_token": position.pe_token,
        "pe_sell": position.pe_sell_price,
        "management_log": position.management_log or [],
    }


def _write_straddle_journal(state: dict, ctx: AgentContext) -> None:
    """Append to legacy StraddlePosition.management_log + write v2 JournalEntry."""
    if state.get("error"):
        ctx.journal.record({
            "portfolio_id": ctx.portfolio_id,
            "agent_run_id": ctx.run_id,
            "kind": "error",
            "title": "Straddle run errored",
            "body": state["error"],
            "meta": {"state": _scrub(state)},
        })
        return

    p = state.get("position") or {}
    action = state.get("action") or {}
    analysis = state.get("analysis") or {}

    # 1. Append to legacy management_log
    try:
        from trading.models import StraddlePosition

        position = StraddlePosition.objects.get(id=p["id"])
        log = list(position.management_log or [])
        log.append({
            "ts": datetime.now().isoformat(),
            "source": "v2-agent",
            "action": action.get("action"),
            "reason": action.get("reason"),
            "urgency": action.get("urgency"),
            "new_strike": action.get("new_strike"),
            "net_pnl_inr": analysis.get("net_pnl_inr"),
            "premium_decayed_pct": analysis.get("premium_decayed_pct"),
        })
        position.management_log = log
        position.save(update_fields=["management_log", "updated_at"])
    except Exception:  # noqa: BLE001
        pass

    # 2. v2 JournalEntry
    ctx.journal.record({
        "portfolio_id": ctx.portfolio_id,
        "agent_run_id": ctx.run_id,
        "kind": "adjustment",
        "title": f"Straddle {action.get('action', '?')} · {p.get('underlying', '?')} {p.get('strike', '?')}",
        "body": action.get("reason", ""),
        "meta": {
            "position": p,
            "action": action,
            "analysis": {
                "net_pnl_inr": analysis.get("net_pnl_inr"),
                "premium_decayed_pct": analysis.get("premium_decayed_pct"),
                "vix_phase": analysis.get("vix_phase"),
            },
            "execution": state.get("execution"),
        },
    })


def _scrub(state: dict) -> dict:
    """Strip non-JSON-serializable bits from state before writing to journal meta."""
    return {k: v for k, v in state.items() if k not in {"_seq"}}


# Sanity check at import time — assert the Protocol is satisfied.
assert isinstance(ShortStraddleStrategy(), Strategy)
