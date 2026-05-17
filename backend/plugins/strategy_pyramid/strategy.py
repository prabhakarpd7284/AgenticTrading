"""Pyramid options strategy plugin — bridge port over legacy run_pyramid().

The legacy implementation lives in `trading.pyramid.strategy` and is a
pure-Python simulation that processes 5-min option candles and produces
a PyramidResult (entries · exit · realized P&L · log). No LLM call.

This plugin:
  1. Picks an option leg (underlying + strike + CE/PE + expiry) from config.
  2. Pulls today's 5-min candles for that option token.
  3. Runs `trading.pyramid.strategy.run_pyramid(...)` against them.
  4. Returns a plan-shaped result (entries, exit, P&L projection).

If today's market is closed or no candles are available, the plugin returns
a stub with `error="no candles"` so the orchestrator can render gracefully.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Literal

from apps.agents_core.domain.contracts import (
    AgentContext,
    AgentEvent,
    Strategy,
    StrategySchema,
)


class PyramidStrategy:
    name = "pyramid"
    version = "1.0.0"
    asset_class = "options"

    def schema(self) -> StrategySchema:
        return StrategySchema(
            name=self.name,
            version=self.version,
            asset_class=self.asset_class,
            required_retrievers=[],
            params={
                "type": "object",
                "properties": {
                    "underlying":  {"type": "string", "default": "NIFTY"},
                    "strike":      {"type": "integer", "description": "ATM ± offset"},
                    "option_type": {"type": "string", "enum": ["CE", "PE"], "default": "CE"},
                    "expiry":      {"type": "string", "description": "DDMMMYY (e.g. 22MAY26). Empty = nearest weekly."},
                    "capital":     {"type": "number", "minimum": 10000, "default": 100000},
                    "risk_pct":    {"type": "number", "minimum": 0.1, "maximum": 5.0, "default": 2.0},
                    "max_pyramids": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "lookback_days": {
                        "type": "integer", "minimum": 1, "maximum": 10, "default": 3,
                        "description": "How many days of 5-min option candles to fetch (the simulator needs at least one trading session of intraday data; on Sundays/Mondays default to 3 to cover last Thu-Fri).",
                    },
                },
                "required": ["underlying", "option_type"],
            },
        )

    def build_graph(self, ctx: AgentContext):
        from langgraph.graph import StateGraph, END

        async def fetch_data(state: dict) -> dict:
            cfg = state["config"]
            underlying = (cfg.get("underlying") or "NIFTY").upper()
            opt_type: Literal["CE", "PE"] = cfg.get("option_type", "CE")
            expiry = cfg.get("expiry") or _default_weekly_expiry(underlying)

            seq = _next(state)
            ctx.publisher.emit(AgentEvent(
                seq=seq, node="fetch_data", type="info",
                payload={"underlying": underlying, "option_type": opt_type, "expiry": expiry},
            ))

            # Resolve strike — if not given, pick the strike *closest to spot*
            # from the scrip master's actual grid (works for both indices and
            # stocks; stocks step by 5/10, NIFTY by 50, BANKNIFTY by 100).
            strike = cfg.get("strike")
            spot = await asyncio.to_thread(_fetch_spot, underlying)
            if not strike:
                available = await asyncio.to_thread(_available_strikes, underlying, expiry)
                if available and spot:
                    strike = min(available, key=lambda s: abs(s - spot))
                else:
                    strike = int(round(spot / 50) * 50) if spot else 0

            symbol_token = await asyncio.to_thread(_resolve_option_token, underlying, int(strike), opt_type, expiry)
            # If the resolver snapped to a nearby strike, use that as the
            # authoritative strike going forward.
            resolved_strike = int(symbol_token.get("strike_used") or strike)
            snapped_from = symbol_token.get("strike_snapped_from")
            state.update({
                "underlying": underlying, "strike": resolved_strike,
                "option_type": opt_type, "expiry": expiry, "spot": spot,
                "symbol": symbol_token.get("symbol", ""),
                "token": symbol_token.get("token", ""),
                "strike_snapped_from": snapped_from,
            })

            if not state["token"]:
                available = await asyncio.to_thread(_available_strikes, underlying, expiry)
                near = sorted(available, key=lambda s: abs(s - int(strike)))[:5] if available else []
                hint = (
                    f"Nearest listed strikes: {near}. Pick one of these."
                    if near
                    else "No listed strikes at all for this expiry — try `next weekly` or check the expiry format."
                )
                state["candles_raw"] = []
                state["fetch_error"] = (
                    f"No option token found for {underlying} {strike} {opt_type} {expiry}. {hint}"
                )
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="fetch_data", type="error",
                    payload={"detail": state["fetch_error"], "strike": strike,
                              "spot": spot, "nearest_strikes": near},
                ))
                return state

            if snapped_from and snapped_from != resolved_strike:
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="fetch_data", type="info",
                    payload={"detail": f"Strike {snapped_from} not listed — snapped to nearest {resolved_strike}.",
                              "strike_snapped_from": snapped_from, "strike": resolved_strike},
                ))

            lookback_days = int(cfg.get("lookback_days", 3))
            candles = await asyncio.to_thread(_fetch_option_candles, state["token"], lookback_days)
            state["candles_raw"] = candles
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="fetch_data", type="state",
                payload={"symbol": state["symbol"], "strike": state["strike"],
                         "expiry": state["expiry"], "candles": len(candles), "spot": spot},
            ))
            return state

        async def simulate(state: dict) -> dict:
            cfg = state["config"]
            candles_raw = state.get("candles_raw") or []
            if not candles_raw:
                detail = state.get("fetch_error") or (
                    f"No 5-min candles returned for {state.get('symbol') or '?'} "
                    f"({state.get('underlying')} {state.get('strike')} {state.get('option_type')} {state.get('expiry')}) — "
                    "market closed or strike too illiquid for intraday data."
                )
                state["plan"] = {
                    "error": detail,
                    "symbol": state.get("symbol"),
                    "underlying": state.get("underlying"),
                    "strike": state.get("strike"),
                    "expiry": state.get("expiry"),
                    "spot": state.get("spot"),
                }
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="simulate", type="error",
                    payload={"error": detail},
                ))
                return state

            result = await asyncio.to_thread(
                _run_pyramid,
                candles_raw, state["symbol"] or f"{state['underlying']}_{state['option_type']}",
                int(cfg.get("capital", 100_000)),
                float(cfg.get("risk_pct", 2.0)),
                int(cfg.get("max_pyramids", 5)),
                int(cfg.get("lot_size", _lot_size(state["underlying"]))),
            )
            state["plan"] = result
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="simulate", type="result",
                payload={
                    "entries": len(result.get("entries", [])),
                    "exit_price": result.get("exit_price"),
                    "exit_reason": result.get("exit_reason"),
                    "total_lots": result.get("total_lots"),
                    "pnl_inr": result.get("total_pnl_rupees"),
                },
            ))
            return state

        async def journal_step(state: dict) -> dict:
            plan = state.get("plan") or {}
            try:
                ctx.journal.record({
                    "portfolio_id": ctx.portfolio_id,
                    "agent_run_id": ctx.run_id,
                    "kind": "plan",
                    "title": f"Pyramid {state.get('option_type')} {state.get('underlying')} {state.get('strike')}",
                    "body": f"{len(plan.get('entries', []))} entries · exit {plan.get('exit_price')} · P&L ₹{plan.get('total_pnl_rupees', 0):,.0f}",
                    "meta": {"plan": plan, "symbol": state.get("symbol")},
                })
            except Exception:  # noqa: BLE001
                pass
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="journal", type="info", payload={"ok": True},
            ))
            return state

        g = StateGraph(dict)
        for n, fn in [
            ("fetch_data", fetch_data),
            ("simulate", simulate),
            ("journal", journal_step),
        ]:
            g.add_node(n, fn)
        g.add_edge("fetch_data", "simulate")
        g.add_edge("simulate", "journal")
        g.add_edge("journal", END)
        g.set_entry_point("fetch_data")
        return g.compile()


# ─────────────────────────────────────────────────────────────────────────
# Helpers (sync)
# ─────────────────────────────────────────────────────────────────────────
def _next(state: dict) -> int:
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


def _lot_size(underlying: str) -> int:
    return {"NIFTY": 65, "BANKNIFTY": 30, "SENSEX": 20}.get(underlying, 1)


def _default_weekly_expiry(underlying: str) -> str:
    """Nearest NFO expiry for the underlying. Works for both indices (OPTIDX)
    and stocks (OPTSTK). Picks the soonest expiry regardless of weekly/monthly.
    """
    from datetime import datetime, date
    from trading.services.ticker_service import ticker_service

    ticker_service._ensure_loaded()
    today = date.today()
    candidates: set[date] = set()
    for inst in ticker_service._nfo_by_key.values():
        if inst.get("name") != underlying:
            continue
        if inst.get("instrumenttype") not in ("OPTIDX", "OPTSTK"):
            continue
        try:
            d = datetime.strptime(inst.get("expiry", ""), "%d%b%Y").date()
        except Exception:  # noqa: BLE001
            continue
        if d >= today:
            candidates.add(d)
    if not candidates:
        return today.strftime("%d%b%y").upper()
    return min(candidates).strftime("%d%b%y").upper()


def _fetch_spot(underlying: str) -> float:
    """Spot LTP for an index or equity stock.

    Falls back to the last daily close when live LTP returns 0 — happens
    on weekends, holidays, or pre-market when the broker quote endpoint
    has no fresh tick. Without the fallback, the strike picker thinks
    spot=0, picks the smallest available strike (deep OTM), and the
    option has 0 volume → empty plan.
    """
    from trading.services.ticker_service import ticker_service

    # 1) Live LTP path
    try:
        if underlying in ("NIFTY", "BANKNIFTY", "SENSEX"):
            from trading.options.data_service import OptionsDataService
            ods = OptionsDataService()
            if underlying == "BANKNIFTY":
                ltp = float(ods.fetch_banknifty_spot().get("ltp", 0))
            else:
                ltp = float(ods.fetch_nifty_spot().get("ltp", 0))
        else:
            from trading.services.data_service import BrokerClient
            broker = BrokerClient.get_instance(); broker.ensure_login()
            token = ticker_service.get_token(underlying)
            if not token:
                return 0.0
            r = broker.ltp(ticker_service.resolve_exchange(underlying), underlying, token)
            ltp = float(r.get("ltp", 0) if isinstance(r, dict) else 0)
        if ltp > 0:
            return ltp
    except Exception:  # noqa: BLE001
        pass

    # 2) Fallback — last daily close from the broker's historical endpoint.
    # Always returns the most recent trading day even when today is a
    # weekend/holiday, so strike selection stays sane.
    try:
        from trading.services.data_service import BrokerClient
        from trading.utils.time_utils import intraday_session_date
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(underlying)
        if not token:
            return 0.0
        end = intraday_session_date().strftime("%Y-%m-%d 15:30")
        start = (intraday_session_date() - timedelta(days=10)).strftime("%Y-%m-%d 09:15")
        exch = ticker_service.resolve_exchange(underlying)
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange=exch) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        if raw:
            return float(raw[-1][4])  # last day's close
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _available_strikes(underlying: str, expiry: str) -> list[int]:
    """Sorted strikes (INR) from the scrip master for this underlying+expiry."""
    from datetime import datetime
    from trading.services.ticker_service import ticker_service

    ticker_service._ensure_loaded()
    # Normalise the caller-supplied underlying (NIFTY50 → NIFTY, …) so
    # the lookup hits Angel One's canonical scrip-master name.
    underlying = ticker_service.normalize_underlying(underlying)
    try:
        target = datetime.strptime(expiry, "%d%b%y").date()
    except Exception:  # noqa: BLE001
        try:
            target = datetime.strptime(expiry, "%d%b%Y").date()
        except Exception:  # noqa: BLE001
            return []
    strikes: set[int] = set()
    for inst in ticker_service._nfo_by_key.values():
        if inst.get("name") != underlying:
            continue
        if inst.get("instrumenttype") not in ("OPTIDX", "OPTSTK"):
            continue
        try:
            d = datetime.strptime(inst.get("expiry", ""), "%d%b%Y").date()
        except Exception:  # noqa: BLE001
            continue
        if d != target:
            continue
        try:
            strikes.add(int(float(inst.get("strike", 0)) / 100))
        except (ValueError, TypeError):
            continue
    return sorted(strikes)


def _resolve_option_token(underlying: str, strike: int, option_type: str, expiry: str) -> dict:
    """Look up an option contract; auto-snap to the nearest valid strike
    when the requested one doesn't exist in the scrip master.

    The snap is the safety net for callers (UI / planner) that pass a
    spot-rounded number that isn't a real listed strike. Without this
    we returned an empty result and the operator saw "No option token
    found for NIFTY 23668 CE …" — true but unhelpful when 23650 and
    23700 are both available a few rupees away.
    """
    from trading.services.ticker_service import ticker_service
    try:
        underlying = ticker_service.normalize_underlying(underlying)
        opts = ticker_service.get_nfo_options(underlying, strike, expiry)
        if option_type in opts:
            sym, tok = opts[option_type]
            return {"symbol": sym, "token": tok, "strike_used": strike}

        # Auto-snap — find the closest LISTED strike and retry.
        listed = _available_strikes(underlying, expiry)
        if listed:
            nearest = min(listed, key=lambda s: abs(s - strike))
            if nearest != strike:
                opts = ticker_service.get_nfo_options(underlying, nearest, expiry)
                if option_type in opts:
                    sym, tok = opts[option_type]
                    return {
                        "symbol": sym, "token": tok,
                        "strike_used": nearest,
                        "strike_snapped_from": strike,
                    }
    except Exception:  # noqa: BLE001
        pass
    return {"symbol": "", "token": ""}


def _fetch_option_candles(token: str, lookback_days: int) -> list:
    """Pull 5-min candles for an option token. If the requested window
    returns empty (weekend, holiday, fresh strike), auto-widen up to 10
    days back before giving up — saves the user from a silently-empty plan
    just because today happens to be Sunday.
    """
    if not token:
        return []
    try:
        from trading.services.data_service import BrokerClient
        broker = BrokerClient.get_instance(); broker.ensure_login()
        today = date.today()
        end = today.strftime("%Y-%m-%d 15:30")
        # Try the requested window first, then progressively widen.
        for days in sorted({lookback_days, 3, 5, 7, 10}):
            if days < lookback_days:
                continue
            start = (today - timedelta(days=days)).strftime("%Y-%m-%d 09:15")
            candles = broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange="NFO") or []
            if candles:
                return candles
        return []
    except Exception:  # noqa: BLE001
        return []


def _run_pyramid(candles_raw: list, symbol: str, capital: int, risk_pct: float, max_pyramids: int, lot_size: int) -> dict:
    """Adapt legacy `run_pyramid` output to a JSON-safe dict."""
    from trading.pyramid.strategy import Candle, PyramidConfig, run_pyramid

    candles = [Candle.from_raw(r) for r in candles_raw]
    cfg = PyramidConfig(
        lot_size=lot_size,
        initial_capital=float(capital),
        initial_risk_pct=risk_pct,
        max_pyramids=max_pyramids,
    )
    result = run_pyramid(candles, symbol=symbol, config=cfg)
    return {
        "symbol": result.symbol,
        "entries": [
            {
                "bar_index": e.bar_index, "timestamp": e.timestamp,
                "price": e.price, "lots": e.lots, "sl_at_entry": e.sl_at_entry,
                "reason": e.reason,
            }
            for e in result.entries
        ],
        "exit_price": result.exit_price,
        "exit_time": result.exit_time,
        "exit_reason": result.exit_reason,
        "total_lots": result.total_lots,
        "total_cost": result.total_cost,
        "realized_pnl": result.realized_pnl,
        "peak_unrealized": result.peak_unrealized,
        "peak_lots": result.peak_lots,
        "lot_size": result.lot_size,
        "avg_entry": result.avg_entry,
        "pnl_per_lot": result.pnl_per_lot,
        "total_pnl_points": result.total_pnl_points,
        "total_pnl_rupees": result.total_pnl_rupees,
        "log_tail": result.log[-15:] if result.log else [],
    }


assert isinstance(PyramidStrategy(), Strategy)
