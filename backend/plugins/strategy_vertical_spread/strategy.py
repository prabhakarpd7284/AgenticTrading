"""Vertical spread strategy plugin — planner-only (no lifecycle yet).

A vertical spread is a defined-risk options structure:
  * Bull call spread: BUY ATM CE + SELL OTM higher CE      (debit, bullish)
  * Bear put spread:  BUY ATM PE + SELL OTM lower PE       (debit, bearish)

This plugin picks an ATM strike, picks the second leg `width` points away,
fetches real CE/PE LTPs from Angel One via the legacy OptionsDataService,
and returns a structured plan: legs, net debit, max profit/loss, breakeven,
size given a capital budget.

No LLM call by default — this is a deterministic structurer. The orchestrator
can attach LLM reasoning around it.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Literal

from apps.agents_core.domain.contracts import (
    AgentContext,
    AgentEvent,
    Strategy,
    StrategySchema,
)


class VerticalSpreadStrategy:
    name = "vertical_spread"
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
                    "underlying": {"type": "string", "default": "NIFTY"},
                    "side":      {"type": "string", "enum": ["BULL", "BEAR"], "default": "BULL"},
                    "expiry":    {"type": "string", "description": "DDMMMYY (e.g. 29MAY26). Empty = nearest monthly."},
                    "width":     {"type": "number", "minimum": 25, "default": 100,
                                   "description": "Strike distance between legs (points)."},
                    "capital":   {"type": "number", "minimum": 10000, "default": 200000},
                    "max_lots":  {"type": "integer", "minimum": 1, "default": 10},
                    "strike_step": {"type": "number", "default": 50, "description": "ATM rounding granularity"},
                },
                "required": ["underlying", "side"],
            },
        )

    def build_graph(self, ctx: AgentContext):
        from langgraph.graph import StateGraph, END

        async def fetch_data(state: dict) -> dict:
            cfg = state["config"]
            underlying = (cfg.get("underlying") or "NIFTY").upper()
            seq = _next(state)
            ctx.publisher.emit(AgentEvent(
                seq=seq, node="fetch_data", type="info",
                payload={"underlying": underlying},
            ))
            spot = await asyncio.to_thread(_fetch_spot, underlying)
            state["underlying"] = underlying
            state["spot"] = spot
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="fetch_data", type="state",
                payload={"spot": spot},
            ))
            return state

        async def pick_legs(state: dict) -> dict:
            cfg = state["config"]
            spot = state.get("spot") or 0.0
            underlying = state["underlying"]
            side: Literal["BULL", "BEAR"] = cfg.get("side", "BULL").upper()
            expiry = cfg.get("expiry") or _default_monthly_expiry(underlying)

            # Use the actual scrip-master strike grid (stock options can step
            # by ₹5 or ₹10; indices step by 50 / 100). This makes us pick
            # strikes that genuinely exist instead of guessing via rounding.
            available = await asyncio.to_thread(_available_strikes, underlying, expiry)
            if not available:
                state["legs_meta_error"] = f"No option chain found for {underlying} {expiry}"
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="pick_legs", type="error",
                    payload={"detail": state["legs_meta_error"]},
                ))
                return state

            atm = min(available, key=lambda s: abs(s - spot))
            # `width` is "n strikes away" so we always land on a real strike.
            n_away = int(cfg.get("width_strikes") or _default_n_away(available, cfg.get("width", 0)))
            atm_idx = available.index(atm)
            if side == "BULL":
                short_idx = min(len(available) - 1, atm_idx + n_away)
                long_strike, short_strike, opt_type = atm, available[short_idx], "CE"
            else:
                short_idx = max(0, atm_idx - n_away)
                long_strike, short_strike, opt_type = atm, available[short_idx], "PE"

            state["legs_meta"] = {
                "side": side, "option_type": opt_type, "expiry": expiry,
                "long_strike": int(long_strike), "short_strike": int(short_strike),
                "atm": int(atm), "width": float(abs(long_strike - short_strike)),
                "n_away": n_away,
            }
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="pick_legs", type="result",
                payload=state["legs_meta"],
            ))
            return state

        async def fetch_legs(state: dict) -> dict:
            if state.get("legs_meta_error"):
                state["long_ltp"] = 0.0; state["short_ltp"] = 0.0
                return state
            meta = state["legs_meta"]
            underlying = state["underlying"]
            long_ltp, short_ltp = await asyncio.to_thread(
                _fetch_leg_prices,
                underlying, meta["long_strike"], meta["short_strike"],
                meta["option_type"], meta["expiry"],
            )
            state["long_ltp"] = long_ltp
            state["short_ltp"] = short_ltp
            if not long_ltp or not short_ltp:
                state["leg_ltp_warning"] = (
                    f"LTP came back zero — strike likely illiquid or market closed "
                    f"(long={long_ltp}, short={short_ltp})."
                )
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="fetch_legs", type="state",
                payload={"long_ltp": long_ltp, "short_ltp": short_ltp, "warning": state.get("leg_ltp_warning")},
            ))
            return state

        async def size_and_compute(state: dict) -> dict:
            cfg = state["config"]
            if state.get("legs_meta_error"):
                state["plan"] = {"error": state["legs_meta_error"], "underlying": state["underlying"], "spot": state.get("spot")}
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="size_and_compute", type="error",
                    payload=state["plan"],
                ))
                return state

            meta = state["legs_meta"]
            long_ltp = float(state.get("long_ltp") or 0.0)
            short_ltp = float(state.get("short_ltp") or 0.0)
            if not long_ltp or not short_ltp:
                state["plan"] = {
                    "error": state.get("leg_ltp_warning") or "Could not fetch leg LTPs.",
                    "underlying": state["underlying"], "side": meta["side"], "option_type": meta["option_type"],
                    "expiry": meta["expiry"], "long_strike": meta["long_strike"], "short_strike": meta["short_strike"],
                    "spot": state.get("spot"),
                }
                ctx.publisher.emit(AgentEvent(
                    seq=_next(state), node="size_and_compute", type="error",
                    payload=state["plan"],
                ))
                return state

            net_debit = max(0.0, long_ltp - short_ltp)
            net_credit = max(0.0, short_ltp - long_ltp)

            lot_size = await asyncio.to_thread(_get_lot_size, state["underlying"])
            max_lots = int(cfg.get("max_lots", 10))
            capital = float(cfg.get("capital", 200_000))

            # For a debit spread: cost per lot = net_debit * lot_size
            cost_per_lot = (net_debit if net_debit > 0 else 0.0) * lot_size
            lots = 1
            if cost_per_lot > 0:
                lots = max(1, min(max_lots, int(capital // cost_per_lot)))

            width_pts = meta["width"]
            max_profit_per_lot = (width_pts - net_debit) * lot_size if net_debit > 0 else width_pts * lot_size
            max_loss_per_lot = net_debit * lot_size if net_debit > 0 else (width_pts - net_credit) * lot_size
            breakeven = (
                meta["long_strike"] + net_debit if meta["option_type"] == "CE"
                else meta["long_strike"] - net_debit
            )

            plan = {
                "underlying": state["underlying"],
                "side": meta["side"],
                "option_type": meta["option_type"],
                "expiry": meta["expiry"],
                "long_strike": meta["long_strike"],
                "short_strike": meta["short_strike"],
                "width": width_pts,
                "long_ltp": long_ltp,
                "short_ltp": short_ltp,
                "net_debit": round(net_debit, 2),
                "net_credit": round(net_credit, 2),
                "lots": lots,
                "lot_size": lot_size,
                "qty_per_leg": lots * lot_size,
                "max_profit_inr": round(max_profit_per_lot * lots, 2),
                "max_loss_inr": round(max_loss_per_lot * lots, 2),
                "breakeven": round(breakeven, 2),
                "capital_used": round(cost_per_lot * lots, 2),
                "rr_ratio": round(max_profit_per_lot / max_loss_per_lot, 2) if max_loss_per_lot else None,
                "spot": state.get("spot"),
            }
            state["plan"] = plan
            ctx.publisher.emit(AgentEvent(
                seq=_next(state), node="size_and_compute", type="result",
                payload={
                    "lots": plan["lots"],
                    "net_debit": plan["net_debit"],
                    "max_profit_inr": plan["max_profit_inr"],
                    "max_loss_inr": plan["max_loss_inr"],
                    "breakeven": plan["breakeven"],
                    "rr_ratio": plan["rr_ratio"],
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
                    "title": f"{plan.get('side')} {plan.get('option_type')} spread {plan.get('long_strike')}/{plan.get('short_strike')} on {plan.get('underlying')}",
                    "body": f"Net debit {plan.get('net_debit')} · max profit {plan.get('max_profit_inr')} · max loss {plan.get('max_loss_inr')} · BE {plan.get('breakeven')}",
                    "meta": {"plan": plan},
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
            ("pick_legs", pick_legs),
            ("fetch_legs", fetch_legs),
            ("size_and_compute", size_and_compute),
            ("journal", journal_step),
        ]:
            g.add_node(n, fn)
        g.add_edge("fetch_data", "pick_legs")
        g.add_edge("pick_legs", "fetch_legs")
        g.add_edge("fetch_legs", "size_and_compute")
        g.add_edge("size_and_compute", "journal")
        g.add_edge("journal", END)
        g.set_entry_point("fetch_data")
        return g.compile()


# ─────────────────────────────────────────────────────────────────────────
# Helpers (sync, called via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────
def _next(state: dict) -> int:
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


def _default_monthly_expiry(underlying: str = "NIFTY") -> str:
    """Pick the nearest monthly expiry for the given underlying from the
    loaded NFO scrip master. Works for both index (OPTIDX) and stock (OPTSTK)
    options.
    """
    from datetime import date, datetime
    from trading.services.ticker_service import ticker_service

    ticker_service._ensure_loaded()
    today = date.today()
    candidates: set[date] = set()
    for inst in ticker_service._nfo_by_key.values():
        if inst.get("name") != underlying or inst.get("instrumenttype") not in ("OPTIDX", "OPTSTK"):
            continue
        try:
            d = datetime.strptime(inst.get("expiry", ""), "%d%b%Y").date()
        except Exception:  # noqa: BLE001
            continue
        if d < today:
            continue
        candidates.add(d)
    if not candidates:
        return today.strftime("%d%b%y").upper()
    # Monthly = latest expiry within each month (NSE convention).
    by_month: dict[tuple[int, int], date] = {}
    for d in candidates:
        k = (d.year, d.month)
        if k not in by_month or d > by_month[k]:
            by_month[k] = d
    return min(by_month.values()).strftime("%d%b%y").upper()


def _available_strikes(underlying: str, expiry: str) -> list[int]:
    """Sorted list of strikes (in INR) that exist in the scrip master for
    this underlying + expiry. Works for both index and stock options.
    """
    from datetime import datetime
    from trading.services.ticker_service import ticker_service

    ticker_service._ensure_loaded()
    try:
        target = datetime.strptime(expiry, "%d%b%y").date()
    except Exception:  # noqa: BLE001
        try:
            target = datetime.strptime(expiry, "%d%b%Y").date()
        except Exception:  # noqa: BLE001
            return []

    strikes: set[int] = set()
    for inst in ticker_service._nfo_by_key.values():
        if inst.get("name") != underlying or inst.get("instrumenttype") not in ("OPTIDX", "OPTSTK"):
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


def _default_n_away(strikes: list[int], width_pts: float = 0) -> int:
    """How many strikes to walk for the short leg.

    If the caller passed an explicit width in points (e.g. 100 for NIFTY),
    convert it to # of strikes based on the grid spacing. Otherwise default
    to a sensible structure: 3 strikes away on a tight grid, 1 on a coarse
    grid.
    """
    if len(strikes) < 2:
        return 1
    grid = strikes[1] - strikes[0]  # rough step
    if width_pts and grid > 0:
        return max(1, int(round(width_pts / grid)))
    if grid <= 10:
        return 3          # stock options (5-10 step) → 3 strikes ~ 15-30 pts wide
    if grid <= 50:
        return 2          # NIFTY weekly → 2 strikes = 100 pts
    return 1              # BANKNIFTY (100 step) → 1 strike = 100 pts


def _fetch_spot(underlying: str) -> float:
    """Return current LTP for the underlying."""
    try:
        if underlying in ("NIFTY", "BANKNIFTY", "SENSEX"):
            from trading.options.data_service import (
                OptionsDataService, NIFTY_SPOT_TOKEN, BANKNIFTY_SPOT_TOKEN,
            )
            ods = OptionsDataService()
            if underlying == "BANKNIFTY":
                return float(ods.fetch_banknifty_spot().get("ltp", 0))
            return float(ods.fetch_nifty_spot().get("ltp", 0))

        # Equity stock — use the equity broker LTP via the symbol master
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(underlying)
        if not token:
            return 0.0
        r = broker.ltp("NSE", underlying, token)
        return float(r.get("ltp", 0) if isinstance(r, dict) else 0)
    except Exception:  # noqa: BLE001
        return 0.0


def _fetch_leg_prices(underlying: str, long_strike: int, short_strike: int, option_type: str, expiry: str) -> tuple[float, float]:
    from trading.services.ticker_service import ticker_service
    from trading.options.data_service import OptionsDataService

    ods = OptionsDataService()
    long_opts = ticker_service.get_nfo_options(underlying, long_strike, expiry)
    short_opts = ticker_service.get_nfo_options(underlying, short_strike, expiry)
    if option_type not in long_opts or option_type not in short_opts:
        return 0.0, 0.0
    long_sym, long_tok = long_opts[option_type]
    short_sym, short_tok = short_opts[option_type]
    long_ltp = float(ods.fetch_option_ltp(long_sym, long_tok).get("ltp", 0))
    short_ltp = float(ods.fetch_option_ltp(short_sym, short_tok).get("ltp", 0))
    return long_ltp, short_ltp


def _get_lot_size(underlying: str) -> int:
    """Index lot sizes (Jan 2026+). For equity stocks, look up the scrip master."""
    if underlying == "NIFTY":     return 65
    if underlying == "BANKNIFTY": return 30
    if underlying == "SENSEX":    return 20
    try:
        from trading.services.ticker_service import ticker_service
        ticker_service._ensure_loaded()
        for inst in ticker_service._nfo_by_key.values():
            if inst.get("name") == underlying and inst.get("instrumenttype") == "OPTSTK":
                ls = inst.get("lotsize")
                if ls:
                    return int(ls)
    except Exception:  # noqa: BLE001
        pass
    return 1


assert isinstance(VerticalSpreadStrategy(), Strategy)
