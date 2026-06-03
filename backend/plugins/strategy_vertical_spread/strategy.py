"""Directional vertical spread strategy — LangGraph plugin.

Nodes (one timeline event per node, canonical names so the Agent
Console's Plan / Risk / Execute tabs populate without aliases):

  fetch_market   →  classify_bias  →  planner  →  risk
                 →  execute        →  journal

This is the production wrapper around the PoC's bias + selector + lifecycle
logic. Market data comes through the `MarketDataPort` so the same graph
works against Angel, Fyers, Zerodha, or paper-broker fakes interchangeably.

Configurable params (StrategySchema):
  - underlying        NIFTY | BANKNIFTY | SENSEX
  - mode              AUTO | BULL_PUT | BEAR_CALL | IRON_CONDOR
                      AUTO = derive from bias; explicit modes skip the classifier
  - lots              integer ≥ 1
  - sell_pct          default 0.80
  - buy_pct           default 0.60
  - max_width         default 200 points
  - profit_take_pct   default 0.70
  - risk_cap_pct      default 0.08   (options profile)

The strategy emits one AgentEvent per node. Failed risk gates still
record a journal entry — every decision (taken or rejected) is logged.
"""
from __future__ import annotations

from typing import Any

from apps.agents_core.domain.contracts import (
    AgentContext,
    AgentEvent,
    Strategy,
    StrategySchema,
)

from .bias import BIAS_TO_MODE, Bias, classify
from .lifecycle import PROFIT_TAKE_PCT
from .selector import (
    BUY_PCT, MAX_WIDTH_POINTS, SELL_PCT,
    Leg, SpreadPick, atm_strike,
    select_bear_call, select_bull_put, select_iron_condor,
)


__all__ = ["VerticalSpreadStrategy"]


# Lot sizes per underlying — single source for derived metrics inside the
# graph. Keeps the strategy honest about absolute ₹-values without making
# every node re-derive them.
_LOT_SIZES: dict[str, int] = {
    "NIFTY":      75,
    "BANKNIFTY":  30,
    "SENSEX":     20,
    "FINNIFTY":   65,
    "MIDCPNIFTY": 120,
}


def _next_seq(state: dict, publisher=None) -> int:
    """Return the next event seq.

    Prefers the run-level publisher's counter (started at 0 by the run
    entry point, then incremented per emit including the `init` event)
    so plugin events never collide with the run's bookend events on the
    `(run, seq)` unique constraint. Falls back to a state-local counter
    when called without a publisher (tests, etc.).

    Without this alignment, the init event takes seq=1, then the first
    plugin node also picks seq=1, IntegrityError fires, and on some DB
    backends the transaction aborts so NO subsequent step persists either.
    """
    if publisher is not None and hasattr(publisher, "next_seq"):
        return publisher.next_seq()
    state["_seq"] = state.get("_seq", 0) + 1
    return state["_seq"]


class VerticalSpreadStrategy:
    name = "vertical_spread"
    version = "1.0.0"
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
                    "underlying": {
                        "type": "string",
                        "enum": ["NIFTY", "BANKNIFTY", "SENSEX"],
                        "default": "NIFTY",
                    },
                    "expiry": {
                        "type": ["string", "null"],
                        "description": (
                            "Canonical DDMMMYYYY expiry (e.g. \"28MAY2026\"). "
                            "Null/missing picks the nearest weekly via the broker."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["AUTO", "BULL_PUT", "BEAR_CALL", "IRON_CONDOR"],
                        "default": "AUTO",
                        "description": "AUTO derives from bias classifier; explicit modes skip it.",
                    },
                    "lots": {"type": "integer", "minimum": 1, "default": 1},
                    "sell_pct": {"type": "number", "default": SELL_PCT,
                                  "minimum": 0.3, "maximum": 0.95},
                    "buy_pct": {"type": "number", "default": BUY_PCT,
                                 "minimum": 0.1, "maximum": 0.9},
                    "max_width": {"type": "integer", "default": MAX_WIDTH_POINTS,
                                   "minimum": 50, "maximum": 1000},
                    "profit_take_pct": {"type": "number", "default": PROFIT_TAKE_PCT,
                                         "minimum": 0.3, "maximum": 0.95},
                    "risk_cap_pct": {"type": "number", "default": 0.08,
                                      "minimum": 0.005, "maximum": 0.15,
                                      "description": "Max-loss cap as fraction of capital."},
                },
                "required": ["underlying"],
            },
        )

    # ────────────────────────────────────────────────────────────────
    # LangGraph build — six nodes
    # ────────────────────────────────────────────────────────────────
    def build_graph(self, ctx: AgentContext):
        from langgraph.graph import END, StateGraph
        import logging
        import traceback as _tb
        _node_logger = logging.getLogger(__name__)

        cfg = ctx.config or {}
        underlying = cfg.get("underlying", "NIFTY")
        expiry = cfg.get("expiry") or None      # None → nearest weekly
        mode = cfg.get("mode", "AUTO")
        sell_pct = float(cfg.get("sell_pct", SELL_PCT))
        buy_pct = float(cfg.get("buy_pct", BUY_PCT))
        max_width = int(cfg.get("max_width", MAX_WIDTH_POINTS))
        lots = int(cfg.get("lots", 1))
        risk_cap_pct = float(cfg.get("risk_cap_pct", 0.08))

        # Capital base used by the risk node to compute the per-trade ₹-cap.
        # Reads `capital` from the portfolio (or defaults to ₹5L) — done
        # once per graph build, not per-node, since portfolios don't change
        # mid-run. Wrapped so a missing FK column or non-existent portfolio
        # doesn't poison the graph.
        def capital_for_check() -> float:
            try:
                from apps.trading.models import Portfolio
                p = Portfolio.objects.filter(id=ctx.portfolio_id).first()
                if p and getattr(p, "capital", None) is not None:
                    return float(p.capital)
            except Exception:  # noqa: BLE001
                pass
            return 500_000.0

        # Closure-scoped cache shared between fetch_market and planner.
        # The options-chain snapshot is the single most expensive call in
        # the graph (broker network round-trips + scrip-master lookup);
        # the snapshot fetched in fetch_market is reused by planner so we
        # only pay for it once per run. Stored OUTSIDE state because
        # OptionsChainSnapshot is a dataclass — not JSON-serializable —
        # and `run.result = state` would choke when persisting.
        _chain_cache: dict = {}

        def _safe(node_name: str, fn):
            """Wrap a graph node so any uncaught exception becomes:
              - a per-node error event (visible on the agent console),
              - a structured entry in state["node_errors"], AND
              - a log.exception with the traceback (visible in celery.log).
            The graph keeps moving so downstream nodes still record state.
            Without this, a single typo in any node turns the entire run
            into a generic 'Failed' badge with an opaque message."""
            async def wrapper(state: dict) -> dict:
                try:
                    return await fn(state)
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    tb = _tb.format_exc().splitlines()[-6:]
                    _node_logger.exception(
                        "vertical_spread.node_failed node=%s err=%s",
                        node_name, err,
                    )
                    state.setdefault("node_errors", {})[node_name] = err
                    state.setdefault("node_tracebacks", {})[node_name] = tb
                    try:
                        ctx.publisher.emit(AgentEvent(
                            seq=_next_seq(state, ctx.publisher),
                            node=node_name,
                            type="error",
                            payload={"error": err, "traceback": tb},
                        ))
                    except Exception:  # noqa: BLE001
                        pass
                    return state
            return wrapper

        # ── Node 1: fetch market data ──
        # Spot + VIX come from the options-chain snapshot directly. We
        # CANNOT use DefaultMarketData.ltp("NIFTY") for indices — the
        # legacy ticker_service only knows equity symbols (RELIANCE-EQ,
        # MFSL-EQ, …) and returns no token for the index name. The chain
        # endpoint is the canonical source for both spot (the underlying
        # LTP the adapter just looked up) and VIX (cross-fetched alongside
        # by AngelOneAdapter via getMarketData on the VIX token).
        #
        # The snapshot is cached in the closure for planner to reuse, so
        # we only pay for one broker round-trip per run instead of two.
        async def fetch_market(state: dict) -> dict:
            mkt = ctx.market_data
            warnings: list[str] = []

            snapshot = None
            try:
                snapshot = mkt.options_chain(underlying, expiry=expiry)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"options_chain: {type(e).__name__}: {e}")

            spot = getattr(snapshot, "spot", None) if snapshot else None
            vix_today = getattr(snapshot, "vix", None) if snapshot else None

            if snapshot is None:
                warnings.append("broker returned no chain")
            elif not spot:
                warnings.append(f"chain returned but spot=0 (source={snapshot.source})")
            else:
                # Cache the snapshot — planner will reuse it instead of
                # re-calling options_chain. Major perf win for live brokers.
                _chain_cache["snapshot"] = snapshot

            # Momentum candles — optional, classifier degrades gracefully
            # without them. Use "1d" (the _BROKER_INTERVAL key) not "ONE_DAY".
            # For indices `mkt.candles` is also broken (same equity-ticker
            # issue as `ltp`); we tolerate the failure here and the
            # classifier will fall back to bias=RANGE.
            try:
                rows = mkt.candles(underlying, "1d", 6) or []
                if len(rows) >= 4:
                    state["today_open"] = rows[-1].get("o") or rows[-1].get("open")
                    state["prev_close"] = rows[-2].get("c") or rows[-2].get("close")
                    state["close_3d_ago"] = rows[-4].get("c") or rows[-4].get("close")
            except Exception as e:  # noqa: BLE001
                warnings.append(f"candles: {type(e).__name__}: {e}")

            state["spot"] = spot
            state["vix_today"] = vix_today
            state["vix_yesterday"] = state.get("vix_yesterday") or vix_today
            if snapshot:
                state["chain_source"] = snapshot.source
                state["chain_expiry"] = snapshot.expiry
                state["chain_atm"] = snapshot.atm_strike
                state["chain_pcr_oi"] = snapshot.pcr_oi
            if warnings:
                state.setdefault("_warnings", []).extend(warnings)

            ctx.publisher.emit(AgentEvent(
                seq=_next_seq(state, ctx.publisher),
                node="fetch_market", type="state",
                payload={
                    "status": "ok" if spot else "degraded",
                    "underlying": underlying,
                    "spot": spot,
                    "vix": vix_today,
                    "today_open": state.get("today_open"),
                    "prev_close": state.get("prev_close"),
                    "close_3d_ago": state.get("close_3d_ago"),
                    "chain_source": state.get("chain_source"),
                    "chain_expiry": state.get("chain_expiry"),
                    "chain_atm": state.get("chain_atm"),
                    "chain_pcr_oi": state.get("chain_pcr_oi"),
                    "warnings": warnings,
                },
            ))
            return state

        # ── Node 2: classify bias ──
        # Three input sources, in priority order:
        #   1. Explicit `mode` from config (operator picked Bull Put / etc.)
        #   2. Classifier on spot + today_open + 3d momentum + VIX direction
        #   3. Fallback to RANGE when market data is insufficient
        async def classify_bias_node(state: dict) -> dict:
            source = "fallback"
            if mode != "AUTO":
                source = "explicit_mode"
                bias = {
                    "BULL_PUT": Bias.UP.value,
                    "BEAR_CALL": Bias.DOWN.value,
                    "IRON_CONDOR": Bias.RANGE.value,
                }.get(mode, Bias.RANGE.value)
                bias_reason = f"operator picked mode={mode}"
            elif state.get("spot") and state.get("today_open"):
                source = "classifier"
                reading = classify(
                    today_open=state["today_open"],
                    prev_close=state.get("prev_close") or state["spot"],
                    close_3d_ago=state.get("close_3d_ago") or state["spot"],
                    spot_now=state["spot"],
                    vix_today=state.get("vix_today") or 0.0,
                    vix_yesterday=state.get("vix_yesterday") or 0.0,
                )
                bias = reading.bias.value
                bias_reason = reading.reason
            else:
                bias = Bias.RANGE.value
                bias_reason = "no spot or open price — defaulting to RANGE"

            # Pre-resolve the structural mode so downstream nodes don't
            # have to re-derive it. Plain enum lookup, no I/O.
            resolved_mode = BIAS_TO_MODE.get(Bias(bias), "IRON_CONDOR")

            state["bias"] = bias
            state["bias_reason"] = bias_reason
            state["bias_source"] = source
            state["resolved_mode"] = resolved_mode

            ctx.publisher.emit(AgentEvent(
                seq=_next_seq(state, ctx.publisher),
                node="classify_bias", type="result",
                payload={
                    "status": "ok",
                    "bias": bias,
                    "reason": bias_reason,
                    "source": source,
                    "structure": resolved_mode,
                },
            ))
            return state

        # ── Node 3: planner ── (was `select_strikes`)
        # Renamed to "planner" so the Agent Console's Plan tab populates
        # directly — no alias-emit needed. The Stream feed still shows
        # the full payload below, including credit/max-loss/breakeven.
        # Single emit per call (success or failure paths).
        async def planner(state: dict) -> dict:
            def _emit_fail(reason: str) -> dict:
                state["pick"] = None
                state["pick_error"] = reason
                ctx.publisher.emit(AgentEvent(
                    seq=_next_seq(state, ctx.publisher),
                    node="planner", type="error",
                    payload={"status": "skipped", "reason": reason},
                ))
                return state

            spot = state.get("spot")
            if not spot:
                return _emit_fail("no spot — fetch_market degraded")

            atm = atm_strike(spot)
            bias = state.get("bias") or Bias.RANGE.value
            resolved = state.get("resolved_mode") or BIAS_TO_MODE.get(
                Bias(bias), "IRON_CONDOR",
            )
            state["resolved_mode"] = resolved
            state["atm"] = atm

            # Prefer the snapshot fetch_market already loaded — saves a
            # broker round-trip. Only re-fetch if fetch_market couldn't get
            # one (degraded run) AND we have a spot from somewhere.
            snapshot = _chain_cache.get("snapshot")
            if snapshot is None:
                try:
                    snapshot = ctx.market_data.options_chain(underlying, expiry=expiry)
                except Exception as e:  # noqa: BLE001
                    return _emit_fail(f"chain_fetch_failed: {type(e).__name__}: {e}")
            if snapshot is None:
                return _emit_fail("broker returned empty chain")

            chain = _legs_from_snapshot(snapshot)
            if getattr(snapshot, "atm_strike", None):
                atm = snapshot.atm_strike
                state["atm"] = atm
            state["chain_source"] = snapshot.source
            state["chain_expiry"] = snapshot.expiry
            state["chain_vix"] = snapshot.vix
            state["chain_pcr_oi"] = snapshot.pcr_oi

            atm_pe = next((L for L in chain if L.strike == atm and L.opt == "PE"), None)
            atm_ce = next((L for L in chain if L.strike == atm and L.opt == "CE"), None)
            if not atm_pe or not atm_ce:
                return _emit_fail(f"chain missing ATM legs at {atm}")

            if resolved == "BULL_PUT":
                pick = select_bull_put(chain, atm, atm_pe.ltp,
                                        sell_pct=sell_pct, buy_pct=buy_pct, max_width=max_width)
            elif resolved == "BEAR_CALL":
                pick = select_bear_call(chain, atm, atm_ce.ltp,
                                         sell_pct=sell_pct, buy_pct=buy_pct, max_width=max_width)
            else:
                pick = select_iron_condor(chain, atm, atm_pe.ltp, atm_ce.ltp,
                                           sell_pct=sell_pct, buy_pct=buy_pct, max_width=max_width)
            if not pick:
                return _emit_fail(f"no spread for {resolved} (80/60 rule found no match)")

            # Compute derived metrics ONCE here. Risk + execute + journal
            # all read these from state — single source of truth, no drift.
            lot_size = _LOT_SIZES.get(underlying, 75)
            max_profit = pick.credit * lot_size * lots
            max_loss = max(0.0, pick.width - pick.credit) * lot_size * lots
            breakeven = (pick.sell.strike - pick.credit) if resolved == "BULL_PUT" \
                else (pick.sell.strike + pick.credit) if resolved == "BEAR_CALL" \
                else None
            state["pick"] = {
                "mode": pick.mode,
                "underlying": underlying,
                "expiry": state.get("chain_expiry"),
                "atm": atm,
                "sell_strike": pick.sell.strike,
                "sell_price": pick.sell.sell_price,
                "buy_strike": pick.buy.strike,
                "buy_price": pick.buy.buy_price,
                "sell_2_strike": pick.sell_2.strike if pick.sell_2 else None,
                "buy_2_strike": pick.buy_2.strike if pick.buy_2 else None,
                "credit": round(pick.credit, 2),
                "width": pick.width,
                "lots": lots,
                "lot_size": lot_size,
                "max_profit": round(max_profit, 2),
                "max_loss": round(max_loss, 2),
                "breakeven": breakeven,
                "chain_source": state.get("chain_source"),
            }
            ctx.publisher.emit(AgentEvent(
                seq=_next_seq(state, ctx.publisher),
                node="planner", type="result",
                payload={"status": "planned", **state["pick"]},
            ))
            return state

        # ── Node 4: risk gate ──
        async def risk_gate(state: dict) -> dict:
            pick = state.get("pick")
            if not pick:
                # Planner already emitted the reason; risk's job is just to
                # propagate the skip cleanly.
                state["risk"] = {
                    "approved": False,
                    "status": "skipped",
                    "reason": state.get("pick_error") or "no plan to risk-check",
                }
                ctx.publisher.emit(AgentEvent(
                    seq=_next_seq(state, ctx.publisher),
                    node="risk", type="error",
                    payload=state["risk"],
                ))
                return state

            risk_cap_inr = capital_for_check() * risk_cap_pct
            draft = {
                "portfolio_id": ctx.portfolio_id,
                "symbol": f"{underlying}_{pick['mode']}",
                "side": "SELL",
                "qty": lots,
                "instrument": "option_spread",
                "credit": pick["credit"],
                "width": pick["width"],
                "lots": lots,
                "risk_cap_pct": risk_cap_pct,
                "max_loss_inr": pick["max_loss"],
            }
            try:
                decision = ctx.risk.validate(draft)
                d = decision.model_dump()
            except Exception as e:  # noqa: BLE001
                d = {"approved": False, "reason": f"risk_engine_error: {type(e).__name__}: {e}"}

            # Add our own option-spread cap as a fail-safe — RiskEngine
            # may not yet understand option_spread instruments.
            exceeds_cap = pick["max_loss"] > risk_cap_inr
            if exceeds_cap and d.get("approved", False):
                d["approved"] = False
                d["reason"] = (f"max_loss ₹{pick['max_loss']:,.0f} > cap "
                                f"₹{risk_cap_inr:,.0f} ({risk_cap_pct*100:.0f}% of capital)")

            state["risk"] = {
                **d,
                "status": "approved" if d.get("approved") else "rejected",
                "max_loss": pick["max_loss"],
                "risk_cap_inr": round(risk_cap_inr, 2),
                "risk_cap_pct": risk_cap_pct,
                "exceeds_cap": exceeds_cap,
            }
            ctx.publisher.emit(AgentEvent(
                seq=_next_seq(state, ctx.publisher),
                node="risk",
                type="result" if state["risk"]["approved"] else "error",
                payload=state["risk"],
            ))
            return state

        # ── Node 5: execute (paper) ──
        async def execute(state: dict) -> dict:
            risk = state.get("risk") or {}
            if not risk.get("approved"):
                state["executed"] = False
                state["executed_reason"] = risk.get("reason") or "risk not approved"
                ctx.publisher.emit(AgentEvent(
                    seq=_next_seq(state, ctx.publisher),
                    node="execute", type="error",
                    payload={
                        "status": "skipped",
                        "reason": state["executed_reason"],
                        "executed": False,
                    },
                ))
                return state

            pick = state["pick"]
            primary_side = "PE" if pick["mode"] in ("BULL_PUT", "IRON_CONDOR") else "CE"
            orders = [
                {"side": "SELL", "opt": primary_side,
                 "strike": pick["sell_strike"], "lots": lots,
                 "price": pick["sell_price"]},
                {"side": "BUY", "opt": primary_side,
                 "strike": pick["buy_strike"], "lots": lots,
                 "price": pick["buy_price"]},
            ]
            if pick["mode"] == "IRON_CONDOR":
                orders.extend([
                    {"side": "SELL", "opt": "CE",
                     "strike": pick["sell_2_strike"], "lots": lots, "price": None},
                    {"side": "BUY", "opt": "CE",
                     "strike": pick["buy_2_strike"], "lots": lots, "price": None},
                ])
            state["intended_orders"] = orders
            state["executed"] = True
            ctx.publisher.emit(AgentEvent(
                seq=_next_seq(state, ctx.publisher),
                node="execute", type="result",
                payload={
                    "status": "executed",
                    "executed": True,
                    "mode": pick["mode"],
                    "lots": lots,
                    "orders": orders,
                    "expected_profit": pick["max_profit"],
                    "max_loss": pick["max_loss"],
                },
            ))
            return state

        # ── Node 6: journal + summary ──
        # Composes a single comprehensive summary event AND persists it
        # via the journal port. The summary is what the Agent Console's
        # Stream tab renders as the "final" row — operator can scan one
        # event and understand: what did the strategy decide, why, what
        # was actually traded, what would the P&L look like.
        async def journal_step(state: dict) -> dict:
            # Stringify UUIDs / Decimals for JSON storage.
            from decimal import Decimal
            from uuid import UUID
            def _safe(v):
                if isinstance(v, UUID): return str(v)
                if isinstance(v, Decimal): return float(v)
                if isinstance(v, dict): return {k: _safe(x) for k, x in v.items()}
                if isinstance(v, (list, tuple)): return [_safe(x) for x in v]
                return v

            pick = state.get("pick") or {}
            risk = state.get("risk") or {}
            executed = bool(state.get("executed"))
            node_errors = state.get("node_errors") or {}

            # Resolve the headline outcome — one of three.
            if executed:
                outcome = "ENTERED"
                outcome_reason = (
                    f"{pick.get('mode')} on {underlying} {pick.get('expiry')} — "
                    f"credit ₹{pick.get('credit'):.2f} per lot"
                )
            elif risk.get("status") == "rejected":
                outcome = "BLOCKED"
                outcome_reason = risk.get("reason") or "risk rejected"
            elif not pick:
                outcome = "NO_PLAN"
                outcome_reason = state.get("pick_error") or "planner produced no plan"
            else:
                outcome = "SKIPPED"
                outcome_reason = "did not execute"

            summary = _safe({
                "status": "summary",
                "outcome": outcome,
                "outcome_reason": outcome_reason,
                "underlying": underlying,
                "bias": state.get("bias"),
                "bias_source": state.get("bias_source"),
                "structure": state.get("resolved_mode"),
                "spread": pick or None,
                "risk": risk or None,
                "executed": executed,
                "orders": state.get("intended_orders") or [],
                "node_errors": node_errors,
                "warnings": state.get("_warnings") or [],
            })

            try:
                ctx.journal.record({
                    "kind": "vertical_spread_entry",
                    "title": f"{outcome}: {underlying} {state.get('resolved_mode', '—')} "
                              f"@ {pick.get('atm', '—')}",
                    "body": outcome_reason,
                    "meta": summary,
                    "portfolio_id": str(ctx.portfolio_id),
                    "agent_run_id": str(ctx.run_id),
                })
            except Exception as e:  # noqa: BLE001
                node_errors["journal"] = f"{type(e).__name__}: {e}"
                summary["node_errors"] = node_errors

            ctx.publisher.emit(AgentEvent(
                seq=_next_seq(state, ctx.publisher),
                node="journal",
                type="result" if outcome == "ENTERED" else "info",
                payload=summary,
            ))
            return state

        # ── wire the graph ──
        g = StateGraph(dict)
        for node_name, fn in [
            ("fetch_market", fetch_market),
            ("classify_bias", classify_bias_node),
            ("planner", planner),           # renamed from select_strikes
            ("risk", risk_gate),
            ("execute", execute),
            ("journal", journal_step),
        ]:
            g.add_node(node_name, _safe(node_name, fn))
        g.add_edge("fetch_market", "classify_bias")
        g.add_edge("classify_bias", "planner")
        g.add_edge("planner", "risk")
        g.add_edge("risk", "execute")
        g.add_edge("execute", "journal")
        g.add_edge("journal", END)
        g.set_entry_point("fetch_market")
        return g.compile()


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _legs_from_snapshot(snapshot) -> list[Leg]:
    """Coerce an OptionsChainSnapshot into the flat Leg list the selector
    operates on. Accepts duck-typed snapshots (anything with `.rows`).

    Each row → up to two legs (CE, PE). Strikes with no data are skipped.
    """
    out: list[Leg] = []
    for row in getattr(snapshot, "rows", []) or []:
        for opt_name in ("CE", "PE"):
            q = getattr(row, opt_name.lower(), None)
            if not q:
                continue
            out.append(Leg(
                strike=q.strike, opt=q.opt,
                ltp=float(q.ltp or 0),
                bid=float(q.bid or 0),
                ask=float(q.ask or 0),
                oi=int(q.oi or 0),
            ))
    return out


# Plugin protocol assertion — fails at import-time if shape drifts
assert isinstance(VerticalSpreadStrategy(), Strategy)
