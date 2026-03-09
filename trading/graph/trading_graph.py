"""
LangGraph Trading Orchestrator.

Clean linear flow:
    fetch_data → retrieve_context → planner → risk → execute → journal

Includes:
    - Audit logging at every decision point
    - Position reconciliation on startup
    - Backtest mode support

No supervisor complexity. No multi-agent branching. Just works.
"""
import os
import sys
from datetime import date, datetime
from typing import Optional

from logzero import logger
from langgraph.graph import StateGraph, END

# Bootstrap Django
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import django
    django.setup()

from trading.graph.state import TradingState, RiskResult, ExecutionResult
from trading.services.data_service import DataService
from trading.services.risk_engine import validate_trade
from trading.services.broker_service import BrokerService
from trading.rag.retriever import retrieve_context, retrieve_portfolio_context
from trading.agents.planner import run_planner


# ──────────────────────────────────────────────
# Singletons (initialized once per graph build)
# ──────────────────────────────────────────────
_data_service = DataService()
_broker_service = BrokerService()


# ──────────────────────────────────────────────
# Audit helper (non-blocking)
# ──────────────────────────────────────────────
def _audit(event_type: str, symbol: str = "", risk_details: dict = None,
           execution_details: dict = None, trade_journal=None):
    """Write audit log. Never blocks trading flow."""
    try:
        from trading.models import AuditLog
        AuditLog.objects.create(
            event_type=event_type,
            symbol=symbol,
            risk_details=risk_details,
            execution_details=execution_details,
            trade_journal=trade_journal,
        )
    except Exception as e:
        logger.warning(f"Audit write failed (non-fatal): {e}")


# ──────────────────────────────────────────────
# Position Reconciliation
# ──────────────────────────────────────────────
def reconcile_positions():
    """
    On startup: pull broker positions, compare with DB, log mismatches.
    Never assume state is correct.
    """
    from trading.models import TradeJournal, PortfolioSnapshot

    trading_mode = os.getenv("TRADING_MODE", "paper")

    if trading_mode == "paper":
        logger.info("Reconciliation skipped (paper mode)")
        return

    logger.info("=== Position Reconciliation ===")

    try:
        # Pull live positions from broker
        broker_positions = _data_service.fetch_positions()
        broker_data = broker_positions.get("data", []) if isinstance(broker_positions, dict) else []

        # Pull DB state: open/executed trades from today
        db_open = TradeJournal.objects.filter(
            status__in=["EXECUTED", "FILLED", "PARTIAL"],
            trade_date=date.today(),
        )

        db_symbols = {t.symbol for t in db_open}
        broker_symbols = set()

        if broker_data:
            for pos in broker_data:
                sym = pos.get("tradingsymbol", "").replace("-EQ", "")
                broker_symbols.add(sym)

        # Find mismatches
        db_only = db_symbols - broker_symbols
        broker_only = broker_symbols - db_symbols

        if db_only:
            logger.warning(f"RECONCILIATION: In DB but NOT in broker: {db_only}")
            _audit("RECONCILE", risk_details={
                "type": "db_only", "symbols": list(db_only),
                "message": "Positions in DB but not found in broker"
            })

        if broker_only:
            logger.warning(f"RECONCILIATION: In broker but NOT in DB: {broker_only}")
            _audit("RECONCILE", risk_details={
                "type": "broker_only", "symbols": list(broker_only),
                "message": "Positions in broker but not tracked in DB"
            })

        if not db_only and not broker_only:
            logger.info("Reconciliation: OK — positions match")

        # Update portfolio snapshot
        try:
            snap = PortfolioSnapshot.objects.latest()
            snap.open_positions = len(broker_symbols) if broker_data else len(db_symbols)
            snap.open_positions_count = snap.open_positions
            snap.save()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Reconciliation failed: {e}")
        _audit("RECONCILE", risk_details={"error": str(e)})


# ──────────────────────────────────────────────
# Node: fetch_data
# ──────────────────────────────────────────────
def fetch_data_node(state: TradingState) -> dict:
    """
    Fetch and enrich market data for the symbol.
    Extracts symbol and date from user_intent or uses defaults.
    """
    symbol = state.get("symbol", "")
    user_intent = state.get("user_intent", "")

    if not symbol:
        words = user_intent.upper().split()
        skip = {"BUY", "SELL", "TRADE", "PLAN", "FOR", "ON", "THE", "A", "AN", "AT", "TO", "IN", "OF"}
        for w in words:
            if w.isalpha() and len(w) >= 2 and w not in skip:
                symbol = w
                break

    if not symbol:
        return {
            "error": "Could not determine symbol from intent. Please specify a stock symbol.",
            "market_data": None,
            "market_data_raw": "",
            "symbol": "",
        }

    today = date.today().isoformat()
    logger.info(f"Fetching data for {symbol} on {today}")

    try:
        data = _data_service.fetch_intraday(symbol, today)

        if "error" in data:
            # Data fetch failed but don't block workflow —
            # planner can still work with RAG context alone
            fallback = (
                f"Market data unavailable for {symbol} on {today}.\n"
                f"Reason: {data['error']}\n"
                f"Planner should rely on RAG context and strategy rules."
            )
            logger.warning(f"Data fetch issue for {symbol}: {data['error']} — continuing with RAG only")
            return {
                "symbol": symbol,
                "market_data": data,
                "market_data_raw": fallback,
                "error": None,  # don't propagate — let planner decide
            }

        return {
            "symbol": symbol,
            "market_data": data,
            "market_data_raw": data.get("summary", ""),
            "error": None,
        }

    except Exception as e:
        logger.exception(f"Data fetch failed: {e}")
        fallback = (
            f"Market data unavailable for {symbol} on {today}.\n"
            f"Reason: {str(e)}\n"
            f"Planner should rely on RAG context and strategy rules."
        )
        return {
            "symbol": symbol,
            "market_data": None,
            "market_data_raw": fallback,
            "error": None,  # don't block workflow
        }


# ──────────────────────────────────────────────
# Node: retrieve_context
# ──────────────────────────────────────────────
def retrieve_context_node(state: TradingState) -> dict:
    """Pull RAG context from Postgres: recent trades + strategy docs."""
    symbol = state.get("symbol", "")

    if not symbol:
        portfolio_ctx = retrieve_portfolio_context()
        return {"rag_context": portfolio_ctx}

    try:
        ctx = retrieve_context(symbol)
        portfolio_ctx = retrieve_portfolio_context()
        full_context = f"{ctx}\n\n---\n\n{portfolio_ctx}"
        return {"rag_context": full_context}

    except Exception as e:
        logger.exception(f"RAG retrieval failed: {e}")
        return {"rag_context": f"RAG context unavailable: {str(e)}"}


# ──────────────────────────────────────────────
# Node: planner
# ──────────────────────────────────────────────
def planner_node(state: TradingState) -> dict:
    """Run the planner agent to produce a TradePlan."""
    user_intent = state.get("user_intent", "")
    market_data_raw = state.get("market_data_raw", "No market data available")
    rag_context = state.get("rag_context", "No context available")

    if state.get("error"):
        return {
            "trade_plan": None,
            "planner_raw": f"Skipped due to upstream error: {state['error']}",
            "error": state["error"],
        }

    try:
        plan = run_planner(user_intent, market_data_raw, rag_context)

        if "error" in plan:
            return {
                "trade_plan": None,
                "planner_raw": plan["error"],
                "error": plan["error"],
            }

        symbol = plan.get("symbol", state.get("symbol", ""))

        return {
            "trade_plan": plan,
            "planner_raw": str(plan),
            "symbol": symbol,
            "error": None,
        }

    except Exception as e:
        logger.exception(f"Planner node failed: {e}")
        return {
            "trade_plan": None,
            "planner_raw": str(e),
            "error": f"Planner failed: {str(e)}",
        }


# ──────────────────────────────────────────────
# Node: risk_validator
# ──────────────────────────────────────────────
def risk_node(state: TradingState) -> dict:
    """Run deterministic risk validation on the trade plan."""
    plan = state.get("trade_plan")

    if not plan:
        return {
            "risk_approved": False,
            "risk_result": RiskResult(
                approved=False,
                reason="No trade plan to validate",
            ).model_dump(),
        }

    # Low confidence = no trade
    if plan.get("confidence", 0) < 0.01:
        reason = (
            f"Planner confidence too low: {plan.get('confidence', 0):.2f}. "
            f"Reason: {plan.get('reasoning', 'N/A')}"
        )
        _audit("RISK_REJECT", symbol=plan.get("symbol", ""),
               risk_details={"reason": reason, "confidence": plan.get("confidence", 0)})
        return {
            "risk_approved": False,
            "risk_result": RiskResult(approved=False, reason=reason).model_dump(),
        }

    # Get portfolio state
    from trading.models import PortfolioSnapshot
    try:
        snap = PortfolioSnapshot.objects.latest()
        capital = snap.capital
        daily_loss = snap.daily_loss
        open_positions = snap.open_positions
    except Exception:
        capital = float(os.getenv("DEFAULT_CAPITAL", "100000"))
        daily_loss = 0.0
        open_positions = 0
        logger.warning(f"No portfolio snapshot — using default capital: {capital}")

    approved, reason, details = validate_trade(
        plan=plan,
        capital=capital,
        daily_loss=daily_loss,
        open_positions=open_positions,
    )

    risk_result = RiskResult(
        approved=approved,
        reason=reason,
        risk_amount=details.get("risk_amount", 0),
        risk_pct_of_capital=details.get("risk_pct_of_capital", 0),
    )

    # ── Audit: log every risk decision ──
    event = "RISK_APPROVE" if approved else "RISK_REJECT"
    _audit(event, symbol=plan.get("symbol", ""), risk_details={
        "approved": approved,
        "reason": reason,
        "details": details,
        "plan_summary": {
            "symbol": plan["symbol"], "side": plan["side"],
            "entry": plan["entry_price"], "sl": plan["stop_loss"],
            "target": plan["target"], "qty": plan["quantity"],
            "confidence": plan.get("confidence", 0),
        },
        "capital": capital,
        "daily_loss": daily_loss,
        "open_positions": open_positions,
    })

    logger.info(f"Risk {'APPROVED' if approved else 'REJECTED'}: {reason}")

    return {
        "risk_approved": approved,
        "risk_result": risk_result.model_dump(),
    }


# ──────────────────────────────────────────────
# Node: execute
# ──────────────────────────────────────────────
def execute_node(state: TradingState) -> dict:
    """Execute the trade via broker (paper or live)."""
    plan = state.get("trade_plan")

    if not plan:
        return {
            "execution_result": ExecutionResult(
                success=False, message="No trade plan"
            ).model_dump(),
        }

    result = _broker_service.place_order(
        symbol=plan["symbol"],
        side=plan["side"],
        quantity=plan["quantity"],
        price=plan["entry_price"],
    )

    exec_result = ExecutionResult(
        success=result["success"],
        order_id=result.get("order_id", ""),
        fill_price=result.get("fill_price", 0),
        fill_quantity=result.get("fill_quantity", 0),
        message=result.get("message", ""),
        mode=result.get("mode", "paper"),
    )

    # ── Audit: log execution ──
    _audit("EXECUTION", symbol=plan["symbol"], execution_details={
        "success": result["success"],
        "order_id": result.get("order_id", ""),
        "fill_price": result.get("fill_price", 0),
        "mode": result.get("mode", "paper"),
        "message": result.get("message", ""),
    })

    return {"execution_result": exec_result.model_dump()}


# ──────────────────────────────────────────────
# Node: journal_writer
# ──────────────────────────────────────────────
def journal_node(state: TradingState) -> dict:
    """Save the trade to the journal (Django ORM)."""
    from trading.models import TradeJournal

    plan = state.get("trade_plan")
    risk_result = state.get("risk_result") or {}
    exec_result = state.get("execution_result") or {}
    risk_approved = state.get("risk_approved", False)

    if not plan:
        return {"journal_id": None}

    # Determine status
    if not risk_approved:
        status = TradeJournal.Status.REJECTED
    elif exec_result.get("success"):
        mode = exec_result.get("mode", "paper")
        status = TradeJournal.Status.PAPER if mode == "paper" else TradeJournal.Status.EXECUTED
    else:
        status = TradeJournal.Status.APPROVED

    try:
        journal = TradeJournal.objects.create(
            symbol=plan["symbol"],
            side=plan["side"],
            entry_price=plan["entry_price"],
            stop_loss=plan["stop_loss"],
            target=plan["target"],
            quantity=plan["quantity"],
            reasoning=plan.get("reasoning", ""),
            confidence=plan.get("confidence", 0),
            status=status,
            order_id=exec_result.get("order_id", ""),
            fill_price=exec_result.get("fill_price"),
            fill_quantity=exec_result.get("fill_quantity"),
            risk_approved=risk_approved,
            risk_reason=risk_result.get("reason", ""),
            trade_date=date.today(),
        )

        logger.info(f"Journal saved: ID={journal.id} | {journal}")
        return {"journal_id": journal.id}

    except Exception as e:
        logger.exception(f"Journal save failed: {e}")
        return {"journal_id": None, "error": f"Journal save failed: {str(e)}"}


# ──────────────────────────────────────────────
# Conditional edge: risk gate
# ──────────────────────────────────────────────
def risk_gate(state: TradingState) -> str:
    """Route based on risk approval."""
    if state.get("risk_approved"):
        return "execute"
    else:
        return "journal"


# ──────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────
def build_trading_graph() -> StateGraph:
    """
    Construct and compile the trading workflow graph.

    Flow:
        fetch_data → retrieve_context → planner → risk
            → (approved) → execute → journal → END
            → (rejected) → journal → END
    """
    graph = StateGraph(TradingState)

    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("planner", planner_node)
    graph.add_node("risk", risk_node)
    graph.add_node("execute", execute_node)
    graph.add_node("journal", journal_node)

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "retrieve_context")
    graph.add_edge("retrieve_context", "planner")
    graph.add_edge("planner", "risk")

    graph.add_conditional_edges(
        "risk",
        risk_gate,
        {
            "execute": "execute",
            "journal": "journal",
        },
    )

    graph.add_edge("execute", "journal")
    graph.add_edge("journal", END)

    return graph.compile()


# ──────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────
def run_trading_workflow(
    user_intent: str,
    symbol: str = "",
    run_reconciliation: bool = True,
) -> dict:
    """
    Run the full trading workflow.

    Args:
        user_intent: What the user wants, e.g. "Plan a BUY trade for MFSL"
        symbol: Optional explicit symbol. If empty, extracted from intent.
        run_reconciliation: If True, reconcile positions before trading.

    Returns:
        Final state dict with all results.
    """
    # ── Position reconciliation on startup ──
    if run_reconciliation:
        reconcile_positions()

    graph = build_trading_graph()

    initial_state: TradingState = {
        "user_intent": user_intent,
        "symbol": symbol,
        "market_data": None,
        "market_data_raw": None,
        "rag_context": None,
        "trade_plan": None,
        "planner_raw": None,
        "risk_approved": None,
        "risk_result": None,
        "execution_result": None,
        "journal_id": None,
        "error": None,
    }

    logger.info(f"=== Trading workflow started | intent='{user_intent}' | symbol='{symbol}' ===")

    result = graph.invoke(initial_state)

    logger.info(
        f"=== Workflow complete | "
        f"risk={'APPROVED' if result.get('risk_approved') else 'REJECTED'} | "
        f"journal_id={result.get('journal_id')} ==="
    )

    return result
