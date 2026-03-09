"""
Straddle Management Graph — LangGraph workflow for options lifecycle management.

Flow:
    fetch_market_data
        → analyze_position
            → generate_action   (LLM with straddle prompt)
                → validate_action   (deterministic, no LLM)
                    → (approved) execute_action → journal_action → END
                    → (rejected) journal_action → END

Designed to be run:
  - Once at session start (full analysis + recommendation)
  - Repeatedly every 15-30 min throughout the trading day
  - On demand after any significant NIFTY move (>0.5%)

No supervisor. No multi-agent branching. Just works.
"""
import os
import sys
import json
import time
import subprocess
import tempfile
from datetime import date, datetime
from typing import Optional

from logzero import logger
from langgraph.graph import StateGraph, END

# Bootstrap Django
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    import django
    django.setup()

from trading.options.straddle.state import (
    StraddleState, StraddleAction, ActionValidation, StraddleExecutionResult
)
from trading.options.straddle.analyzer import analyze_straddle
from trading.options.straddle.prompts import (
    STRADDLE_SYSTEM_PROMPT, STRADDLE_ACTION_TOOL,
    build_cli_straddle_prompt, build_straddle_prompt, REQUIRED_ACTION_FIELDS,
)
from trading.options.data_service import OptionsDataService
from trading.services.broker_service import BrokerService

# ──────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────
_options_data = OptionsDataService()
_broker       = BrokerService()
PLANNER_MODE  = os.getenv("PLANNER_MODE", "cli")


# ──────────────────────────────────────────────
# Audit helper (mirrors equity graph pattern)
# ──────────────────────────────────────────────
def _audit(event_type: str, symbol: str = "", details: dict = None):
    try:
        from trading.models import AuditLog
        AuditLog.objects.create(
            event_type=event_type,
            symbol=symbol,
            risk_details=details,
        )
    except Exception as e:
        logger.warning(f"Straddle audit write failed (non-fatal): {e}")


# ──────────────────────────────────────────────
# JSON parser (shared with equity planner)
# ──────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON: {text[:200]}")


# ──────────────────────────────────────────────
# Node 1: fetch_market_data
# ──────────────────────────────────────────────
def fetch_market_data_node(state: StraddleState) -> dict:
    """
    Fetch NIFTY spot, VIX, CE/PE prices, and 5-min candles from Angel One.
    Single API call batch per cycle.
    """
    today = date.today().isoformat()

    ce_symbol = state.get("ce_symbol", "")
    ce_token  = state.get("ce_token", "")
    pe_symbol = state.get("pe_symbol", "")
    pe_token  = state.get("pe_token", "")

    if not (ce_symbol and ce_token and pe_symbol and pe_token):
        return {
            "error": "Missing CE/PE symbol or token. Register the position first.",
            "market_snapshot": None,
            "nifty_candles": None,
        }

    logger.info(f"Fetching straddle market data | CE={ce_symbol} | PE={pe_symbol}")

    try:
        snapshot = _options_data.fetch_straddle_snapshot(
            ce_symbol=ce_symbol, ce_token=ce_token,
            pe_symbol=pe_symbol, pe_token=pe_token,
            date_str=today,
        )

        if not snapshot.get("nifty") or not snapshot["nifty"].get("ltp"):
            return {
                "error": "NIFTY spot data unavailable from broker",
                "market_snapshot": None,
                "nifty_candles": None,
            }

        logger.info(
            f"Market snapshot: NIFTY={snapshot['nifty']['ltp']:.2f} | "
            f"VIX={snapshot['vix'].get('ltp', 0):.2f} | "
            f"CE={snapshot['ce'].get('ltp', 0):.2f} | "
            f"PE={snapshot['pe'].get('ltp', 0):.2f} | "
            f"Candles={len(snapshot['candles'])}"
        )

        return {
            "market_snapshot": snapshot,
            "nifty_candles":   snapshot["candles"],
            "error": None,
        }

    except Exception as e:
        logger.exception(f"Market data fetch failed: {e}")
        return {
            "error": f"Market data fetch failed: {e}",
            "market_snapshot": None,
            "nifty_candles":   None,
        }


# ──────────────────────────────────────────────
# Node 2: analyze_position
# ──────────────────────────────────────────────
def analyze_position_node(state: StraddleState) -> dict:
    """
    Run the pure-Python straddle analyzer.
    Computes P&L, delta, market phase, scenarios.
    No LLM — purely deterministic.
    """
    if state.get("error"):
        return {"analysis": None}

    snapshot = state.get("market_snapshot", {})
    candles  = state.get("nifty_candles", [])

    nifty   = snapshot.get("nifty", {})
    vix     = snapshot.get("vix",   {})
    ce_data = snapshot.get("ce",    {})
    pe_data = snapshot.get("pe",    {})

    try:
        analysis = analyze_straddle(
            underlying    = state.get("underlying", "NIFTY"),
            strike        = state.get("strike", 0),
            expiry        = state.get("expiry", ""),
            lot_size      = state.get("lot_size", 65),
            lots          = state.get("lots", 1),
            ce_sell_price = state.get("ce_sell_price", 0.0),
            pe_sell_price = state.get("pe_sell_price", 0.0),
            ce_ltp        = ce_data.get("ltp", 0.0),
            pe_ltp        = pe_data.get("ltp", 0.0),
            nifty_spot    = nifty.get("ltp", 0.0),
            nifty_prev_close = nifty.get("prev_close", 0.0),
            vix_current   = vix.get("ltp", 0.0),
            vix_prev_close= vix.get("prev_close", 0.0),
            candles       = candles,
        )

        logger.info(
            f"Analysis: P&L={analysis.net_pnl_inr:+,.0f} INR | "
            f"Delta={analysis.net_delta:+.2f} | "
            f"Phase={analysis.market_phase} | "
            f"VIX={analysis.vix_phase} | "
            f"DTE={analysis.days_to_expiry}"
        )

        if analysis.is_underwater:
            logger.warning("⚠️  POSITION UNDERWATER — combined premium > sold")
        if analysis.expiry_tomorrow:
            logger.warning("⚠️  EXPIRY TOMORROW — gamma risk is extreme")

        return {"analysis": analysis.model_dump()}

    except Exception as e:
        logger.exception(f"Straddle analysis failed: {e}")
        return {"analysis": None, "error": f"Analysis failed: {e}"}


# ──────────────────────────────────────────────
# Node 3: generate_action (LLM)
# ──────────────────────────────────────────────
def generate_action_node(state: StraddleState) -> dict:
    """
    Run the straddle management LLM to recommend a management action.
    Uses same dual-mode pattern as equity planner (CLI or API).
    """
    if state.get("error"):
        return {"recommended_action": None, "planner_raw": state["error"]}

    analysis = state.get("analysis")
    if not analysis:
        return {
            "recommended_action": None,
            "planner_raw": "No analysis available",
            "error": "Cannot generate action without analysis",
        }

    analysis_text    = analysis.get("summary_text", "")
    position_history = _get_position_history(state.get("position_id"))

    mode = os.getenv("PLANNER_MODE", "cli").lower()

    try:
        if mode == "api":
            result = _generate_action_api(analysis_text, position_history)
        else:
            result = _generate_action_cli(analysis_text, position_history)

        if "error" in result:
            return {"recommended_action": None, "planner_raw": result["error"], "error": result["error"]}

        # Validate required fields
        missing = [f for f in REQUIRED_ACTION_FIELDS if f not in result]
        if missing:
            err = f"Action missing required fields: {missing}"
            logger.error(err)
            return {"recommended_action": None, "planner_raw": str(result), "error": err}

        # Coerce types
        result["confidence"] = float(result.get("confidence", 0.0))
        result["hedge_lots"] = int(result.get("hedge_lots", 0))
        result.setdefault("hedge_side", "NONE")
        result.setdefault("pe_stop_loss", None)
        result.setdefault("pe_target", None)

        logger.info(
            f"Action: {result['action']} | urgency={result['urgency']} | "
            f"conf={result['confidence']:.2f} | CE={result['ce_action']} | PE={result['pe_action']}"
        )

        _audit("PLANNER_RES", symbol=state.get("underlying", "NIFTY"), details={
            "action": result["action"],
            "urgency": result["urgency"],
            "confidence": result["confidence"],
            "reasoning": result.get("reasoning", ""),
        })

        return {"recommended_action": result, "planner_raw": str(result), "error": None}

    except Exception as e:
        logger.exception(f"Action generation failed: {e}")
        return {"recommended_action": None, "planner_raw": str(e), "error": str(e)}


def _get_position_history(position_id: Optional[int]) -> str:
    """Fetch today's management log from StraddlePosition."""
    if not position_id:
        return ""
    try:
        from trading.models import StraddlePosition
        pos = StraddlePosition.objects.get(id=position_id)
        log = pos.management_log or []
        if not log:
            return ""
        lines = [
            f"  {entry.get('time', '?')} | {entry.get('action', '?')} | {entry.get('note', '')}"
            for entry in log[-5:]  # last 5 actions
        ]
        return "\n".join(lines)
    except Exception:
        return ""


def _generate_action_cli(analysis_text: str, position_history: str) -> dict:
    """Generate straddle action via Claude CLI."""
    import shutil

    claude_path = os.getenv("CLAUDE_CLI_PATH", "").strip() or shutil.which("claude") or ""
    if not claude_path:
        # Try login shell
        for shell_cmd in [["zsh", "-lc", "which claude"], ["bash", "-lc", "which claude"]]:
            try:
                r = subprocess.run(shell_cmd, capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    claude_path = r.stdout.strip()
                    break
            except Exception:
                continue

    if not claude_path:
        return {"error": "Claude CLI not found. Set CLAUDE_CLI_PATH or use PLANNER_MODE=api"}

    prompt = build_cli_straddle_prompt(analysis_text, position_history)
    env    = os.environ.copy()
    env.pop("CLAUDECODE", None)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(prompt)
            tmp_path = tmp.name

        t0     = time.time()
        result = subprocess.run(
            f'"{claude_path}" --print < "{tmp_path}"',
            capture_output=True, text=True, timeout=120, env=env, shell=True,
        )
        latency_ms = int((time.time() - t0) * 1000)

        if result.returncode != 0:
            return {"error": f"Claude CLI failed (exit {result.returncode}): {result.stderr[:200]}"}

        raw = result.stdout.strip()
        if not raw:
            return {"error": "Claude CLI returned empty response"}

        parsed = _parse_json(raw)
        logger.info(f"CLI action generated in {latency_ms}ms")
        return parsed

    except subprocess.TimeoutExpired:
        return {"error": "Claude CLI timed out (120s)"}
    except ValueError as e:
        return {"error": f"JSON parse failed: {e}"}
    except Exception as e:
        return {"error": f"CLI generation failed: {e}"}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _generate_action_api(analysis_text: str, position_history: str) -> dict:
    """Generate straddle action via Anthropic API with tool_use."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}

    model_name = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    user_msg   = build_straddle_prompt(analysis_text, position_history)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        t0     = time.time()
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.2,
            system=STRADDLE_SYSTEM_PROMPT,
            tools=[STRADDLE_ACTION_TOOL],
            tool_choice={"type": "tool", "name": "recommend_straddle_action"},
            messages=[{"role": "user", "content": user_msg}],
        )
        latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"API action generated in {latency_ms}ms")

        for block in response.content:
            if block.type == "tool_use" and block.name == "recommend_straddle_action":
                return block.input

        return {"error": "LLM did not produce a tool_use response"}

    except Exception as e:
        return {"error": f"API generation failed: {e}"}


# ──────────────────────────────────────────────
# Node 4: validate_action (deterministic)
# ──────────────────────────────────────────────
def validate_action_node(state: StraddleState) -> dict:
    """
    Deterministic validation of the LLM's recommended action.
    No LLM. Hard rules only.

    Rules:
    1. Low confidence (<0.6) → convert to MONITOR
    2. Underwater + HOLD → override to CLOSE_BOTH IMMEDIATE
    3. Expiry tomorrow + ROLL → override to CLOSE_BOTH
    4. HEDGE_FUTURES without hedge_lots → reject
    5. CLOSE_BOTH on highly profitable position without urgency → warn but allow
    """
    action_dict = state.get("recommended_action")
    analysis    = state.get("analysis", {})

    if not action_dict:
        return {
            "action_approved": False,
            "validation_result": ActionValidation(
                approved=False, reason="No action to validate"
            ).model_dump(),
        }

    action       = action_dict.get("action", "HOLD")
    urgency      = action_dict.get("urgency", "MONITOR")
    confidence   = float(action_dict.get("confidence", 0.0))
    is_underwater = analysis.get("is_underwater", False)
    expiry_tomorrow = analysis.get("expiry_tomorrow", False)
    net_pnl_inr  = analysis.get("net_pnl_inr", 0.0)

    override_action = None

    # Rule 1: Low confidence → MONITOR
    if confidence < 0.6:
        approved = True
        reason   = f"Low confidence ({confidence:.2f}) — converted to MONITOR"
        override_action = "MONITOR"
        action_dict["action"]  = "HOLD"
        action_dict["urgency"] = "MONITOR"
        logger.info(f"Validator: {reason}")
        _audit("RISK_APPROVE", symbol=state.get("underlying", "NIFTY"),
               details={"rule": "low_confidence", "override": "MONITOR"})
        return {
            "action_approved": True,
            "validation_result": ActionValidation(
                approved=True, reason=reason, override_action="HOLD/MONITOR"
            ).model_dump(),
            "recommended_action": action_dict,
        }

    # Rule 2: Underwater + HOLD → override to CLOSE_BOTH IMMEDIATE
    if is_underwater and action == "HOLD":
        reason = "Position is underwater. Overriding HOLD → CLOSE_BOTH IMMEDIATE (hard stop rule)"
        action_dict["action"]  = "CLOSE_BOTH"
        action_dict["urgency"] = "IMMEDIATE"
        action_dict["ce_action"] = "CLOSE"
        action_dict["pe_action"] = "CLOSE"
        logger.warning(f"Validator override: {reason}")
        _audit("RISK_REJECT", symbol=state.get("underlying", "NIFTY"),
               details={"rule": "underwater_hold_override", "original": action})
        return {
            "action_approved": True,
            "validation_result": ActionValidation(
                approved=True, reason=reason, override_action="CLOSE_BOTH"
            ).model_dump(),
            "recommended_action": action_dict,
        }

    # Rule 3: Expiry tomorrow + ROLL → override to CLOSE_BOTH
    if expiry_tomorrow and action == "ROLL":
        reason = "Cannot ROLL on expiry day. Overriding → CLOSE_BOTH IMMEDIATE"
        action_dict["action"]  = "CLOSE_BOTH"
        action_dict["urgency"] = "IMMEDIATE"
        action_dict["ce_action"] = "CLOSE"
        action_dict["pe_action"] = "CLOSE"
        logger.warning(f"Validator override: {reason}")
        _audit("RISK_REJECT", symbol=state.get("underlying", "NIFTY"),
               details={"rule": "expiry_roll_override"})
        return {
            "action_approved": True,
            "validation_result": ActionValidation(
                approved=True, reason=reason, override_action="CLOSE_BOTH"
            ).model_dump(),
            "recommended_action": action_dict,
        }

    # Rule 4: HEDGE without lots
    if action == "HEDGE_FUTURES" and action_dict.get("hedge_lots", 0) == 0:
        reason = "HEDGE_FUTURES action requires hedge_lots > 0. Rejected."
        logger.error(f"Validator rejected: {reason}")
        return {
            "action_approved": False,
            "validation_result": ActionValidation(
                approved=False, reason=reason
            ).model_dump(),
        }

    # All clear
    reason = f"Action {action}/{urgency} approved. Confidence: {confidence:.2f}"
    logger.info(f"Validator approved: {reason}")
    _audit("RISK_APPROVE", symbol=state.get("underlying", "NIFTY"),
           details={"action": action, "urgency": urgency, "confidence": confidence})

    return {
        "action_approved": True,
        "validation_result": ActionValidation(approved=True, reason=reason).model_dump(),
    }


# ──────────────────────────────────────────────
# Node 5: execute_action
# ──────────────────────────────────────────────
def execute_action_node(state: StraddleState) -> dict:
    """
    Execute the validated management action via BrokerService.
    In paper mode: simulates fills. In live mode: real NFO orders.
    """
    action_dict = state.get("recommended_action", {})
    action      = action_dict.get("action", "HOLD")

    if action in ("HOLD", "MONITOR"):
        return {
            "execution_result": StraddleExecutionResult(
                success=True,
                actions_taken=["HOLD — no orders placed"],
                message="HOLD: monitoring only",
            ).model_dump()
        }

    actions_taken = []
    orders        = []
    lot_size      = state.get("lot_size", 65)
    lots          = state.get("lots", 1)
    qty           = lot_size * lots
    snapshot      = state.get("market_snapshot", {})

    # ── Close CE ──
    if action in ("CLOSE_BOTH", "CLOSE_CE") or action_dict.get("ce_action") == "CLOSE":
        ce_ltp = snapshot.get("ce", {}).get("ltp", 0)
        result = _broker.place_order(
            symbol       = state.get("ce_symbol", ""),
            side         = "BUY",           # BUY to close a short call
            quantity     = qty,
            price        = ce_ltp,
            product_type = "CARRYFORWARD",  # NFO options require CARRYFORWARD
            exchange     = "NFO",
            symbol_token = state.get("ce_token", ""),
        )
        orders.append(result)
        actions_taken.append(f"CLOSED_CE @ {ce_ltp:.2f}")
        logger.info(f"CE closed: {result.get('message', '')}")

    # ── Close PE ──
    if action in ("CLOSE_BOTH", "CLOSE_PE") or action_dict.get("pe_action") == "CLOSE":
        pe_ltp = snapshot.get("pe", {}).get("ltp", 0)
        result = _broker.place_order(
            symbol       = state.get("pe_symbol", ""),
            side         = "BUY",           # BUY to close a short put
            quantity     = qty,
            price        = pe_ltp,
            product_type = "CARRYFORWARD",
            exchange     = "NFO",
            symbol_token = state.get("pe_token", ""),
        )
        orders.append(result)
        actions_taken.append(f"CLOSED_PE @ {pe_ltp:.2f}")
        logger.info(f"PE closed: {result.get('message', '')}")

    # ── Hedge with futures ──
    if action == "HEDGE_FUTURES" and action_dict.get("hedge_lots", 0) > 0:
        hedge_side = action_dict.get("hedge_side", "BUY")
        hedge_lots = action_dict.get("hedge_lots", 1)
        nifty_ltp  = snapshot.get("nifty", {}).get("ltp", 0)
        result = _broker.place_order(
            symbol       = "NIFTY",
            side         = hedge_side,
            quantity     = 75 * hedge_lots,  # NIFTY futures lot size = 75
            price        = nifty_ltp,
            product_type = "CARRYFORWARD",
            exchange     = "NFO",
        )
        orders.append(result)
        actions_taken.append(f"HEDGE_FUTURES {hedge_side} {hedge_lots}L @ {nifty_ltp:.2f}")
        logger.info(f"Futures hedge: {result.get('message', '')}")

    success = all(o.get("success", False) for o in orders) if orders else True

    exec_result = StraddleExecutionResult(
        success=success,
        actions_taken=actions_taken,
        orders=orders,
        message=" | ".join(actions_taken),
        mode=os.getenv("TRADING_MODE", "paper"),
    )

    _audit("EXECUTION", symbol=state.get("underlying", "NIFTY"), details={
        "action": action,
        "actions_taken": actions_taken,
        "success": success,
    })

    return {"execution_result": exec_result.model_dump()}


# ──────────────────────────────────────────────
# Node 6: journal_action
# ──────────────────────────────────────────────
def journal_action_node(state: StraddleState) -> dict:
    """
    Update StraddlePosition in the database.
    Appends management event to position_log.
    """
    from trading.models import StraddlePosition

    position_id = state.get("position_id")
    if not position_id:
        logger.warning("No position_id — cannot journal straddle action")
        return {"journal_id": None}

    analysis      = state.get("analysis", {})
    action_dict   = state.get("recommended_action", {})
    exec_result   = state.get("execution_result", {})
    action        = action_dict.get("action", "HOLD")

    try:
        pos = StraddlePosition.objects.get(id=position_id)

        # Update current prices
        snapshot = state.get("market_snapshot", {})
        pos.ce_current_price = snapshot.get("ce", {}).get("ltp", pos.ce_current_price)
        pos.pe_current_price = snapshot.get("pe", {}).get("ltp", pos.pe_current_price)
        pos.net_delta        = analysis.get("net_delta", 0.0)
        pos.current_pnl_inr  = analysis.get("net_pnl_inr", 0.0)
        pos.last_updated     = datetime.now()

        # Update status
        if action in ("CLOSE_BOTH",) and exec_result.get("success"):
            pos.status       = StraddlePosition.Status.CLOSED
            pos.action_taken = action
            pos.closed_at    = datetime.now()
        elif action in ("CLOSE_CE", "CLOSE_PE") and exec_result.get("success"):
            pos.status       = StraddlePosition.Status.PARTIAL
            pos.action_taken = action
        elif action == "HEDGE_FUTURES" and exec_result.get("success"):
            pos.status       = StraddlePosition.Status.HEDGED
            pos.action_taken = action

        # Append to management log
        log_entry = {
            "time":     datetime.now().strftime("%H:%M"),
            "action":   action,
            "urgency":  action_dict.get("urgency", "MONITOR"),
            "nifty":    analysis.get("nifty_spot", 0),
            "pnl_inr":  analysis.get("net_pnl_inr", 0),
            "note":     action_dict.get("reasoning", "")[:100],
            "executed": exec_result.get("success", False),
        }
        if pos.management_log is None:
            pos.management_log = []
        pos.management_log.append(log_entry)

        pos.save()
        logger.info(f"Straddle journal updated: ID={position_id} | action={action} | P&L={pos.current_pnl_inr:+,.0f}")
        return {"journal_id": position_id}

    except Exception as e:
        logger.exception(f"Straddle journal save failed: {e}")
        return {"journal_id": None, "error": f"Journal save failed: {e}"}


# ──────────────────────────────────────────────
# Conditional edge: action gate
# ──────────────────────────────────────────────
def action_gate(state: StraddleState) -> str:
    """Route to execute if approved, else skip to journal."""
    action_dict = state.get("recommended_action", {})
    action      = action_dict.get("action", "HOLD")

    if state.get("action_approved") and action not in ("HOLD",):
        return "execute"
    return "journal"


# ──────────────────────────────────────────────
# Build the straddle graph
# ──────────────────────────────────────────────
def build_straddle_graph() -> StateGraph:
    """
    Construct and compile the straddle management workflow.

    Flow:
        fetch_market_data → analyze_position → generate_action → validate_action
            → (approved + non-HOLD) → execute_action → journal_action → END
            → (HOLD or rejected) → journal_action → END
    """
    graph = StateGraph(StraddleState)

    graph.add_node("fetch_market_data", fetch_market_data_node)
    graph.add_node("analyze_position",  analyze_position_node)
    graph.add_node("generate_action",   generate_action_node)
    graph.add_node("validate_action",   validate_action_node)
    graph.add_node("execute_action",    execute_action_node)
    graph.add_node("journal_action",    journal_action_node)

    graph.set_entry_point("fetch_market_data")
    graph.add_edge("fetch_market_data", "analyze_position")
    graph.add_edge("analyze_position",  "generate_action")
    graph.add_edge("generate_action",   "validate_action")

    graph.add_conditional_edges(
        "validate_action",
        action_gate,
        {"execute": "execute_action", "journal": "journal_action"},
    )

    graph.add_edge("execute_action", "journal_action")
    graph.add_edge("journal_action", END)

    return graph.compile()


# ──────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────
def run_straddle_workflow(
    position_id: int,
    underlying: str,
    strike: int,
    expiry: str,
    lot_size: int,
    lots: int,
    ce_symbol: str,
    ce_token: str,
    pe_symbol: str,
    pe_token: str,
    ce_sell_price: float,
    pe_sell_price: float,
) -> dict:
    """
    Run one full straddle management cycle.

    Designed to be called:
    - Once at session start
    - Every 15-30 min via cron/loop
    - On-demand via manage_straddle CLI

    Returns:
        Final StraddleState dict with all results.
    """
    graph = build_straddle_graph()

    initial_state: StraddleState = {
        "position_id":    position_id,
        "underlying":     underlying,
        "strike":         strike,
        "expiry":         expiry,
        "lot_size":       lot_size,
        "lots":           lots,
        "ce_symbol":      ce_symbol,
        "ce_token":       ce_token,
        "pe_symbol":      pe_symbol,
        "pe_token":       pe_token,
        "ce_sell_price":  ce_sell_price,
        "pe_sell_price":  pe_sell_price,
        "nifty_candles":  None,
        "market_snapshot": None,
        "analysis":       None,
        "recommended_action": None,
        "planner_raw":    None,
        "action_approved": None,
        "validation_result": None,
        "execution_result": None,
        "journal_id":     None,
        "error":          None,
    }

    logger.info(
        f"=== Straddle workflow started | "
        f"{underlying} {strike} {expiry} | "
        f"CE={ce_symbol} | PE={pe_symbol} ==="
    )

    result = graph.invoke(initial_state)

    action  = (result.get("recommended_action") or {}).get("action", "N/A")
    pnl_inr = (result.get("analysis") or {}).get("net_pnl_inr", 0)

    logger.info(
        f"=== Straddle workflow complete | "
        f"action={action} | "
        f"P&L={pnl_inr:+,.0f} INR | "
        f"journal_id={result.get('journal_id')} ==="
    )

    return result
