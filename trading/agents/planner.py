"""
Planner Agent — Dual-mode: Claude CLI (Max plan) + Anthropic API fallback.

Primary:  Claude Code CLI (`claude --print`) — uses your Max plan, no API credits.
Fallback: Official Anthropic SDK with tool_use — uses API credits.

The CLI mode sends a structured prompt and parses JSON output.
The API mode uses tool_choice to force structured TradePlan output.

Every prompt and response is audit-logged.
"""
import os
import json
import time
import subprocess
import shutil
from typing import Optional

from logzero import logger
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Planner mode: "cli" (default, uses Max plan) or "api" (uses API credits)
# ──────────────────────────────────────────────
PLANNER_MODE = os.getenv("PLANNER_MODE", "cli")  # cli | api


# ──────────────────────────────────────────────
# TradePlan tool schema (for API mode forced structured output)
# ──────────────────────────────────────────────
TRADE_PLAN_TOOL = {
    "name": "generate_trade_plan",
    "description": (
        "Generate a structured trade plan based on market data, "
        "strategy rules, and recent trade history. "
        "You MUST call this tool with your analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "NSE stock symbol in caps, e.g. MFSL, RELIANCE",
            },
            "side": {
                "type": "string",
                "enum": ["BUY", "SELL"],
                "description": "Trade direction — BUY or SELL",
            },
            "entry_price": {
                "type": "number",
                "description": "Planned entry price in INR",
            },
            "stop_loss": {
                "type": "number",
                "description": "Stop loss price in INR. Below entry for BUY, above for SELL.",
            },
            "target": {
                "type": "number",
                "description": "Target/take-profit price in INR. Above entry for BUY, below for SELL.",
            },
            "quantity": {
                "type": "integer",
                "description": "Number of shares. Must be >= 1.",
                "minimum": 1,
            },
            "reasoning": {
                "type": "string",
                "description": "Why this trade — 2-3 sentences max, based on price action analysis.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in this trade, 0.0 to 1.0. Below 0.6 means don't trade.",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [
            "symbol", "side", "entry_price", "stop_loss",
            "target", "quantity", "reasoning", "confidence",
        ],
    },
}


# ──────────────────────────────────────────────
# System prompt for the planner
# ──────────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = """You are a professional Indian stock market trader with 25 years of experience.
You analyze intraday market data and produce precise, actionable trade plans.

RULES:
1. Every trade must have a clear entry, stop_loss, and target.
2. Stop loss MUST be on the correct side (below entry for BUY, above for SELL).
3. Target MUST be on the correct side (above entry for BUY, below for SELL).
4. Risk:Reward ratio should be at least 1.5:1.
5. Quantity should be reasonable given the price (think position sizing for 500,000 INR capital).
6. Reasoning should be 2-3 sentences explaining WHY — not just restating numbers.
7. Confidence between 0.0 and 1.0. Be honest. Below 0.6 means "don't trade."
8. If the data doesn't support a trade, set confidence to 0.0 and explain why.
9. Consider the recent trade history — avoid repeating losing patterns.

You analyze based on:
- Price action (new highs, new lows, range expansion/compression)
- Intraday structure (open-high/open-low candles, doji patterns)
- Pivot points and key levels
- Recent trade performance on this symbol
- Active strategy rules provided in context
"""


def build_planner_prompt(
    user_intent: str,
    market_data_summary: str,
    rag_context: str,
) -> str:
    """Build the user message for the planner LLM."""
    return f"""USER REQUEST: {user_intent}

{market_data_summary}

---

{rag_context}

---

Based on the above market data and context, produce a trade plan.
If conditions are not favorable, set confidence to 0.0 and explain in reasoning."""


def _build_cli_prompt(
    user_intent: str,
    market_data_summary: str,
    rag_context: str,
) -> str:
    """
    Build a single prompt for Claude CLI that forces JSON output.
    CLI mode doesn't support tool_use, so we use a strict JSON instruction.
    """
    return f"""{PLANNER_SYSTEM_PROMPT}

{build_planner_prompt(user_intent, market_data_summary, rag_context)}

OUTPUT INSTRUCTIONS:
You MUST output ONLY a valid JSON object with exactly these keys:
- "symbol" (string, e.g. "RELIANCE")
- "side" (string, "BUY" or "SELL")
- "entry_price" (number, in INR)
- "stop_loss" (number, in INR)
- "target" (number, in INR)
- "quantity" (integer, >= 1)
- "reasoning" (string, 2-3 sentences)
- "confidence" (number, 0.0 to 1.0)

Output ONLY the JSON object. No markdown fences, no explanation, no extra text."""


def _audit_log(event_type: str, symbol: str = "", prompt: str = "",
               response: str = "", model_name: str = "",
               latency_ms: int = None, trade_journal=None):
    """Write an audit log entry. Fails silently — never blocks trading."""
    try:
        from trading.models import AuditLog
        AuditLog.objects.create(
            event_type=event_type,
            symbol=symbol,
            prompt=prompt[:10000],
            response=response[:10000],
            model_name=model_name,
            latency_ms=latency_ms,
            trade_journal=trade_journal,
        )
    except Exception as e:
        logger.warning(f"Audit log write failed (non-fatal): {e}")


def _parse_json_response(raw: str) -> dict:
    """
    Parse JSON from LLM response. Handles markdown fences and extra text.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object from text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# ──────────────────────────────────────────────
# CLI Mode: Uses Claude Code CLI (Max plan, no API credits)
# ──────────────────────────────────────────────
def _run_planner_cli(
    user_intent: str,
    market_data_summary: str,
    rag_context: str,
) -> dict:
    """
    Run planner via Claude Code CLI (`claude --print`).
    Uses your Max plan — no API credits consumed.
    """
    model_name = "claude-cli (Max plan)"

    # Find the claude CLI binary.
    # Priority: CLAUDE_CLI_PATH env var > shutil.which > common paths > login shell
    logger.info("=== Claude CLI Discovery ===")

    claude_path = os.getenv("CLAUDE_CLI_PATH", "").strip()
    if claude_path:
        exists = os.path.isfile(claude_path)
        exe = os.access(claude_path, os.X_OK) if exists else False
        logger.info(f"  CLAUDE_CLI_PATH={claude_path} | exists={exists} | executable={exe}")
        if not (exists and exe):
            claude_path = ""

    if not claude_path:
        claude_path = shutil.which("claude") or ""
        logger.info(f"  shutil.which('claude') = {claude_path or 'NOT FOUND'}")

    if not claude_path:
        home = os.path.expanduser("~")
        logger.info(f"  HOME={home}")
        import glob as _glob
        candidates = [
            "/usr/local/bin/claude",                          # Linux / Homebrew Intel
            "/opt/homebrew/bin/claude",                       # macOS ARM Homebrew
            f"{home}/.claude/local/claude",                   # Claude Code local install
            f"{home}/.npm-global/bin/claude",                 # npm global (custom prefix)
            f"{home}/.nvm/versions/node/*/bin/claude",        # nvm installs
        ]
        expanded = []
        for c in candidates:
            expanded.extend(_glob.glob(c)) if "*" in c else expanded.append(c)
        for fallback in expanded:
            exists = os.path.isfile(fallback)
            exe = os.access(fallback, os.X_OK) if exists else False
            logger.info(f"  Checking {fallback}: exists={exists} executable={exe}")
            if exists and exe:
                claude_path = fallback
                break

    if not claude_path:
        # Last resort: ask a login shell (picks up full PATH from .zshrc/.bashrc)
        for shell_cmd in [
            ["zsh", "-lc", "which claude"],
            ["bash", "-lc", "which claude"],
        ]:
            try:
                which_result = subprocess.run(
                    shell_cmd, capture_output=True, text=True, timeout=5,
                )
                found = which_result.stdout.strip() if which_result.returncode == 0 else ""
                logger.info(f"  {' '.join(shell_cmd)} => '{found}' (rc={which_result.returncode})")
                if found:
                    claude_path = found
                    break
            except Exception as e:
                logger.info(f"  {' '.join(shell_cmd)} => EXCEPTION: {e}")
                continue

    logger.info(f"  FINAL claude_path = {claude_path or 'NOT FOUND'}")
    logger.info(f"  System PATH = {os.environ.get('PATH', 'NOT SET')}")
    logger.info("=== End CLI Discovery ===")

    if not claude_path:
        return {
            "error": (
                "Claude CLI not found. Either:\n"
                "  1. Set CLAUDE_CLI_PATH in .env to the full path "
                "(run 'which claude' in your terminal to find it)\n"
                "  2. Or switch to PLANNER_MODE=api with a valid ANTHROPIC_API_KEY"
            )
        }

    # Build prompt
    prompt = _build_cli_prompt(user_intent, market_data_summary, rag_context)
    full_prompt_for_audit = prompt

    _audit_log("PLANNER_REQ", symbol="", prompt=full_prompt_for_audit, model_name=model_name)

    logger.info(f"Planner (CLI) invoked | intent='{user_intent[:80]}'")

    try:
        t0 = time.time()

        # Call Claude CLI with --print for non-interactive output.
        # Unset CLAUDECODE to bypass the nested-session check (when running
        # inside a Claude Code session like Cowork). Keep all other vars.
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        import tempfile
        tmp_path = None
        try:
            # Write prompt to temp file for reliable delivery
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False,
                dir=tempfile.gettempdir(),
            ) as tmp:
                tmp.write(prompt)
                tmp_path = tmp.name

            # Method 1: shell redirect (works in nested Claude Code sessions)
            result = subprocess.run(
                f'"{claude_path}" --print < "{tmp_path}"',
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                shell=True,
            )

            # If Method 1 fails (e.g. "Not logged in"), try Method 2: direct pipe
            if result.returncode != 0 or "Not logged in" in result.stdout:
                logger.warning(
                    f"CLI shell redirect failed (rc={result.returncode}, "
                    f"out={result.stdout[:100]}), trying direct pipe..."
                )
                result = subprocess.run(
                    [claude_path, "--print"],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        latency_ms = int((time.time() - t0) * 1000)

        if result.returncode != 0:
            error_msg = f"Claude CLI failed (exit {result.returncode}): {result.stderr[:200]}"
            logger.error(error_msg)
            _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                        response=error_msg, model_name=model_name, latency_ms=latency_ms)
            return {"error": error_msg}

        raw_output = result.stdout.strip()

        if not raw_output:
            error_msg = "Claude CLI returned empty response"
            logger.error(error_msg)
            _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                        response=error_msg, model_name=model_name, latency_ms=latency_ms)
            return {"error": error_msg}

        # Parse JSON from response
        plan_dict = _parse_json_response(raw_output)

        # Validate required fields
        required = ["symbol", "side", "entry_price", "stop_loss", "target", "quantity", "reasoning", "confidence"]
        missing = [f for f in required if f not in plan_dict]
        if missing:
            error_msg = f"Trade plan missing required fields: {missing}"
            logger.error(error_msg)
            _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                        response=raw_output, model_name=model_name, latency_ms=latency_ms)
            return {"error": error_msg}

        # Ensure correct types
        plan_dict["entry_price"] = float(plan_dict["entry_price"])
        plan_dict["stop_loss"] = float(plan_dict["stop_loss"])
        plan_dict["target"] = float(plan_dict["target"])
        plan_dict["quantity"] = int(plan_dict["quantity"])
        plan_dict["confidence"] = float(plan_dict["confidence"])

        response_str = json.dumps(plan_dict, indent=2)

        logger.info(
            f"Planner (CLI) output: {plan_dict['side']} {plan_dict['quantity']}x {plan_dict['symbol']} "
            f"@ {plan_dict['entry_price']} | conf={plan_dict['confidence']} | {latency_ms}ms"
        )

        _audit_log("PLANNER_RES", symbol=plan_dict.get("symbol", ""),
                    response=response_str, model_name=model_name, latency_ms=latency_ms)

        return plan_dict

    except subprocess.TimeoutExpired:
        error_msg = "Claude CLI timed out (120s). Market may be volatile — try again."
        logger.error(error_msg)
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=error_msg, model_name=model_name)
        return {"error": error_msg}

    except ValueError as e:
        error_msg = f"Failed to parse planner JSON: {e}"
        logger.error(error_msg)
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=str(e), model_name=model_name)
        return {"error": error_msg}

    except Exception as e:
        logger.exception(f"Planner (CLI) failed: {e}")
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=str(e), model_name=model_name)
        return {"error": f"Planner CLI failed: {str(e)}"}


# ──────────────────────────────────────────────
# API Mode: Uses Anthropic SDK with tool_use (requires API credits)
# ──────────────────────────────────────────────
def _run_planner_api(
    user_intent: str,
    market_data_summary: str,
    rag_context: str,
    model: Optional[str] = None,
) -> dict:
    """
    Run planner via official Anthropic SDK with tool_use.
    Requires API credits (ANTHROPIC_API_KEY).

    Docs: https://platform.claude.com/docs/en/api/messages
    """
    import anthropic

    model_name = model or os.getenv("LLM_MODEL", "claude-opus-4-5-20251101")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set in environment"}

    user_msg = build_planner_prompt(user_intent, market_data_summary, rag_context)
    full_prompt_for_audit = f"[SYSTEM]\n{PLANNER_SYSTEM_PROMPT}\n\n[USER]\n{user_msg}"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Planner (API) invoked | model={model_name} | intent='{user_intent[:80]}'")

        t0 = time.time()

        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.2,
            system=PLANNER_SYSTEM_PROMPT,
            tools=[TRADE_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "generate_trade_plan"},
            messages=[{"role": "user", "content": user_msg}],
        )

        latency_ms = int((time.time() - t0) * 1000)

        plan_dict = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "generate_trade_plan":
                plan_dict = block.input
                break

        if not plan_dict:
            error_msg = "LLM did not produce a tool_use response with trade plan"
            logger.error(error_msg)
            _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                        response=error_msg, model_name=model_name, latency_ms=latency_ms)
            return {"error": error_msg}

        required = ["symbol", "side", "entry_price", "stop_loss", "target", "quantity", "reasoning", "confidence"]
        missing = [f for f in required if f not in plan_dict]
        if missing:
            error_msg = f"Trade plan missing required fields: {missing}"
            logger.error(error_msg)
            _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                        response=error_msg, model_name=model_name, latency_ms=latency_ms)
            return {"error": error_msg}

        response_str = json.dumps(plan_dict, indent=2)

        logger.info(
            f"Planner (API) output: {plan_dict['side']} {plan_dict['quantity']}x {plan_dict['symbol']} "
            f"@ {plan_dict['entry_price']} | conf={plan_dict['confidence']} | {latency_ms}ms"
        )

        _audit_log("PLANNER_REQ", symbol=plan_dict.get("symbol", ""),
                    prompt=full_prompt_for_audit, model_name=model_name)
        _audit_log("PLANNER_RES", symbol=plan_dict.get("symbol", ""),
                    response=response_str, model_name=model_name, latency_ms=latency_ms)

        usage = response.usage
        logger.info(
            f"Token usage: input={usage.input_tokens} output={usage.output_tokens} "
            f"total={usage.input_tokens + usage.output_tokens}"
        )

        return plan_dict

    except anthropic.AuthenticationError as e:
        error_msg = f"Anthropic auth failed: {e}. Check ANTHROPIC_API_KEY."
        logger.error(error_msg)
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=str(e), model_name=model_name)
        return {"error": error_msg}

    except anthropic.BadRequestError as e:
        error_str = str(e)
        if "credit balance" in error_str.lower():
            error_msg = "Anthropic API: Insufficient credits. Use PLANNER_MODE=cli or top up at console.anthropic.com"
        else:
            error_msg = f"Anthropic bad request: {e}"
        logger.error(error_msg)
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=error_str, model_name=model_name)
        return {"error": error_msg}

    except anthropic.RateLimitError as e:
        error_msg = f"Rate limit hit: {e}. Try again shortly."
        logger.error(error_msg)
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=str(e), model_name=model_name)
        return {"error": error_msg}

    except anthropic.APIError as e:
        error_msg = f"Anthropic API error: {e}"
        logger.error(error_msg)
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=str(e), model_name=model_name)
        return {"error": error_msg}

    except Exception as e:
        logger.exception(f"Planner (API) failed: {e}")
        _audit_log("PLANNER_ERR", prompt=full_prompt_for_audit,
                    response=str(e), model_name=model_name)
        return {"error": f"Planner API failed: {str(e)}"}


# ──────────────────────────────────────────────
# Main entry point: routes to CLI or API mode
# ──────────────────────────────────────────────
def run_planner(
    user_intent: str,
    market_data_summary: str,
    rag_context: str,
    model: Optional[str] = None,
) -> dict:
    """
    Run the planner agent.

    Mode is controlled by PLANNER_MODE env var:
      - "cli" (default): Uses Claude Code CLI with your Max plan. Free, no API credits.
      - "api": Uses official Anthropic SDK. Requires API credits.

    Returns:
        dict with TradePlan fields, or {"error": "..."} on failure.
    """
    mode = os.getenv("PLANNER_MODE", "cli").lower()

    if mode == "api":
        logger.info("Planner mode: API (Anthropic SDK)")
        return _run_planner_api(user_intent, market_data_summary, rag_context, model)
    else:
        logger.info("Planner mode: CLI (Claude Code, Max plan)")
        return _run_planner_cli(user_intent, market_data_summary, rag_context)
