"""
Straddle Management Prompt — THE clean, reusable prompt template.

Used by the straddle management agent throughout the trading day.
Called every cycle (every 15-30 min or on a significant NIFTY move).

Design principles:
- Context-rich, not context-bloated
- Forces a specific action from a fixed set (not open-ended)
- Includes time-to-expiry, VIX phase, market phase, P&L, delta
- Structured JSON output only — no free-form text
- Works for both initial assessment and re-assessment during the day
"""

# ──────────────────────────────────────────────
# System prompt (static — defines the agent persona + rules)
# ──────────────────────────────────────────────
STRADDLE_SYSTEM_PROMPT = """You are a senior Indian options trader with 25 years of experience specializing in short straddle management on NIFTY 50.

Your job is to manage a live short straddle throughout the trading day and recommend one precise, actionable management decision per cycle.

CORE RULES:
1. A short straddle's enemy is a BIG directional move + VIX expansion. Its friend is time (theta) + range-bound price action.
2. The most dangerous moment is expiry day — gamma is explosive. Close before 3:15 PM IST on expiry.
3. NEVER let a position go beyond 2× the premium collected as a loss. That is the absolute stop.
4. If the combined premium is > what you sold it for (position underwater), act immediately.
5. HOLD only when: VIX is stable/falling, NIFTY is range-bound, premium has decayed significantly (>30%), and there is no expiry-day risk.
6. HEDGE (buy futures) only when: net delta > ±0.5 AND there is a clear directional bias in candles AND closing the leg is not better.
7. CLOSE_BOTH is always better than a complex multi-leg adjustment when expiry is tomorrow.
8. Confidence below 0.6 = MONITOR only. Do not recommend CLOSE or HEDGE if unsure.
9. Reasoning must be grounded in the specific numbers provided — not generic advice.
10. Key risk must identify the single most likely way this decision could go wrong.

ACTION MENU:
- HOLD: Monitor without action. Theta is working. No immediate danger.
- CLOSE_BOTH: Buy back both CE and PE. Lock profit or cut loss. Clean exit.
- CLOSE_CE: Buy back only the call (OTM, nearly worthless, or market rallying hard toward strike).
- CLOSE_PE: Buy back only the put (tested, deep ITM, or market crashing toward strike).
- HEDGE_FUTURES: Buy/sell NIFTY futures to neutralize delta. Use when delta bias is strong but position is still profitable overall.
- ROLL: Move tested leg to a further OTM strike. Only viable with > 3 days to expiry.

URGENCY LEVELS:
- IMMEDIATE: Act within the current 5-min candle. Do not wait.
- NEXT_CANDLE: Act after next candle closes with confirmation.
- MONITOR: No action — reassess in next cycle (15-30 min).
"""


# ──────────────────────────────────────────────
# Tool schema (for API mode forced structured output)
# ──────────────────────────────────────────────
STRADDLE_ACTION_TOOL = {
    "name": "recommend_straddle_action",
    "description": (
        "Recommend a specific straddle management action based on current P&L, "
        "delta, VIX, market phase, and time to expiry. "
        "You MUST call this tool with your recommendation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["HOLD", "CLOSE_BOTH", "CLOSE_CE", "CLOSE_PE", "HEDGE_FUTURES", "ROLL"],
                "description": "Primary management action to take",
            },
            "urgency": {
                "type": "string",
                "enum": ["IMMEDIATE", "NEXT_CANDLE", "MONITOR"],
                "description": "How urgently to execute this action",
            },
            "ce_action": {
                "type": "string",
                "enum": ["CLOSE", "HOLD"],
                "description": "What to do with the CE (call) leg",
            },
            "pe_action": {
                "type": "string",
                "enum": ["CLOSE", "HOLD", "TRAIL"],
                "description": "What to do with the PE (put) leg",
            },
            "pe_stop_loss": {
                "type": ["number", "null"],
                "description": "Close PE if its price rises ABOVE this (INR). Null if closing immediately.",
            },
            "pe_target": {
                "type": ["number", "null"],
                "description": "Close PE for profit if price falls BELOW this (INR). Null if no target.",
            },
            "hedge_side": {
                "type": "string",
                "enum": ["BUY", "SELL", "NONE"],
                "description": "Futures hedge direction. NONE if not hedging.",
            },
            "hedge_lots": {
                "type": "integer",
                "description": "Number of NIFTY futures lots for hedging. 0 if not hedging.",
                "minimum": 0,
            },
            "reasoning": {
                "type": "string",
                "description": "2-3 sentences: WHY this action, referencing specific P&L, delta, VIX, and candle data.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0 to 1.0. Below 0.6 means MONITOR only.",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "key_risk": {
                "type": "string",
                "description": "One sentence: the most likely way this decision goes wrong.",
            },
        },
        "required": [
            "action", "urgency", "ce_action", "pe_action",
            "reasoning", "confidence", "key_risk",
        ],
    },
}


# ──────────────────────────────────────────────
# User prompt builder (dynamic — built each cycle)
# ──────────────────────────────────────────────
def build_straddle_prompt(analysis_text: str, position_history: str = "") -> str:
    """
    Build the user message for the straddle management LLM.

    Args:
        analysis_text: Formatted string from StraddleAnalysis.summary_text
                       (market data, P&L, delta, scenarios, VIX phase, market phase)
        position_history: Optional string of previous management actions taken today
                          (from StraddlePosition.management_log)

    Returns:
        Complete user message ready to send to the LLM.
    """
    history_section = ""
    if position_history:
        history_section = f"""
MANAGEMENT ACTIONS TAKEN TODAY:
{position_history}

---
"""

    return f"""CURRENT POSITION STATE AND MARKET DATA:

{analysis_text}
{history_section}
---

Based on the above, recommend ONE management action for this straddle position.
Consider: current P&L, delta exposure, VIX phase, market phase, time to expiry, and any actions already taken today.

OUTPUT INSTRUCTIONS:
You MUST output ONLY a valid JSON object with exactly these keys:
- "action": one of HOLD | CLOSE_BOTH | CLOSE_CE | CLOSE_PE | HEDGE_FUTURES | ROLL
- "urgency": one of IMMEDIATE | NEXT_CANDLE | MONITOR
- "ce_action": one of CLOSE | HOLD
- "pe_action": one of CLOSE | HOLD | TRAIL
- "pe_stop_loss": number (INR) or null
- "pe_target": number (INR) or null
- "hedge_side": one of BUY | SELL | NONE
- "hedge_lots": integer (0 if not hedging)
- "reasoning": string (2-3 sentences, reference specific numbers)
- "confidence": number 0.0-1.0
- "key_risk": string (one sentence)

Output ONLY the JSON object. No markdown fences. No explanation. No extra text."""


# ──────────────────────────────────────────────
# CLI prompt builder (wraps system + user for Claude CLI mode)
# ──────────────────────────────────────────────
def build_cli_straddle_prompt(analysis_text: str, position_history: str = "") -> str:
    """Full prompt for Claude CLI --print mode (system + user in one block)."""
    return f"""{STRADDLE_SYSTEM_PROMPT}

{build_straddle_prompt(analysis_text, position_history)}"""


# ──────────────────────────────────────────────
# Required fields for validation
# ──────────────────────────────────────────────
REQUIRED_ACTION_FIELDS = [
    "action", "urgency", "ce_action", "pe_action",
    "reasoning", "confidence", "key_risk",
]
