"""
Straddle management state — flows through every node in the straddle graph.

StraddleState:   LangGraph TypedDict (mutable across nodes)
StraddleAction:  Pydantic schema — structured LLM output (what to DO)
StraddleAnalysis: Pydantic schema — pure-Python computed analysis (what IS)
"""
from typing import TypedDict, Optional, Literal, List
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Structured LLM output: what action to take
# ──────────────────────────────────────────────
class StraddleAction(BaseModel):
    """
    Strict schema for straddle management agent output.
    Every field maps to an executable decision or a human-readable explanation.
    The LLM MUST produce this — no free-text management decisions.
    """

    action: Literal[
        "HOLD",           # Do nothing — let theta work
        "CLOSE_BOTH",     # Buy back CE + PE immediately
        "CLOSE_CE",       # Buy back CE only (CE is at risk or already worthless)
        "CLOSE_PE",       # Buy back PE only (PE is tested / dangerous)
        "HEDGE_FUTURES",  # Buy/sell NIFTY futures to neutralize delta
        "ROLL",           # Roll tested leg to different strike
    ] = Field(description="Primary management action")

    urgency: Literal[
        "IMMEDIATE",    # Execute within current candle
        "NEXT_CANDLE",  # Execute after next 5-min candle closes
        "MONITOR",      # No action yet — reassess next cycle
    ] = Field(description="How urgently to act")

    ce_action: Literal["CLOSE", "HOLD"] = Field(
        description="What to do with the CE leg"
    )

    pe_action: Literal["CLOSE", "HOLD", "TRAIL"] = Field(
        description="What to do with the PE leg"
    )

    pe_stop_loss: Optional[float] = Field(
        default=None,
        description="If trailing PE: close PE if price rises ABOVE this level (INR). None if closing immediately.",
    )

    pe_target: Optional[float] = Field(
        default=None,
        description="Close PE for profit if price falls BELOW this level (INR). None if no target set.",
    )

    hedge_side: Optional[Literal["BUY", "SELL", "NONE"]] = Field(
        default="NONE",
        description="Futures hedge direction. BUY to hedge short PE (NIFTY bearish). SELL to hedge short CE (NIFTY bullish).",
    )

    hedge_lots: int = Field(
        default=0,
        description="Number of NIFTY futures lots for hedging. Usually 1.",
        ge=0,
    )

    reasoning: str = Field(
        description="2-3 sentences: WHY this action, grounded in current P&L, delta, VIX, and market phase."
    )

    confidence: float = Field(
        description="0.0 to 1.0. Below 0.6 = monitor only, don't act.",
        ge=0.0,
        le=1.0,
    )

    key_risk: str = Field(
        description="Single sentence: what could go wrong if this action is taken (or not taken)."
    )


# ──────────────────────────────────────────────
# Pure-Python computed analysis (no LLM)
# ──────────────────────────────────────────────
class ExpiryScenario(BaseModel):
    """P&L at a specific NIFTY expiry level."""
    label: str
    nifty_level: float
    ce_expiry_value: float
    pe_expiry_value: float
    net_pnl_inr: float


class StraddleAnalysis(BaseModel):
    """
    All computed analytics for a straddle position.
    Produced by analyzer.py — pure Python, no LLM.
    Passed as context to the LLM planner.
    """

    # Live market
    nifty_spot: float
    nifty_prev_close: float
    nifty_gap_pts: float
    nifty_gap_pct: float
    vix_current: float
    vix_prev_close: float
    vix_change_pct: float
    vix_phase: Literal["CALM", "ELEVATED", "SPIKE"]  # <15 / 15-22 / >22

    # Option legs (current live prices)
    ce_ltp: float
    pe_ltp: float
    ce_sell_price: float
    pe_sell_price: float

    # P&L
    combined_sold: float
    combined_current: float
    net_pnl_pts: float
    net_pnl_inr: float
    premium_decayed_pct: float  # % of total premium already captured

    # Delta exposure
    ce_delta: float      # approximate (negative for short call)
    pe_delta: float      # approximate (positive for short put)
    net_delta: float     # net position delta
    delta_bias: Literal["LONG", "SHORT", "NEUTRAL"]

    # Risk flags
    is_underwater: bool         # combined_current > combined_sold
    stop_triggered: bool        # combined > 1x sold (hard stop level)
    expiry_tomorrow: bool       # days_to_expiry <= 1
    days_to_expiry: int

    # Market phase
    market_phase: Literal[
        "CRASH",        # NIFTY down >1.5% + new lows, VIX spiking
        "CHOP",         # Range-bound, no clear direction
        "RECOVERY",     # Bouncing from lows, VIX falling
        "RALLY",        # NIFTY up >0.5% from low, approaching strike
        "CLOSE",        # After 2:45 PM — intraday close protocol
    ]

    # Key levels
    pe_itm_by: float     # how many pts PE is ITM (> 0 means PE is ITM)
    ce_itm_by: float     # how many pts CE is ITM (> 0 means CE is ITM)
    nearest_itm_leg: Literal["CE", "PE", "BOTH_OTM"]

    # Expiry scenarios
    scenarios: List[ExpiryScenario]

    # Summary for LLM context
    summary_text: str


# ──────────────────────────────────────────────
# Validation result (deterministic, no LLM)
# ──────────────────────────────────────────────
class ActionValidation(BaseModel):
    approved: bool
    reason: str
    override_action: Optional[str] = None  # If validator overrides LLM action


# ──────────────────────────────────────────────
# Execution result (option orders)
# ──────────────────────────────────────────────
class StraddleExecutionResult(BaseModel):
    success: bool
    actions_taken: List[str] = []   # ["CLOSED_CE @ 96.40", "CLOSED_PE @ 280.00"]
    orders: List[dict] = []          # raw order responses
    message: str = ""
    mode: Literal["paper", "live"] = "paper"


# ──────────────────────────────────────────────
# Graph state (flows through all straddle nodes)
# ──────────────────────────────────────────────
class StraddleState(TypedDict):
    """Full state passed between straddle graph nodes."""

    # Position identity
    position_id: Optional[int]          # StraddlePosition.id from DB
    underlying: str                     # "NIFTY"
    strike: int                         # 24200
    expiry: str                         # "2026-03-10"
    lot_size: int                       # 65 for NIFTY

    # Leg tokens (Angel One NFO tokens)
    ce_symbol: str                      # "NIFTY10MAR2624200CE"
    ce_token: str                       # "45482"
    pe_symbol: str                      # "NIFTY10MAR2624200PE"
    pe_token: str                       # "45483"

    # Original sell prices (set when position registered)
    ce_sell_price: float
    pe_sell_price: float
    lots: int

    # Live market data (fetched each cycle)
    nifty_candles: Optional[list]       # 5-min candles from BrokerClient
    market_snapshot: Optional[dict]     # {nifty_spot, vix, ce_ltp, pe_ltp, ...}

    # Computed analysis (from analyzer.py)
    analysis: Optional[dict]            # serialized StraddleAnalysis

    # LLM recommendation
    recommended_action: Optional[dict]  # serialized StraddleAction
    planner_raw: Optional[str]          # raw LLM output for debugging

    # Validation
    action_approved: Optional[bool]
    validation_result: Optional[dict]   # serialized ActionValidation

    # Execution
    execution_result: Optional[dict]    # serialized StraddleExecutionResult

    # Journal
    journal_id: Optional[int]           # StraddlePosition.id after update

    # Error handling
    error: Optional[str]
