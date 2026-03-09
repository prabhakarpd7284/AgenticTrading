"""
Shared state definition for the LangGraph trading workflow.

This state flows through every node in the graph:
    fetch_data -> retrieve_context -> planner -> risk -> execute -> journal
"""
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Structured output: TradePlan
# ──────────────────────────────────────────────
class TradePlan(BaseModel):
    """
    Strict schema for planner output.
    The LLM MUST produce this — no free-text trading decisions.
    """
    symbol: str = Field(description="NSE stock symbol in caps, e.g. MFSL")
    side: Literal["BUY", "SELL"] = Field(description="Trade direction")
    entry_price: float = Field(description="Planned entry price")
    stop_loss: float = Field(description="Stop loss price")
    target: float = Field(description="Target/take-profit price")
    quantity: int = Field(description="Number of shares", ge=1)
    reasoning: str = Field(description="Why this trade — 2-3 sentences max")
    confidence: float = Field(
        description="Confidence in this trade, 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )


# ──────────────────────────────────────────────
# Risk validation result
# ──────────────────────────────────────────────
class RiskResult(BaseModel):
    approved: bool
    reason: str
    risk_amount: float = 0.0
    risk_pct_of_capital: float = 0.0


# ──────────────────────────────────────────────
# Execution result
# ──────────────────────────────────────────────
class ExecutionResult(BaseModel):
    success: bool
    order_id: str = ""
    fill_price: float = 0.0
    fill_quantity: int = 0
    message: str = ""
    mode: Literal["paper", "live"] = "paper"


# ──────────────────────────────────────────────
# Graph state (flows through all nodes)
# ──────────────────────────────────────────────
class TradingState(TypedDict):
    """Full state passed between graph nodes."""

    # User input
    user_intent: str

    # Market data (from data_service)
    symbol: str
    market_data: Optional[dict]       # enriched OHLCV summary
    market_data_raw: Optional[str]    # stringified for LLM context

    # RAG context (from retriever)
    rag_context: Optional[str]        # recent trades + strategy rules

    # Planner output
    trade_plan: Optional[dict]        # serialized TradePlan
    planner_raw: Optional[str]        # raw LLM output for debugging

    # Risk validation
    risk_approved: Optional[bool]
    risk_result: Optional[dict]       # serialized RiskResult

    # Execution
    execution_result: Optional[dict]  # serialized ExecutionResult

    # Journal
    journal_id: Optional[int]         # TradeJournal.id after save

    # Error handling
    error: Optional[str]
