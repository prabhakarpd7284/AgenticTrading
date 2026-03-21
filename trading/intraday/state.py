"""
Intraday Agent State — all typed state flowing through the LangGraph pipeline.

Goal: Make money through proven price structures on liquid NSE stocks.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SetupType(str, Enum):
    """Proven intraday price structures that have statistical edge."""
    ORB_LONG = "ORB_LONG"           # Opening Range Breakout — long
    ORB_SHORT = "ORB_SHORT"         # Opening Range Breakout — short
    PDH_BREAK = "PDH_BREAK"         # Previous Day High breakout
    PDL_BREAK = "PDL_BREAK"         # Previous Day Low breakdown
    GAP_AND_GO = "GAP_AND_GO"       # Gap holds, trade in gap direction
    GAP_FILL = "GAP_FILL"           # Gap reverting, trade the fill
    VWAP_RECLAIM = "VWAP_RECLAIM"   # Price reclaims VWAP from below → long
    VWAP_REJECT = "VWAP_REJECT"     # Price rejects at VWAP from below → short
    LEVEL_BOUNCE_LONG = "LEVEL_BOUNCE_LONG"    # Bounce off support level → long
    LEVEL_BOUNCE_SHORT = "LEVEL_BOUNCE_SHORT"  # Rejection at resistance level → short


class TradeBias(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Phase(str, Enum):
    PREMARKET = "PREMARKET"       # Before 9:15 — scan + build watchlist
    OPENING = "OPENING"           # 9:15–9:30 — capture opening range
    ACTIVE = "ACTIVE"             # 9:30–15:00 — monitor + trade
    CLOSING = "CLOSING"           # 15:00–15:15 — close open positions
    POSTMARKET = "POSTMARKET"     # After 15:15 — journal + report


@dataclass
class StockSetup:
    """A scored stock with identified potential setup."""
    symbol: str
    token: str
    bias: TradeBias
    score: float                          # 0–100, higher = stronger setup
    setups: List[SetupType] = field(default_factory=list)

    # Previous day stats
    prev_open: float = 0.0
    prev_high: float = 0.0
    prev_low: float = 0.0
    prev_close: float = 0.0
    prev_volume: int = 0
    prev_atr: float = 0.0                # Average True Range (5-day)

    # Premarket analysis
    gap_pct: float = 0.0                  # Gap from prev close to today open
    nr_days: int = 0                      # Narrow Range streak (NR4=4, NR7=7)
    near_pdh: bool = False                # Close near prev day high
    near_pdl: bool = False                # Close near prev day low

    # Today's live data (filled during market hours)
    today_open: float = 0.0
    orb_high: float = 0.0                 # Opening range (first 15 min) high
    orb_low: float = 0.0                  # Opening range (first 15 min) low
    vwap: float = 0.0
    current_price: float = 0.0
    current_volume: int = 0

    # Camarilla pivot levels (computed from prev day OHLC — zero API cost)
    pivot_s3: float = 0.0                 # Buy zone
    pivot_s4: float = 0.0                 # Strong support / trend break
    pivot_r3: float = 0.0                 # Sell zone
    pivot_r4: float = 0.0                 # Strong resistance / trend break
    pivot_p: float = 0.0                  # Central pivot

    # Indicator confluence (filled during live scan)
    confluence_score: int = 0             # 0-100
    confluence_bias: str = "NEUTRAL"      # LONG / SHORT / NEUTRAL

    reason: str = ""


@dataclass
class IntradaySignal:
    """A triggered trade signal from a price structure."""
    symbol: str
    token: str
    setup_type: SetupType
    bias: TradeBias
    entry_price: float
    stop_loss: float
    target: float
    risk_reward: float
    confidence: float                     # 0.0–1.0
    reason: str = ""
    timestamp: str = ""
    side: str = ""                        # "BUY" or "SELL" (derived from bias if empty)
    near_pivot: str = ""                  # nearest level name (e.g. "Cam_R3", "PDH")

    def __post_init__(self):
        if not self.side:
            self.side = "BUY" if self.bias == TradeBias.LONG else "SELL"


@dataclass
class IntradayState:
    """Full state for the intraday agent graph."""
    # Config
    trading_date: str = ""
    phase: Phase = Phase.PREMARKET
    capital: float = 500000.0
    max_positions: int = 3
    daily_loss: float = 0.0

    # Watchlist (from premarket scan)
    universe: List[str] = field(default_factory=list)   # All symbols to scan
    watchlist: List[StockSetup] = field(default_factory=list)  # Top picks

    # Live monitoring
    signals: List[IntradaySignal] = field(default_factory=list)
    open_positions: int = 0
    trades_today: List[Dict[str, Any]] = field(default_factory=list)

    # LLM analysis
    premarket_analysis: str = ""
    market_context: str = ""

    # Errors
    errors: List[str] = field(default_factory=list)
