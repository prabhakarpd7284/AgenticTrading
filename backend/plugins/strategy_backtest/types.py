"""Shared types for the backtester engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Protocol


class PnLMode(Enum):
    """How P&L is denominated."""
    POINTS = "points"    # Raw price difference (screener)
    RUPEES = "rupees"    # qty × price diff (swing/intraday)


class TimeframeMode(Enum):
    """How the engine progresses through time."""
    DAILY = "daily"         # Walk-forward on daily bars
    INTRADAY = "intraday"   # Bar-by-bar replay


class TradeSide(Enum):
    BUY = "BUY"
    SHORT = "SHORT"


class TradeState(Enum):
    PENDING = auto()
    OPEN = auto()
    PARTIAL = auto()     # Partial exit done, remainder running
    CLOSED = auto()


# ──────────────────────────────────────────────
# Bar — normalized candle used throughout engine
# ──────────────────────────────────────────────

@dataclass(slots=True)
class Bar:
    """Single OHLCV candle. The universal data unit for the engine."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    @staticmethod
    def from_dict(d: dict) -> Bar:
        """Convert a candle dict (from DataService) to a Bar."""
        return Bar(
            timestamp=d.get("timestamp", d.get("date", "")),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=int(d.get("volume", 0)),
        )


# ──────────────────────────────────────────────
# Signals — strategy-agnostic entry/exit
# ──────────────────────────────────────────────

@dataclass
class EntrySignal:
    """Strategy-agnostic trade entry signal.

    Any entry detector (OK cycles, screener, EMA bounce, future strategies)
    produces EntrySignals. The engine doesn't know what strategy created it.
    """
    symbol: str
    timestamp: str
    side: TradeSide
    entry_price: float
    stoploss: float
    target: float
    risk_points: float
    atr: float = 0.0

    # Strategy metadata — carried through to Trade and stats breakdown
    metadata: dict = field(default_factory=dict)
    # Expected keys: trade_type (str), phase (str), strategy (str),
    #                confidence (float), trend (str)


@dataclass
class ExitSignal:
    """Returned by ExitCheck when an exit triggers."""
    price: float
    reason: str
    partial: bool = False        # True = partial exit (book some qty, keep rest)
    partial_fraction: float = 0.5  # Fraction to exit (0.5 = 50%)
