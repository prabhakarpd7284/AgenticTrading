"""
Candle Store — in-memory rolling buffers with tick-to-candle aggregation.

Builds 1m candles from live ticks, then auto-aggregates into 5m and 15m.
All higher timeframes are derived from 1m — single source of truth.

Memory budget: 50 symbols × 375 bars × 3 TFs ≈ 10MB.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Optional


@dataclass
class CandleBar:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    is_complete: bool = False

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


# Max bars per timeframe (full trading day)
_MAX_BARS = {"1m": 375, "5m": 78, "15m": 26}

# Aggregation periods (how many 1m bars per higher TF bar)
_AGG_PERIODS = {"5m": 5, "15m": 15}

# Minute boundaries for each timeframe
# 5m bars start at :15, :20, :25, ... (market opens 9:15)
# 15m bars start at :15, :30, :45, :00
_TF_MINUTES = {"5m": 5, "15m": 15}


class CandleStore:
    """
    Per-symbol multi-timeframe candle storage.

    Ingests ticks or raw 1m bars, maintains rolling buffers for 1m/5m/15m.
    Higher timeframes are aggregated from 1m automatically.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bars: dict[str, deque[CandleBar]] = {
            tf: deque(maxlen=max_bars) for tf, max_bars in _MAX_BARS.items()
        }
        self._current_1m: Optional[CandleBar] = None
        self._current_higher: dict[str, list[CandleBar]] = {"5m": [], "15m": []}

        # Previous day OHLC for pivot computation (set externally)
        self.prev_day_high: float = 0
        self.prev_day_low: float = 0
        self.prev_day_close: float = 0

        # Today's running VWAP components
        self._cum_tp_vol: float = 0.0
        self._cum_vol: int = 0

    @property
    def vwap(self) -> float:
        """Session VWAP from all 1m bars."""
        return round(self._cum_tp_vol / self._cum_vol, 2) if self._cum_vol > 0 else 0.0

    @property
    def ltp(self) -> float:
        """Last traded price (from forming or last completed bar)."""
        if self._current_1m:
            return self._current_1m.close
        bars = self.bars["1m"]
        return bars[-1].close if bars else 0.0

    def ingest_tick(self, ltp: float, volume: int = 0, timestamp: datetime = None) -> list[str]:
        """
        Process a tick and return list of timeframes that got a new completed bar.

        Returns e.g. ["1m"] or ["1m", "5m"] or ["1m", "5m", "15m"].
        """
        if timestamp is None:
            timestamp = datetime.now()

        completed_tfs = []
        current_minute = timestamp.replace(second=0, microsecond=0)

        # Should we close the current 1m bar and start a new one?
        if self._current_1m is None:
            self._current_1m = CandleBar(
                timestamp=current_minute, open=ltp, high=ltp, low=ltp, close=ltp,
                volume=volume,
            )
        elif current_minute > self._current_1m.timestamp:
            # Close current bar
            self._current_1m.is_complete = True
            self._close_1m_bar(self._current_1m)
            completed_tfs.append("1m")

            # Check higher timeframe aggregation
            completed_tfs += self._try_aggregate_higher(current_minute)

            # Start new bar
            self._current_1m = CandleBar(
                timestamp=current_minute, open=ltp, high=ltp, low=ltp, close=ltp,
                volume=volume,
            )
        else:
            # Update forming bar
            self._current_1m.high = max(self._current_1m.high, ltp)
            self._current_1m.low = min(self._current_1m.low, ltp)
            self._current_1m.close = ltp
            self._current_1m.volume += volume

        return completed_tfs

    def add_completed_bar(self, tf: str, bar: CandleBar):
        """Add a pre-built completed bar (for bootstrap from REST candles)."""
        bar.is_complete = True
        self.bars[tf].append(bar)

        # Update VWAP if 1m bar
        if tf == "1m":
            tp = (bar.high + bar.low + bar.close) / 3
            self._cum_tp_vol += tp * bar.volume
            self._cum_vol += bar.volume

    def seed_from_candles(self, raw_candles: list, tf: str = "1m"):
        """
        Seed store from REST API candles.

        raw_candles: list of [timestamp_str, open, high, low, close, volume]
        """
        for row in raw_candles:
            ts = row[0] if isinstance(row[0], datetime) else _parse_timestamp(row[0])
            bar = CandleBar(
                timestamp=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=int(row[5]) if len(row) > 5 else 0,
                is_complete=True,
            )
            self.add_completed_bar(tf, bar)

        # If seeding 1m bars, also build 5m and 15m from today's 1m
        # But preserve any pre-seeded higher TF bars (e.g., prev day warmup)
        if tf == "1m" and self.bars["1m"]:
            self._append_higher_from_1m()

    def get_closes(self, tf: str, n: int = 0) -> list[float]:
        """Get last N closing prices. 0 = all available."""
        bars = self.bars[tf]
        subset = list(bars)[-n:] if n else list(bars)
        return [b.close for b in subset]

    def get_bars_as_dicts(self, tf: str, n: int = 0) -> list[dict]:
        """Get last N bars as dicts (for indicator functions that expect dicts)."""
        bars = self.bars[tf]
        subset = list(bars)[-n:] if n else list(bars)
        return [b.as_dict() for b in subset]

    def get_last_bar(self, tf: str) -> Optional[CandleBar]:
        """Get the most recent completed bar for a timeframe."""
        bars = self.bars[tf]
        return bars[-1] if bars else None

    def bar_count(self, tf: str) -> int:
        return len(self.bars[tf])

    # ── Internal ──

    def _close_1m_bar(self, bar: CandleBar):
        """Finalize a 1m bar: store it, update VWAP."""
        self.bars["1m"].append(bar)
        tp = (bar.high + bar.low + bar.close) / 3
        self._cum_tp_vol += tp * bar.volume
        self._cum_vol += bar.volume

    def _try_aggregate_higher(self, new_minute: datetime) -> list[str]:
        """Check if any higher TF bar should close based on the new minute boundary."""
        completed = []
        for tf, period in _TF_MINUTES.items():
            # A higher TF bar closes when the new minute is on its boundary
            # Market opens at 9:15, so boundaries are relative to that
            minutes_since_open = (new_minute.hour * 60 + new_minute.minute) - (9 * 60 + 15)
            if minutes_since_open > 0 and minutes_since_open % period == 0:
                # Aggregate last N 1m bars into one higher TF bar
                n = period
                recent = list(self.bars["1m"])[-n:]
                if len(recent) >= n:
                    agg = _aggregate_bars(recent)
                    agg.timestamp = recent[0].timestamp
                    self.bars[tf].append(agg)
                    completed.append(tf)
        return completed

    def _append_higher_from_1m(self):
        """Build 5m/15m bars from 1m bars, appending to (not replacing) existing higher TF bars."""
        all_1m = list(self.bars["1m"])

        for tf, period in _TF_MINUTES.items():
            # Find the last timestamp already in this TF buffer
            existing = list(self.bars[tf])
            last_existing_ts = existing[-1].timestamp if existing else None

            i = 0
            while i + period <= len(all_1m):
                chunk = all_1m[i:i + period]
                first_min = chunk[0].timestamp.hour * 60 + chunk[0].timestamp.minute
                minutes_since_open = first_min - (9 * 60 + 15)
                if minutes_since_open >= 0 and minutes_since_open % period == 0:
                    agg = _aggregate_bars(chunk)
                    agg.timestamp = chunk[0].timestamp
                    # Only append if this bar is newer than what we already have
                    if last_existing_ts is None or agg.timestamp > last_existing_ts:
                        self.bars[tf].append(agg)
                        last_existing_ts = agg.timestamp
                    i += period
                else:
                    i += 1


def _aggregate_bars(bars: list[CandleBar]) -> CandleBar:
    """Combine multiple bars into one OHLCV bar."""
    return CandleBar(
        timestamp=bars[0].timestamp,
        open=bars[0].open,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=bars[-1].close,
        volume=sum(b.volume for b in bars),
        is_complete=True,
    )


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse Angel One timestamp format."""
    # "2026-03-20T09:15:00+05:30" → datetime
    clean = ts_str.replace("+05:30", "").replace("T", " ")
    try:
        return datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.strptime(clean[:16], "%Y-%m-%d %H:%M")
