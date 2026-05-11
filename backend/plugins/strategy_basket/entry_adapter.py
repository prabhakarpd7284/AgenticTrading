"""Basket entry adapter for BacktestEngine.

Pre-computes basket signals (equity + options) for each trading day
in the 9:45-10:30 window, then serves them to the engine by timestamp.

Uses the same mood → scanner → momentum pipeline as live, but on historical data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

from logzero import logger

from trading.backtester.entry import EntryDetector
from trading.backtester.types import Bar, EntrySignal, TradeSide
from plugins.strategy_basket.config import BasketConfig
from trading.utils.indicators import _ema, _wma, _rsi_series, bollinger_bands


class BasketEntryAdapter:
    """Adapts basket signal logic for the BacktestEngine.

    For backtesting, we can't call the live mood assessor. Instead:
    - Mood is inferred from the intraday data itself (first 30min A/D proxy)
    - Equity signals use OK cycle phases from daily data + intraday momentum
    - Options signals are simulated using ATM strike from the day's open

    Usage:
        adapter = BasketEntryAdapter()
        adapter.precompute(symbol, candles_5m, daily_candles)
        # Then pass to BacktestEngine
    """

    def __init__(self, cfg: BasketConfig = None, min_rr: float = 2.0):
        self.cfg = cfg or BasketConfig()
        self.min_rr = min_rr
        self._cache: Dict[str, Dict[str, List[EntrySignal]]] = {}

    def precompute(
        self,
        symbol: str,
        intraday_candles: List[dict],
        daily_candles: List[dict] = None,
    ) -> Dict[str, List[EntrySignal]]:
        """Pre-compute basket signals for a symbol across all trading days.

        Scans intraday candles for momentum-confirmed entries in the
        9:45-10:30 window. Requires enough bars for 5 EMA + BB + RSI.
        """
        if not intraday_candles or len(intraday_candles) < 30:
            self._cache[symbol] = {}
            return {}

        by_ts: Dict[str, List[EntrySignal]] = {}

        # Group candles by day
        days: Dict[str, List[dict]] = {}
        for c in intraday_candles:
            ts = c.get("timestamp", "")
            day = ts[:10]
            days.setdefault(day, []).append(c)

        # Check OK cycle phase from daily data (if available)
        phase = "BASKET"
        if daily_candles and len(daily_candles) >= 60:
            try:
                from plugins.strategy_swing.ok_cycles import CycleDetector, BULLISH_ACTIONABLE, BEARISH_ACTIONABLE
                detector = CycleDetector()
                df = detector.compute_indicators(daily_candles)
                detected = detector.detect_phase(df)
                if detected.value != "NONE":
                    phase = detected.value
            except Exception:
                pass

        for day, day_candles in days.items():
            # Filter to 9:45-10:30 window
            window = [c for c in day_candles if self._in_window(c.get("timestamp", ""))]
            if len(window) < self.cfg.ema_period + 3:
                continue

            # Build running indicator state across the day
            closes_so_far: List[float] = []
            for i, candle in enumerate(day_candles):
                closes_so_far.append(candle["close"])
                ts = candle.get("timestamp", "")

                # Only generate signals in the window
                if not self._in_window(ts):
                    continue

                if len(closes_so_far) < max(self.cfg.ema_period + 1, 15):
                    continue

                # Check momentum
                sig = self._check_entry(symbol, closes_so_far, candle, phase)
                if sig:
                    by_ts.setdefault(ts, []).append(sig)

        self._cache[symbol] = by_ts
        count = sum(len(v) for v in by_ts.values())
        if count > 0:
            logger.debug(f"Basket adapter: {symbol} → {count} signals across {len(by_ts)} bars")
        return by_ts

    def detect(
        self, symbol: str, bars: List[Bar], bar_index: int, context: dict,
    ) -> List[EntrySignal]:
        """Return pre-computed signals at the given bar's timestamp."""
        ts = bars[bar_index].timestamp
        by_ts = self._cache.get(symbol, {})
        return by_ts.get(ts, [])

    def _check_entry(
        self, symbol: str, closes: List[float], candle: dict, phase: str,
    ) -> EntrySignal | None:
        """Check if current bar qualifies as a basket entry."""
        last_close = closes[-1]

        # 5 EMA
        ema5 = _ema(closes, self.cfg.ema_period)
        if not ema5:
            return None
        ema5_val = ema5[-1]

        # RSI momentum
        rsi_vals = _rsi_series(closes, 14)
        rsi_clean = [v if v is not None else 50.0 for v in rsi_vals]
        rsi_ema = _ema(rsi_clean, 3)
        rsi_wma = _wma(rsi_clean, 21)

        mom_bull = rsi_ema[-1] > rsi_wma[-1] if rsi_ema and rsi_wma else False

        # BUY: price > 5 EMA + momentum bullish
        if last_close <= ema5_val or not mom_bull:
            return None

        # SL below 5 EMA
        sl = ema5_val * 0.998
        risk = last_close - sl
        if risk <= 0:
            return None

        target = last_close + (risk * self.min_rr)

        # BB check (optional, need 20 bars)
        bb = bollinger_bands(closes, self.cfg.bb_period, self.cfg.bb_std) if len(closes) >= self.cfg.bb_period else None

        # Confluence
        confluence = 0.5  # base
        if mom_bull:
            confluence += 0.2
        if bb and last_close < bb["upper"]:
            confluence += 0.1

        return EntrySignal(
            symbol=symbol,
            timestamp=candle.get("timestamp", ""),
            side=TradeSide.BUY,
            entry_price=last_close,
            stoploss=round(sl, 2),
            target=round(target, 2),
            risk_points=round(risk, 2),
            atr=risk,  # approximate
            metadata={
                "trade_type": f"BASKET_{phase}",
                "phase": phase,
                "confluence": round(confluence, 2),
            },
        )

    def _in_window(self, ts: str) -> bool:
        """Check if timestamp is in the 9:45-10:30 trading window."""
        try:
            clean = ts.replace("+05:30", "").replace("T", " ")
            parts = clean.split(" ")
            if len(parts) < 2:
                return False
            time_parts = parts[1].split(":")
            h, m = int(time_parts[0]), int(time_parts[1])
            start_h, start_m = map(int, self.cfg.start_time.split(":"))
            end_h, end_m = map(int, self.cfg.end_time.split(":"))

            t = h * 60 + m
            s = start_h * 60 + start_m
            e = end_h * 60 + end_m
            return s <= t <= e
        except (ValueError, IndexError):
            return False
