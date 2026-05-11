"""Entry detectors — protocol + adapters for existing signal generators."""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from trading.backtester.types import Bar, EntrySignal, TradeSide


# ──────────────────────────────────────────────
# Protocol — any strategy implements this
# ──────────────────────────────────────────────

class EntryDetector(Protocol):
    """Protocol for entry signal generators.

    Any existing or future detector adapts to this interface.
    The engine calls detect() on each bar and opens trades from signals.
    """

    def detect(
        self,
        symbol: str,
        bars: List[Bar],
        bar_index: int,
        context: dict,
    ) -> List[EntrySignal]:
        """Return entry signals at the given bar index.

        Args:
            symbol: Stock symbol
            bars: All bars for this symbol (up to and including bar_index)
            bar_index: Current bar index to evaluate
            context: Strategy-specific context (weekly bars, pre-computed indicators, etc.)
        """
        ...


# ──────────────────────────────────────────────
# Adapter: OK Cycle (daily swing)
# ──────────────────────────────────────────────

class OKCycleAdapter:
    """Wraps CycleDetector.analyze() → EntrySignal."""

    def __init__(self, min_rr: float = 2.0, slippage_pct: float = 0.1):
        self.min_rr = min_rr
        self.slippage_pct = slippage_pct
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            from plugins.strategy_swing.ok_cycles import CycleDetector
            self._detector = CycleDetector()
        return self._detector

    def detect(
        self, symbol: str, bars: List[Bar], bar_index: int, context: dict,
    ) -> List[EntrySignal]:
        from plugins.strategy_swing.ok_cycles import BULLISH_ACTIONABLE, BEARISH_ACTIONABLE, CyclePhase

        detector = self._get_detector()
        daily_slice = [{"date": b.timestamp, "open": b.open, "high": b.high,
                        "low": b.low, "close": b.close, "volume": b.volume}
                       for b in bars[:bar_index + 1]]

        if len(daily_slice) < 60:
            return []

        weekly_slice = context.get(f"weekly_{symbol}", [])
        if not weekly_slice:
            from plugins.strategy_swing.ok_scanner import OKScanner
            scanner = OKScanner()
            weekly_slice = scanner._daily_to_weekly(daily_slice)

        result = detector.analyze(symbol, daily_slice, weekly_slice)

        is_buy = result.phase in BULLISH_ACTIONABLE
        is_short = result.phase in BEARISH_ACTIONABLE
        if not is_buy and not is_short:
            return []

        bar = bars[bar_index]
        side = TradeSide.BUY if is_buy else TradeSide.SHORT
        ema_mid = result.ema_mid

        if side == TradeSide.BUY:
            sl = ema_mid * 0.995
            risk = bar.close - sl
        else:
            sl = ema_mid * 1.005
            risk = sl - bar.close

        if risk <= 0:
            return []

        if side == TradeSide.BUY:
            target = bar.close + (risk * self.min_rr)
        else:
            target = bar.close - (risk * self.min_rr)

        return [EntrySignal(
            symbol=symbol,
            timestamp=bar.timestamp,
            side=side,
            entry_price=bar.close,
            stoploss=round(sl, 2),
            target=round(target, 2),
            risk_points=round(risk, 2),
            metadata={
                "trade_type": result.phase.value,
                "phase": result.phase.value,
                "trend_daily": result.trend_daily.value,
                "trend_weekly": result.trend_weekly.value,
                "aligned": result.aligned,
                "confidence": result.confidence,
            },
        )]


# ──────────────────────────────────────────────
# Adapter: OK Intraday (3m/5m/15m + RSI momentum)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Adapter: Screener (intraday strategies)
# ──────────────────────────────────────────────

class ScreenerEntryAdapter:
    """Wraps pre-collected screener Signal list → EntrySignal by timestamp.

    The screener generates signals by replaying 1m ticks through the live
    ScreenerEngine. This adapter takes those pre-generated signals and
    indexes them for the BacktestEngine to consume.

    Usage:
        # 1. Run screener replay to collect signals
        signals = run_screener_replay(symbols, from_date, to_date, strategies)
        # 2. Feed to adapter
        adapter = ScreenerEntryAdapter(signals)
        # 3. Pass to engine
        engine = BacktestEngine(config, entry=adapter, exits=[...])
    """

    def __init__(self, signals: list = None):
        self._by_ts: Dict[str, List[EntrySignal]] = {}
        if signals:
            self.load_signals(signals)

    def load_signals(self, signals: list):
        """Convert screener Signal objects to EntrySignals indexed by timestamp."""
        for sig in signals:
            ts = str(sig.timestamp) if hasattr(sig, "timestamp") else ""
            entry = EntrySignal(
                symbol=sig.symbol,
                timestamp=ts,
                side=TradeSide.BUY if sig.side == "BUY" else TradeSide.SHORT,
                entry_price=sig.entry,
                stoploss=sig.stoploss,
                target=sig.target,
                risk_points=sig.risk_points,
                metadata={
                    "trade_type": sig.strategy,
                    "strategy": sig.strategy,
                    "confidence": sig.confidence,
                },
            )
            self._by_ts.setdefault(ts, []).append(entry)

    def detect(
        self, symbol: str, bars: List[Bar], bar_index: int, context: dict,
    ) -> List[EntrySignal]:
        ts = bars[bar_index].timestamp
        return [s for s in self._by_ts.get(ts, []) if s.symbol == symbol]


# ──────────────────────────────────────────────
# Adapter: OK Intraday (3m/5m/15m + RSI momentum)
# ──────────────────────────────────────────────

class IntradayCycleAdapter:
    """Wraps IntradayCycleDetector.scan() → EntrySignal list.

    Unlike the daily adapter which checks one bar at a time,
    this pre-computes all signals for a symbol and returns them
    indexed by timestamp for the engine to consume.
    """

    def __init__(self, sl_atr_mult: float = 1.5, min_rr: float = 2.0):
        self.sl_atr_mult = sl_atr_mult
        self.min_rr = min_rr
        self._cache: Dict[str, Dict[str, List[EntrySignal]]] = {}

    def precompute(self, symbol: str, candles: List[dict]) -> Dict[str, List[EntrySignal]]:
        """Pre-compute all signals for a symbol. Returns {timestamp: [signals]}."""
        from plugins.strategy_swing.ok_intraday import IntradayCycleDetector

        detector = IntradayCycleDetector(sl_atr_mult=self.sl_atr_mult)
        raw_signals = detector.scan(symbol, candles, min_rr=self.min_rr)

        by_ts: Dict[str, List[EntrySignal]] = {}
        for sig in raw_signals:
            entry = EntrySignal(
                symbol=sig.symbol,
                timestamp=sig.timestamp,
                side=TradeSide.BUY if sig.action == "BUY" else TradeSide.SHORT,
                entry_price=sig.close,
                stoploss=sig.sl,
                target=sig.target,
                risk_points=sig.risk_points,
                atr=sig.atr,
                metadata={
                    "trade_type": sig.trade_type or sig.phase.value,
                    "phase": sig.phase.value,
                    "rsi": sig.rsi,
                    "trend": sig.trend.value,
                },
            )
            by_ts.setdefault(sig.timestamp, []).append(entry)

        self._cache[symbol] = by_ts
        return by_ts

    def detect(
        self, symbol: str, bars: List[Bar], bar_index: int, context: dict,
    ) -> List[EntrySignal]:
        """Return pre-computed signals at the given bar's timestamp."""
        ts = bars[bar_index].timestamp
        by_ts = self._cache.get(symbol, {})
        return by_ts.get(ts, [])
