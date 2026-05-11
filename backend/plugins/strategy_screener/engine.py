"""
Screener Engine — orchestrates data, indicators, conditions, and signal output.

This is the main loop that:
1. Receives ticks/bars from TickStream
2. Updates CandleStores
3. Recomputes indicators on bar close
4. Evaluates all strategies across all symbols
5. Emits Signals to registered output handlers (CLI, Telegram, DB)
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from logzero import logger

# ── Pivot cache — avoid re-fetching prev-day OHLC on restart ──
_PIVOT_CACHE_DIR = Path("/tmp/screener_cache")

from plugins.strategy_screener.candle_store import CandleStore
from plugins.strategy_screener.indicator_engine import IndicatorEngine
from plugins.strategy_screener.conditions import Condition
from plugins.strategy_screener.strategies import Strategy, STRATEGIES
from plugins.strategy_screener.signals import Signal


class ScreenerEngine:
    """
    Core screener loop.

    Usage:
        engine = ScreenerEngine(symbols=["RELIANCE", "TCS", ...])
        engine.add_output_handler(my_handler)  # called with each Signal
        engine.bootstrap()                      # seed candle stores from REST
        engine.start()                          # begin live processing
    """

    def __init__(
        self,
        symbols: list[str],
        strategies: list[Strategy] = None,
        capital: float = 500000,
        max_risk_pct: float = 1.0,
    ):
        self.symbols = symbols
        self.strategies = [s for s in (strategies or STRATEGIES) if s.enabled]
        self.capital = capital
        self.max_risk_pct = max_risk_pct

        # Per-symbol state (dedup symbols)
        self.symbols = list(dict.fromkeys(symbols))  # preserve order, remove dupes
        self.stores: dict[str, CandleStore] = {sym: CandleStore(sym) for sym in self.symbols}
        self.indicator_engine = IndicatorEngine()

        # Strategy cooldown tracking: (symbol, strategy_name) → last_signal_bar_index
        self._cooldowns: dict[tuple[str, str], int] = {}
        self._bar_counters: dict[str, int] = {}  # per-symbol bar counter

        # Global per-symbol cooldown — no stock fires more than 1 signal per 20 bars (~100min on 5m)
        self._symbol_last_signal: dict[str, int] = {}
        self.symbol_cooldown_bars: int = 20

        # Signal output handlers
        self._handlers: list[Callable[[Signal], None]] = []

        # Stats
        self.signals_emitted: int = 0
        self.bars_processed: int = 0
        self._start_time: Optional[float] = None

    def add_output_handler(self, handler: Callable[[Signal], None]):
        """Register a function to call when a signal fires."""
        self._handlers.append(handler)

    def bootstrap(self, fetch_candles_fn: Callable = None):
        """
        Seed candle stores with historical data from REST API.

        fetch_candles_fn: (symbol, token, start, end, interval) → list of raw candles
        """
        if fetch_candles_fn is None:
            return

        from trading.services.ticker_service import ticker_service
        from trading.services.data_service import BrokerClient

        broker = BrokerClient.get_instance()
        broker.ensure_login()

        from trading.utils.time_utils import (
            last_trading_day, can_fetch_candles, cap_end_time, is_market_open,
        )

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        prev_day = last_trading_day(now).isoformat()
        market_has_candles = can_fetch_candles(now)

        logger.info(
            f"Bootstrap: {len(self.symbols)} symbols | "
            f"market_open={is_market_open(now)} | candles_available={market_has_candles} | "
            f"prev_day={prev_day}"
        )

        # Step 1: Fetch prev-day OHLC via batch market data — with disk cache
        _PIVOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _PIVOT_CACHE_DIR / f"pivots_{prev_day}.json"
        pivot_data: dict[str, dict] = {}

        # Try cache first
        if cache_file.exists():
            try:
                pivot_data = json.loads(cache_file.read_text())
                logger.info(f"  Loaded {len(pivot_data)} pivots from cache ({cache_file.name})")
            except Exception:
                pivot_data = {}

        # Fetch missing symbols from API
        token_map = {}
        for symbol in self.symbols:
            if symbol in pivot_data:
                # Apply cached pivots
                pd = pivot_data[symbol]
                self.stores[symbol].prev_day_high = pd["high"]
                self.stores[symbol].prev_day_low = pd["low"]
                self.stores[symbol].prev_day_close = pd["close"]
                continue
            token = ticker_service.get_token(symbol)
            if token:
                token_map[token] = symbol

        if token_map:
            logger.info(f"  Fetching pivots for {len(token_map)} symbols via batch API...")
            all_tokens = list(token_map.keys())
            for i in range(0, len(all_tokens), 50):
                batch = all_tokens[i:i + 50]
                try:
                    fetched = broker.market_data_batch({"NSE": batch}, mode="OHLC")
                    for item in fetched:
                        tok = str(item.get("symbolToken", ""))
                        sym = token_map.get(tok)
                        if sym and sym in self.stores:
                            prev_close = float(item.get("close", 0))
                            high = float(item.get("high", 0))
                            low = float(item.get("low", 0))
                            if prev_close > 0:
                                h = high if high > 0 else prev_close * 1.01
                                l = low if low > 0 else prev_close * 0.99
                                self.stores[sym].prev_day_high = h
                                self.stores[sym].prev_day_low = l
                                self.stores[sym].prev_day_close = prev_close
                                pivot_data[sym] = {"high": h, "low": l, "close": prev_close}
                    logger.info(f"  Batch {i // 50 + 1}: {len(fetched)} symbols")
                except Exception as e:
                    logger.error(f"  Batch pivot fetch failed: {e}")

            # Save to cache
            try:
                cache_file.write_text(json.dumps(pivot_data))
                logger.info(f"  Cached {len(pivot_data)} pivots to {cache_file.name}")
            except Exception as e:
                logger.debug(f"  Cache write failed: {e}")

        # Step 2: Seed prev-day 5m candles (for indicator warmup) — with disk cache
        candle_cache_file = _PIVOT_CACHE_DIR / f"candles_5m_{prev_day}.json"
        cached_candles: dict[str, list] = {}

        if candle_cache_file.exists():
            try:
                cached_candles = json.loads(candle_cache_file.read_text())
                logger.info(f"  Loaded prev-day 5m candles from cache ({len(cached_candles)} symbols)")
            except Exception:
                cached_candles = {}

        symbols_to_fetch = [s for s in self.symbols if s not in cached_candles]
        if symbols_to_fetch:
            logger.info(f"  Fetching prev-day 5m candles for {len(symbols_to_fetch)} symbols...")
            for symbol in symbols_to_fetch:
                token = ticker_service.get_token(symbol)
                if not token:
                    continue
                try:
                    raw = broker.fetch_candles(
                        token, f"{prev_day} 09:15", f"{prev_day} 15:30", "FIVE_MINUTE"
                    )
                    if raw:
                        cached_candles[symbol] = raw
                except Exception as e:
                    logger.debug(f"  {symbol} 5m fetch failed: {e}")

            try:
                candle_cache_file.write_text(json.dumps(cached_candles))
                logger.info(f"  Cached {len(cached_candles)} symbols of 5m candles")
            except Exception:
                pass

        # Apply cached 5m candles to stores
        for symbol in self.symbols:
            if symbol in cached_candles:
                self.stores[symbol].seed_from_candles(cached_candles[symbol], "5m")

        # Step 3: Seed today's 1m candles if market has started
        if market_has_candles:
            seeded = 0
            for symbol in self.symbols:
                token = ticker_service.get_token(symbol)
                if not token:
                    continue
                try:
                    end_time = cap_end_time(today, now)
                    raw_1m = broker.fetch_candles(
                        token, f"{today} 09:15", end_time, "ONE_MINUTE"
                    )
                    if raw_1m:
                        self.stores[symbol].seed_from_candles(raw_1m, "1m")
                        seeded += 1
                except Exception as e:
                    logger.debug(f"  {symbol} 1m seed failed: {e}")
            logger.info(f"  Seeded {seeded}/{len(self.symbols)} with today's 1m candles")
        else:
            logger.info("  Pre-market: today's bars will come from websocket")

        # Step 3: Initialize indicators for seeded symbols
        for symbol in self.symbols:
            for tf in ["1m", "5m", "15m"]:
                if self.stores[symbol].bar_count(tf) > 0:
                    self.indicator_engine.update(symbol, tf, self.stores[symbol])

    def on_tick(self, symbol: str, ltp: float, volume: int = 0, timestamp: datetime = None):
        """
        Process a single tick for a symbol.

        Called by TickStream (websocket) or polling loop.
        """
        store = self.stores.get(symbol)
        if store is None:
            return

        completed_tfs = store.ingest_tick(ltp, volume, timestamp)

        # For each completed timeframe bar, update indicators and evaluate
        all_signals = []
        for tf in completed_tfs:
            self.bars_processed += 1
            self._bar_counters[symbol] = self._bar_counters.get(symbol, 0) + 1

            snap = self.indicator_engine.update(symbol, tf, store)
            all_signals.extend(self._evaluate_strategies(symbol, tf, store, timestamp))

        # Dedup: only emit the highest-confidence signal per symbol per tick
        # This prevents 3 strategies all firing on the same candle and overwhelming the trader
        if all_signals:
            # Global per-symbol cooldown — skip if this symbol signalled recently
            bar_idx = self._bar_counters.get(symbol, 0)
            last_sym_fire = self._symbol_last_signal.get(symbol, -999)
            if bar_idx - last_sym_fire < self.symbol_cooldown_bars:
                return

            best = max(all_signals, key=lambda s: (s.confidence, s.risk_reward))
            self.signals_emitted += 1
            self._symbol_last_signal[symbol] = bar_idx
            self._emit_signal(best)

    def on_batch_tick(self, ticks: list[dict]):
        """
        Process a batch of ticks from polling.

        ticks: [{"symbol": "RELIANCE", "ltp": 2850.5, "volume": 12345}, ...]
        """
        now = datetime.now()
        for tick in ticks:
            self.on_tick(
                symbol=tick["symbol"],
                ltp=tick["ltp"],
                volume=tick.get("volume", 0),
                timestamp=now,
            )

    def _evaluate_strategies(
        self, symbol: str, tf: str, store: CandleStore, now: datetime = None,
    ) -> list[Signal]:
        """Evaluate all strategies for a symbol when a bar closes."""
        if now is None:
            now = datetime.now()

        signals = []
        bar_idx = self._bar_counters.get(symbol, 0)

        for strategy in self.strategies:
            # Only evaluate if this timeframe is relevant to at least one condition
            if not any(c.timeframe == tf for c in strategy.conditions):
                continue

            # Check active window
            if not (strategy.active_window[0] <= now.time() <= strategy.active_window[1]):
                continue

            # Check cooldown
            last_fire = self._cooldowns.get((symbol, strategy.name), -999)
            if bar_idx - last_fire < strategy.cooldown_bars:
                continue

            # Evaluate ALL conditions
            all_met = True
            reasons = []

            for cond in strategy.conditions:
                snap = self.indicator_engine.get(symbol, cond.timeframe)
                prev_snap = self.indicator_engine.get_previous(symbol, cond.timeframe)

                if snap is None:
                    all_met = False
                    break

                bars = list(store.bars.get(cond.timeframe, []))
                if not cond.evaluate(snap, bars, prev_snap, now):
                    all_met = False
                    break

                reasons.append(cond.description)

            if not all_met:
                continue

            # All conditions met — build signal
            signal = self._build_signal(symbol, strategy, store, reasons, now)
            if signal and signal.risk_reward >= strategy.min_rr:
                signals.append(signal)
                self._cooldowns[(symbol, strategy.name)] = bar_idx

        return signals

    def _build_signal(
        self, symbol: str, strategy: Strategy, store: CandleStore,
        reasons: list[str], now: datetime,
    ) -> Optional[Signal]:
        """Compute entry, SL, target and build a Signal."""
        # Use 5m snapshot as primary (most conditions operate on 5m)
        snap = self.indicator_engine.get(symbol, "5m")
        if snap is None:
            snap = self.indicator_engine.get(symbol, "15m")
        if snap is None:
            return None

        side = strategy.side
        if side == "BOTH":
            # Determine side from bias
            side = "BUY" if snap.last_close > snap.vwap else "SELL"

        entry = strategy.entry_rule.compute(snap, store)
        sl = strategy.stoploss_rule.compute(snap, store, side, entry)
        target = strategy.target_rule.compute(snap, side, entry, sl)

        risk = abs(entry - sl)
        if risk == 0:
            return None

        reward = abs(target - entry)
        rr = round(reward / risk, 2)

        # Validate direction sanity
        if side == "BUY" and sl >= entry:
            return None
        if side == "SELL" and sl <= entry:
            return None

        # Confidence based on how many optional indicators align
        confidence = self._compute_confidence(snap, side, symbol)

        # Capture indicator context for review
        indicators = {
            "sma_9": snap.sma_9,
            "bb_upper": snap.bb_upper,
            "bb_lower": snap.bb_lower,
            "rsi_14": snap.rsi_14,
            "atr_14": snap.atr_14,
            "vwap": snap.vwap,
            "macd_hist": snap.macd_histogram,
        }

        return Signal(
            timestamp=now,
            symbol=symbol,
            strategy=strategy.name,
            side=side,
            entry=entry,
            stoploss=sl,
            target=target,
            risk_reward=rr,
            risk_points=round(risk, 2),
            reasons=reasons,
            confidence=confidence,
            indicators=indicators,
            timeframes_checked=list(set(c.timeframe for c in strategy.conditions)),
        )

    def _compute_confidence(self, snap, side: str, symbol: str = None) -> float:
        """Score 0-1 based on how many indicators agree with the trade direction."""
        votes = 0
        checks = 0

        # VWAP alignment
        if snap.vwap > 0:
            checks += 1
            if side == "BUY" and snap.last_close > snap.vwap:
                votes += 1
            elif side == "SELL" and snap.last_close < snap.vwap:
                votes += 1

        # RSI zone
        checks += 1
        if side == "BUY" and snap.rsi_14 < 70:
            votes += 1
        elif side == "SELL" and snap.rsi_14 > 30:
            votes += 1

        # MACD
        if snap.macd_histogram != 0:
            checks += 1
            if side == "BUY" and snap.macd_histogram > 0:
                votes += 1
            elif side == "SELL" and snap.macd_histogram < 0:
                votes += 1

        # EMA alignment
        if snap.ema_9 and snap.ema_21:
            checks += 1
            if side == "BUY" and snap.ema_9 > snap.ema_21:
                votes += 1
            elif side == "SELL" and snap.ema_9 < snap.ema_21:
                votes += 1

        # Volume surge bonus (soft signal — boosts confidence, not a hard gate)
        if symbol:
            store = self.stores.get(symbol)
            if store:
                bars_5m = list(store.bars.get("5m", []))
                if len(bars_5m) >= 11:
                    prior = bars_5m[-11:-1]
                    avg_vol = sum(b.volume for b in prior) / len(prior) if prior else 0
                    cur_vol = bars_5m[-1].volume
                    if avg_vol > 0 and cur_vol > avg_vol * 1.3:
                        checks += 1
                        votes += 1  # volume confirms

        return round(votes / checks, 2) if checks > 0 else 0.5

    def _emit_signal(self, signal: Signal):
        """Send signal to all registered handlers."""
        for handler in self._handlers:
            try:
                handler(signal)
            except Exception as e:
                logger.error(f"Signal handler error: {e}")

    def get_stats(self) -> dict:
        return {
            "symbols": len(self.symbols),
            "strategies": len(self.strategies),
            "bars_processed": self.bars_processed,
            "signals_emitted": self.signals_emitted,
            "stores": {
                sym: {tf: store.bar_count(tf) for tf in ["1m", "5m", "15m"]}
                for sym, store in self.stores.items()
            },
        }
