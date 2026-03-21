"""
Real-Time Monitor — watches the watchlist during market hours.

Runs every 5 minutes from 9:15 to 15:15 IST.
For each stock in the watchlist:
  1. Fetch latest 5-min candles
  2. Run all structure detectors
  3. If signal triggered → send to risk engine → execute

This is the heartbeat of the intraday agent.
"""
import os
import sys
import time
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from logzero import logger

from trading.utils.time_utils import can_fetch_candles, cap_end_time
from trading.utils.pnl_utils import compute_equity_pnl, calc_position_size

# Bootstrap Django
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import django
    django.setup()

from trading.services.data_service import DataService, BrokerClient
from trading.services.risk_engine import validate_trade
from trading.services.broker_service import BrokerService
from trading.intraday.state import IntradaySignal, IntradayState, Phase, StockSetup
from trading.intraday.structures import detect_all_structures, calculate_vwap


class IntradayMonitor:
    """
    Monitors watchlist stocks in real-time and triggers trades
    when price structures confirm.
    """

    def __init__(self, state: IntradayState):
        self.state = state
        self.data_service = DataService()
        self._candle_cache: Dict[str, List[Dict]] = {}  # symbol → candles
        self._volume_avgs: Dict[str, float] = {}         # symbol → avg 5-min volume

    def run_scan_cycle(self) -> List[IntradaySignal]:
        """
        Single monitoring cycle — fetch data + detect structures for all watchlist stocks.

        Returns list of triggered signals.
        """
        all_signals: List[IntradaySignal] = []
        today_str = self.state.trading_date or date.today().strftime("%Y-%m-%d")

        # Classify market regime from NIFTY data (one LTP call, cached)
        try:
            from trading.intraday.regime import classify_regime
            from trading.options.data_service import OptionsDataService
            svc = OptionsDataService()
            nifty = svc.fetch_nifty_spot()
            if nifty.get("ltp"):
                self._regime = classify_regime(
                    nifty_open=nifty.get("open", 0),
                    nifty_high=nifty.get("high", 0),
                    nifty_low=nifty.get("low", 0),
                    nifty_current=nifty.get("ltp", 0),
                    nifty_prev_close=nifty.get("prev_close", 0),
                )
            else:
                self._regime = None
        except Exception:
            self._regime = None

        regime_str = f" | Regime: {self._regime.classification} (gap={self._regime.nifty_gap_pct:+.1f}% range={self._regime.nifty_range_pct:.1f}%)" if self._regime else ""
        logger.info(f"--- Monitor cycle: {len(self.state.watchlist)} stocks | "
                     f"Open: {self.state.open_positions}/{self.state.max_positions}{regime_str} ---")

        # Don't scan if we've hit max positions
        if self.state.open_positions >= self.state.max_positions:
            logger.info("Max positions reached — monitoring only, no new entries")
            return []

        # Scan top 5 by score to stay within API budget (5 candle calls vs 10)
        scan_list = sorted(self.state.watchlist, key=lambda s: s.score, reverse=True)[:5]

        for setup in scan_list:
            try:
                # Fetch today's 5-min candles
                candles = self._fetch_intraday_candles(setup.symbol, setup.token, today_str)
                if not candles or len(candles) < 4:
                    continue

                # Update current price
                setup.current_price = candles[-1]["close"]
                setup.current_volume = sum(c.get("volume", 0) for c in candles)
                setup.today_open = candles[0]["open"]

                # Calculate average 5-min volume for this stock
                avg_vol = self._volume_avgs.get(setup.symbol, 0)
                if avg_vol == 0 and len(candles) > 3:
                    avg_vol = sum(c.get("volume", 0) for c in candles) / len(candles)
                    self._volume_avgs[setup.symbol] = avg_vol

                # Compute indicator confluence for this stock
                from trading.utils.indicators import compute_indicator_confluence
                closes = [c["close"] for c in candles]
                confluence = compute_indicator_confluence(
                    closes, candles,
                    prev_high=setup.prev_high, prev_low=setup.prev_low,
                    prev_close=setup.prev_close,
                )
                setup.confluence_score = confluence.get("confluence_score", 0)
                setup.confluence_bias = confluence.get("confluence_bias", "NEUTRAL")

                # Build level map for this stock
                from trading.intraday.levels import LevelMap
                try:
                    from trading.services.data_service import DataService
                    ds = DataService()
                    hist = ds.fetch_historical(setup.symbol,
                        (date.today() - timedelta(days=10)).isoformat(),
                        (date.today() - timedelta(days=1)).isoformat(), "ONE_DAY")
                    level_map = LevelMap.build(hist or [], setup.today_open, candles)
                except Exception:
                    level_map = None

                # Run structure detectors (existing V1 setups)
                signals = detect_all_structures(setup, candles, avg_vol)

                # Run level bounce detector (V2)
                if level_map:
                    try:
                        from trading.intraday.level_bounce import detect_level_bounce
                        from trading.intraday.level_retest import detect_level_retest
                        from trading.intraday.vwap_fade import detect_vwap_fade

                        atr = setup.prev_atr or (setup.prev_high - setup.prev_low)
                        bounce_sigs = detect_level_bounce(level_map, candles, atr, avg_vol)
                        retest_sigs = detect_level_retest(level_map, candles, atr, avg_vol)
                        fade_sigs = detect_vwap_fade(level_map, candles, atr, avg_vol)

                        for s in bounce_sigs + retest_sigs + fade_sigs:
                            s.symbol = setup.symbol
                            s.token = setup.token
                        signals.extend(bounce_sigs + retest_sigs + fade_sigs)
                    except Exception as e:
                        logger.warning(f"  [{setup.symbol}] V2 detectors failed: {e}")

                # Apply Sweet Spot Filter to ALL signals (V1 + V2)
                if level_map and signals:
                    from trading.intraday.sweet_spot import evaluate_signal
                    atr = setup.prev_atr or (setup.prev_high - setup.prev_low)
                    filtered = []
                    for sig in signals:
                        enriched = evaluate_signal(sig, level_map, atr)
                        if enriched:
                            filtered.append(enriched)
                        else:
                            logger.info(f"  [{setup.symbol}] {sig.setup_type.value} rejected — not at level")
                    signals = filtered

                if signals:
                    # Sort by confidence, take best
                    signals.sort(key=lambda s: -s.confidence)
                    best = signals[0]

                    # Regime-based confidence adjustment
                    if hasattr(self, '_regime') and self._regime:
                        from trading.intraday.regime import adjust_confidence
                        original_conf = best.confidence
                        best.confidence = adjust_confidence(
                            best.setup_type.value, best.confidence, self._regime
                        )
                        if best.confidence != original_conf:
                            logger.info(f"  [{setup.symbol}] Regime {self._regime.classification}: "
                                        f"conf {original_conf:.2f} → {best.confidence:.2f}")

                    # Multi-timeframe confirmation (skip for level bounces — they're self-confirming)
                    is_level_setup = best.setup_type.value.startswith("LEVEL_BOUNCE")
                    if is_level_setup or self._confirm_multi_tf(setup, best, today_str):
                        all_signals.append(best)
                        regime_tag = f" | Regime: {self._regime.classification}" if hasattr(self, '_regime') and self._regime else ""
                        level_tag = f" | Level: {best.near_pivot}" if best.near_pivot else ""
                        logger.info(f"  SIGNAL: {best.symbol} {best.setup_type.value} "
                                    f"@ {best.entry_price:.2f} | Conf {best.confidence:.2f} | "
                                    f"Confluence {setup.confluence_score}{regime_tag}{level_tag}")

                        self._update_watchlist_outcome(setup, best)
                    else:
                        logger.info(f"  SIGNAL FILTERED: {best.symbol} {best.setup_type.value} "
                                    f"— higher TF disagrees")

            except Exception as e:
                logger.error(f"  {setup.symbol} monitor error: {e}")

        return all_signals

    def process_signal(self, signal: IntradaySignal) -> Dict[str, Any]:
        """
        Process a triggered signal through risk engine → execution.

        Returns trade result dict.
        """
        from trading.models import TradeJournal, AuditLog, PortfolioSnapshot

        # Convert signal to trade plan (same format as directional planner output)
        plan = {
            "symbol": signal.symbol,
            "side": "BUY" if signal.bias.value == "LONG" else "SELL",
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "target": signal.target,
            "quantity": self._calc_position_size(signal),
            "confidence": signal.confidence,
            "rationale": signal.reason,
            "setup_type": signal.setup_type.value,
        }

        logger.info(f"Processing signal: {signal.symbol} {plan['side']} "
                     f"{plan['quantity']}x @ {plan['entry_price']:.2f}")

        # 1. Risk engine — deterministic, no exceptions
        approved, reason, risk_details = validate_trade(
            plan=plan,
            capital=self.state.capital,
            daily_loss=self.state.daily_loss,
            open_positions=self.state.open_positions,
        )

        if not approved:
            logger.warning(f"  RISK REJECTED: {signal.symbol} — {reason}")

            # Journal the rejection
            try:
                from datetime import date as _date
                TradeJournal.objects.create(
                    symbol=signal.symbol,
                    side=plan["side"],
                    entry_price=plan["entry_price"],
                    stop_loss=plan["stop_loss"],
                    target=plan["target"],
                    quantity=plan["quantity"],
                    status="REJECTED",
                    reasoning=f"[{signal.setup_type.value}] {signal.reason}",
                    confidence=signal.confidence,
                    risk_approved=False,
                    risk_reason=reason,
                    trade_date=_date.today(),
                )
            except Exception as e:
                logger.warning(f"Failed to journal rejected trade for {signal.symbol}: {e}")

            return {
                "action": "REJECTED",
                "symbol": signal.symbol,
                "reason": reason,
                "signal": signal,
            }

        # 2. Execute order
        trading_mode = os.getenv("TRADING_MODE", "paper")

        try:
            broker_svc = BrokerService()
            exec_result = broker_svc.place_order(
                symbol=signal.symbol,
                side=plan["side"],
                quantity=plan["quantity"],
                price=plan["entry_price"],
                order_type="LIMIT",
            )
        except Exception as e:
            logger.error(f"  Execution failed: {e}")
            exec_result = {"status": "FAILED", "error": str(e)}

        # 3. Journal the trade
        status = "EXECUTED" if trading_mode == "live" else "PAPER"
        try:
            from datetime import date as _date
            TradeJournal.objects.create(
                symbol=signal.symbol,
                side=plan["side"],
                entry_price=plan["entry_price"],
                stop_loss=plan["stop_loss"],
                target=plan["target"],
                quantity=plan["quantity"],
                status=status,
                reasoning=f"[{signal.setup_type.value}] {signal.reason}",
                confidence=signal.confidence,
                risk_approved=True,
                risk_reason="Approved by risk engine",
                order_id=exec_result.get("order_id", ""),
                trade_date=_date.today(),
            )
        except Exception as e:
            logger.error(f"  Journal failed: {e}")

        # 4. Audit log
        try:
            AuditLog.objects.create(
                event_type="EXECUTION",
                symbol=signal.symbol,
                prompt=f"Signal: {signal.setup_type.value} for {signal.symbol}",
                response=str(plan),
                risk_details=risk_details,
                execution_details=exec_result,
            )
        except Exception as e:
            logger.warning(f"Audit log write failed (non-fatal): {e}")

        # 5. Update state
        self.state.open_positions += 1
        self.state.trades_today.append({
            "symbol": signal.symbol,
            "side": plan["side"],
            "entry": plan["entry_price"],
            "sl": plan["stop_loss"],
            "target": plan["target"],
            "qty": plan["quantity"],
            "setup": signal.setup_type.value,
            "status": status,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        logger.info(f"  TRADED: {signal.symbol} {plan['side']} {plan['quantity']}x "
                     f"@ {plan['entry_price']:.2f} | SL {plan['stop_loss']:.2f} | "
                     f"Target {plan['target']:.2f} | {status}")

        # Update WatchlistEntry to TRADED
        try:
            from trading.models import WatchlistEntry
            WatchlistEntry.objects.filter(
                symbol=signal.symbol, scan_date=date.today(),
            ).update(outcome="TRADED", triggered_setup=signal.setup_type.value)
        except Exception:
            pass

        return {
            "action": "TRADED",
            "symbol": signal.symbol,
            "plan": plan,
            "execution": exec_result,
            "signal": signal,
        }

    def _calc_position_size(self, signal: IntradaySignal) -> int:
        """
        Calculate position size based on risk per trade.
        Risk = 1% of capital. Size = risk / (entry - SL).
        """
        max_risk_pct = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
        risk_amount = self.state.capital * (max_risk_pct / 100)

        if signal.bias.value == "LONG":
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
        else:
            risk_per_share = abs(signal.stop_loss - signal.entry_price)

        if risk_per_share <= 0:
            return 0

        qty = int(risk_amount / risk_per_share)

        # Cap by max position size (% of capital)
        max_pos_pct = float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0"))
        max_value = self.state.capital * (max_pos_pct / 100)
        max_qty_by_value = int(max_value / signal.entry_price) if signal.entry_price > 0 else 0

        qty = min(qty, max_qty_by_value)
        return max(qty, 1) if qty > 0 else 0

    def _fetch_intraday_candles(
        self, symbol: str, token: str, today: str
    ) -> List[Dict]:
        """Fetch today's 5-min candles from broker."""
        if not can_fetch_candles():
            return []

        self.data_service._ensure_broker()

        if not token:
            from trading.services.ticker_service import ticker_service
            token = ticker_service.get_token(symbol) or ""
            if not token:
                return []

        start = f"{today} 09:15"
        end = cap_end_time(today)

        raw = self.data_service._broker.fetch_candles(token, start, end, "FIVE_MINUTE")
        if not raw:
            return []

        # Convert to candle dicts
        candles = []
        for row in raw:
            candles.append({
                "timestamp": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": int(row[5]),
            })

        return candles

    def _update_watchlist_outcome(self, setup, signal):
        """Update WatchlistEntry when a structure triggers."""
        try:
            from trading.models import WatchlistEntry
            WatchlistEntry.objects.filter(
                symbol=setup.symbol,
                scan_date=date.today(),
            ).update(
                outcome="TRIGGERED",
                triggered_setup=signal.setup_type.value,
            )
        except Exception as e:
            logger.debug(f"Watchlist outcome update failed: {e}")

    def _confirm_multi_tf(self, setup, signal, today_str: str) -> bool:
        """
        Multi-timeframe confirmation using technical indicators.

        Checks 15-min candles for:
        1. Candle direction agrees with signal
        2. MACD histogram agrees with signal direction
        3. Price vs VWAP agrees

        Must pass at least 2 of 3 checks to confirm.
        """
        try:
            now = datetime.now()
            # Skip MTF check in first 30 min (not enough 15-min candles)
            if now.hour == 9 and now.minute < 45:
                return True

            token = setup.token
            if not token:
                return True

            self.data_service._ensure_broker()
            end = cap_end_time(today_str)
            raw = self.data_service._broker.fetch_candles(
                token, f"{today_str} 09:15", end, "FIFTEEN_MINUTE"
            )

            if not raw or len(raw) < 3:
                return True

            closes_15m = [float(c[4]) for c in raw]
            candles_15m = [{"high": float(c[2]), "low": float(c[3]),
                            "close": float(c[4]), "volume": int(c[5])} for c in raw]

            votes_for = 0
            votes_against = 0
            is_long = signal.bias.value == "LONG"

            # Check 1: Last 15-min candle direction
            last_open = float(raw[-1][1])
            last_close = float(raw[-1][4])
            if (is_long and last_close > last_open) or (not is_long and last_close < last_open):
                votes_for += 1
            else:
                votes_against += 1

            # Check 2: MACD histogram direction
            from trading.utils.indicators import macd as macd_calc
            m = macd_calc(closes_15m, fast=8, slow=17, signal_period=5)
            if (is_long and m["histogram"] > 0) or (not is_long and m["histogram"] < 0):
                votes_for += 1
            else:
                votes_against += 1

            # Check 3: Price vs VWAP
            from trading.utils.indicators import vwap as vwap_calc
            v = vwap_calc(candles_15m)
            if v > 0:
                if (is_long and closes_15m[-1] > v) or (not is_long and closes_15m[-1] < v):
                    votes_for += 1
                else:
                    votes_against += 1

            return votes_for >= 2  # Need at least 2 of 3 checks

        except Exception as e:
            logger.debug(f"MTF check failed for {setup.symbol} (allowing signal): {e}")
            return True
