"""Basket signal generation — equity legs + ATM index option leg.

Equity: OK scanner phases + 5 EMA + Bollinger Bands momentum check.
Options: ATM CE (bullish) or PE (bearish) on the index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

from logzero import logger

from plugins.strategy_basket.config import BasketConfig
from plugins.strategy_basket.mood import MarketMood, MoodAssessment
from trading.utils.indicators import _ema, _wma, _rsi_series, bollinger_bands


@dataclass
class BasketSignal:
    """A single leg signal for the morning basket."""
    symbol: str
    side: str                # BUY or SELL
    leg_type: str            # "equity" or "option"
    entry_price: float
    stoploss: float
    risk_points: float
    phase: str = ""          # OK cycle phase code
    confluence: float = 0.0  # 0-1 score

    # Momentum context
    ema5: float = 0.0
    bb_lower: float = 0.0
    bb_upper: float = 0.0
    bb_mid: float = 0.0

    # Options-specific
    option_type: str = ""    # CE or PE
    strike: int = 0
    expiry: str = ""
    option_symbol: str = ""
    option_token: str = ""

    def to_dict(self) -> dict:
        d = {
            "symbol": self.symbol, "side": self.side, "leg_type": self.leg_type,
            "entry_price": self.entry_price, "stoploss": self.stoploss,
            "risk_points": self.risk_points, "phase": self.phase,
            "confluence": self.confluence,
        }
        if self.leg_type == "option":
            d.update({
                "option_type": self.option_type, "strike": self.strike,
                "expiry": self.expiry, "option_symbol": self.option_symbol,
            })
        return d


class BasketSignalGenerator:
    """Generate equity + option signals for the morning basket."""

    def __init__(self, cfg: BasketConfig = None, data_svc=None):
        self.cfg = cfg or BasketConfig()
        self._data_svc = data_svc

    def _ensure_data_svc(self):
        if self._data_svc is None:
            from trading.services.data_service import DataService
            self._data_svc = DataService()

    def generate(self, mood: MoodAssessment) -> List[BasketSignal]:
        """Generate all basket signals based on mood assessment.

        Returns list of BasketSignal (equity + option legs combined).
        """
        signals: List[BasketSignal] = []

        if mood.mood == MarketMood.NEUTRAL:
            logger.info("Basket: mood NEUTRAL — no signals")
            return signals

        # Equity legs
        equity = self.generate_equity_signals(mood)
        signals.extend(equity)

        # Options leg
        option = self.generate_option_signal(mood)
        if option:
            signals.append(option)

        logger.info(f"Basket: {len(equity)} equity + {1 if option else 0} option signals")
        return signals

    def generate_equity_signals(self, mood: MoodAssessment) -> List[BasketSignal]:
        """Find equity legs from OK scanner + momentum filters."""
        self._ensure_data_svc()

        from plugins.strategy_swing.ok_scanner import OKScanner
        from plugins.strategy_swing.ok_cycles import CyclePhase, BULLISH_ACTIONABLE, BEARISH_ACTIONABLE

        scan_date = date.today().strftime("%Y-%m-%d")
        scanner = OKScanner()

        from apps.market_data.constants import NIFTY_50_SYMBOLS
        results = scanner.scan(list(NIFTY_50_SYMBOLS), scan_date=scan_date)

        # Filter by mood-aligned phases
        if mood.is_bullish:
            allowed = set(self.cfg.buy_phases)
            candidates = [r for r in results if r.phase.value in allowed]
            side = "BUY"
        else:
            allowed = set(self.cfg.sell_phases)
            candidates = [r for r in results if r.phase.value in allowed]
            side = "SELL"

        if not candidates:
            logger.info(f"Basket equity: no {side} candidates from scanner")
            return []

        # Check momentum (5 EMA + BB + RSI) on intraday candles
        confirmed: List[BasketSignal] = []

        for r in candidates:
            try:
                candles = self._data_svc.fetch_intraday_candles(
                    r.symbol, scan_date, scan_date, self.cfg.candle_interval
                )
                if not candles or len(candles) < self.cfg.ema_period + 1:
                    continue

                sig = self._check_momentum(r.symbol, candles, side, r.phase.value, r.confidence)
                if sig:
                    confirmed.append(sig)

            except Exception as e:
                logger.warning(f"Basket equity skip {r.symbol}: {e}")

        # Sort by confluence, take top N
        confirmed.sort(key=lambda s: -s.confluence)
        top = confirmed[:self.cfg.max_equity_legs]
        logger.info(f"Basket equity: {len(confirmed)} confirmed, taking top {len(top)}")
        return top

    def _check_momentum(
        self, symbol: str, candles: list, side: str, phase: str, phase_confidence: float
    ) -> Optional[BasketSignal]:
        """Check 5 EMA + BB + RSI momentum on intraday candles."""
        closes = [c["close"] for c in candles]
        last_close = closes[-1]

        # 5 EMA
        ema5_series = _ema(closes, self.cfg.ema_period)
        if not ema5_series:
            return None
        ema5 = ema5_series[-1]

        # Bollinger Bands
        bb = bollinger_bands(closes, self.cfg.bb_period, self.cfg.bb_std) if len(closes) >= self.cfg.bb_period else None

        # RSI momentum: EMA(3) vs WMA(21)
        rsi_vals = _rsi_series(closes, 14)
        rsi_clean = [v if v is not None else 50.0 for v in rsi_vals]
        rsi_ema = _ema(rsi_clean, 3)
        rsi_wma = _wma(rsi_clean, 21)
        mom_bull = rsi_ema[-1] > rsi_wma[-1] if rsi_ema and rsi_wma else False
        mom_bear = rsi_ema[-1] < rsi_wma[-1] if rsi_ema and rsi_wma else False

        # BUY check: price > 5 EMA, momentum bullish
        if side == "BUY":
            if last_close <= ema5 or not mom_bull:
                return None
            sl = ema5 * 0.998  # SL 0.2% below 5 EMA
            if bb:
                sl = min(sl, bb["lower"])  # Or BB lower, whichever is tighter
        else:
            # SELL check: price < 5 EMA, momentum bearish
            if last_close >= ema5 or not mom_bear:
                return None
            sl = ema5 * 1.002
            if bb:
                sl = max(sl, bb["upper"])

        risk = abs(last_close - sl)
        if risk <= 0:
            return None

        # Confluence score
        score = phase_confidence
        if mom_bull and side == "BUY":
            score += 0.2
        if mom_bear and side == "SELL":
            score += 0.2
        if bb and side == "BUY" and last_close < bb["upper"]:
            score += 0.1  # Room to run within BB

        return BasketSignal(
            symbol=symbol,
            side=side,
            leg_type="equity",
            entry_price=round(last_close, 2),
            stoploss=round(sl, 2),
            risk_points=round(risk, 2),
            phase=phase,
            confluence=round(min(score, 1.0), 2),
            ema5=round(ema5, 2),
            bb_lower=round(bb["lower"], 2) if bb else 0,
            bb_upper=round(bb["upper"], 2) if bb else 0,
            bb_mid=round(bb["middle"], 2) if bb else 0,
        )

    def generate_option_signal(self, mood: MoodAssessment) -> Optional[BasketSignal]:
        """Generate ATM index option signal based on mood."""
        from trading.options.data_service import find_atm_strike, find_option_token
        from trading.services.data_service import BrokerClient
        from trading.utils.expiry_utils import next_expiry_date, iso_to_angel

        option_type = "CE" if mood.is_bullish else "PE"
        spot = mood.nifty_spot

        # ATM strike
        atm = find_atm_strike(spot, step=self.cfg.strike_step)

        # Expiry
        expiry_date = next_expiry_date(self.cfg.index)
        if not expiry_date:
            logger.warning("Basket option: no expiry found")
            return None

        expiry_angel = iso_to_angel(expiry_date.isoformat())
        if not expiry_angel:
            logger.warning(f"Basket option: could not normalize expiry {expiry_date}")
            return None

        # Token lookup
        result = find_option_token(self.cfg.index, atm, expiry_angel, option_type)
        if not result:
            logger.warning(f"Basket option: token not found for {self.cfg.index} {atm} {expiry_angel} {option_type}")
            return None

        opt_symbol, opt_token = result

        # LTP
        broker = BrokerClient.get_instance()
        ltp_data = broker.ltp("NFO", opt_symbol, opt_token)
        premium = ltp_data.get("ltp", 0)

        if premium <= 0:
            logger.warning(f"Basket option: no LTP for {opt_symbol}")
            return None

        # SL = 30% loss on premium
        sl = round(premium * (1 - self.cfg.option_sl_pct / 100), 2)
        risk = round(premium - sl, 2)

        logger.info(
            f"Basket option: {option_type} {atm} {expiry_angel} "
            f"premium=₹{premium:.2f} SL=₹{sl:.2f}"
        )

        return BasketSignal(
            symbol=opt_symbol,
            side="BUY",
            leg_type="option",
            entry_price=premium,
            stoploss=sl,
            risk_points=risk,
            phase=f"ATM_{option_type}",
            confluence=mood.confidence,
            option_type=option_type,
            strike=atm,
            expiry=expiry_date.isoformat(),
            option_symbol=opt_symbol,
            option_token=opt_token,
        )
