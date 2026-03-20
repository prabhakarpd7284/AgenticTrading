"""
Premarket Scanner — find today's best setups before 9:15 AM.

Runs on previous N days of data for each stock in the universe.
Scores stocks on multiple criteria and outputs a ranked watchlist.

Scoring criteria (total 100 points):
  - ATR / volatility (20 pts) — need movement to make money
  - Volume vs average (15 pts) — liquidity = reliable fills
  - Narrow range setup (20 pts) — NR4/NR7 = compressed spring
  - Near key level (20 pts) — close to PDH/PDL = ready to break
  - Trend clarity (15 pts) — clear direction > chop
  - Gap potential (10 pts) — based on global cues / pre-open if available
"""
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from logzero import logger

# Bootstrap Django
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import django
    django.setup()

from trading.services.data_service import DataService
from trading.intraday.state import StockSetup, TradeBias
from trading.intraday.universe import get_universe


class PremarketScanner:
    """
    Scans the stock universe and scores each stock for intraday setup quality.
    Run this before market open to build the day's watchlist.
    """

    def __init__(self, lookback_days: int = 5, top_n: int = 10):
        """
        Args:
            lookback_days: Number of previous trading days to analyze
            top_n: Number of stocks to include in final watchlist
        """
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.data_service = DataService()

    def scan(
        self,
        universe: str = "high_volume",
        trading_date: Optional[str] = None,
    ) -> List[StockSetup]:
        """
        Scan the universe and return ranked watchlist.

        Args:
            universe: 'nifty50', 'high_volume', or 'banknifty'
            trading_date: Today's date (for calculating lookback). Default: today.

        Returns:
            List of StockSetup, sorted by score descending. Top N stocks.
        """
        symbols = get_universe(universe)
        if not trading_date:
            trading_date = date.today().strftime("%Y-%m-%d")

        # Validate universe against fresh symbol master — skip delisted/renamed tickers
        from trading.services.ticker_service import ticker_service
        validity = ticker_service.validate_universe(symbols)
        invalid = [s for s, valid in validity.items() if not valid]
        if invalid:
            logger.warning(f"Skipping {len(invalid)} invalid tickers (not in NSE master): {invalid}")
            symbols = [s for s in symbols if validity.get(s, False)]

        logger.info(f"=== PREMARKET SCAN: {len(symbols)} stocks | universe={universe} | date={trading_date} ===")

        # Calculate date range for historical data
        to_date = datetime.strptime(trading_date, "%Y-%m-%d").date() - timedelta(days=1)
        from_date = to_date - timedelta(days=self.lookback_days + 5)  # Extra buffer for weekends

        setups: List[StockSetup] = []

        for symbol in symbols:
            try:
                setup = self._analyze_stock(symbol, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))
                if setup:
                    setups.append(setup)
                    logger.info(f"  {symbol}: score={setup.score:.0f} bias={setup.bias.value} setups={[s.value for s in setup.setups]}")
            except Exception as e:
                logger.error(f"  {symbol}: scan failed — {e}")

        # Sort by score and return top N
        setups.sort(key=lambda s: s.score, reverse=True)
        watchlist = setups[:self.top_n]

        # ── Batch enrichment: 52wk range, live volume, OI in ONE API call ──
        watchlist = self._enrich_batch(watchlist)

        logger.info(f"=== WATCHLIST ({len(watchlist)} stocks) ===")
        for i, s in enumerate(watchlist, 1):
            logger.info(f"  {i}. {s.symbol} — score {s.score:.0f} | bias {s.bias.value} | {s.reason}")

        # Persist to WatchlistEntry for tracking
        self._persist_watchlist(watchlist, trading_date)

        return watchlist

    def _analyze_stock(self, symbol: str, from_date: str, to_date: str) -> Optional[StockSetup]:
        """
        Fetch historical data and score a single stock.
        """
        # Fetch daily candles
        candles = self.data_service.fetch_historical(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval="ONE_DAY",
        )

        if not candles or len(candles) < 3:
            return None

        # Get token for this symbol
        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol) or ""

        # Use the most recent days for analysis
        recent = candles[-self.lookback_days:] if len(candles) >= self.lookback_days else candles
        last_day = recent[-1]
        prev_days = recent[:-1] if len(recent) > 1 else recent

        # Calculate ATR (Average True Range over lookback)
        atr = self._calc_atr(recent)

        # Calculate average volume
        avg_volume = sum(c["volume"] for c in recent) / len(recent) if recent else 0

        # Calculate Camarilla pivots (pure math from prev day OHLC — zero API cost)
        from trading.utils.indicators import camarilla_pivots, bollinger_bands as bb_calc, rsi as rsi_calc
        pivots = camarilla_pivots(last_day["high"], last_day["low"], last_day["close"])

        # Build setup
        setup = StockSetup(
            symbol=symbol,
            token=token,
            bias=TradeBias.NEUTRAL,
            score=0.0,
            prev_open=last_day["open"],
            prev_high=last_day["high"],
            prev_low=last_day["low"],
            prev_close=last_day["close"],
            prev_volume=last_day["volume"],
            prev_atr=atr,
            pivot_s3=pivots["S3"],
            pivot_s4=pivots["S4"],
            pivot_r3=pivots["R3"],
            pivot_r4=pivots["R4"],
            pivot_p=pivots["P"],
        )

        # ── Score the stock ──
        score = 0.0
        reasons = []

        # 1. ATR / Volatility score (0-20 pts)
        #    Higher ATR relative to price = more movement = more profit potential
        atr_pct = (atr / last_day["close"] * 100) if last_day["close"] > 0 else 0
        if atr_pct >= 2.0:
            score += 20
            reasons.append(f"high ATR {atr_pct:.1f}%")
        elif atr_pct >= 1.5:
            score += 15
            reasons.append(f"good ATR {atr_pct:.1f}%")
        elif atr_pct >= 1.0:
            score += 10
            reasons.append(f"moderate ATR {atr_pct:.1f}%")
        else:
            score += 5  # Low vol stocks still get some points

        # 2. Volume score (0-15 pts)
        #    Last day volume vs average — spike means interest
        if avg_volume > 0 and last_day["volume"] > 0:
            vol_ratio = last_day["volume"] / avg_volume
            if vol_ratio >= 1.5:
                score += 15
                reasons.append(f"volume spike {vol_ratio:.1f}x")
            elif vol_ratio >= 1.2:
                score += 10
                reasons.append(f"above-avg volume {vol_ratio:.1f}x")
            elif vol_ratio >= 0.8:
                score += 7
            else:
                score += 3  # Low volume = low interest

        # 3. Narrow Range score (0-20 pts)
        #    NR4 / NR7 = range compression → expansion coming
        nr_count = self._calc_narrow_range(recent)
        setup.nr_days = nr_count
        if nr_count >= 7:
            score += 20
            reasons.append(f"NR7 — extreme compression")
            from trading.intraday.state import SetupType
            setup.setups.append(SetupType.ORB_LONG)  # ORB works best on NR days
        elif nr_count >= 4:
            score += 15
            reasons.append(f"NR4 — compression building")
            from trading.intraday.state import SetupType
            setup.setups.append(SetupType.ORB_LONG)
        elif nr_count >= 2:
            score += 8
            reasons.append(f"NR{nr_count}")

        # 3b. Bollinger Squeeze confirmation (bonus, 0-5 pts)
        daily_closes = [c["close"] for c in recent]
        if len(daily_closes) >= 5:
            bb = bb_calc(daily_closes, period=min(len(daily_closes), 20))
            if bb.get("squeeze"):
                score += 5
                reasons.append("BB squeeze — breakout imminent")

        # 4. Near key level score (0-20 pts)
        #    Close near PDH or PDL = ready to break
        if len(prev_days) >= 1:
            day_before = prev_days[-1]
            proximity_high = abs(last_day["close"] - day_before["high"]) / atr if atr > 0 else 99
            proximity_low = abs(last_day["close"] - day_before["low"]) / atr if atr > 0 else 99

            if proximity_high < 0.3:
                score += 20
                setup.near_pdh = True
                reasons.append("near prev high — breakout potential")
                from trading.intraday.state import SetupType
                if SetupType.PDH_BREAK not in setup.setups:
                    setup.setups.append(SetupType.PDH_BREAK)
            elif proximity_high < 0.6:
                score += 12
                setup.near_pdh = True
                reasons.append("approaching prev high")

            if proximity_low < 0.3:
                score += 20
                setup.near_pdl = True
                reasons.append("near prev low — breakdown potential")
                from trading.intraday.state import SetupType
                if SetupType.PDL_BREAK not in setup.setups:
                    setup.setups.append(SetupType.PDL_BREAK)
            elif proximity_low < 0.6:
                score += 12
                setup.near_pdl = True
                reasons.append("approaching prev low")

        # 5. Trend clarity score (0-15 pts)
        #    Consecutive up/down closes = clear direction
        if len(recent) >= 3:
            closes = [c["close"] for c in recent]
            up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
            down_days = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
            total = up_days + down_days

            if total > 0:
                trend_strength = max(up_days, down_days) / total
                if trend_strength >= 0.8:
                    score += 15
                    bias = TradeBias.LONG if up_days > down_days else TradeBias.SHORT
                    reasons.append(f"strong {bias.value} trend ({max(up_days, down_days)}/{total} days)")
                elif trend_strength >= 0.6:
                    score += 10
                    bias = TradeBias.LONG if up_days > down_days else TradeBias.SHORT
                    reasons.append(f"moderate {bias.value} trend")
                else:
                    score += 5
                    bias = TradeBias.NEUTRAL
                    reasons.append("choppy / no trend")

                setup.bias = bias

        # 6. Gap potential (0-10 pts)
        #    Big moves the previous day suggest continuation or reversal
        last_change_pct = ((last_day["close"] - last_day["open"]) / last_day["open"] * 100) if last_day["open"] > 0 else 0
        if abs(last_change_pct) >= 2.0:
            score += 10
            reasons.append(f"big move {last_change_pct:+.1f}% — gap likely")
        elif abs(last_change_pct) >= 1.0:
            score += 5
            reasons.append(f"decent move {last_change_pct:+.1f}%")

        setup.score = min(score, 100)  # Cap at 100
        setup.reason = " | ".join(reasons)

        return setup

    def _calc_atr(self, candles: List[Dict], period: int = 5) -> float:
        """Calculate Average True Range."""
        if len(candles) < 2:
            return candles[0]["high"] - candles[0]["low"] if candles else 0

        true_ranges = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]
            l = candles[i]["low"]
            pc = candles[i-1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            true_ranges.append(tr)

        if not true_ranges:
            return 0.0

        # Use last `period` TRs
        recent_trs = true_ranges[-period:]
        return round(sum(recent_trs) / len(recent_trs), 2)

    def _calc_narrow_range(self, candles: List[Dict]) -> int:
        """
        Count consecutive narrow range days from most recent.
        NR4 = today's range is narrowest in 4 days.
        NR7 = today's range is narrowest in 7 days.
        """
        if len(candles) < 2:
            return 0

        ranges = [c["high"] - c["low"] for c in candles]
        last_range = ranges[-1]
        count = 0

        for i in range(len(ranges) - 2, -1, -1):
            if last_range <= ranges[i]:
                count += 1
            else:
                break

        return count + 1  # Include today

    def _enrich_batch(self, setups: List[StockSetup]) -> List[StockSetup]:
        """
        Batch-enrich watchlist with live market data from ONE getMarketData call.
        Adds: 52wk range position, live volume, OI — boosts scoring for high-edge setups.
        """
        if not setups:
            return setups

        try:
            from trading.services.data_service import BrokerClient
            from trading.services.ticker_service import ticker_service

            b = BrokerClient.get_instance()
            b.ensure_login()

            tokens = []
            token_map = {}
            for s in setups:
                tok = s.token or ticker_service.get_token(s.symbol) or ""
                if tok:
                    tokens.append(tok)
                    token_map[tok] = s

            if not tokens:
                return setups

            # ONE API call for all watchlist stocks with FULL data
            fetched = b.market_data_batch({"NSE": tokens}, mode="FULL")

            for item in fetched:
                tok = str(item.get("symbolToken", ""))
                setup = token_map.get(tok)
                if not setup:
                    continue

                ltp = float(item.get("ltp", 0))
                high_52w = float(item.get("52WeekHigh", 0))
                low_52w = float(item.get("52WeekLow", 0))
                live_vol = int(item.get("tradeVolume", 0))
                oi = int(item.get("opnInterest", 0))
                pct_change = float(item.get("percentChange", 0))

                bonus = 0
                bonus_reasons = []

                # 52-week range position: near breakout = high value
                if high_52w > 0 and low_52w > 0 and ltp > 0:
                    range_52w = high_52w - low_52w
                    if range_52w > 0:
                        position_in_range = (ltp - low_52w) / range_52w
                        if position_in_range > 0.95:
                            bonus += 8
                            bonus_reasons.append("near 52wk HIGH — breakout zone")
                        elif position_in_range < 0.05:
                            bonus += 8
                            bonus_reasons.append("near 52wk LOW — bounce or break")
                        elif position_in_range > 0.80:
                            bonus += 4
                            bonus_reasons.append("upper 52wk range")

                # High OI = institutional interest (options-active stocks)
                if oi > 0:
                    bonus += 3
                    bonus_reasons.append(f"OI: {oi:,.0f}")

                # Pre-open gap from prev close (if market is open)
                if ltp > 0 and setup.prev_close > 0:
                    gap = (ltp - setup.prev_close) / setup.prev_close * 100
                    setup.gap_pct = round(gap, 2)
                    if abs(gap) >= 1.0:
                        bonus += 5
                        bonus_reasons.append(f"gap {gap:+.1f}%")

                if bonus_reasons:
                    setup.score = min(setup.score + bonus, 100)
                    setup.reason += " | " + " | ".join(bonus_reasons)

            # Re-sort after enrichment bonus
            setups.sort(key=lambda s: s.score, reverse=True)

        except Exception as e:
            logger.warning(f"Batch enrichment failed (non-fatal): {e}")

        return setups

    def _persist_watchlist(self, setups: List[StockSetup], trading_date: str):
        """Save watchlist to WatchlistEntry model for tracking outcomes."""
        try:
            from trading.models import WatchlistEntry
            scan_date = datetime.strptime(trading_date, "%Y-%m-%d").date()

            for setup in setups:
                WatchlistEntry.objects.update_or_create(
                    symbol=setup.symbol,
                    scan_date=scan_date,
                    defaults={
                        "score": setup.score,
                        "bias": setup.bias.value,
                        "setups": [s.value for s in setup.setups],
                        "prev_high": setup.prev_high,
                        "prev_low": setup.prev_low,
                        "prev_close": setup.prev_close,
                        "prev_atr": setup.prev_atr,
                        "reason": setup.reason,
                    },
                )
            logger.info(f"Watchlist persisted: {len(setups)} entries for {trading_date}")
        except Exception as e:
            logger.warning(f"Watchlist persist failed (non-fatal): {e}")
