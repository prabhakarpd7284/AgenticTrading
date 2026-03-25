"""
Intraday Backtester — replay structure detectors + indicators against real candle data.

No LLM calls. No broker calls during simulation. Just math + historical candles.

Usage:
    from trading.services.intraday_backtester import run_intraday_backtest

    results = run_intraday_backtest(
        symbols=["MARUTI", "SBIN", "RELIANCE"],
        from_date="2026-03-10",
        to_date="2026-03-17",
    )

    # Or via management command:
    python manage.py run_backtest_intraday --days 5 --universe high_volume
"""
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from logzero import logger

# Bootstrap Django
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import django
    django.setup()

from trading.intraday.structures import detect_all_structures
from trading.intraday.state import StockSetup, TradeBias, IntradaySignal
from trading.utils.indicators import (
    camarilla_pivots, compute_indicator_confluence, atr as calc_atr,
)
from trading.utils.pnl_utils import compute_equity_pnl, calc_position_size


# ──────────────────────────────────────────────
# Config — tunable parameters
# ──────────────────────────────────────────────
@dataclass
class BacktestConfig:
    """All tunable parameters in one place for optimization."""
    capital: float = 500000.0
    max_positions: int = 3
    max_risk_pct: float = 1.0           # % of capital risked per trade
    max_position_pct: float = 10.0      # max position value as % of capital
    min_rr: float = 1.5                 # minimum risk:reward ratio
    min_confidence: float = 0.55        # minimum signal confidence
    min_confluence: int = 0             # minimum indicator confluence score (0 = disabled)
    mtf_enabled: bool = True            # multi-timeframe filter
    mtf_min_votes: int = 2              # votes needed to confirm (out of 3)
    sl_atr_mult: float = 0.3           # SL distance as multiple of ATR (used by structures)
    target_atr_mult: float = 1.0       # target distance as multiple of ATR


# ──────────────────────────────────────────────
# Trade result
# ──────────────────────────────────────────────
@dataclass
class BacktestTrade:
    date: str
    symbol: str
    setup_type: str
    side: str
    entry: float
    sl: float
    target: float
    exit_price: float
    outcome: str           # "TARGET", "SL", "EOD"
    pnl_per_share: float
    quantity: int
    pnl_inr: float
    confidence: float
    confluence_score: int
    confluence_bias: str
    near_pivot: str = ""   # e.g. "S3", "R3"
    rr_planned: float = 0
    rr_actual: float = 0


# ──────────────────────────────────────────────
# Core backtester
# ──────────────────────────────────────────────
def run_intraday_backtest(
    symbols: List[str],
    from_date: str,
    to_date: str,
    config: BacktestConfig = None,
) -> Dict[str, Any]:
    """
    Replay the full intraday pipeline against historical data.

    Pipeline per stock per day:
      1. Load prev-day OHLC → compute Camarilla pivots + ATR
      2. Load today's 5-min candles
      3. Build StockSetup → run structure detectors
      4. For each signal: compute indicator confluence → MTF filter
      5. Position sizing via risk engine rules
      6. Simulate execution: check candle-by-candle for SL/target hit
      7. Record result

    Returns dict with: trades, summary, by_setup, by_symbol, config
    """
    if config is None:
        config = BacktestConfig()

    from trading.services.data_service import DataService, BrokerClient
    from trading.services.ticker_service import ticker_service

    b = BrokerClient.get_instance()
    b.ensure_login()
    ds = DataService()

    # Expand date range for prev-day lookback
    start_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
    lookback_start = (start_dt - timedelta(days=10)).strftime("%Y-%m-%d")

    trades: List[BacktestTrade] = []
    capital = config.capital
    daily_pnl: Dict[str, float] = {}

    logger.info(f"=== BACKTEST: {len(symbols)} symbols | {from_date} → {to_date} | Config: {config} ===")

    for symbol in symbols:
        token = ticker_service.get_token(symbol) or ""
        if not token:
            logger.warning(f"  {symbol}: no token, skipping")
            continue

        # Fetch daily candles for the full range (one API call per stock)
        daily = ds.fetch_historical(symbol, lookback_start, to_date, "ONE_DAY")
        if not daily or len(daily) < 3:
            logger.warning(f"  {symbol}: insufficient daily data ({len(daily) if daily else 0} candles)")
            continue

        # Build date index for quick lookup
        daily_by_date = {c["date"]: c for c in daily}

        # Iterate through each test date
        current = start_dt
        while current <= end_dt:
            if current.weekday() >= 5:  # skip weekends
                current += timedelta(days=1)
                continue

            test_date = current.strftime("%Y-%m-%d")
            current += timedelta(days=1)

            # Find prev trading day
            prev_day = None
            for d in reversed(daily):
                if d["date"] < test_date:
                    prev_day = d
                    break
            if not prev_day:
                continue

            # Compute prev-day metrics
            # Find lookback candles for ATR
            prev_idx = next((i for i, c in enumerate(daily) if c["date"] == prev_day["date"]), None)
            if prev_idx is None:
                continue
            lookback = daily[max(0, prev_idx - 4): prev_idx + 1]
            atr_val = calc_atr(lookback) if len(lookback) >= 2 else prev_day["high"] - prev_day["low"]
            pivots = camarilla_pivots(prev_day["high"], prev_day["low"], prev_day["close"])

            # Fetch 5-min candles for test date
            raw = b.fetch_candles(token, f"{test_date} 09:15", f"{test_date} 15:30", "FIVE_MINUTE")
            if not raw or len(raw) < 10:
                continue

            candles = [{
                "timestamp": r[0], "open": float(r[1]), "high": float(r[2]),
                "low": float(r[3]), "close": float(r[4]), "volume": int(r[5]),
            } for r in raw]

            # Build StockSetup with Camarilla pivots
            setup = StockSetup(
                symbol=symbol, token=token, bias=TradeBias.NEUTRAL, score=70,
                prev_open=prev_day["open"], prev_high=prev_day["high"],
                prev_low=prev_day["low"], prev_close=prev_day["close"],
                prev_volume=prev_day["volume"], prev_atr=atr_val,
                pivot_s3=pivots["S3"], pivot_s4=pivots["S4"],
                pivot_r3=pivots["R3"], pivot_r4=pivots["R4"],
                pivot_p=pivots["P"],
            )
            setup.today_open = candles[0]["open"]
            setup.current_price = candles[-1]["close"]
            orb = candles[:3]
            setup.orb_high = max(c["high"] for c in orb)
            setup.orb_low = min(c["low"] for c in orb)

            # Detect structures
            avg_vol = sum(c["volume"] for c in candles) / len(candles)
            signals = detect_all_structures(setup, candles, avg_vol)
            if not signals:
                continue

            for signal in signals[:1]:  # take best signal only
                is_long = signal.bias.value == "LONG"

                # Confidence filter
                if signal.confidence < config.min_confidence:
                    continue

                # Indicator confluence
                closes_5m = [c["close"] for c in candles]
                confluence = compute_indicator_confluence(
                    closes_5m, candles,
                    prev_high=prev_day["high"], prev_low=prev_day["low"],
                    prev_close=prev_day["close"],
                )

                # Confluence filter
                if confluence["confluence_score"] < config.min_confluence:
                    continue

                # MTF filter: check confluence bias agrees
                if config.mtf_enabled:
                    conf_bias = confluence["confluence_bias"]
                    agrees = (
                        (is_long and conf_bias in ("LONG", "NEUTRAL")) or
                        (not is_long and conf_bias in ("SHORT", "NEUTRAL"))
                    )
                    if not agrees:
                        continue

                # Position sizing
                qty = calc_position_size(
                    capital, signal.entry_price, signal.stop_loss,
                    config.max_risk_pct, config.max_position_pct,
                )
                if qty <= 0:
                    continue

                # R:R check
                if signal.risk_reward < config.min_rr:
                    continue

                # Simulate: find entry, then check candle-by-candle
                entry_idx = _find_entry_candle(candles, signal)
                if entry_idx is None:
                    continue

                outcome, exit_price = _simulate_trade(
                    candles[entry_idx:], signal, is_long,
                )

                side = "BUY" if is_long else "SELL"
                pnl_per_share = compute_equity_pnl(side, signal.entry_price, exit_price, 1)
                pnl_inr = pnl_per_share * qty

                # Near pivot detection
                near_pivot = ""
                for lvl in ["S3", "S4", "R3", "R4"]:
                    if abs(signal.entry_price - pivots.get(lvl, 0)) / signal.entry_price * 100 < 0.5:
                        near_pivot = lvl
                        break

                rr_actual = abs(pnl_per_share) / abs(signal.entry_price - signal.stop_loss) \
                    if abs(signal.entry_price - signal.stop_loss) > 0 else 0

                trade = BacktestTrade(
                    date=test_date, symbol=symbol,
                    setup_type=signal.setup_type.value,
                    side=side, entry=signal.entry_price,
                    sl=signal.stop_loss, target=signal.target,
                    exit_price=exit_price, outcome=outcome,
                    pnl_per_share=round(pnl_per_share, 2),
                    quantity=qty, pnl_inr=round(pnl_inr, 2),
                    confidence=signal.confidence,
                    confluence_score=confluence["confluence_score"],
                    confluence_bias=confluence["confluence_bias"],
                    near_pivot=near_pivot,
                    rr_planned=signal.risk_reward,
                    rr_actual=round(rr_actual, 2),
                )
                trades.append(trade)
                capital += pnl_inr

                # Track daily P&L
                daily_pnl[test_date] = daily_pnl.get(test_date, 0) + pnl_inr

    # Build summary
    return _build_summary(trades, daily_pnl, config)


# ──────────────────────────────────────────────
# Simulation helpers
# ──────────────────────────────────────────────
def _find_entry_candle(candles: list, signal: IntradaySignal) -> Optional[int]:
    """Find the candle index where entry price is first reached."""
    is_long = signal.bias.value == "LONG"
    for i, c in enumerate(candles):
        if is_long and c["high"] >= signal.entry_price:
            return i
        elif not is_long and c["low"] <= signal.entry_price:
            return i
    return None


def _simulate_trade(
    candles: list, signal: IntradaySignal, is_long: bool
) -> tuple:
    """
    Simulate trade candle-by-candle.
    Returns (outcome, exit_price).
    """
    for c in candles:
        if is_long:
            # Check SL first (worst case)
            if c["low"] <= signal.stop_loss:
                return "SL", signal.stop_loss
            if c["high"] >= signal.target:
                return "TARGET", signal.target
        else:
            if c["high"] >= signal.stop_loss:
                return "SL", signal.stop_loss
            if c["low"] <= signal.target:
                return "TARGET", signal.target

    # EOD close
    return "EOD", candles[-1]["close"]


# ──────────────────────────────────────────────
# Results builder
# ──────────────────────────────────────────────
def _build_summary(
    trades: List[BacktestTrade],
    daily_pnl: Dict[str, float],
    config: BacktestConfig,
) -> Dict[str, Any]:
    """Build comprehensive backtest summary."""

    if not trades:
        return {"trades": [], "summary": {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_pnl_per_trade": 0, "profit_factor": 0,
            "max_drawdown": 0, "outcomes": {},
        }, "config": config}

    wins = [t for t in trades if t.pnl_inr > 0]
    losses = [t for t in trades if t.pnl_inr <= 0]
    total_pnl = sum(t.pnl_inr for t in trades)

    # By setup type
    by_setup = {}
    for t in trades:
        if t.setup_type not in by_setup:
            by_setup[t.setup_type] = {"trades": 0, "wins": 0, "pnl": 0, "sl_hits": 0, "targets": 0}
        s = by_setup[t.setup_type]
        s["trades"] += 1
        s["pnl"] += t.pnl_inr
        if t.pnl_inr > 0:
            s["wins"] += 1
        if t.outcome == "SL":
            s["sl_hits"] += 1
        if t.outcome == "TARGET":
            s["targets"] += 1

    # By symbol
    by_symbol = {}
    for t in trades:
        if t.symbol not in by_symbol:
            by_symbol[t.symbol] = {"trades": 0, "wins": 0, "pnl": 0}
        s = by_symbol[t.symbol]
        s["trades"] += 1
        s["pnl"] += t.pnl_inr
        if t.pnl_inr > 0:
            s["wins"] += 1

    # By outcome
    outcomes = {"TARGET": 0, "SL": 0, "EOD": 0}
    for t in trades:
        outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1

    # Drawdown
    running_pnl = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running_pnl += t.pnl_inr
        if running_pnl > peak:
            peak = running_pnl
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(t.pnl_inr for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_inr for t in losses)) if losses else 1

    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0

    summary = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl": round(total_pnl, 0),
        "avg_pnl_per_trade": round(total_pnl / len(trades), 0),
        "avg_win": round(avg_win, 0),
        "avg_loss": round(avg_loss, 0),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "max_drawdown": round(max_dd, 0),
        "outcomes": outcomes,
        "return_pct": round(total_pnl / config.capital * 100, 2),
        "trading_days": len(daily_pnl),
        "best_day": round(max(daily_pnl.values()), 0) if daily_pnl else 0,
        "worst_day": round(min(daily_pnl.values()), 0) if daily_pnl else 0,
    }

    return {
        "trades": trades,
        "summary": summary,
        "by_setup": by_setup,
        "by_symbol": by_symbol,
        "daily_pnl": daily_pnl,
        "config": config,
    }


# ──────────────────────────────────────────────
# Parameter optimization
# ──────────────────────────────────────────────
def optimize_parameters(
    symbols: List[str],
    from_date: str,
    to_date: str,
) -> Dict[str, Any]:
    """
    Run backtest with multiple parameter combinations to find optimal config.
    Tests confluence thresholds, MTF on/off, min confidence levels.
    """
    param_grid = [
        BacktestConfig(min_confluence=0, mtf_enabled=False, min_confidence=0.55),
        BacktestConfig(min_confluence=0, mtf_enabled=True, min_confidence=0.55),
        BacktestConfig(min_confluence=15, mtf_enabled=True, min_confidence=0.55),
        BacktestConfig(min_confluence=25, mtf_enabled=True, min_confidence=0.55),
        BacktestConfig(min_confluence=0, mtf_enabled=True, min_confidence=0.60),
        BacktestConfig(min_confluence=0, mtf_enabled=True, min_confidence=0.65),
        BacktestConfig(min_confluence=15, mtf_enabled=True, min_confidence=0.65),
    ]

    results = []
    for cfg in param_grid:
        r = run_intraday_backtest(symbols, from_date, to_date, cfg)
        s = r["summary"]
        results.append({
            "config": f"conf≥{cfg.min_confidence} confl≥{cfg.min_confluence} mtf={cfg.mtf_enabled}",
            "trades": s["total_trades"],
            "win_rate": s["win_rate"],
            "pnl": s["total_pnl"],
            "profit_factor": s["profit_factor"],
            "avg_pnl": s["avg_pnl_per_trade"],
            "max_dd": s["max_drawdown"],
            "full": r,
        })

    # Sort by profit factor (most reliable metric)
    results.sort(key=lambda x: x["profit_factor"], reverse=True)
    return {"runs": results, "best": results[0] if results else None}
