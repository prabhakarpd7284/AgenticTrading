"""
Backtest Engine — Replay historical candles through planner + risk.

"If planner loses in backtest, it will lose live."

How it works:
    1. Load historical OHLCV candles for a symbol and date range
    2. For each candle (or session), simulate market data
    3. Run planner → risk → simulated execution
    4. Track P&L, win rate, drawdown
    5. Store results in BacktestResult model

No broker calls. No real orders. Just truth.
"""
import os
import sys
from datetime import date, datetime, timedelta
from typing import List, Optional

from logzero import logger

# Bootstrap Django
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import django
    django.setup()


def _simulate_candle_data(candle: dict, symbol: str) -> str:
    """
    Convert a historical candle dict into the same summary format
    that DataService.fetch_intraday() would produce.
    """
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    vol = candle.get("volume", 0)
    dt = candle.get("date", "unknown")

    body = abs(c - o)
    range_ = h - l
    is_bullish = c > o
    is_doji = body < (range_ * 0.1) if range_ > 0 else False
    is_oh = abs(o - h) < 0.01
    is_ol = abs(o - l) < 0.01

    pivot = (h + l) / 2

    summary = (
        f"MARKET DATA FOR {symbol} ({dt}):\n"
        f"  Open: {o:.2f} | High: {h:.2f} | Low: {l:.2f} | Close: {c:.2f}\n"
        f"  Volume: {vol:,}\n"
        f"  Range: {range_:.2f} | Body: {body:.2f}\n"
        f"  Bullish: {is_bullish} | Doji: {is_doji}\n"
        f"  Open=High: {is_oh} | Open=Low: {is_ol}\n"
        f"  Pivot: {pivot:.2f}\n"
        f"  Current price: {c:.2f}"
    )
    return summary


def _simulate_pnl(plan: dict, candle: dict) -> dict:
    """
    Simulate trade outcome against the candle's high/low.

    Logic:
        BUY trade:
            - If low <= stop_loss → loss (hit SL)
            - If high >= target → win (hit target)
            - Else → close at candle close (partial outcome)
        SELL trade:
            - If high >= stop_loss → loss (hit SL)
            - If low <= target → win (hit target)
            - Else → close at candle close
    """
    entry = plan["entry_price"]
    sl = plan["stop_loss"]
    target = plan["target"]
    qty = plan["quantity"]
    side = plan["side"]

    h, l, c = candle["high"], candle["low"], candle["close"]

    if side == "BUY":
        if l <= sl:
            # Stop loss hit
            exit_price = sl
            outcome = "STOP_LOSS"
        elif h >= target:
            # Target hit
            exit_price = target
            outcome = "TARGET_HIT"
        else:
            # Close at candle close
            exit_price = c
            outcome = "CLOSE_AT_EOD"
        pnl = (exit_price - entry) * qty
    else:  # SELL
        if h >= sl:
            exit_price = sl
            outcome = "STOP_LOSS"
        elif l <= target:
            exit_price = target
            outcome = "TARGET_HIT"
        else:
            exit_price = c
            outcome = "CLOSE_AT_EOD"
        pnl = (entry - exit_price) * qty

    return {
        "exit_price": round(exit_price, 2),
        "pnl": round(pnl, 2),
        "outcome": outcome,
        "side": side,
        "entry": entry,
        "sl": sl,
        "target": target,
        "qty": qty,
    }


def run_backtest(
    symbol: str,
    candles: List[dict],
    initial_capital: float = 500000.0,
    user_intent_template: str = "Plan a trade for {symbol}",
    model: Optional[str] = None,
) -> dict:
    """
    Run backtest over a list of historical candles.

    Args:
        symbol: Stock symbol (e.g. "MFSL")
        candles: List of dicts with keys: date, open, high, low, close, volume
        initial_capital: Starting capital in INR
        user_intent_template: Template for planner intent (use {symbol} placeholder)
        model: Optional LLM model override

    Returns:
        dict with backtest summary and per-trade details
    """
    from trading.rag.retriever import retrieve_context
    from trading.agents.planner import run_planner
    from trading.services.risk_engine import validate_trade

    trades = []
    capital = initial_capital
    daily_loss = 0.0
    wins = 0
    losses = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak_capital = capital

    logger.info(f"=== BACKTEST START: {symbol} | {len(candles)} candles | Capital: {capital:,.0f} INR ===")

    for i, candle in enumerate(candles):
        candle_date = candle.get("date", f"candle_{i}")

        # 1. Simulate market data summary
        market_summary = _simulate_candle_data(candle, symbol)

        # 2. Pull RAG context (real — uses DB history)
        try:
            rag_context = retrieve_context(symbol)
        except Exception:
            rag_context = "No context available."

        # 3. Run planner
        intent = user_intent_template.format(symbol=symbol)
        plan = run_planner(
            user_intent=intent,
            market_data_summary=market_summary,
            rag_context=rag_context,
            model=model,
        )

        if "error" in plan:
            trades.append({
                "candle": candle_date,
                "action": "SKIP",
                "reason": f"Planner error: {plan['error']}",
                "pnl": 0,
            })
            logger.warning(f"  [{candle_date}] Planner error: {plan['error']}")
            continue

        # 4. Run risk engine
        approved, reason, risk_details = validate_trade(
            plan=plan,
            capital=capital,
            daily_loss=daily_loss,
            open_positions=0,  # backtest: one trade at a time
        )

        if not approved:
            trades.append({
                "candle": candle_date,
                "action": "REJECTED",
                "reason": reason,
                "plan": plan,
                "pnl": 0,
            })
            logger.info(f"  [{candle_date}] Risk REJECTED: {reason}")
            continue

        # 5. Simulate execution and P&L
        sim = _simulate_pnl(plan, candle)
        pnl = sim["pnl"]
        total_pnl += pnl
        capital += pnl

        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
            daily_loss += abs(pnl)

        # Track drawdown
        if capital > peak_capital:
            peak_capital = capital
        drawdown = (peak_capital - capital) / peak_capital * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        trades.append({
            "candle": candle_date,
            "action": "TRADED",
            "plan": plan,
            "simulation": sim,
            "pnl": pnl,
            "capital_after": round(capital, 2),
            "drawdown_pct": round(drawdown, 2),
        })

        logger.info(
            f"  [{candle_date}] {sim['side']} {sim['qty']}x @ {sim['entry']:.2f} → "
            f"{sim['outcome']} @ {sim['exit_price']:.2f} | P&L: {pnl:+,.0f} | Capital: {capital:,.0f}"
        )

    # Summary
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    skipped = len([t for t in trades if t["action"] in ("SKIP", "REJECTED")])

    summary = {
        "symbol": symbol,
        "candles_processed": len(candles),
        "trades_executed": total_trades,
        "trades_skipped": skipped,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "final_capital": round(capital, 2),
        "initial_capital": initial_capital,
        "return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
    }

    logger.info(
        f"=== BACKTEST COMPLETE: {symbol} | "
        f"Trades: {total_trades} ({wins}W/{losses}L) | "
        f"Win Rate: {win_rate:.0f}% | "
        f"P&L: {total_pnl:+,.0f} INR | "
        f"Return: {summary['return_pct']:+.1f}% | "
        f"Max DD: {max_drawdown:.1f}% ==="
    )

    return {
        "summary": summary,
        "trades": trades,
    }


def run_backtest_from_csv(
    symbol: str,
    csv_path: str,
    initial_capital: float = 500000.0,
    model: Optional[str] = None,
) -> dict:
    """
    Run backtest from a CSV file with columns: date,open,high,low,close,volume

    Convenience wrapper around run_backtest().
    """
    import csv

    candles = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append({
                "date": row["date"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(float(row.get("volume", 0))),
            })

    if not candles:
        return {"error": "No candles found in CSV"}

    logger.info(f"Loaded {len(candles)} candles from {csv_path}")
    return run_backtest(symbol, candles, initial_capital, model=model)
