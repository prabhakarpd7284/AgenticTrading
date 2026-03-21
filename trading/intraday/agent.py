"""
Intraday Agent — LangGraph orchestration for the full trading day.

Pipeline:
  premarket_scan → llm_analysis → monitor_loop → detect_structures →
    confirm_signal → risk_check → execute → journal → daily_review

This is the main entry point. Call run_intraday_agent() to start.
"""
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from logzero import logger

# Bootstrap Django
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import django
    django.setup()

from trading.intraday.state import IntradayState, Phase, StockSetup
from trading.intraday.scanner import PremarketScanner
from trading.intraday.monitor import IntradayMonitor
from trading.intraday.prompts import build_premarket_prompt, build_signal_confirmation_prompt
from trading.agents.planner import _run_planner_cli, _run_planner_api


# ──────────────────────────────────────────────
# LLM call (reuse existing planner infrastructure)
# ──────────────────────────────────────────────

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call LLM using the same dual-mode (CLI/API) as the directional planner.
    Returns raw text response.
    """
    planner_mode = os.getenv("PLANNER_MODE", "cli").lower()
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    if planner_mode == "api":
        try:
            import anthropic
            model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return f"ERROR: {e}"
    else:
        # CLI mode (Claude Code Max plan)
        import shutil, subprocess, tempfile
        try:
            claude_path = os.getenv("CLAUDE_CLI_PATH", "").strip() or shutil.which("claude") or ""
            if not claude_path:
                for shell_cmd in [["zsh", "-lc", "which claude"], ["bash", "-lc", "which claude"]]:
                    try:
                        r = subprocess.run(shell_cmd, capture_output=True, text=True, timeout=5)
                        if r.returncode == 0 and r.stdout.strip():
                            claude_path = r.stdout.strip()
                            break
                    except Exception:
                        continue
            if not claude_path:
                return "ERROR: Claude CLI not found"

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                tmp.write(full_prompt)
                tmp_path = tmp.name

            env = os.environ.copy()
            env.pop("CLAUDECODE", None)
            result = subprocess.run(
                f'"{claude_path}" --print < "{tmp_path}"',
                capture_output=True, text=True, timeout=120, env=env, shell=True,
            )
            os.unlink(tmp_path)

            if result.returncode != 0:
                return f"ERROR: CLI exit {result.returncode}"
            return result.stdout.strip() or "No response from CLI"
        except Exception as e:
            logger.error(f"LLM CLI call failed: {e}")
            return f"ERROR: {e}"


# ──────────────────────────────────────────────
# Agent nodes
# ──────────────────────────────────────────────

def premarket_scan_node(state: IntradayState) -> IntradayState:
    """
    Node 1: Run premarket scanner to build watchlist.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: PREMARKET SCAN")
    logger.info("=" * 60)

    state.phase = Phase.PREMARKET

    scanner = PremarketScanner(lookback_days=5, top_n=10)
    watchlist = scanner.scan(
        universe="high_volume",
        trading_date=state.trading_date,
    )

    state.watchlist = watchlist
    logger.info(f"Watchlist built: {len(watchlist)} stocks")

    return state


def llm_analysis_node(state: IntradayState) -> IntradayState:
    """
    Node 2: LLM analyzes the watchlist and provides premarket commentary.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: LLM PREMARKET ANALYSIS")
    logger.info("=" * 60)

    if not state.watchlist:
        state.premarket_analysis = "No stocks in watchlist — nothing to analyze."
        return state

    system_prompt, user_prompt = build_premarket_prompt(state)

    analysis = _call_llm(system_prompt, user_prompt)
    state.premarket_analysis = analysis

    logger.info(f"Premarket analysis:\n{analysis[:500]}...")

    return state


def monitor_and_trade_node(state: IntradayState) -> IntradayState:
    """
    Node 3: Real-time monitoring loop.
    Runs until market close (or max iterations in paper mode).

    In live mode: loops every 5 min from 9:30 to 15:15
    In paper/test mode: runs a single scan cycle
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: MONITOR & TRADE")
    logger.info("=" * 60)

    state.phase = Phase.ACTIVE
    monitor = IntradayMonitor(state)

    trading_mode = os.getenv("TRADING_MODE", "paper")

    # Don't try to fetch intraday candles before market open
    now = datetime.now()
    if now.hour < 9 or (now.hour == 9 and now.minute < 16):
        logger.info("Market not open yet — skipping intraday scan. "
                     "Premarket scan complete, run again after 9:16 AM for live structures.")
        return state

    if trading_mode == "live":
        # Live mode: continuous monitoring loop
        _run_live_loop(monitor, state)
    else:
        # Paper/test mode: single scan cycle
        logger.info("Paper mode — running single scan cycle")
        signals = monitor.run_scan_cycle()

        for signal in signals:
            if state.open_positions >= state.max_positions:
                logger.info("Max positions reached, stopping")
                break

            # In paper mode, skip LLM confirmation for speed
            result = monitor.process_signal(signal)
            logger.info(f"  Result: {result['action']} — {result.get('reason', result.get('symbol', ''))}")

    return state


def _run_live_loop(monitor: IntradayMonitor, state: IntradayState):
    """
    Continuous monitoring for live trading.
    Runs from 9:30 AM to 3:15 PM IST, checking every 5 minutes.
    """
    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    scan_interval = 300  # 5 minutes

    while True:
        now = datetime.now(ist)
        current_time = now.time()

        # Market hours check
        from datetime import time as dt_time
        market_open = dt_time(9, 30)
        market_close = dt_time(15, 15)

        if current_time < market_open:
            wait = (datetime.combine(now.date(), market_open) - datetime.combine(now.date(), current_time)).seconds
            logger.info(f"Waiting {wait}s for market open at 9:30 AM...")
            time.sleep(min(wait, 60))
            continue

        if current_time >= market_close:
            logger.info("Market closing (3:15 PM). Squaring off open positions...")
            state.phase = Phase.CLOSING
            _square_off_all(state)
            break

        # 3:00 PM warning — stop new entries, prepare to close
        if current_time >= dt_time(15, 0):
            logger.info("3:00 PM — no new entries. Closing open positions.")
            _square_off_all(state)
            break

        # Run scan cycle
        signals = monitor.run_scan_cycle()

        for signal in signals:
            if state.open_positions >= state.max_positions:
                break

            # LLM confirmation for live trades
            setup = next((s for s in state.watchlist if s.symbol == signal.symbol), None)
            if setup:
                confirmed = _confirm_signal_with_llm(signal, setup)
                if not confirmed:
                    logger.info(f"  LLM rejected signal for {signal.symbol}")
                    continue

            result = monitor.process_signal(signal)
            logger.info(f"  Result: {result['action']}")

        # Check daily loss limit
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
        daily_loss_pct = (state.daily_loss / state.capital * 100) if state.capital > 0 else 0
        if daily_loss_pct >= max_daily_loss:
            logger.warning(f"DAILY LOSS LIMIT HIT: {daily_loss_pct:.1f}% >= {max_daily_loss}%. Stopping.")
            break

        # Sleep until next cycle
        logger.info(f"  Next scan in {scan_interval}s...")
        time.sleep(scan_interval)


def _square_off_all(state: IntradayState):
    """
    Close all open intraday positions before market close.
    Intraday rule: never carry positions overnight.
    """
    from trading.models import TradeJournal
    from trading.services.broker_service import BrokerService
    from trading.services.data_service import BrokerClient

    open_trades = TradeJournal.objects.filter(
        trade_date=date.today(),
        status__in=["EXECUTED", "FILLED", "PAPER"],
    )

    if not open_trades.exists():
        logger.info("No open positions to square off.")
        return

    b = BrokerClient.get_instance()
    broker_svc = BrokerService(smart_api=b.smart_api)

    for trade in open_trades:
        exit_side = "SELL" if trade.side == "BUY" else "BUY"

        # Get current LTP for exit price
        ltp = 0.0
        try:
            data = b.ltp("NSE", f"{trade.symbol}-EQ", "")
            ltp = data.get("ltp", trade.entry_price)
        except Exception:
            ltp = trade.entry_price  # fallback to entry

        result = broker_svc.place_order(
            symbol=trade.symbol,
            side=exit_side,
            quantity=trade.quantity,
            price=ltp,
            order_type="MARKET",
            product_type="INTRADAY",
        )

        if result.get("success"):
            # Compute P&L
            if trade.side == "BUY":
                pnl = (ltp - trade.entry_price) * trade.quantity
            else:
                pnl = (trade.entry_price - ltp) * trade.quantity

            trade.pnl = round(pnl, 2)
            trade.pnl_percent = round(pnl / (trade.entry_price * trade.quantity) * 100, 2)
            trade.fill_price = ltp
            trade.status = "FILLED"
            trade.save()

            state.daily_loss += max(0, -pnl)
            logger.info(
                f"  SQUARED OFF: {exit_side} {trade.quantity}x {trade.symbol} "
                f"@ {ltp:.2f} | P&L: {pnl:+,.0f} INR [{result.get('mode', 'paper')}]"
            )
        else:
            logger.error(f"  Square-off FAILED for {trade.symbol}: {result.get('message')}")


def _confirm_signal_with_llm(signal, setup) -> bool:
    """Ask LLM to confirm or reject a signal (live mode only)."""
    try:
        prompt = build_signal_confirmation_prompt(signal, setup, "")
        response = _call_llm(
            "You are confirming an intraday trade signal. Be concise.",
            prompt,
        )
        return "TAKE" in response.upper()
    except Exception:
        return True  # If LLM fails, trust the structure


def daily_review_node(state: IntradayState) -> IntradayState:
    """
    Node 4: End of day review and reporting.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: DAILY REVIEW")
    logger.info("=" * 60)

    state.phase = Phase.POSTMARKET

    trades = state.trades_today
    wins = sum(1 for t in trades if t.get("status") in ("EXECUTED", "PAPER"))
    total_pnl = 0  # Would need position tracking for real P&L

    logger.info(f"Trades today: {len(trades)}")
    for t in trades:
        logger.info(f"  {t['time']} | {t['symbol']} {t['side']} {t['qty']}x "
                     f"@ {t['entry']:.2f} | SL {t['sl']:.2f} | "
                     f"Target {t['target']:.2f} | {t['setup']} | {t['status']}")

    return state


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run_intraday_agent(
    trading_date: Optional[str] = None,
    capital: float = 500000.0,
    universe: str = "high_volume",
    skip_llm: bool = False,
    max_positions: int = 3,
) -> IntradayState:
    """
    Run the full intraday agent pipeline.

    Args:
        trading_date: Date to trade (default: today)
        capital: Starting capital in INR
        universe: Stock universe tier
        skip_llm: Skip LLM analysis (for fast testing)
        max_positions: Maximum concurrent positions

    Returns:
        Final IntradayState with all results
    """
    if not trading_date:
        trading_date = date.today().strftime("%Y-%m-%d")

    # Initialize state
    state = IntradayState(
        trading_date=trading_date,
        capital=capital,
        max_positions=max_positions,
    )

    # Load capital: prefer real broker margin, fallback to DB snapshot
    try:
        from trading.services.data_service import BrokerClient
        b = BrokerClient.get_instance()
        b.ensure_login()
        margin = b.margin_available()
        net = float(margin.get("net", 0))
        if net > 0:
            state.capital = net
            logger.info(f"Loaded capital from broker margin: {state.capital:,.0f} INR")
        else:
            # Fallback to portfolio snapshot (paper mode or margin not available)
            from trading.models import PortfolioSnapshot
            snap = PortfolioSnapshot.objects.order_by("-created_at").first()
            if snap:
                state.capital = float(snap.capital)
                logger.info(f"Loaded capital from portfolio: {state.capital:,.0f} INR")
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info(f"INTRADAY AGENT: {trading_date}")
    logger.info(f"Capital: {state.capital:,.0f} INR | Universe: {universe} | Max Pos: {max_positions}")
    logger.info("=" * 60)

    # Phase 1: Premarket scan
    state = premarket_scan_node(state)

    # Phase 2: LLM analysis (optional)
    if not skip_llm and state.watchlist:
        state = llm_analysis_node(state)

    # Phase 3: Monitor and trade
    if state.watchlist:
        state = monitor_and_trade_node(state)
    else:
        logger.warning("Empty watchlist — no stocks to monitor")

    # Phase 4: Daily review
    state = daily_review_node(state)

    return state
