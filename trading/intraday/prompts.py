"""
Intraday Agent Prompts — LLM is used for analysis, NOT for price calculation.

The LLM's job:
  1. Premarket: synthesize multiple signals into a narrative + trade bias
  2. Signal confirmation: given a triggered structure, confirm or reject
  3. End-of-day: review trades, extract lessons

The LLM does NOT:
  - Calculate entry/SL/target (that's deterministic in structures.py)
  - Override risk engine (that's @RiskGuard's job)
  - Decide position size (that's capital-based math)
"""

PREMARKET_SYSTEM_PROMPT = """You are @IntradayAgent, an expert Indian stock market intraday trader.

YOUR GOAL: Make money through proven price structures on liquid NSE stocks.

You are analyzing the premarket scan results for today's trading session.
Your job is to:
1. Review the scored watchlist of stocks
2. Identify the BEST 3-5 setups for today
3. Assign a trade bias (LONG/SHORT/NEUTRAL) to each
4. Flag any risks or concerns (earnings, events, sector weakness)

PROVEN STRUCTURES YOU TRADE:
- ORB (Opening Range Breakout): First 15 min high/low breakout with volume
- PDH/PDL Break: Yesterday's high/low break with momentum
- Gap and Go: Gap holds, ride momentum in gap direction
- Gap Fill: Gap reverting, trade back toward previous close
- VWAP Reclaim/Reject: Price action around VWAP as dynamic S/R

RULES (NEVER BREAK):
- Only trade liquid NIFTY 50 stocks
- Every trade must have defined SL before entry
- R:R must be >= 1.5 minimum
- Max 3 positions at any time
- Max 1% risk per trade, max 3% daily loss
- Close everything before 3:15 PM IST
- If unsure, DON'T TRADE. Cash is a position.

Capital: {capital} INR | Mode: {trading_mode}
"""

PREMARKET_USER_PROMPT = """Today is {date}. Here are the premarket scan results:

=== TOP WATCHLIST STOCKS ===
{watchlist_summary}

=== MARKET CONTEXT ===
{market_context}

Based on this analysis:
1. Which stocks have the STRONGEST setups for today?
2. What price structures should we watch for in each?
3. Any stocks to AVOID despite high scores?
4. What's the overall market bias?

Respond in this format:
MARKET BIAS: [BULLISH/BEARISH/NEUTRAL]

TOP PICKS:
1. [SYMBOL] - [BIAS] - [Setup to watch] - [Key level to watch]
2. ...

AVOID: [Any stocks to skip and why]

RISK NOTES: [Any concerns for today]
"""


SIGNAL_CONFIRMATION_PROMPT = """A price structure has triggered for {symbol}:

SIGNAL: {setup_type}
Direction: {bias}
Entry: {entry_price}
Stop Loss: {stop_loss}
Target: {target}
Risk:Reward: {risk_reward}
Confidence: {confidence}

CONTEXT:
- Previous Day: O={prev_open} H={prev_high} L={prev_low} C={prev_close}
- Today Open: {today_open}
- Current Price: {current_price}
- VWAP: {vwap}
- ORB Range: {orb_high} - {orb_low}
- Premarket Score: {score}/100
- Premarket Bias: {premarket_bias}

RECENT CANDLES:
{recent_candles}

Should we take this trade? Respond ONLY with:
DECISION: TAKE or SKIP
REASON: [one sentence]
ADJUSTED_CONFIDENCE: [0.0 to 1.0]
"""


DAILY_REVIEW_PROMPT = """End of day review for {date}:

=== TODAY'S TRADES ===
{trades_summary}

=== SUMMARY ===
Trades taken: {trades_taken}
Wins: {wins} | Losses: {losses}
Total P&L: {total_pnl} INR
Capital: {capital} INR

What worked? What didn't? What should we do differently tomorrow?
Keep it concise — 3 bullet points max.
"""


def build_premarket_prompt(state) -> tuple:
    """Build the premarket analysis prompt pair (system, user)."""
    import os

    watchlist_lines = []
    for i, s in enumerate(state.watchlist, 1):
        watchlist_lines.append(
            f"{i}. {s.symbol} (Score: {s.score:.0f}/100)\n"
            f"   Bias: {s.bias.value} | ATR: {s.prev_atr:.2f} | "
            f"Close: {s.prev_close:.2f}\n"
            f"   PDH: {s.prev_high:.2f} | PDL: {s.prev_low:.2f}\n"
            f"   Near PDH: {s.near_pdh} | Near PDL: {s.near_pdl} | "
            f"NR: {s.nr_days} days\n"
            f"   Setups: {[st.value for st in s.setups]}\n"
            f"   Reason: {s.reason}"
        )

    system = PREMARKET_SYSTEM_PROMPT.format(
        capital=state.capital,
        trading_mode=os.getenv("TRADING_MODE", "paper"),
    )

    user = PREMARKET_USER_PROMPT.format(
        date=state.trading_date,
        watchlist_summary="\n\n".join(watchlist_lines) if watchlist_lines else "No stocks scanned yet.",
        market_context=state.market_context or "No additional market context available.",
    )

    return system, user


def build_signal_confirmation_prompt(signal, setup, recent_candles_text: str) -> str:
    """Build the signal confirmation prompt."""
    return SIGNAL_CONFIRMATION_PROMPT.format(
        symbol=signal.symbol,
        setup_type=signal.setup_type.value,
        bias=signal.bias.value,
        entry_price=f"{signal.entry_price:.2f}",
        stop_loss=f"{signal.stop_loss:.2f}",
        target=f"{signal.target:.2f}",
        risk_reward=f"{signal.risk_reward:.2f}",
        confidence=f"{signal.confidence:.2f}",
        prev_open=f"{setup.prev_open:.2f}",
        prev_high=f"{setup.prev_high:.2f}",
        prev_low=f"{setup.prev_low:.2f}",
        prev_close=f"{setup.prev_close:.2f}",
        today_open=f"{setup.today_open:.2f}",
        current_price=f"{setup.current_price:.2f}",
        vwap=f"{setup.vwap:.2f}",
        orb_high=f"{setup.orb_high:.2f}",
        orb_low=f"{setup.orb_low:.2f}",
        score=f"{setup.score:.0f}",
        premarket_bias=setup.bias.value,
        recent_candles=recent_candles_text,
    )
