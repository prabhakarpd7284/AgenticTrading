# AgenticTrading — Business Goals & Milestones

## Mission
Build a consistently profitable AI-driven trading operation on the Indian stock market.
Capital: ₹5,00,000 | Mode: Paper → Live | Target: Sharpe > 1.5, DD < 10%

---

## Phase 1: Validate (Apr 1 – May 31, 2026)
**Paper trade V2 system for 60 days. Prove the edge is real.**

### Targets
- [ ] Sharpe ratio > 1.5 (annualized, daily returns)
- [ ] Max drawdown < 10% of capital (< ₹50,000)
- [ ] Profit factor > 1.5
- [ ] Win rate > 50% on equity
- [ ] Daily loss limit never breached (₹15,000)
- [ ] Zero unhedged options exposure (spreads only when VIX > 20)
- [ ] 60 consecutive trading days tracked

### Daily Operations
- `run_trading_day` starts at 8:00 AM, runs to 3:30 PM
- `run_screener --telegram` running in parallel
- Dashboard monitored for anomalies
- End-of-day review logged in event journal
- Weekly backtest comparison (V2 vs actual)

### Exit Criteria for Phase 2
All 7 targets met for 4 consecutive weeks.

---

## Phase 2: Go Live — 1 Lot (Jun 2026)
**Switch TRADING_MODE=live. ₹5 lakh capital. 1 NIFTY lot.**

### Targets
- [ ] 30 profitable days out of first 45
- [ ] Max DD < 8% live (tighter than paper)
- [ ] Slippage < 0.1% on equity, < 0.5% on options
- [ ] No manual intervention required for 5 consecutive days
- [ ] Broker reconciliation matches 100% (positions + P&L)

### Risk Controls (hard-coded, non-negotiable)
- Daily loss limit: ₹15,000 → auto-pause all trading
- Single trade max loss: ₹5,000 (1% of capital)
- Options max loss: ₹6,750 per spread (1.4% of capital)
- VIX > 20 → no straddles (spreads only)
- VIX > 35 → no options at all

---

## Phase 3: Scale — 5 Lots (Jul-Aug 2026)
**After 30 profitable live days, scale to 5 lots.**

### Targets
- [ ] Capital: ₹25 lakh (5x)
- [ ] Same risk % (1% per trade = ₹25,000 risk)
- [ ] Sharpe maintained > 1.5 at scale
- [ ] Add BANKNIFTY as second underlying
- [ ] Add 2-3 high-volume stocks for equity Level Bounce

---

## Phase 4: Institutional (Sep 2026+)
**6-month track record → seek external capital.**

### Requirements
- [ ] Audited track record (180+ trading days)
- [ ] SEBI PMS/AIF registration (if managing external capital)
- [ ] Risk reporting dashboard for investors
- [ ] Multi-strategy attribution (equity vs options contribution)
- [ ] Drawdown recovery demonstrated (recovered from >5% DD)

---

## Current System Capabilities

### Equity (@DirectionalTrader)
- Level Bounce detector (local extremes at scored levels)
- Level Break + Retest, VWAP Fade
- Sweet Spot Filter (rejects signals not at significant levels)
- TradeManager (BE@0.5R, partial@1R, trail 0.3 ATR)
- Regime-adaptive confidence (trending boost, range kill)
- Multi-timeframe confirmation
- Backtester with slippage + commissions

### Options (@OptionsStrategist)
- Adaptive engine: straddle + bear/bull spreads based on VIX
- Simple lifecycle (1.3x stop, shift max 2x/day)
- VIX gate (no straddles when VIX > 20)
- Direct execution (no LLM for CLOSE/SHIFT decisions)
- Realized P&L tracking across rolls

### Infrastructure
- Centralized BrokerClient (rate-limited, cached)
- Config manager (50+ params, 3 presets)
- 10-page Streamlit dashboard
- Screener (6 strategies, WebSocket + REST, Telegram alerts)
- Event logging (JSONL) for audit trail

---

## Key Lessons from March 2026

1. **Unhedged straddles in high VIX = catastrophe.** Three positions lost ₹133K.
   Fix: Adaptive engine with defined-risk spreads.

2. **Position sizing matters more than win rate.** 58% WR but 0.1% risk = no money.
   Fix: 1% risk per trade, 15% position cap.

3. **Profit capture is as important as entry.** Winners exited at 0.3R avg.
   Fix: Partial at 1R, trail at 0.3 ATR.

4. **Level-based trading beats breakout trading.** Gold analysis: +83,850 at levels.
   Fix: Level Bounce detector matches 85% of gold signals.

5. **One bad day can wipe a month.** Mar 11: -74,816 (15% of capital).
   Fix: Daily loss limit + options risk cap + VIX gate.
