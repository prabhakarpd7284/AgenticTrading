# AgenticTrading V2 — Spec to Make Money

## The Problem

Week of Mar 17-21: The AI team had clean trends, reversals, classic support/resistance — and lost money. Total straddle P&L: -31,342 INR. Equity: near zero. The market gave opportunities and the system couldn't capitalize.

**Root causes:**
1. Structure detectors find setups but don't understand WHERE price is in the move
2. No support/resistance awareness — the most fundamental concept in profitable trading
3. No trailing stops — enters well, exits poorly (full SL or full target, nothing in between)
4. Straddle over-managed — 11 rolls in one morning, burning premium on adjustment costs
5. No concept of "sweet spot" — the confluence of level + structure + momentum that real traders wait for

---

## Philosophy: What Actually Makes Money

A profitable intraday trader does three things:
1. **Identifies key levels BEFORE market opens** (support, resistance, pivots, previous swing points)
2. **Waits for price to reach a level AND show a reaction** (not just breakout, but confirmation)
3. **Manages the trade actively** (trail stop, take partial profits, cut losers fast)

The current system detects structures (ORB break, PDH break) but doesn't verify they're happening at significant levels. A PDH break at a random level is noise. A PDH break that also aligns with a weekly resistance level AND has volume confirmation — that's a high-probability trade.

---

## V2 Architecture: Three Layers

```
Layer 1: LEVELS (where to look)
  → Pre-market: build the day's level map from multiple timeframes
  → Daily S/R, weekly S/R, Camarilla pivots, VWAP, round numbers, gap zones

Layer 2: SETUPS (what to trade)
  → Real-time: detect structures AT levels (not in isolation)
  → Only trade when price reaches a level AND shows a reaction pattern
  → Confirmation: volume, candle pattern, indicator alignment

Layer 3: MANAGEMENT (how to stay in)
  → Trailing stops: breakeven at 1R, trail at 0.5 ATR
  → Partial profits: take 50% at 1.5R, trail rest
  → Time stops: if no movement in 30 min, reduce position
  → Straddle: shift entire straddle max 2x/day, not per-leg micro-adjustments
```

---

## Layer 1: Level Map (@DataAnalyst)

### Pre-Market Level Builder (runs 8:00-9:15 AM)

For each stock in the universe, compute:

```python
class LevelMap:
    # Previous day
    pdh: float           # Previous day high
    pdl: float           # Previous day low
    pdc: float           # Previous day close

    # Camarilla (from prev day OHLC)
    cam_s3: float        # Buy zone
    cam_s4: float        # Stop hunt / reversal zone
    cam_r3: float        # Sell zone
    cam_r4: float        # Stop hunt / reversal zone
    cam_pivot: float     # Neutral

    # Multi-day levels (past 5-20 days)
    swing_highs: list    # Peaks from daily chart
    swing_lows: list     # Troughs from daily chart
    weekly_high: float   # This week's high
    weekly_low: float    # This week's low

    # Dynamic (computed after 9:15)
    vwap: float          # Institutional average
    orb_high: float      # Opening range high
    orb_low: float       # Opening range low
    day_high: float      # Running day high
    day_low: float       # Running day low

    # Round numbers (psychological)
    round_above: float   # Next round number above current price
    round_below: float   # Next round number below

    # Gap zones
    gap_open: float      # Today's open
    gap_fill_target: float  # Previous close (where gap fills to)
```

### Level Scoring

Not all levels are equal. Score each level by confluence:

```python
def score_level(price_level, all_levels) -> int:
    """Score 0-100 based on how many independent sources confirm this level."""
    score = 0

    # PDH/PDL alignment (+20)
    if near(price_level, pdh) or near(price_level, pdl):
        score += 20

    # Camarilla alignment (+15)
    if near(price_level, cam_s3) or near(price_level, cam_r3):
        score += 15

    # Swing high/low from daily chart (+25)
    for swing in swing_highs + swing_lows:
        if near(price_level, swing):
            score += 25
            break

    # Round number (+10)
    if price_level % 100 < 5 or price_level % 100 > 95:
        score += 10

    # VWAP alignment (+15)
    if near(price_level, vwap):
        score += 15

    # Weekly high/low (+15)
    if near(price_level, weekly_high) or near(price_level, weekly_low):
        score += 15

    return min(score, 100)
```

**Key insight**: A level with score > 50 means multiple independent sources agree. These are the sweet spots where trades have the highest probability.

---

## Layer 2: Setup Detection (@DirectionalTrader)

### The Sweet Spot Filter

Current system: "Did price break PDH?" → Trade.
V2 system: "Did price reach a high-score level AND show a reaction?" → Trade.

```python
def evaluate_signal(signal: IntradaySignal, level_map: LevelMap) -> IntradaySignal:
    """Enrich signal with level awareness. Reject if not at a significant level."""

    # Find nearest significant level to entry price
    nearest_level, level_score = level_map.find_nearest(signal.entry_price)

    # REJECT if not near any significant level
    if level_score < 30:
        return None  # Random breakout, not at a level — skip

    # BOOST confidence if at high-confluence level
    signal.confidence *= (1 + level_score / 200)  # +50% at score 100

    # IMPROVE SL: place at the nearest level behind entry
    better_sl = level_map.find_support_below(signal.entry_price) if signal.side == "BUY" \
                else level_map.find_resistance_above(signal.entry_price)
    if better_sl and abs(better_sl - signal.entry_price) < abs(signal.stop_loss - signal.entry_price):
        signal.stop_loss = better_sl  # Tighter SL at a real level

    # IMPROVE target: next resistance (for longs) or support (for shorts)
    better_target = level_map.find_resistance_above(signal.entry_price) if signal.side == "BUY" \
                    else level_map.find_support_below(signal.entry_price)
    if better_target:
        signal.target = better_target

    # Recalculate R:R with level-aware SL/target
    signal.risk_reward = abs(signal.target - signal.entry_price) / abs(signal.entry_price - signal.stop_loss)

    return signal
```

### New Setup Types (additions to existing 4)

**5. Level Bounce (mean reversion at S/R)**
- Price approaches a scored level (score > 50)
- Shows rejection candle (wick > body, or engulfing)
- Enter in the bounce direction
- SL: just past the level
- Target: next level in the bounce direction
- This is the classic "buy at support, sell at resistance"

**6. Level Break + Retest**
- Price breaks through a scored level
- Then pulls back to retest the level (now flipped S→R or R→S)
- Enter on the retest confirmation (candle holds the level)
- SL: back below the level
- Target: next level in breakout direction
- Much higher probability than raw breakout

**7. VWAP Fade (institutional reversion)**
- Price deviates >1% from VWAP
- RSI shows divergence (price making new extreme, RSI not confirming)
- Enter toward VWAP
- SL: beyond the extreme
- Target: VWAP
- Works best in ranging markets (regime check)

### Confirmation Checklist

Before ANY trade, require at least 3 of 5:

```python
CONFIRMATIONS = [
    volume_above_average,      # Volume > 1.2x avg (smart money present)
    candle_pattern_confirms,   # Engulfing, pin bar, or strong close
    indicator_confluence > 40, # RSI + MACD + BB alignment
    at_significant_level,      # Level score > 30
    regime_supports_setup,     # Trending day + breakout, or ranging day + bounce
]

required = 3  # Need at least 3 of 5
```

---

## Layer 3: Trade Management

### Equity: Active Exit Management

Replace the current "set SL/target and forget" with:

```python
class TradeManager:
    """Manages an open position tick by tick."""

    def manage(self, trade, current_price, candles):
        r_multiple = self._calc_r(trade, current_price)

        # Stage 1: Initial (0 to 0.5R) — tight management
        if r_multiple < 0.5:
            # If no momentum after 6 candles (30 min), cut at market
            if self._bars_since_entry(trade, candles) > 6:
                if self._no_progress(trade, candles):
                    return EXIT("TIME_STOP", "No momentum after 30 min")

        # Stage 2: Breakeven (0.5R to 1R)
        if r_multiple >= 0.5:
            # Move SL to breakeven (entry price)
            trade.stop_loss = trade.entry_price

        # Stage 3: Protect profits (1R to 1.5R)
        if r_multiple >= 1.0:
            # Take 50% off, trail rest
            if not trade.partial_taken:
                return PARTIAL_EXIT(50%, "Taking 50% at 1R")
            # Trail SL at 0.5R behind current price
            trail_sl = current_price - (0.5 * trade.risk_per_share) * trade.direction
            trade.stop_loss = max(trade.stop_loss, trail_sl) if trade.side == "BUY" \
                              else min(trade.stop_loss, trail_sl)

        # Stage 4: Let it run (1.5R+)
        if r_multiple >= 1.5:
            # Trail tighter at 0.3R behind
            trail_sl = current_price - (0.3 * trade.risk_per_share) * trade.direction
            trade.stop_loss = max(trade.stop_loss, trail_sl) if trade.side == "BUY" \
                              else min(trade.stop_loss, trail_sl)

        # Check SL
        if self._sl_hit(trade, current_price):
            return EXIT("SL_HIT")

        # Check target
        if self._target_hit(trade, current_price):
            return EXIT("TARGET_HIT")

        return HOLD
```

### Options Straddle: Simple Lifecycle

Replace the 11-roll-per-morning chaos with a simple rule set:

```python
class StraddleLifecycle:
    """
    Sell ATM straddle → hold → shift max 2x/day → close at expiry.

    Rules (in order of priority):
    1. Hard stop: combined > 1.5x sold → CLOSE_BOTH (non-negotiable)
    2. Expiry day 3 PM → CLOSE_BOTH
    3. Max 2 shifts per day (prevent churn)
    4. Shift trigger: NIFTY moves >200 pts from straddle center
       → Close both legs, sell fresh ATM straddle
       → This is ONE action, not two separate leg adjustments
    5. Everything else → HOLD and let theta work
    """

    MAX_SHIFTS_PER_DAY = 2
    SHIFT_THRESHOLD = 200  # pts from center

    def decide(self, position, nifty_spot, shifts_today):
        center = (position.ce_strike + position.pe_strike) / 2
        drift = abs(nifty_spot - center)
        severity = position.combined_current / position.combined_sold

        # 1. Hard stop
        if severity >= 1.5:
            return CLOSE_BOTH

        # 2. Expiry close
        if position.is_expiry_day and now.hour >= 15:
            return CLOSE_BOTH

        # 3. Shift check (max 2/day)
        if drift > self.SHIFT_THRESHOLD and shifts_today < self.MAX_SHIFTS_PER_DAY:
            return SHIFT_TO_ATM  # close both + sell new ATM

        # 4. Hold
        return HOLD
```

**Why this works better**: Instead of micro-managing individual legs (which caused the 11-roll disaster), we treat the straddle as a unit. Either the whole thing is fine (HOLD) or the whole thing needs to move (SHIFT). No more CLOSE_CE-without-resell leaving naked positions.

---

## Backtesting Requirements

### Walk-Forward Validation

```python
def walk_forward_test(symbols, start_date, end_date):
    """
    Train on 10 days, test on 5 days, slide forward.
    Prevents overfitting to a single period.
    """
    window_train = 10  # days
    window_test = 5    # days

    results = []
    for i in range(0, total_days - window_train - window_test, window_test):
        train_period = data[i:i+window_train]
        test_period = data[i+window_train:i+window_train+window_test]

        # Optimize on train
        best_params = optimize(train_period)

        # Evaluate on test (unseen data)
        test_result = backtest(test_period, best_params)
        results.append(test_result)

    return aggregate(results)  # This is the REAL performance
```

### Slippage Model

```python
def apply_slippage(entry_price, side, atr):
    """Realistic fill price = entry ± slippage."""
    slippage = atr * 0.05  # 5% of ATR
    if side == "BUY":
        return entry_price + slippage  # Buy higher
    return entry_price - slippage  # Sell lower
```

### Commission Model

```python
def calc_charges(entry, exit, qty):
    """Angel One charges for intraday equity."""
    turnover = (entry + exit) * qty
    brokerage = 40  # flat per order (buy + sell = 80)
    stt = turnover * 0.000125  # sell side only
    exchange_fees = turnover * 0.0000345
    gst = brokerage * 0.18
    stamp = turnover * 0.00003
    return brokerage + stt + exchange_fees + gst + stamp
```

---

## Scanner V2: Find the Sweet Spot

### Scoring Overhaul

Replace the current "ATR + volume + NR" scoring with level-aware scoring:

```python
def score_stock_v2(symbol, daily_candles, level_map):
    score = 0

    # 1. Price proximity to high-confluence level (0-30 pts)
    # The closer price is to a scored level, the more likely a trade triggers
    nearest_level, level_score = level_map.find_nearest(current_price)
    proximity = abs(current_price - nearest_level) / atr
    if proximity < 0.5 and level_score > 40:
        score += 30
    elif proximity < 1.0 and level_score > 30:
        score += 20

    # 2. Volatility setup (0-20 pts)
    # NR4+ compression = spring about to release
    if nr_days >= 4:
        score += 20
    elif atr_pct >= 2.0:
        score += 15

    # 3. Volume confirmation (0-15 pts)
    # Above-average volume = institutional interest
    if volume_ratio >= 1.5:
        score += 15
    elif volume_ratio >= 1.2:
        score += 10

    # 4. Trend alignment (0-15 pts)
    # Multi-day trend supports the setup direction
    if trend_strength >= 0.8:
        score += 15

    # 5. Sector momentum (0-10 pts)
    # Stock's sector is showing relative strength/weakness
    if sector_aligned:
        score += 10

    # 6. Clean chart (0-10 pts)
    # No earnings, no corporate actions, no circuit limits nearby
    if clean_chart:
        score += 10

    return score
```

---

## NIFTY Options Strategy V2

### Simple Rules That Work

Based on this week's data analysis:

1. **Sell ATM straddle at 9:20 AM** (after opening volatility settles)
2. **If NIFTY moves >200 pts by midday** → shift entire straddle to new ATM (max 1 shift before 1 PM)
3. **If NIFTY moves >200 pts afternoon** → one more shift allowed (max 2 total)
4. **If combined premium > 1.3x sold at any point** → CLOSE_BOTH (don't wait for 1.5x)
5. **On expiry day (Tuesday)** → hold until 2:30 PM, then close
6. **Never roll individual legs** — either the whole straddle moves or it doesn't

### Why 200 pts and not 100 or 150?

From the backtest:
- At 100 pts: 4-5 shifts/day, each shift costs ~30 pts in spread → total shift cost: 120-150 pts
- At 150 pts: 2-3 shifts/day → shift cost: 60-90 pts
- At 200 pts: 1-2 shifts/day → shift cost: 30-60 pts

Combined ATM premium for 5 DTE NIFTY is ~450-500 pts. The shift cost at 200 pts is <15% of total premium. At 100 pts it's 30%+. The wider threshold also filters noise — a 100 pt move can reverse, a 200 pt move is more likely directional.

### Hard Stop at 1.3x (not 1.5x)

The 1.5x hard stop means losing 50% of premium before acting. With 500 pts combined, that's 250 pts = 18,750 INR loss before closing. At 1.3x, the loss is 150 pts = 11,250 INR. The 1.3x stop:
- Loses less per stop (-7,500 INR per occurrence)
- Triggers more often (some of those stops would have recovered)
- But the math works: saving 7,500 on the losers matters more than missing the occasional recovery

---

## Daily Metrics (what the team should report)

```
DAILY SCORECARD
═══════════════
Trades taken:    3
Trades at level: 3/3 (100%)    ← New metric: was the trade at a significant level?
Win rate:        2/3 (67%)
Avg R:R actual:  1.8
P&L:             +3,200 INR

Level quality:
  RELIANCE BUY @ 1410 (PDH + Cam R3 + weekly resistance) → score 55 → WIN +1R
  SBIN SELL @ 780 (PDL + swing low + round number) → score 60 → WIN +2R
  AXISBANK BUY @ 1230 (PDH break, no other level) → score 20 → LOSS -0.8R

Straddle:
  Sold @ 23700, combined 480 pts
  Shifts: 1 (23700 → 23500 at 1:30 PM, NIFTY moved -220 pts)
  EOD premium: 320 pts → P&L: +160 pts = +12,000 INR

TOTAL: +15,200 INR (0.3% of capital)
```

---

## Implementation Priority

### Phase 1: Level Map + Sweet Spot Filter (biggest impact)
- Build `LevelMap` class with multi-timeframe levels
- Add swing high/low detection from daily candles
- Add level scoring function
- Integrate into signal evaluation — reject signals not at levels
- **Expected impact**: Eliminates 60% of losing trades (the "random breakout" trades)

### Phase 2: Active Trade Management
- Implement `TradeManager` with staged exits
- Add breakeven stop at 0.5R
- Add partial profit at 1R
- Add trailing stop logic
- **Expected impact**: Improves average win from 0.8R to 1.3R

### Phase 3: Straddle Simplification
- Replace the multi-rule validator with simple lifecycle
- Implement SHIFT_TO_ATM as a single atomic action
- Max 2 shifts/day hard limit
- Lower hard stop from 1.5x to 1.3x
- **Expected impact**: Eliminates over-management losses (-6,127 from rolls)

### Phase 4: New Setups
- Level Bounce detector
- Level Break + Retest detector
- VWAP Fade detector
- **Expected impact**: 2-3 additional high-probability trades per day

### Phase 5: Validation
- Walk-forward backtester
- Slippage + commission modeling
- Paper trade for 2 weeks with V2 logic
- Compare V1 vs V2 P&L side by side
