# AlphaDesk trading rules — owner-approved (2026-10-21)

These rules apply to every automated strategy (the scalping/pivot agent and any
other non-manual origin). They are deterministic — no LLM decides any of them —
and they sit alongside the existing 10-criterion RiskEngine, which stays the last
gate before execution (CLAUDE.md invariant #1).

## 1. Losing-streak cooldown
- **Loss day** = a trading day whose net P&L for the portfolio (after costs, realized
  + any EOD-closed positions) is below zero. Days with no trades don't count either way.
- **Trigger:** 3 consecutive loss days.
- **Cooldown:** the next **2 NSE trading days** — no new automated entries. Manual
  (UI) orders are still allowed.
- **Shadow mode during cooldown:** strategies keep running and their would-be trades
  are recorded (as events, not orders) so we can see whether the cooldown helped.
- **Re-entry:** after the cooldown, risk per trade is **halved (50%)** until the first
  profitable day, then returns to 100%.

## 2. Trailing stop-loss
- Every automated trade has an initial stop (from the setup).
- When open profit reaches **1R** (one unit of the initial risk), the stop moves to
  **breakeven** (entry price).
- After that, the stop trails **1.5 × ATR** behind the best price reached (ATR on the
  strategy's trading timeframe).
- The stop **only ever moves in the trade's favour** (never loosened).
- All intraday positions are flat by **15:14 IST** regardless (data has no bars
  15:15–15:29 since 2026-08-03; also the book's "exit 15 min before close").

## 3. Nightly news-sentiment filter (from the 21:00 scan)
- **Strong negative** sentiment on a stock → **no longs** in it next day.
- **Strong positive** sentiment → **no shorts** in it next day.
- **Results / major corporate event** due next day → **skip the stock** entirely.
- Sentiment **never forces** a trade and never affects position size.
- If sentiment couldn't be computed, the filter does not block (and the report says so).

## 4. Existing limits (unchanged, RiskEngine)
Risk per trade ≤ 1% of capital · daily loss limit 3% · position ≤ 10% of capital ·
minimum reward:risk · max open positions · kill switch / AI-pause · paper by default.

## 5. Nightly scan — 21:00 IST before each trading day
1. Refresh data: today's NSE bhavcopy (all stocks, daily) + today's 1-min candles
   for NIFTY 50 and the indices.
2. Next-day levels per NIFTY 50 stock: floor pivots, CPR (+ width), Camarilla,
   two-day relationships (from the Pivot Boss module).
3. News: headlines per stock (Google News, Economic Times, Livemint, Business
   Standard feeds) + official NSE corporate announcements and the results/event
   calendar. Headlines and links only.
4. Sentiment: Claude (Max plan, CLI) scores each headline — sentiment −1..+1,
   materiality, confidence — rolled up per stock.
5. Output per stock: levels, two-day pattern, news + sentiment, event flags,
   final allowed directions (long / short / both / none). Saved to the database,
   shown on a UI page, and summarised to Telegram.
