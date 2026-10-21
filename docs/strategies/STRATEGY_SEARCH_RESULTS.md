# Intraday strategy search — NSE (NIFTY 50), honest results

> **Verdict (Strategist, 2026-10-21). Read this first.**
> 1. **Nothing survived.** 0 of 18 pre-registered selection tests qualified in IS. There were no finalists, and no strategy is fit to build or trade.
> 2. **About 221 configuration runs over 34 distinct signal rules** were tried: Pivot Boss, the 6 live screener strategies, and 9 pre-registered rules. Every intraday rule loses after costs except one post-hoc lead, C5, at about +0.04 R. That lead fails the IS rule and the live R:R gate and is gone at 0.07% slippage.
> 3. **Costs are now solved** (dATR stops give cost ≈ 0.04–0.06 R), but **no rule shows a gross edge above about +0.08 R** on 1m NIFTY 50 data. The bottleneck is the edge itself, and 126 sessions cannot measure an edge that small.
> 4. **The 2026-09-01 → 10-19 OOS window is spent.** Any future claim must be validated on sessions after 2026-10-19 or on never-seen pre-2026-04 history.
> 5. **Next step (see "THE PLAN"):** one frozen test of C5's daily cross-sectional reversal hypothesis on about 10 years of never-seen NSE bhavcopy, which has the power to detect it. If it fails, stop single-stock intraday directional research at this capital.

A team of roles runs this search. Each role appends its own section below and
does not rewrite the others. Losing results are reported as losses.

| Item | Value |
|---|---|
| Universe | NIFTY 50 constituents (`apps.market_data.constants.NIFTY_50_SYMBOLS`, 49 names) and the indices |
| Data | Angel `1m` bars, 2026-04-20 → 2026-10-19 (~125 sessions). NSE bhavcopy `1d` for all stocks |
| **In-sample (IS)** | **2026-04-20 → 2026-08-31.** Every role except the OOS validator works only on these dates |
| **Out-of-sample (OOS)** | **2026-09-01 → 2026-10-19.** Touched once, by the OOS-validator role, with `--allow-oos` |
| Prior work | `PIVOT_BOSS_NSE_RESULTS.md`: every Pivot Boss setup lost money after costs. Costs and slippage came to about 0.2 R per trade at a 20% notional cap |
| Owner rules | `TRADING_RULES.md`: breakeven, then ATR trail; losing-streak cooldown; flat by 15:14 |
| Raw outputs | `backend/research_output/strategy_search/` (gitignored) |

---

## Harness (Engineer role, 2026-10-21)

Code is in `backend/plugins/strategy_search/`. The CLI is
`backend/apps/market_data/management/commands/strategy_search.py` and the tests
are in `backend/tests/unit/test_strategy_search.py` (33 tests, pure Python, no
DB). The harness reads the DB only and never calls a broker.
`plugins/strategy_screener/` is not modified.

### How to run

```bash
cd backend
# config as loaded from settings + list of enabled screener strategies
nice -n 10 .venv/bin/python manage.py strategy_search info

# live screener strategies (all enabled ones, or a subset) on IS
nice -n 10 .venv/bin/python manage.py strategy_search screener --period is
nice -n 10 .venv/bin/python manage.py strategy_search screener --period is \
    --strategies "Breakout Long,Breakdown Short" --sensitivity-cap 100,500 --label scr_breakouts_is

# any new strategy: a factory returning an object with .name and .generate(SymbolData)
nice -n 10 .venv/bin/python manage.py strategy_search run \
    --source plugins.strategy_search.examples:OpeningRangeBreakout \
    --source-kwargs '{"minutes": 15}' --period is

# tests
.venv/bin/python -m pytest -q -p no:warnings tests/unit/test_strategy_search.py
```

Flags: `--symbols A,B` (default: NIFTY 50) · `--start/--end` (narrow the
period) · `--risk-pct 0.5` · `--max-open N` / `--max-position-pct P` (override
the RiskEngine settings) · `--sensitivity-cap 100,500` (extra sim passes with
larger notional caps on the same signals, e.g. MIS leverage) ·
`--sensitivity-max-open N` · `--no-trail` · `--no-cooldown` · `--min-rr 1.5`
(enforce the RiskEngine R:R gate) · `--slippage-pct` · `--warmup 3` (sessions
before start, used for indicator warm-up only) · `--label` · `--out-dir`.

**OOS guard.** `--period oos`, `--period all`, or any `--end` after
2026-08-31 is refused (`CommandError: REFUSED …`) unless `--allow-oos` is
passed. The guard is enforced in three layers: `periods.resolve_period`, the
`CandleSource`, which raises before running any query past its `max_date`, and
the trading calendar, which never reads sessions past `max_date`.

Outputs go to `research_output/strategy_search/<label>/<config>/`:

- `trades.csv`: one row per signal. `status` is `taken`, `shadow` (cooldown)
  or `skipped`, with `skip_reason`. Rows carry fills, qty, notional, risk ₹,
  gross, costs, net, R net/gross, MFE/MAE in R and the exit reason.
- `daily.csv`: per-day status (normal, cooldown or reduced), net P&L, shadow
  net and max concurrent positions.
- `summary.json` and `summary.md`: n, win %, average win and loss, PF, gross,
  costs, net, R per trade (net, gross and cost), max drawdown (per trade and
  per day), trades per day, cost share of gross wins, % of trades
  notional-capped, exit reasons, skips, phase-1 rejects, and a per-strategy
  breakdown.

Python API, for use from scripts or notebooks:

```python
from plugins.strategy_search.periods import resolve_period
from plugins.strategy_search.runner import run_search
from plugins.strategy_search.sim import SimConfig, Intent
res = run_search(source, symbols, resolve_period("is"),
                 [SimConfig.from_settings(label="base"),
                  SimConfig.from_settings(label="cap500", max_position_pct=500)],
                 out_dir="research_output/strategy_search/x")
res[0].summary()
```

### How it works

1. **Candle source** (`data.py`). Loads one symbol at a time: its 1m bars for
   the period plus `--warmup` earlier sessions, and its bhavcopy history.
   Intraday bars are cut to 09:15–15:14 IST. Each `DaySession` exposes
   `bars`, `prev_daily` (the prior session's bhavcopy) and `daily_history`
   (sessions strictly before the day). The day's own bhavcopy bar is never
   exposed, because it is known only after the close. `resample(bars, n)`
   builds 3, 5 or 15m bars aligned to 09:15 that never span sessions. A
   missing minute gives a thinner bar, not a synthetic one. The session
   calendar is the set of `RELIANCE` bhavcopy dates. Beyond the data it falls
   back to `market_calendar`. Peak RSS in the smoke run was about 165 MB.
2. **Signal source.** `generate(SymbolData)` returns `Intent`s with these
   fields: `strategy`, `symbol`, `day`, `side` ±1, `signal_t` (epoch seconds
   when the signal is *known*, i.e. the close of the signal bar), an absolute
   `stop` (which defines 1R), an optional `target`, `trail`, `trail_tf`
   (minutes, for the trail ATR), an optional `time_exit_t`, `priority` and
   `meta`. See `examples.py` for a template.
3. **Phase 1: path per intent** (`sim.resolve_intent`, per symbol). The path
   does not depend on position size.
   - Entry is at the **open of the first 1m bar with `t ≥ signal_t`**, with
     0.02% slippage against the trade. The intent is rejected if that fill is
     already beyond the stop or target, or if the bar opens after 14:30.
   - Stops and targets are checked **intrabar on every 1m bar**, including
     the entry bar. **If a bar touches both, the stop counts first.** A bar
     that opens through the stop fills at its open (worse). A bar that opens
     through the target also fills at its open (better).
   - **Owner trail:** once the best price since entry reaches +1R of the
     entry fill, the stop moves to the entry price. After that it is
     `max(entry, best − 1.5 × ATR)`, mirrored for shorts, and it only ratchets.
   - The trail is updated after each 1m bar from that bar's extreme, using the
     ATR published at the bar's close. The new stop takes effect from the next
     bar. A bar that tags +1R and then reverses within the same minute is
     therefore not stopped at breakeven.
   - The ATR is the live window: Wilder ATR(14) over the last 16 completed
     `trail_tf` bars, continuous across sessions. This is what
     `trailing_stop.atr_for_symbol` reads.
   - An optional time exit fills at a bar open. Otherwise the trade exits at
     the **close of the last bar ≤ 15:14**. Exit reasons are `stop`,
     `breakeven`, `trail`, `target`, `time` and `eod`.
4. **Phase 2: portfolio** (`sim.PortfolioSimulator`). Candidates from all
   symbols are walked in entry-time order. Ties go to higher `priority`, then
   symbol. The rules applied:
   - **Cooldown:** 3 consecutive loss days (net < 0, counting only days with
     trades) block entries on the next 2 NSE trading days, which correctly
     skips weekends and holidays. Signals on those days are recorded as
     `shadow` rows with their would-be P&L. After that, risk is ×0.5 until the
     first profitable day. Days inside the cooldown don't count toward a new
     streak. A day with net exactly 0 breaks the streak.
   - One position per symbol.
   - **RiskEngine** (from `settings.ALPHADESK`, currently max open positions
     **3** and position notional ≤ **10%** of capital):
     - risk ≤ 1% × multiplier;
     - no new entries once the realized net day loss is ≥ 3%;
     - a trade is rejected if day loss + its risk > 4.5%;
     - optionally, the min R:R.
   - **Sizing:** qty = ⌊₹5,00,000 × 0.5% × multiplier ÷ |entry fill −
     stop|⌋, capped by the position cap. Capital is fixed (no compounding).
   - A position frees its slot for entries strictly after the 1m bar it
     exited in.
   - **Costs** per executed order (`costs.py`):
     - brokerage min(₹20, 0.1%);
     - STT 0.025% on the sell side;
     - NSE transaction charge 0.00297%;
     - SEBI ₹10/crore;
     - stamp duty 0.003% on the buy side;
     - GST 18% on brokerage + transaction + SEBI.
5. **Sensitivity.** Phase 1 runs once. Each extra `SimConfig` (for example
   `--sensitivity-cap 500`) only re-sizes the same candidates.

### Live-screener adapter (`screener_adapter.py`)

The adapter runs the real `ScreenerEngine`, `CandleStore`, `IndicatorEngine`,
conditions and `STRATEGIES` unchanged. For each (symbol, day) it builds a
fresh single-symbol engine, the way the live `bootstrap()` does before the
open:

- **Pivots:** the prior session's bhavcopy H/L/C.
- **Warm-up:** the prior session's 5m bars, resampled from the DB, are seeded
  into the 5m buffer. Then the 5m and 15m indicators are initialised.
- **Replay:** each 1m bar is replayed as four ticks (O, H, L, C) carrying the
  bar's naive-IST timestamp.

The engine's own code applies candle aggregation, VWAP, per-strategy
cooldowns, the 20-bar symbol cooldown and the one-signal-per-tick dedupe. A
1m bar is completed by the next minute's first tick, so the signal timestamp
equals the next bar's start (`signal_t`), and the simulator fills at that
bar's open, not at the engine's `last_close`.

The one change is for speed: the adapter skips recomputing the 1m indicator
snapshot, which no enabled strategy reads. On RELIANCE, HDFCBANK and TCS for
2026-06-01 → 06-12 this produced **identical signals** to the unmodified
engine (32/34/25) and ran about 5× faster (0.4 s vs 2.3 s per symbol for 10
sessions).

**Exits are assumed.** The live screener only sends alerts and has no exit
logic. The simulator brackets each trade at the signal's own stoploss and
target, then applies the owner trail and flat-by-15:14.

| Strategy (enabled) | Side | Window | Stop | Target |
|---|---|---|---|---|
| Breakout Long / Breakdown Short | L / S | 09:20–14:30 | 5-bar 5m swing low/high ∓ **₹2 fixed** (not %) | 2.0R |
| VWAP Bounce Long / VWAP Rejection Short | L / S | 10:30–14:00 | 1.2 × ATR14 (5m) | 1.5R |
| Pivot Rejection Long / Short (classic S1/R1) | L / S | 10:00–14:00 | 1.0 × ATR14 (5m) | 1.5R |
| BB Squeeze Breakout | L | 09:45–14:00 | 1.2 × ATR14 (5m) | 1.5R |
| EMA Crossover Trend | L | 10:00–14:00 | 1.2 × ATR14 (5m) | 1.5R |

BB Fade and Morning Range Break are `enabled=False` and are not run. The
trail ATR uses 5m bars.

### Assumptions and known limits

- R targets are set by the engine from its `last_close` entry. The real
  next-open fill shifts the realised R:R slightly. R in the results is always
  measured from the actual fill to the initial stop.
- The first IS session (2026-04-20) has no prior session in the DB, so it gets
  no pivots and no 5m warm-up. Pivot strategies cannot fire that day.
- The daily-loss gate uses **realized** net P&L only. The live
  `portfolio.day_pnl` may include unrealized P&L.
- `min_rr` is off by default. ~~The trail-only strategies have no target, so
  the RiskEngine R:R gate is not applicable to them.~~ **Corrected in Step 6:**
  the live RiskEngine is stricter. It rejects any draft with no target
  (`_check_fields`: target ≤ 0) and any draft with `round(rr, 2) < 1.5`.
  Pass `--live-rr-gate` (`SimConfig.live_rr_gate=True`) to reproduce that
  exactly. The "real RiskEngine" claim in Steps 2–5 covers only max open
  positions, the position-% cap, max risk per trade and the daily-loss gate.
  Criterion 7 (R:R and target required) and criterion 8 (confidence and
  regime) were not modelled.
- `trail_next_stop` and `CooldownTracker` are research-local copies. Tests
  check them against the live `apps.trading.services.trailing_stop.next_stop`
  (500 random cases) and `cooldown.evaluate_cooldown` (60 random 40-day
  sequences); both match. They will fail loudly if the live rules change.
- The same caveats as the Pivot Boss study apply: 1m granularity, today's
  NIFTY 50 list (survivorship bias), no bars after 15:14, and costs as stated.

### Smoke test (IS only: RELIANCE + HDFCBANK, 2026-05-04 → 05-15)

These runs only check that the plumbing works. They say nothing about
whether any strategy has an edge.

| Run | n | win % | PF | net ₹ | R/trade (gross / cost) | trades/day | notes |
|---|---|---|---|---|---|---|---|
| screener, base (10% cap, 3 open) | 34 | 47.1% | 0.48 | −4,152 | −0.212 (−0.020 / 0.191) | 3.4 | 78 signals, 44 skipped `symbol_busy`, 100% notional-capped |
| screener, cap 100% | 34 | 47.1% | 0.83 | −7,384 | −0.092 (−0.020 / 0.071) | 3.4 | larger size → costs shrink in R, but the loss in ₹ grows |
| example ORB15 | 14 | 35.7% | 0.36 | −2,922 | −0.415 | 1.4 | the cooldown fired twice (5/7–8, 5/15) and reduced risk on 5/11 |

With the 10% RiskEngine cap, every screener trade was notional-capped and
costs came to about **0.19 R per trade**. That is the same cost drag the
Pivot Boss study found. A strategy needs a gross edge well above 0.2 R per
trade at this size, or wider stops relative to price.

---

## Step 1 — Bias verification (IS) (Bias-analyst role, 2026-10-21)

**Question.** Are the Pivot Boss "survivors" real, tradeable edges after NSE
costs, or mechanical artefacts? The survivors are open-vs-PP side persistence,
PP untouched → directional day, Camarilla H3/L3 acting as continuation, and
small-gap fill.

**Scope.** IN-SAMPLE only (2026-04-20 → 08-31). The data is 49 NIFTY 50
names, **4,494 stock-days** over 92 sessions; the first session has no prior
bhavcopy, and a few days with incomplete 1m data are dropped. ATR-bucketed
cells need 15 prior bhavcopy sessions, so they start mid-May. No OOS date was
read.

### Verdicts

| Claim | Verdict | One-line reason |
|---|---|---|
| Open > PP → close > PP (and the mirror) | **ARTEFACT** | Observed persistence (66.6% / 68.0%) sits *at or below* the mechanical null (67.4% / 69.6%). It is just "the open is already on that side". |
| Open side of PP as a direction (09:16 → 15:14) | **ARTEFACT** | Signed return +0.012% (t +0.36). Market-neutral +0.017% (t +0.89). The continuation rate (45% / 50%) is *below* the unconditional base. Every harness variant is ≈ 0 R before slippage. |
| First-15 / first-30-min close vs PP | **ARTEFACT** | −0.015% / −0.049% signed (t −0.5 / −1.5). |
| PP untouched by 10:15 → directional day | **ARTEFACT** (as a trade) | The day's extension is already in the price by 10:15. The rest of the day is −0.029% signed (t −0.4), market-neutral +0.001%. Days where PP *was* touched do no worse. |
| Open outside CPR → trade that side | **ARTEFACT** | Above TC long −0.044%, below BC short +0.035%. That is the IS index drift (−0.037%/day) and nothing else. |
| H3/L3 first touch = continuation | **ARTEFACT** (geometry) | H4/L4 comes first 68% of the resolved races against L3/H3. A driftless random walk gives 2/3, because H4 is half as far. With a symmetric stop (prior close) it is 49 / 51. Before slippage: −0.01 to −0.03 R. |
| Small-gap fill (0.25–1.0% gaps) | **REAL BUT TOO SMALL** | +0.055 / +0.064 R per trade before slippage (t +1.0 / +1.3, weak). Slippage alone is 0.19 R. Net −0.31 R even at MIS size. |
| Tiny-gap fill (< 0.25%) | **REAL BUT TOO SMALL** | Fill rates are high (62% after 09:16 by 10:14 at 0–0.25%). But the average target is 0.11–0.18% and round-trip friction is 0.12–0.17%, so it cannot pay. Harness: −1.4 R gross. |

**Nothing in this step is a REAL EDGE AFTER COSTS.** The descriptive Pivot
Boss facts replicate. As trade signals they are either mechanical (the
condition is already in the price) or too small to survive friction. The
Step 2 suggestion to use them as filters should be treated as a hypothesis
with no IS support from this step: none of them shifts the forward return
from the entry moment.

### How it was measured

- **Levels** come from the prior bhavcopy only. The **open** is the 09:15 1m
  open. The **tradeable entry** is the **09:16 open** (the next bar after the
  signal is known), or 09:30 / 09:45 / 10:15 for the intraday versions. The
  exit is the **15:14 close**.
- The day's own bhavcopy close is used **only** as an outcome, to replicate the
  Pivot Boss "close vs PP" numbers. It never forms a signal.
- **Signed return** = side × return, where side is +1 above PP and −1 below.
  *cont %* = share of days with signed return > 0.
- **Net %** = signed return − 0.04% slippage − exact NSE MIS costs at ₹50k
  notional (the RiskEngine 10% cap) and ₹5L (100%, MIS sensitivity). These
  hypothetical trades have no stop.
- **t** is day-clustered: one mean per date across the 49 stocks, then a t-test
  over dates. The 49 stocks are strongly correlated within a day, so the
  pooled n overstates the evidence about 49×. Because t is on the mean of
  daily means, its sign can differ from the pooled mean in thin cells.
- **Mechanical null**: for each open > PP day, the null is P(open·(1+r) > PP)
  with r drawn from the *same stock's* unconditional IS open→close returns.
  The observed rate is compared with the mean of that.
- **Harness runs** (`plugins/strategy_search`) cover every tradeable variant:
  next-1m-open entry, stop first, 0.02% slippage per side, owner trail
  (breakeven at +1R, then 1.5×ATR on 5m), flat by 15:14, no entries after
  14:30. Configs:
  - `base`: RiskEngine 10% cap, 3 open positions, cooldown. When several
    signals share a bar, the larger |open−PP| % goes first.
  - `cap100` / `cap500`: notional cap 100% / 500% (MIS sensitivity).
  - `all_cap10` / `all_cap500`: every signal taken (no slot limit, no
    daily-loss gate, no cooldown), which measures the per-trade edge.

### 1a. Open vs PP persistence against a mechanical null (bhavcopy close)

| Condition | n | Close on the same side of PP | Mechanical null | Excess | Close beyond the open, same direction | Unconditional, all days |
|---|---|---|---|---|---|---|
| open > PP | 2244 | 66.6% | 67.4% | **−0.8 pp** | 45.1% | close > open 47.0% |
| open < PP | 2242 | 68.0% | 69.6% | **−1.6 pp** | 50.4% | close < open 53.0% |

The 64–69% "persistence" (book: 49–53%) is fully explained by where the open
already is. Once the open is above PP, the close is no more likely to be
above the open than on an average day. It is slightly *less* likely.

### 1b. Open side of PP as a hypothetical trade (no stop)

| Condition | n | Days | Cont % | Mean % | Median % | t | Net % @₹50k | Net % @₹5L | Net win % @₹50k |
|---|---|---|---|---|---|---|---|---|---|
| Long every stock, 09:16→15:14 (drift baseline) | 4494 | 92 | 47.1% | −0.037 | −0.069 | −0.69 | −0.209 | −0.122 | 40.1% |
| open > PP → long, 09:16→15:14 | 2244 | 92 | 46.7% | −0.025 | −0.080 | −0.76 | −0.197 | −0.109 | 39.7% |
| open < PP → short, 09:16→15:14 | 2242 | 92 | 52.1% | +0.048 | +0.055 | +1.38 | −0.124 | −0.037 | 45.3% |
| Both sides, 09:16→15:14 | 4486 | 92 | 49.4% | +0.012 | −0.015 | +0.36 | −0.161 | −0.073 | 42.5% |
| Both sides, **market-neutral** (minus that day's mean over the 49) | 4486 | 92 | 49.4% | +0.017 | −0.012 | +0.89 | −0.156 | −0.068 | 42.0% |
| Both sides, 09:15 open (auction price) →15:14 | 4486 | 92 | 47.7% | −0.026 | −0.056 | −0.78 | −0.199 | −0.111 | 41.3% |
| Both sides, bhavcopy O→C (descriptive) | 4486 | 92 | 47.7% | −0.019 | −0.064 | −0.55 | −0.191 | −0.103 | 42.1% |

The short side looks better only because IS had a negative intraday drift.
Market-neutral, the effect is +1.7 bp per trade against about 16 bp of
friction at the 10% cap and about 7 bp at 100%.

### 1c. By open distance from PP and by gap (09:16→15:14, signed by PP side)

Distance from PP in daily-ATR14 units:

| Side, bucket | n | Days | Cont % | Mean % | t | Net % @₹50k | Net % @₹5L |
|---|---|---|---|---|---|---|---|
| long, 0–0.25 ATR | 1031 | 78 | 48.1% | −0.008 | −0.46 | −0.181 | −0.093 |
| long, 0.25–0.5 | 556 | 74 | 43.9% | −0.083 | −1.29 | −0.256 | −0.168 |
| long, 0.5–1 | 285 | 61 | 42.1% | −0.144 | −1.25 | −0.316 | −0.229 |
| long, ≥ 1 | 47 | 23 | 48.9% | +0.108 | +0.39 | −0.064 | +0.023 |
| short, 0–0.25 ATR | 1096 | 78 | 55.6% | +0.121 | +1.74 | −0.051 | +0.037 |
| short, 0.25–0.5 | 535 | 73 | 49.9% | +0.016 | +0.13 | −0.156 | −0.068 |
| short, 0.5–1 | 213 | 47 | 42.7% | −0.130 | +0.49 | −0.302 | −0.215 |
| short, ≥ 1 | 39 | 21 | 35.9% | −0.746 | −0.49 | −0.918 | −0.831 |

Distance from PP in % of PP:

| Side, bucket | n | Days | Cont % | Mean % | t | Net % @₹50k | Net % @₹5L |
|---|---|---|---|---|---|---|---|
| long, 0–0.25% | 609 | 91 | 49.9% | +0.025 | +0.33 | −0.147 | −0.060 |
| long, 0.25–0.5% | 507 | 88 | 47.9% | +0.041 | −0.98 | −0.132 | −0.044 |
| long, 0.5–1% | 631 | 88 | 44.4% | −0.083 | −2.01 | −0.256 | −0.168 |
| long, 1–2% | 406 | 75 | 43.8% | −0.097 | +0.33 | −0.269 | −0.182 |
| long, ≥ 2% | 91 | 38 | 47.3% | +0.009 | −0.09 | −0.163 | −0.075 |
| short, 0–0.25% | 673 | 91 | 55.0% | +0.075 | +1.04 | −0.097 | −0.010 |
| short, 0.25–0.5% | 521 | 87 | 54.3% | +0.093 | +2.04 | −0.079 | +0.008 |
| short, 0.5–1% | 636 | 86 | 52.4% | +0.120 | +1.62 | −0.052 | +0.035 |
| short, 1–2% | 340 | 61 | 43.8% | −0.146 | +0.30 | −0.318 | −0.231 |
| short, ≥ 2% | 72 | 32 | 45.8% | −0.257 | −0.10 | −0.428 | −0.342 |

The further the open is from PP, the **worse** the continuation. Mild
mean-reversion beyond about 0.5 ATR or 1%, not persistence. The best cells
are "short, near PP": +0.12% at 0–0.25 ATR (t 1.7) and +0.09% at 0.25–0.5%
(t 2.0). They are 2 of 18 cells, carry the −0.037% drift tailwind, have no
mirror on the long side, and are still negative at the 10% cap. They were
**not** promoted. This is multiple-testing noise on one regime.

Gap (09:15 open vs prior close), selected rows. All 20 cells are in
`tables.md`.

| Cell | n | Mean % | t | Net % @₹50k |
|---|---|---|---|---|
| short, \|gap\| 0–0.25%, gap down (with PP side) | 415 | +0.146 | +1.99 | −0.027 |
| long, \|gap\| 0–0.25%, gap up (with PP side) | 399 | −0.014 | +0.39 | −0.187 |
| long, \|gap\| 0.5–1%, gap up | 540 | −0.113 | −1.86 | −0.285 |
| short, \|gap\| 0.5–1%, gap down | 451 | −0.007 | +1.44 | −0.179 |
| long, \|gap\| 1–2%, gap up | 272 | +0.093 | +0.42 | −0.080 |
| short, \|gap\| ≥ 2%, gap down | 43 | −0.247 | −0.44 | −0.418 |

Gap size does not create a usable directional edge either. No cell is net
positive at the 10% cap with n ≥ 50 except noise-level ones.

### 1d. Intraday versions (entry at the next 1m open, exit 15:14, signed)

| Condition | n | Days | Cont % | Mean % | Median % | t | Net % @₹50k | Net % @₹5L |
|---|---|---|---|---|---|---|---|---|
| First-15m close (09:29) vs PP → enter 09:30 | 4491 | 92 | 48.5% | −0.015 | −0.028 | −0.48 | −0.187 | −0.099 |
| First-30m close (09:44) vs PP → enter 09:45 | 4494 | 92 | 46.9% | −0.049 | −0.067 | −1.54 | −0.221 | −0.134 |
| PP untouched 09:15–10:14, above → long 10:15 | 1001 | 89 | 45.0% | −0.048 | −0.099 | −1.01 | −0.221 | −0.133 |
| PP untouched 09:15–10:14, below → short 10:15 | 990 | 89 | 50.8% | −0.010 | +0.018 | +1.53 | −0.183 | −0.095 |
| PP untouched, both sides | 1991 | 92 | 47.9% | −0.029 | −0.038 | −0.41 | −0.202 | −0.114 |
| PP untouched, market-neutral | 1991 | 92 | 49.4% | +0.001 | −0.013 | −0.11 | −0.171 | −0.083 |
| PP touched by 10:15 (reference, side = open vs PP) | 2495 | 92 | 52.1% | +0.029 | +0.048 | +0.72 | −0.143 | −0.055 |
| CPR: open above TC → long 09:16 | 1969 | 89 | 45.9% | −0.044 | −0.104 | −1.40 | −0.217 | −0.129 |
| CPR: open below BC → short 09:16 | 1903 | 92 | 51.8% | +0.035 | +0.049 | +1.32 | −0.137 | −0.050 |
| CPR: open inside (long, reference) | 622 | 92 | 48.9% | −0.019 | −0.032 | +0.40 | −0.191 | −0.104 |

"PP untouched → 65% close beyond L1" (Pivot Boss A) is a statement about the
whole day. The move that makes it true has mostly happened by 10:15. What is
left to trade from 10:15 is zero.

### 1e. Stability inside IS

| Condition | Apr 20–Jun 30: n / mean % / t | Jul 1–Aug 31: n / mean % / t |
|---|---|---|
| Open side of PP, 09:16→15:14 | 2347 / +0.011 / +0.20 | 2139 / +0.013 / +0.32 |
| PP untouched by 10:15, 10:15→15:14 | 1129 / −0.044 / −0.37 | 862 / −0.011 / −0.19 |
| Long every stock (drift) | 2352 / −0.067 / −0.78 | 2142 / −0.004 / −0.04 |

The result is flat in both halves, so no regime dependence is hiding an edge.

### 1f. Gap fill rates (prior close touched, IS)

| \|gap\| % | n | Filled 09:15–10:14 (incl. the 09:15 bar) | Filled after 09:16, by 10:14 | Filled by 15:14 | Mean \|gap\| % |
|---|---|---|---|---|---|
| 0–0.25 | 1400 | 81.4% | 61.5% | 86.8% | 0.11 |
| 0.25–0.5 | 1035 | 62.0% | 51.1% | 73.5% | 0.38 |
| 0.5–1 | 1062 | 47.3% | 41.8% | 64.0% | 0.70 |
| 1–2 | 509 | 21.6% | 20.2% | 37.9% | 1.33 |
| ≥ 2 | 93 | 14.0% | 12.9% | 23.7% | 3.04 |

The "after 09:16" column is the tradeable one. It matches the Pivot Boss
table (60.9% at 0–0.25%), which evidently excluded the opening bar. A fill
inside the 09:15 bar is not tradeable.

### 1g. Harness: every tradeable variant, net of costs

Gross R includes slippage and excludes fees. "Pre-slip R" adds the 0.02%
per-side slippage back. It is shown for comparability with the Pivot Boss
"gross before slippage". The t for pre-slip R is day-clustered.

Every signal taken (`all_cap10`, 10% cap):

| Variant | n | Win % | PF | Net ₹ | R/trade net | Gross R | Pre-slip R (t) | Cost R | Median stop % |
|---|---|---|---|---|---|---|---|---|---|
| pp_open_side: 09:16, stop PP | 3695 | 35.7% | 0.54 | −354,628 | −1.230 | −0.295 | −0.011 (−0.20) | 0.939 | 0.64% |
| pp_open_side_atrstop: stop 0.5×ATR(d) | 3802 | 43.5% | 0.63 | −350,104 | −0.167 | −0.045 | −0.008 (−0.29) | 0.123 | 1.10% |
| pp_first15_side: 09:30, stop PP | 4481 | 33.2% | 0.52 | −413,865 | −0.656 | −0.158 | −0.008 (−0.25) | 0.498 | 0.62% |
| pp_first30_side: 09:45, stop PP | 4490 | 32.9% | 0.48 | −460,957 | −0.625 | −0.177 | −0.042 (−1.87) | 0.447 | 0.67% |
| pp_untouched_1015: 10:15, stop PP | 1991 | 38.5% | 0.52 | −219,481 | −0.258 | −0.094 | −0.044 (−1.14) | 0.164 | 1.16% |
| cpr_open_side: stop far CPR edge | 3494 | 38.3% | 0.55 | −350,561 | −1.039 | −0.247 | −0.011 (−0.22) | 0.783 | 0.81% |
| cam_touch_tgtH4: stop prior close, target H4/L4 | 2511 | 42.1% | 0.46 | −208,070 | −0.596 | −0.158 | −0.026 (−1.48) | 0.438 | 0.57% |
| cam_touch_trail: same, no target | 2657 | 40.2% | 0.56 | −211,311 | −0.568 | −0.148 | −0.021 (−1.26) | 0.421 | 0.59% |
| cam_touch_stopL3: stop opposite band, target H4/L4 | 2599 | 51.2% | 0.53 | −220,137 | −0.210 | −0.063 | −0.019 (−1.91) | 0.147 | 1.10% |
| cam_5mclose_tgtH4: 5m close beyond band, stop C | 1982 | 49.1% | 0.44 | −171,401 | −0.270 | −0.071 | −0.011 (−0.84) | 0.199 | 0.71% |
| gapfill_60m: 0.25–1% gap, target prior close, stop 1×gap, out 10:15 | 1536 | 39.3% | 0.40 | −108,650 | −0.749 | −0.131 | **+0.055 (+1.01)** | 0.618 | 0.56% |
| gapfill_eod: same, no time exit | 1536 | 43.3% | 0.45 | −106,230 | −0.740 | −0.122 | **+0.064 (+1.32)** | 0.618 | 0.56% |
| gapfill_tiny_60m: 0.10–0.25% gap | 267 | 13.9% | 0.05 | −21,415 | −6.532 | −1.404 | +0.018 (+0.70) | 5.019 | 0.18% |

The same signals sized up (`all_cap500`, MIS 5×) cut the cost R to
0.05–0.28, but **every** variant stays negative:

| Variant | R/trade net |
|---|---|
| pp_open_side_atrstop | −0.096 |
| cam_touch_stopL3 | −0.122 |
| cam_5mclose_tgtH4 | −0.143 |
| pp_untouched_1015 | −0.156 |
| gapfill_eod | −0.308 |
| gapfill_60m | −0.317 |
| pp_open_side | −0.572 |

Owner-rules portfolio (`base`: 10% cap, 3 open, cooldown, trail) and the
MIS sensitivity runs:

| Variant | base n / net ₹ / R | cap100 net ₹ / R | cap500 net ₹ / R |
|---|---|---|---|
| pp_open_side | 219 / −16,251 / −0.103 | −39,100 / −0.071 | −29,646 / −0.070 |
| pp_open_side_atrstop | 183 / −29,192 / −0.248 | −53,978 / −0.212 | −53,978 / −0.212 |
| pp_first15_side | 213 / −14,173 / −0.170 | −55,968 / −0.140 | −71,745 / −0.139 |
| pp_first30_side | 219 / −17,557 / −0.248 | −49,120 / −0.166 | −63,508 / −0.163 |
| pp_untouched_1015 | 207 / −20,856 / −0.189 | −55,824 / −0.158 | −59,702 / −0.158 |
| cpr_open_side | 213 / −15,665 / −0.110 | −47,907 / −0.112 | −51,545 / −0.111 |
| cam_touch_tgtH4 | 514 / −42,437 / −0.596 | −164,298 / −0.277 | −216,375 / −0.289 |
| cam_touch_trail | 487 / −32,186 / −0.577 | −144,955 / −0.287 | −196,208 / −0.287 |
| cam_touch_stopL3 | 432 / −40,698 / −0.213 | −114,325 / −0.145 | −121,724 / −0.145 |
| cam_5mclose_tgtH4 | 564 / −48,682 / −0.266 | −177,911 / −0.133 | −189,343 / −0.132 |
| gapfill_60m | 210 / −10,942 / −0.762 | −42,529 / −0.296 | −92,341 / −0.294 |
| gapfill_eod | 228 / −8,545 / −0.670 | −32,051 / −0.304 | −87,814 / −0.282 |
| gapfill_tiny_60m | 130 / −10,660 / −10.8 | −54,416 / −4.12 | −149,858 / −4.15 |

Notes:
- Before slippage every PP/CPR/Camarilla variant sits at −0.04 to −0.01 R.
  That is the same "no raw edge" result as the Pivot Boss study, now for the
  survivors as well.
- Stops at PP or a CPR edge are often a few ticks away. This is where the
  enormous cost R in `all_cap10` comes from (pp_open_side 0.94 R, cpr 0.78 R).
  A vol-based stop (0.5×ATR) cuts friction to 0.12 R but leaves a gross of
  −0.008 R.
- **The H3/L3 "continuation" is geometry, not order flow.** From H3, H4 is
  k/4 away and L3 is k/2 away (k = 1.1 × prior range). A driftless walk hits
  the nearer one first 2/3 of the time. Observed: 1,319 H4-first vs 626
  stop-first, or **67.8%** of resolved trades (Pivot Boss: 54.2 / 23.2 →
  70%). With a symmetric stop at the prior close (also k/4 away) it is 1,029
  vs 1,062 (**49%**). The race result carries no information.
- Gap fill is the only family with positive pre-slippage R (+0.06 R, t ≈ 1,
  same as the Pivot Boss magnet). It is the closest thing to a real effect
  here, and still about 3× too small for its own slippage. The fill is real;
  the edge is too small.

### Reproduce

```bash
cd backend
nice -n 10 .venv/bin/python manage.py strategy_bias      # IS only, ~1 min, ~220 MB RSS
.venv/bin/python -m pytest -q -p no:warnings tests/unit/test_strategy_bias.py   # 7 tests
```

- Code is in `plugins/strategy_search/bias.py` (features, sources, tables) and
  `apps/market_data/management/commands/strategy_bias.py`. The command has no
  `--allow-oos`; `resolve_period("is")` and the `CandleSource` guard both
  refuse dates past 2026-08-31.
- Outputs go to `research_output/strategy_search/bias_is/`:
  - `days.csv`: one row per stock-day with every feature and return.
  - `tables.md`: every table, including all 20 gap cells.
  - `harness_summary.md`, and `<variant>/<config>/` with trades, daily and
    summary files.

---

## Step 2 — Baselines (IS) (Baseline-tester role, 2026-10-21)

**Verdict: every enabled screener strategy loses money in-sample, before and
after costs.** None has a positive gross R/trade at any size. Nothing was
tuned: the live strategy definitions, the adapter's exits (the signal's own
stop/target + owner trail + flat by 15:14) and the owner rules were used
as-is.

### What was run

- **Universe/period:** NIFTY 50 (49 names), IS 2026-04-20 → 2026-08-31,
  93 sessions. No OOS date was loaded (`resolve_period("is")`, CandleSource
  `max_date` = 2026-08-31).
- **Phase 1 once:** the real `ScreenerEngine` with all 8 enabled
  strategies → **14,942 signals**, all resolvable into trades (237 s,
  one symbol at a time, `nice -n 10`).
- **Phase 2, re-sizing the same signals:**
  - *combined*: all strategies compete for the same portfolio slots (what
    the live screener + RiskEngine would actually do);
  - *isolated*: each strategy alone in its own portfolio.
- **Configs** (all: ₹5L, 0.5% risk/trade, 0.02% slippage/side, NSE MIS
  costs, breakeven at +1R then 1.5×ATR(5m) trail, cooldown 3 loss days → 2
  days off → half risk, flat by 15:14, one position per symbol, daily-loss
  gate 3%):

  | Config | Notional cap | Max open | Cooldown | Meaning |
  |---|---|---|---|---|
  | `base` | 10% (RiskEngine) | 3 (RiskEngine) | on | what the system can trade today |
  | `cap100` | 100% | 3 | on | 1× capital per position |
  | `cap500` | 500% | 3 | on | MIS-style 5× leverage sensitivity |
  | `raw` | 500% | 50 | off | **diagnostic only**: almost every signal taken; measures the signal's edge in R, not a tradable ₹ result |

- **Script:** a phase-2 driver over the harness (`collect_candidates` +
  `PortfolioSimulator` + `write_outputs`), saved as
  `backend/research_output/strategy_search/baselines_is/baselines.py`
  (run from `backend/` with `nice -n 10 .venv/bin/python <path>`). Outputs are in
  `backend/research_output/strategy_search/baselines_is/<group>/<config>/`
  (`candidates.pkl` holds the phase-1 output for re-use).

### Two strategies never fire: BB Squeeze Breakout and EMA Crossover Trend

They produced **0 signals in 4,557 symbol-days**. This is a property of the
live screener, not of the adapter:

- The engine seeds only **5m** bars from the prior session
  (`engine.bootstrap` step 2). The **15m** buffer is built from today's bars
  only.
- BB Squeeze needs `bb_upper` on 15m. That needs 20 bars of 15m, which come
  at 14:15 at the earliest, but its window closes at 14:00.
- EMA Crossover needs `macd_histogram` on 15m. That needs 35 bars of 15m,
  and a session has only 25.
- **So in live they can only fire if the engine process stays up across
  sessions without a fresh bootstrap.** The adapter mirrors a daily
  bootstrap. This is worth flagging to whoever owns
  `plugins/strategy_screener/` (not changed here).
- In every table below, "All 8 combined" is effectively six strategies.

### Per-strategy, isolated — `base` (RiskEngine limits: 10% notional, 3 open)

| Strategy | n | win % | avg win ₹ | avg loss ₹ | PF | net ₹ | R/trade | gross R | cost R | max DD ₹ | trades/day | cost share | notional-capped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Breakout Long | 336 | 36.0 | 310 | −413 | 0.42 | −51,255 | −0.264 | −0.117 | 0.147 | 55,239 | 3.61 | 47% | 98% |
| Breakdown Short | 339 | 42.8 | 266 | −338 | 0.59 | −26,885 | −0.202 | −0.034 | 0.168 | 27,874 | 3.65 | 45% | 99% |
| VWAP Bounce Long | 362 | 33.1 | 108 | −181 | 0.30 | −30,727 | −0.667 | −0.179 | 0.488 | 30,787 | 3.89 | 108% | 100% |
| VWAP Rejection Short | 403 | 33.5 | 106 | −170 | 0.31 | −31,147 | −0.595 | −0.093 | 0.503 | 31,522 | 4.33 | 107% | 100% |
| Pivot Rejection Long | 65 | 27.7 | 124 | −175 | 0.27 | −6,004 | −0.773 | −0.270 | 0.503 | 6,407 | 0.70 | 118% | 100% |
| Pivot Rejection Short | 81 | 32.1 | 115 | −184 | 0.30 | −7,138 | −0.659 | −0.175 | 0.483 | 7,161 | 0.87 | 108% | 100% |
| BB Squeeze Breakout | 0 | — | — | — | — | — | — | — | — | — | — | — | — |
| EMA Crossover Trend | 0 | — | — | — | — | — | — | — | — | — | — | — | — |
| **All combined** | **327** | 38.8 | 259 | −358 | **0.46** | **−38,706** | **−0.244** | −0.088 | 0.157 | 42,474 | 3.52 | 50% | 98% |

How to read the columns:

- R/trade = net R. Gross R is before brokerage/taxes but after slippage (the
  0.02%/side slippage is in the fills). Cost R = costs / risk.
- Max DD is on the trade-by-trade equity, in ₹.
- Cost share = costs / gross winning P&L.

**Combined `base`, the realistic number:**

- **−₹38,706 (−7.7% of capital) in 93 sessions.**
- Traded on 69 days, positive on only 16.
- Daily max DD was ₹42,085.
- Cooldown was active for 24 days and half-risk for 27. The 3,815 shadow
  trades on cooldown days would have netted another −₹3.1 L, so the cooldown
  saved money here.

### Per-strategy, isolated — `cap500` (MIS 5× sensitivity, 3 open)

| Strategy | n | win % | avg win ₹ | avg loss ₹ | PF | net ₹ | R/trade | gross R | cost R | max DD ₹ | trades/day | cost share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Breakout Long | 334 | 37.7 | 1,455 | −1,448 | 0.61 | −117,840 | −0.169 | −0.105 | 0.065 | 139,163 | 3.59 | 21% |
| Breakdown Short | 365 | 46.8 | 1,294 | −1,558 | 0.73 | −81,014 | −0.098 | −0.030 | 0.068 | 91,454 | 3.92 | 21% |
| VWAP Bounce Long | 396 | 37.9 | 1,749 | −2,030 | 0.53 | −237,078 | −0.337 | −0.178 | 0.159 | 240,273 | 4.26 | 37% |
| VWAP Rejection Short | 471 | 39.9 | 1,881 | −2,244 | 0.56 | −281,394 | −0.253 | −0.094 | 0.159 | 288,312 | 5.06 | 37% |
| Pivot Rejection Long | 65 | 30.8 | 2,080 | −1,821 | 0.51 | −40,328 | −0.432 | −0.270 | 0.161 | 54,155 | 0.70 | 39% |
| Pivot Rejection Short | 88 | 37.5 | 2,042 | −2,221 | 0.55 | −54,782 | −0.306 | −0.154 | 0.152 | 54,782 | 0.95 | 36% |
| **All combined** | **362** | 43.4 | 1,304 | −1,487 | **0.67** | **−100,055** | **−0.119** | −0.055 | 0.064 | 125,488 | 3.89 | 22% |

- **`cap100` is almost identical to `cap500`.**
  - Combined: n 362, PF 0.68, −₹92,393, R −0.120 (gross −0.055, cost 0.065).
  - Breakout Long: R −0.181. Breakdown Short: −0.098. VWAP Bounce: −0.349.
    VWAP Rejection: −0.261. Pivot Long: −0.441. Pivot Short: −0.326.
  - Only 11% of trades are capped at 100%, so 1× notional already removes
    nearly all of the cap.
- **Larger notional halves the cost per R, but it does not create an
  edge.** Combined net R improves from −0.244 to −0.119, but ₹ losses grow
  2.6× because each trade now carries its full 0.5% risk.
- Gross R barely moves (−0.088 → −0.055). The difference is only which
  trades fit in the 3 slots.

### Signal-level edge — `raw` (diagnostic, ~all signals taken)

| Strategy | n | win % | PF | R/trade | **gross R ± s.e.** | cost R |
|---|---|---|---|---|---|---|
| Breakout Long | 3,453 | 42.3 | 0.63 | −0.168 | **−0.099 ± 0.014** | 0.069 |
| Breakdown Short | 3,533 | 48.3 | 0.78 | −0.088 | **−0.016 ± 0.014** | 0.071 |
| VWAP Bounce Long | 601 | 37.8 | 0.51 | −0.342 | **−0.194 ± 0.043** | 0.147 |
| VWAP Rejection Short | 661 | 39.6 | 0.58 | −0.264 | **−0.112 ± 0.041** | 0.153 |
| Pivot Rejection Long | 74 | 32.4 | 0.44 | −0.402 | **−0.254 ± 0.118** | 0.148 |
| Pivot Rejection Short | 102 | 37.3 | 0.55 | −0.299 | **−0.153 ± 0.103** | 0.146 |
| All | 5,431 | 44.3 | 0.68 | −0.149 | −0.074 ± ~0.012 | 0.075 |

Median MFE across all `raw` trades is 0.57 R.

- Before costs, Breakout Long, VWAP Bounce Long, VWAP Rejection Short and
  Pivot Rejection Long have gross R more than 2 s.e. below 0. Pivot
  Rejection Short is about 1.5 s.e. below 0 (small sample). Only Breakdown
  Short is within noise of zero.
- `raw` still keeps one position per symbol and the 3% daily-loss gate, so
  `n` is less than the 14,942 signals.

### Long vs short (combined portfolio)

| Config | Long: n / win % / PF / R / gross R | Short: n / win % / PF / R / gross R |
|---|---|---|
| base | 173 / 36.4 / 0.45 / −0.244 / −0.098 | 154 / 41.6 / 0.48 / −0.245 / −0.076 |
| cap500 | 185 / 38.9 / 0.65 / −0.127 / −0.063 | 177 / 48.0 / 0.70 / −0.111 / −0.046 |
| raw | 2,613 / 41.2 / 0.62 / −0.186 / −0.112 | 2,818 / 47.1 / 0.74 / −0.116 / −0.040 |

- **The short side is consistently less bad.** Shorts have a higher win
  rate and a gross R about 0.02–0.07 better.
- NIFTY itself was nearly flat over IS (−1.3%, 24,392 → 24,080), so this is
  not just a falling market.
- Both sides lose after costs.

### Closest to break-even before costs (gross R/trade)

1. **Breakdown Short — gross −0.016 R (`raw`), −0.03 to −0.034 R at the
   RiskEngine sizes.**
   - The only strategy whose gross R is within noise of zero
     (±0.014 s.e.).
   - By month (`raw`), gross R went +0.009, +0.005, +0.033 in Apr–Jun, then
     −0.065 and −0.050 in Jul–Aug. It is not stable.
   - It is still clearly negative net: −0.088 R at best, −0.202 R at the
     10% cap.
2. **VWAP Rejection Short (−0.09 to −0.11)** and **Breakout Long (−0.10 to
   −0.12)**: genuinely negative before costs. Breakout Long is negative in
   every IS month (`raw`).
3. **Pivot Rejection Short (−0.15), VWAP Bounce Long (−0.18 to −0.19),
   Pivot Rejection Long (−0.25 to −0.27)**: the worst. The Pivot samples
   are small (65–102 trades), but their gross R is still about 1.5–2.2 s.e.
   below 0.

### Structural observations (no tuning, just what the runs show)

- **Breakout/Breakdown signal flood.**
  - They fire about 3.5 times per symbol-day: 13,438 of the 14,942
    signals.
  - With 3 slots, the combined portfolio is effectively "the first 3
    breakouts after 09:20": 204 of 327 `base` entries are in the 09:xx
    hour.
  - 10,188 later signals were skipped on `max_open_positions`. Most of
    these trades end at EOD (181 of 327), so slots stay full all day.
- **Targets are rarely reached.**
  - Breakouts hit the 2R target about 4% of the time. Their exits are EOD
    or trail.
  - So avg loss > avg win even though the nominal R:R is 1.5–2.
- **The VWAP and Pivot strategies use 1.0–1.2 × ATR(5m) stops.**
  - These are tight relative to price, so costs are about 0.5 R per trade
    at the 10% cap and still about 0.15 R uncapped.
  - Their cost share is above 100% of gross wins at `base`. Even a
    zero-gross version of these setups would lose about 0.15–0.5 R per
    trade.
- **The cooldown helps a losing system.** It suspended 24 of 93 days at
  `base`, and the suspended (shadow) trades lost more than the taken ones.

### Hand-off to the next roles

- Nothing here is a candidate for OOS as-is.
- The only thread worth pulling is the **short-side breakdown** idea, and
  only with fewer, better-filtered entries (one per symbol-day) and wider
  stops relative to price so that costs fall below about 0.07 R. That is a
  design change for the next role, not a tune of this baseline.
- The Pivot Boss survivors (open-vs-PP side, H3/L3 continuation, PP
  untouched) are natural filters to test on it in-sample.

---

## Step 3 — Pre-registration (Researcher role, 2026-10-21)

**This section was written before any Step 3 test was run.** It freezes the
hypotheses, parameters, selection rule and test count. Later roles run
exactly this list. They do not add, drop or re-parameterise anything after
seeing results. If a run exposes a code bug, the code is fixed (and the fix
logged), but no parameter changes. Any new idea that comes up during testing
goes into a *future* pre-registration, marked as post-hoc.

I ran no new tests for this section. It only reads Steps 1–2, the Pivot Boss
study and `TRADING_RULES.md`. No OOS date was read.

### 3.0 What the evidence allows (design constraints)

1. **Cost hurdle, as arithmetic.**
   - Risk is ₹2,500 (0.5% of ₹5L). At the RiskEngine 10% cap (₹50k
     notional), round-trip friction is about 0.13% fees + 0.04% slippage
     ≈ **0.17% of price**. Cost R is therefore ≈ 0.17 / stop%:
     - a 0.6% stop costs ≈ 0.28 R;
     - a 1.0% stop ≈ 0.17 R;
     - a 2.0% stop ≈ 0.085 R.
   - At a 100% cap (MIS), cost R ≈ 0.07 / stop% + 0.016:
     - a 1% stop ≈ 0.09 R;
     - a 2% stop ≈ 0.05 R.
   - **So every new candidate uses a volatility-scaled stop of at least
     0.75 × daily ATR14 (dATR).** On NIFTY 50 names that is roughly 1.1–2%.
     Each candidate takes at most one entry per symbol-day.
2. **Directional "survivor" signals have no forward edge.** Step 1 found
   open-vs-PP side, PP-untouched-by-10:15, open-outside-CPR and
   first-15/30-min side all at ≈ 0 forward return. Under the "only filters
   the evidence supports" rule, **F1 (open side of PP) and F2 (PP untouched
   → continuation) are not pre-registered.** Including them would only add
   to the multiple-testing count.
3. **The measured regime is mildly mean-reverting, not trending.**
   - Opens 0.5–1 dATR from PP continue *less* (1c: long −0.144%, short
     −0.130%, mirror-consistent).
   - Breakout Long has gross R −0.099 ± 0.014 (`raw`). This s.e. is
     trade-level, not day-clustered, so it overstates the confidence.
   - Pivot Boss E: "higher" days show the weakest close > open.
   - Small gaps fill, with pre-slippage R of +0.06.
   - Fades of *pivot/Camarilla levels*, however, lose: Pivot Rejection is
     gross −0.15 to −0.25 R, and the H3/L3 reversal lost in the Pivot Boss
     study. So the mean reversion is about *displacement*, not about the
     levels.
4. **The short side being "less bad" is drift-contaminated.** The IS stock
   drift is −0.037%/day, all of it in Apr–Jun; Jul–Aug is ≈ 0. One-sided
   candidates are therefore guarded by the IS-halves rule in 3.3.
5. **Narrow-range/compression breakouts (NR7, narrow CPR) are excluded.**
   The Pivot Boss study found narrow CPR does *not* predict trend days, and
   the narrowest quintile was the least trendy.

Common definitions, used by every rule below:

| Term | Definition |
|---|---|
| dATR | `bias.daily_atr(session.daily_history, 14)`: the simple mean true range of the 14 bhavcopy sessions before the day. The day is skipped if it is `None`. |
| PP | Floor pivot from `session.prev_daily`: (H+L+C)/3 |
| prev close | `prev_daily.c` (bhavcopy) |
| open | open of the 09:15 1m bar |
| OR30 | high/low of the 1m bars 09:15–09:44 IST |
| bars | `session.resampled(n)`, aligned to 09:15. The signal is known at bar start + n min (`signal_t`) |
| entry | the harness: open of the first 1m bar with t ≥ signal_t, 0.02% slippage per side |
| exits | the stop/target given below, plus the owner trail (breakeven at +1R, then 1.5 × ATR14 on **5m**, `trail_tf=5`, ratchet only), plus flat at 15:14 |
| priority | the tie-break when entries share a bar; higher goes first |

### 3.1 Filter block (re-sizing the Step 2 screener signals, `candidates.pkl`)

**Scope.** This block covers only **Breakout Long** and **Breakdown Short**.
They are the only screener strategies with enough signals (3.4–3.5k `raw`)
and a gross R not hopelessly negative.

**F3 is applied as a strategy-level veto: "no fades".** VWAP Bounce, VWAP
Rejection and Pivot Rejection (Long and Short) are dropped outright. They
fade levels, and all four are gross-negative by more than 1.5 s.e. H3/L3 is
*not* used as a continuation filter, because Step 1 showed its 68% is
geometry (pre-slip −0.01 to −0.03 R).

Filters:

- **S0, one per symbol-day (structural).** Keep only the first signal of
  that strategy for that symbol on that day. Drop the rest *before* the
  portfolio stage. Evidence: the Step 2 signal flood (about 3.5 per
  symbol-day, slots full by 09:xx).
- **F4, gap-context veto.** Let gap = (open − prev close)/prev close. Drop a
  signal when all of the following hold:
  - 0.25% ≤ |gap| ≤ 1.0%;
  - the signal side equals sign(gap), i.e. it trades *away* from the
    unfilled prior close;
  - signal_t is ≤ 10:15 IST;
  - no 1m bar from 09:16 up to the signal bar has touched prev close.

  Evidence: tradeable fill rates after 09:16 of 51% (0.25–0.5%) and 42%
  (0.5–1%) by 10:14 (1f), and gap-fill pre-slip R of +0.06. Gap-direction
  breakouts in that window trade against the magnet.

Configurations (4):

| ID | Signals | Filters |
|---|---|---|
| **FB1** | Breakout Long | S0 |
| **FB2** | Breakdown Short | S0 |
| **FB3** | Breakout Long | S0 + F4 |
| **FB4** | Breakdown Short | S0 + F4 |

Exits stay as in Step 2: the signal's own stop/target, the owner trail and
15:14. These configs are expected to fail. The best `raw` gross is −0.016 R,
and filters that remove signals cannot create the about +0.25 R of gross
edge that base sizing needs. They are run because the Step 2 hand-off asked
for them and because they cost almost nothing.

### 3.2 New candidate strategies (5, frozen parameters)

**C1 — PP-extension fade** (`pp_ext_fade`)

- Evidence: 1c. Opens 0.5–1 dATR from PP continue −0.144% (long) and −0.130%
  (short). That is mirror-consistent, so it is not drift. Fading gives about
  +0.13% over 09:16 → 15:14, with no stop.
- Rule: d = (open − PP)/dATR. If 0.5 ≤ |d| ≤ 1.0, side = −sign(d). The
  signal is known at the 09:15 bar close (signal_t = 09:16), so the entry is
  the 09:16 open.
- Stop: entry − side × 1.0 × dATR.
- Target: PP.
- Exit: trail + 15:14.
- Priority: |d|.
- Expected: ~500 signals in IS, 3 slots/day → n(base) ≈ 200.
- Prior: weak. The cell t-stats were −1.25 / +0.49. At about 1.6% stops the
  +0.13% is ≈ +0.08 R gross, which is borderline even before costs.

**C2 — Failed opening-range breakout fade** (`or30_fail_fade`)

- Evidence: the Breakout Long gross is strongly negative, there is general
  displacement reversion (1c, Pivot Boss E), and the Breakout targets
  (2R) were hit only about 4% of the time.
- Rule, on 5m bars from 09:45 onward:
  - An **excursion** is a 5m close > OR30 high (or < OR30 low).
  - A **failure** is a later 5m bar that closes back inside OR30, below the
    OR30 high (above the OR30 low), with signal_t in [10:00, 13:30].
  - Side: −1 after an upside failure, +1 after a downside failure.
  - Only the first failure per symbol-day counts (either direction).
- Stop: the more distant of (session extreme on the excursion side through
  the signal bar ± 0.10 × dATR) and (entry ∓ 0.75 × dATR). For a short:
  stop = max(session high + 0.1·dATR, entry + 0.75·dATR).
- Target: the opposite OR30 extreme (OR30 low for a short).
- Exit: trail + 15:14.
- Priority: the excursion's max distance beyond OR30, divided by dATR.

**C3 — ORB30 breakdown short, one per day, wide stop** (`orb30_short`)

- This is the Step 2 hand-off thread, redesigned rather than tuned.
- Rule: the first 15m bar (aligned 09:15, so the first eligible bar is
  09:45–09:59) whose close < OR30 low, with signal_t ≤ 12:00. Short at the
  next 1m open.
- Stop: entry + max(OR30 high − entry, 0.75 × dATR).
- Target: none.
- Exit: trail + 15:14.
- Priority: (OR30 low − close)/dATR.

**C4 — ORB30 breakout long (mirror of C3; control)** (`orb30_long`)

- It is identical to C3 with the sides flipped.
- It is pre-registered so that C3 cannot pass on IS short-drift alone. It is
  selection-eligible on the same terms as the other candidates.
- Prior: negative, given Breakout Long.

**C5 — Prior-day cross-sectional reversal** (`xs_reversal_1d`)

- Evidence: the short-term reversal literature (Jegadeesh 1990; Lehmann
  1990) and Pivot Boss E ("higher" days have the weakest close > open,
  mean O→C −0.104%).
- **Caveat:** E pooled IS and OOS. This is the one candidate whose
  motivation has seen OOS-period statistics, and its OOS result must be read
  with that in mind.
- Rule:
  - For each IS day, over the names whose two prior bhavcopy bars and dATR
    exist: z = (prev close − close two sessions back) / dATR, using only 1d
    data before the day.
  - **Short the single highest-z name** if z ≥ 1.0.
  - **Long the single lowest-z name** if z ≤ −1.0.
  - So there are at most 2 trades per day. Entry is the 09:16 open.
- Stop: entry ∓ 1.0 × dATR.
- Target: none.
- Exit: trail + 15:14.
- Priority: |z|.
- Implementation: the factory precomputes the daily ranks from `1d` candles
  for IS dates only, at construction, and `generate()` emits for its own
  symbol. Expected n(base) ≈ 150–180.

None of C1–C5 uses a fixed target that could push R:R below 1 in a
selectable way. `min_rr` stays off, as in Steps 1–2.

### 3.3 Run matrix and finalist selection rule

Every run uses the harness defaults:

- ₹5L capital, 0.5% risk per trade;
- 0.02% slippage per side and the exact NSE MIS costs;
- owner trail and cooldown;
- one position per symbol, a 3% daily-loss gate, no entries after 14:30;
- capital not compounded.

Each of the 9 signal rules (FB1–FB4, C1–C5) runs under three configs:

| Config | Cap | Max open | Cooldown | Role |
|---|---|---|---|---|
| `base` | 10% (RiskEngine) | 3 | on | **primary selection** |
| `cap100` | 100% | 3 | on | conditional selection (needs an owner decision to raise `MAX_POSITION_SIZE_PCT`) |
| `all_cap10` | 10% | unlimited, no daily gate, no cooldown | off | diagnostic only (per-signal edge, gross/pre-slip R); **not** selection-eligible |

**A rule–config qualifies as a finalist only if *all* of the following hold
in IS:**

1. n ≥ 100 trades.
2. Net expectancy ≥ **+0.10 R/trade**.
3. Net PF ≥ **1.10**.
4. **Day-clustered t ≥ 2.0** on the daily net P&L series over all 93 IS
   sessions, with no-trade days counted as 0.
5. **Net P&L > 0 in both IS halves**: 2026-04-20 → 06-30 and 07-01 → 08-31.
   This doubles as the drift guard, because the stock drift was −0.067%/day
   in the first half and ≈ 0 in the second.
6. Max drawdown on daily equity ≤ ₹50,000 (10% of capital).
7. Gross R before fees > 0 in `all_cap10`. The edge must exist at signal
   level and not only through the slot ordering.

Ranking and limits:

- Rules that qualify on `base` are ranked by the t from criterion 4.
- If fewer than 3 qualify on `base`, the remaining slots may be filled by
  rules that meet 1–7 on `cap100`. These are labelled **conditional (MIS
  sizing)**.
- **At most 3 finalists. One rule counts once:** if it passes on both
  configs, `base` wins.
- Each finalist is tagged **"Bonferroni-clean"** if its t ≥ 2.77 and
  **"exploratory"** otherwise. 2.77 is the one-sided 0.05/18 threshold for
  18 selection tests.
- **If nothing qualifies, there are no finalists and OOS stays untouched.**
  The verdict is "no edge found in IS". No best-of-the-losers is sent to OOS.

**OOS pass rule** (for the validator, decided now):

- Each finalist runs once on 2026-09-01 → 10-19, with frozen code and
  parameters, in the config it qualified in.
- **Pass** requires n ≥ 30, net R/trade > 0 and PF > 1.0.
- **Strong pass** requires net R ≥ +0.05 and day-clustered t ≥ 1.0.
- Anything else is a **fail** and is reported as one. OOS is about 33
  sessions, so even a pass is weak evidence.

### 3.4 Multiple-testing accounting

| Item | Count |
|---|---|
| Signal rules | 9 (filter block 4 + candidates 5) |
| Selection-eligible tests | **18** (9 rules × {`base`, `cap100`}) |
| Diagnostic runs | 9 (`all_cap10`), not selection-eligible |
| Total runs | **27** |
| Bonferroni threshold (one-sided, α 0.05 / 18) | t ≥ 2.77 |

Across the whole search the family is much larger:

- Pivot Boss: about 10 setups, plus IS/OOS;
- Step 1: 13 variants × 5 configs, plus about 60 table cells;
- Step 2: 6 strategies × 4 configs;
- Step 3: 18.

That is well over 100 looks at the same 92 IS sessions. **An IS result near
t ≈ 2 is roughly what chance alone would produce somewhere in this family.**
That is why the halves rule and the OOS rule exist and why "exploratory"
finalists must not be traded on IS evidence.

**Stated prior:** given Steps 1–2, I expect 0–1 of the 18 tests to qualify.
The candidates most likely to come close are C1 and C2, the
displacement-reversion ideas. The filter block is expected to fail.

---

## Step 4 — In-sample results (IS-tester role, 2026-10-21)

**Verdict: no finalists. None of the 18 selection-eligible rule–configs meets
the pre-registered rule, so OOS stays untouched.** The closest are C5
`xs_reversal_1d` and C1 `pp_ext_fade`, which are roughly break-even at
t ≈ 0.1–0.8. That is far from the t ≥ 2.0 bar and nowhere near the 2.77
Bonferroni threshold. Every filter-block config and C2–C4 lose money at
every size. The Step 3 prior (0–1 qualifiers, C1/C2 closest, filter block
fails) was right except for C2, which is clearly negative.

### What was run

- **Exactly the Step 3 list:** 9 rules × {`base`, `cap100`, `all_cap10`} =
  27 runs. No parameter was changed and no rule was added.
- **Universe/period:** NIFTY 50 (49 names), IS 2026-04-20 → 2026-08-31, 93
  sessions. `resolve_period("is")` and a CandleSource `max_date` of
  2026-08-31 mean no OOS date was loaded. C5's daily ranks were built from
  `1d` bars ≤ 2026-08-31.
- **Code:**
  - `backend/plugins/strategy_search/prereg_step3.py` holds the signal
    logic: `s0_first_per_day`, `f4_vetoes`, `PPExtFade`, `OR30FailFade`,
    `ORB30(±1)`, `xs_picks` and `XSReversal1D`.
  - It has 25 unit tests in
    `backend/tests/unit/test_strategy_search_prereg.py`. All of
    `tests/unit` passes (351 tests).
- **Driver:** `backend/research_output/strategy_search/step4_is/step4.py`.
  Run it from `backend/` with
  `nice -n 10 .venv/bin/python research_output/strategy_search/step4_is/step4.py`.
  Phase 1 takes 45 s for all 49 names.
- **Outputs** go under `step4_is/<rule>/<config>/` (trades, daily,
  summary). `step4_results.json` holds every metric and every criterion
  check. `step4_table.md` holds the table below.
- **Configs:**
  - `base` = `SimConfig.from_settings()`: 10% cap, 3 open, cooldown on, 3%
    daily-loss gate.
  - `cap100` is the same with `max_position_pct=100`.
  - `all_cap10` is a 10% cap with unlimited open positions, no daily gate
    and no cooldown.

### Implementation choices where the pre-registration left a detail open

These were fixed before any result was seen and are documented in the module
docstrings:

- **Entry used for stop distances.** "Entry" in the stop formulas is the
  harness fill: the open of the first 1m bar with t ≥ signal_t, slipped
  0.02%. 1R is therefore exactly the pre-registered distance (for example,
  1.0 × dATR for C1 and C5).
- **S0 + F4 order.** S0 is applied to the raw signal stream and F4 then
  vetoes from the survivors. So FB3 ⊆ FB1 and FB4 ⊆ FB2, and a vetoed first
  signal is *not* replaced by a later one that day.
- **F4 "up to the signal bar".** This means 1m bars from 09:16 with
  t + 60 s ≤ signal_t. The touch test is low ≤ prev close ≤ high.
- **C2 "first failure only".**
  - The first 5m close back inside OR30 after any excursion is the day's
    only candidate.
  - If its signal_t falls outside [10:00, 13:30] (in practice, a 09:55
    failure), that symbol-day has no trade.
  - "Inside" means OR30 low ≤ close ≤ OR30 high.
- **C3/C4.** Only the first qualifying 15m break in the rule's own
  direction is considered. A break the other way is ignored.
- **C5 ties.** Ties on z are broken by symbol name. Names were ranked
  whether or not they had 1m data that day. If the pick has no 1m session,
  there is no trade (this never happened).

### Data limitation that reduces the candidates' sample

**The DB's bhavcopy (`1d`) history starts on 2026-04-20, the first IS
day.** dATR14 needs 15 prior bars, so under the pre-registered rule ("the
day is skipped if dATR is None") **C1–C5 can only trade from 2026-05-12**.
That leaves 78 of 93 sessions, and only 35 of them fall in IS-half 1.

- The t statistic still uses all 93 sessions, as pre-registered, with
  no-trade days counted as 0.
- This explains why C1 has 490 `all_cap10` signals rather than more. The
  FB rules are unaffected.
- It is a limitation, not a tuning lever. Loading pre-April bhavcopy would
  be a data change and would need a new pre-registration.

### Full table (every configuration)

How to read the table:

- t is the day-clustered t of daily net P&L over 93 sessions, with
  no-trade days counted as 0.
- H1 is 04-20 → 06-30 and H2 is 07-01 → 08-31.
- "capped" is the share of trades where the notional cap bound.
- Gross R is after slippage and before fees.

| Rule | Config | n | win % | PF | net ₹ | R/trade | gross R | cost R | t (day) | IS-H1 ₹ | IS-H2 ₹ | daily DD ₹ | capped | criteria met |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FB1 Breakout Long · S0 | base | 314 | 35.0 | 0.43 | -49,135 | -0.282 | -0.141 | 0.141 | -5.53 | -27,821 | -21,314 | 51,796 | 98% | 1/7 |
| FB1 Breakout Long · S0 | cap100 | 318 | 34.9 | 0.55 | -134,357 | -0.219 | -0.156 | 0.063 | -4.09 | -77,825 | -56,532 | 144,243 | 4% | 1/7 |
| FB1 Breakout Long · S0 | all_cap10 | 3072 | 39.1 | 0.50 | -309,982 | -0.285 | -0.102 | 0.183 | -8.92 | -176,511 | -133,470 | 309,981 | 100% | diag |
| FB2 Breakdown Short · S0 | base | 336 | 44.3 | 0.61 | -25,412 | -0.173 | -0.017 | 0.156 | -2.90 | -10,854 | -14,559 | 25,796 | 99% | 2/7 |
| FB2 Breakdown Short · S0 | cap100 | 349 | 46.7 | 0.74 | -73,455 | -0.095 | -0.030 | 0.065 | -1.75 | -20,585 | -52,870 | 76,832 | 9% | 1/7 |
| FB2 Breakdown Short · S0 | all_cap10 | 3297 | 44.2 | 0.58 | -244,894 | -0.217 | -0.023 | 0.193 | -5.54 | -118,953 | -125,940 | 244,894 | 100% | diag |
| FB3 Breakout Long · S0+F4 | base | 334 | 35.9 | 0.47 | -44,572 | -0.267 | -0.119 | 0.149 | -5.14 | -24,763 | -19,809 | 47,469 | 99% | 2/7 |
| FB3 Breakout Long · S0+F4 | cap100 | 346 | 37.3 | 0.60 | -129,784 | -0.193 | -0.128 | 0.064 | -3.86 | -69,904 | -59,880 | 141,344 | 6% | 1/7 |
| FB3 Breakout Long · S0+F4 | all_cap10 | 2887 | 38.9 | 0.50 | -293,675 | -0.291 | -0.104 | 0.187 | -8.86 | -165,261 | -128,414 | 294,239 | 100% | diag |
| FB4 Breakdown Short · S0+F4 | base | 327 | 43.7 | 0.56 | -29,550 | -0.184 | -0.027 | 0.157 | -3.23 | -12,975 | -16,576 | 29,934 | 98% | 2/7 |
| FB4 Breakdown Short · S0+F4 | cap100 | 347 | 46.1 | 0.69 | -88,185 | -0.110 | -0.043 | 0.067 | -2.10 | -44,525 | -43,660 | 91,562 | 8% | 1/7 |
| FB4 Breakdown Short · S0+F4 | all_cap10 | 3153 | 44.5 | 0.58 | -225,670 | -0.212 | -0.016 | 0.196 | -5.58 | -108,266 | -117,404 | 225,670 | 100% | diag |
| C1 pp_ext_fade | base | 177 | 55.9 | 1.04 | 1,522 | -0.006 | 0.056 | 0.062 | 0.16 | 118 | 1,404 | 8,141 | 97% | 4/7 |
| C1 pp_ext_fade | cap100 | 177 | 58.2 | 1.08 | 7,328 | 0.019 | 0.056 | 0.037 | 0.35 | 3,787 | 3,541 | 20,662 | 0% | 4/7 |
| C1 pp_ext_fade | all_cap10 | 490 | 53.9 | 0.95 | -5,512 | -0.028 | 0.032 | 0.061 | -0.30 | -2,759 | -2,754 | 22,118 | 100% | diag |
| C2 or30_fail_fade | base | 261 | 48.7 | 0.70 | -15,662 | -0.084 | -0.001 | 0.082 | -1.91 | -7,588 | -8,074 | 21,157 | 99% | 2/7 |
| C2 or30_fail_fade | cap100 | 269 | 51.3 | 0.79 | -30,729 | -0.045 | -0.002 | 0.043 | -1.34 | -13,302 | -17,426 | 52,541 | 0% | 1/7 |
| C2 or30_fail_fade | all_cap10 | 2291 | 46.1 | 0.63 | -158,849 | -0.090 | -0.007 | 0.083 | -5.63 | -71,088 | -87,761 | 160,362 | 100% | diag |
| C3 orb30_short | base | 182 | 42.3 | 0.52 | -19,919 | -0.133 | -0.054 | 0.079 | -2.59 | -5,225 | -14,695 | 20,950 | 99% | 2/7 |
| C3 orb30_short | cap100 | 195 | 47.2 | 0.73 | -26,855 | -0.083 | -0.038 | 0.045 | -1.28 | 4,016 | -30,871 | 35,406 | 0% | 2/7 |
| C3 orb30_short | all_cap10 | 1354 | 41.6 | 0.60 | -115,574 | -0.108 | -0.028 | 0.080 | -2.59 | -36,845 | -78,730 | 128,838 | 100% | diag |
| C4 orb30_long | base | 204 | 34.8 | 0.66 | -17,449 | -0.128 | -0.052 | 0.076 | -1.84 | -3,921 | -13,527 | 21,913 | 99% | 2/7 |
| C4 orb30_long | cap100 | 198 | 37.9 | 0.65 | -42,266 | -0.095 | -0.051 | 0.044 | -1.81 | -11,111 | -31,155 | 50,816 | 0% | 1/7 |
| C4 orb30_long | all_cap10 | 1230 | 35.8 | 0.51 | -151,291 | -0.156 | -0.077 | 0.078 | -4.45 | -82,508 | -68,783 | 151,291 | 100% | diag |
| C5 xs_reversal_1d | base | 110 | 50.9 | 1.03 | 620 | 0.009 | 0.071 | 0.062 | 0.10 | -3,814 | 4,434 | 6,796 | 96% | 3/7 |
| C5 xs_reversal_1d | cap100 | 110 | 53.6 | 1.20 | 9,758 | 0.035 | 0.071 | 0.036 | 0.76 | -7,674 | 17,432 | 10,900 | 0% | 4/7 |
| C5 xs_reversal_1d | all_cap10 | 115 | 50.4 | 1.05 | 1,175 | 0.015 | 0.077 | 0.062 | 0.20 | -3,489 | 4,664 | 6,241 | 100% | diag |

Filter-block counts:

- Step 2 had 6,368 Breakout Long and 7,070 Breakdown Short signals.
- S0 left 3,072 (FB1) and 3,297 (FB2).
- F4 vetoed 185 of FB1 (leaving FB3) and 144 of FB2 (leaving FB4).
- C1 lost 8 signals to `entry_beyond_target` in phase 1. The other rules
  had no phase-1 rejects.

### Selection rule, applied mechanically

| Rule/config | Criteria met | Failed criteria |
|---|---|---|
| C5 / cap100 | 4/7 | R ≥ 0.10 (0.035), t ≥ 2.0 (0.76), both halves > 0 (H1 −7,674) |
| C1 / cap100 | 4/7 | R ≥ 0.10 (0.019), PF ≥ 1.10 (1.08), t (0.35) |
| C1 / base | 4/7 | R (−0.006), PF (1.04), t (0.16) |
| C5 / base | 3/7 | R (0.009), PF (1.03), t (0.10), halves (H1 −3,814) |
| C3 / cap100 | 2/7 | R, PF, t, halves (H2 −30,871), all_cap10 gross R < 0 |
| C4 / base, C2 / base, C3 / base, FB2/FB3/FB4 / base | 2/7 | everything except n ≥ 100 and daily DD ≤ 50k |
| all other configs (every cap100 except C1/C3/C5, plus FB1 / base) | 1/7 | also daily DD > ₹50k |

- **Qualifiers on `base`: none. Qualifiers on `cap100`: none.
  Finalists: none.**
- Per the pre-registration, OOS (2026-09-01 → 10-19) **stays untouched**,
  and no best-of-the-losers goes forward.

### Closest configurations (named, not promoted)

1. **C5 `xs_reversal_1d` / cap100.**
   - n 110, PF 1.20, net +₹9,758, +0.035 R/trade, t 0.76.
   - Its gross edge is the best of the search: +0.071 R, and +0.077 R in
     `all_cap10`. The `all_cap10` run is the only diagnostic with net R
     > 0, at +0.015 R.
   - It fails the halves rule: H1 −₹7,674 and H2 +₹17,432. So the whole
     result comes from Jul–Aug.
   - Longs (+0.072 R) beat shorts (+0.001 R).
   - Recall its caveat: its motivation (Pivot Boss E) pooled IS and OOS
     data.
2. **C1 `pp_ext_fade` / cap100 and base.**
   - Gross +0.056 R, which matches the pre-registered "≈ +0.08 R
     borderline" expectation.
   - It is positive in both halves at cap100, but net is only
     +0.019 R (cap100) and −0.006 R (base), with t 0.35 / 0.16.
   - At signal level (`all_cap10`, 490 trades) it is net −0.028 R, and the
     3-slot ordering is not adding a reliable edge.
   - Shorts (fading opens above PP) are slightly better than longs.
   - Exits: 103 of 177 at EOD and 52 at the PP target.
3. **C2 `or30_fail_fade`.** It is gross-flat (−0.001 to −0.007 R) and net
   −0.04 to −0.09 R. Failed OR30 breakouts do not mean-revert enough to pay
   costs, even at about 1R per 0.75 dATR.
4. **C3 / C4 (ORB30 short/long).**
   - Both are gross-negative (−0.03 to −0.08 R).
   - C3 cap100 was positive in H1 (+₹4,016) and badly negative in H2. That
     is the drift contamination the halves rule was written to catch. The
     long control C4 is worse, as predicted.
5. **Filter block (FB1–FB4).**
   - As predicted, S0 and F4 do not rescue the screener breakouts.
   - Breakdown Short stays near gross zero (−0.017 to −0.043 R) and
     Breakout Long stays clearly negative (−0.10 to −0.16 R).
   - F4 slightly improves Breakout Long at base (net −0.267 vs −0.282
     R/trade) and slightly hurts Breakdown Short (−0.184 vs −0.173). Both
     differences are noise-sized.

### Observations (post-hoc — anything pursued from here needs a new pre-registration)

- **Costs are no longer the dominant problem; gross edge is.**
  - With dATR-scaled stops, cost R fell to 0.06–0.08 at the 10% cap and
    0.04 at cap100, as Step 3's arithmetic predicted. Step 2's screener
    configs paid 0.14–0.50.
  - But no rule produced a gross edge above about +0.08 R, and the cost
    floor is about 0.04–0.06 R. The remaining net edge (≤ +0.035 R) is
    statistically indistinguishable from zero over 78 tradable sessions.
- **The ₹ net and the mean R can disagree in sign** (C1 base: +₹1,522 but
  −0.006 R/trade). This is because half-risk and notional-capped trades
  carry different ₹ risk.
- **Most C-rule trades end at 15:14** (C1 58%, C3 87%, C4 83%, C5 85% at `base`). The
  trail rarely engages with ≥ 0.75 dATR stops, so in practice these are
  "open-to-close with a disaster stop" tests.


---

## Step 5 — Out-of-sample + robustness (OOS-validator role, 2026-10-21)

**Bottom line: there was nothing to validate, and nothing found here changes that.**
- Step 4 produced **0 finalists**. Under the Step 3 rule, that means "no edge found in IS".
- As the role brief asked, the closest configurations were still run once on OOS (2026-09-01 → 10-19, 33 sessions) so the owner can see the whole picture. **These runs are labelled NOT PRE-QUALIFIED.** This departs from the Step 3 sentence "if nothing qualifies, OOS stays untouched". Because of that, none of these runs can promote a rule, whatever the result.
- **C1 `pp_ext_fade` fails OOS clearly.** Its 95% CI is entirely below zero.
- ~~**C5 `xs_reversal_1d` passes the OOS pass rule weakly.** It does not reach a strong pass, and its combined IS+OOS CI still includes zero. **Inconclusive.**~~ **Corrected in Step 6:** C5 FAILED the pre-registered IS selection. It is a post-hoc lead only and was not pre-qualified. The OOS rule does not apply to it.
- **C3 `orb30_short` strongly passes OOS.** However, it lost with t −2.59 in IS, and OOS had strong negative intraday drift. Its mirror C4 lost by about the same amount. This is drift, not edge.

### What was run (frozen code, no changes)

- Rules come from `plugins/strategy_search/prereg_step3.py`, unmodified. The harness (`sim.py`, `costs.py`, `rules.py`) is unchanged. Configs are `base`, `cap100` and `all_cap10`, exactly as in Step 4. Universe: NIFTY 50 (49 names).
- **OOS** used `resolve_period("oos", allow_oos=True)` and `CandleSource(max_date=2026-10-19, allow_oos=True)`.
  - The calendar has 33 sessions.
  - Warm-up: 3 sessions, plus the full bhavcopy history for dATR. That history reaches back into IS, which is legitimate because it is the past.
  - The portfolio and cooldown state start fresh on 09-01.
- **Primary OOS runs (top-3 closest from Step 4):** C5/cap100, C1/cap100 and C1/base. C5/base is shown alongside them.
- **For completeness:** C2, C3 and C4 were also run once on OOS. They cost nothing extra, and C3/C4 make a useful drift control. The filter block FB1–FB4 was not run OOS. It would need an OOS screener replay, and it was the worst IS family, with t from −2.9 to −5.5.
- **Robustness** used IS+OOS (2026-04-20 → 10-19, 126 sessions) for C1 and C5 only, since there were no finalists to test. It covers per-month P&L, top-5 stock share, long/short, bootstrap CIs, ±20% parameter perturbation, and notional caps of 10%, 100% and 500%.
- **Sanity checks:**
  - Over the OOS days, the perturbable re-implementations at default parameters (`C1P`, `C5P` in `step5.py`) produced trades identical to the frozen classes: 176/176 for C1 and 52/52 for C5.
  - The IS portion of the IS+OOS runs reproduces Step 4's IS net to the rupee (C1 base +₹1,522, C5 cap100 +₹9,758).

### Market context: OOS was a strong down-drift period

This is the mean open-to-close move per stock-day across the 49 names, from bhavcopy:

| Period | stock-days | mean open→close | % of days down |
|---|---|---|---|
| IS-H1 (04-20 → 06-30) | 2,401 | −0.062% | 53.4% |
| IS-H2 (07-01 → 08-31) | 2,156 | +0.042% | 51.4% |
| **OOS (09-01 → 10-19)** | 1,617 | **−0.156%** | 56.2% |

Most C-rule trades are effectively open-to-close (75–90% exit at 15:14). In this OOS window, any short-biased rule gets a tailwind of roughly +0.1R and any long-biased rule a matching headwind.

### IS vs OOS, per configuration

| Rule / config | IS n | IS R | IS PF | IS t | OOS n | OOS win % | OOS PF | OOS net ₹ | OOS R | OOS gross R | OOS cost R | OOS t | OOS daily DD ₹ | OOS rule result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C5 xs_reversal_1d / cap100** (closest #1) | 110 | +0.035 | 1.20 | 0.76 | 47 | 57.4 | 1.28 | +5,632 | **+0.042** | +0.080 | 0.037 | 0.65 | 6,687 | ~~PASS (not strong)~~ not pre-qualified, no OOS verdict (Step 6) |
| **C1 pp_ext_fade / cap100** (closest #2) | 177 | +0.019 | 1.08 | 0.35 | 63 | 46.0 | 0.52 | −21,755 | **−0.148** | −0.107 | 0.041 | −2.08 | 21,755 | FAIL |
| **C1 pp_ext_fade / base** (closest #3) | 177 | −0.006 | 1.04 | 0.16 | 63 | 42.9 | 0.48 | −11,363 | **−0.201** | −0.137 | 0.065 | −2.39 | 11,363 | FAIL |
| C5 xs_reversal_1d / base | 110 | +0.009 | 1.03 | 0.10 | 48 | 56.2 | 1.20 | +1,950 | +0.027 | +0.087 | 0.061 | 0.50 | 3,667 | ~~PASS (not strong)~~ not pre-qualified, no OOS verdict (Step 6) |
| C5 / all_cap10 (diag) | 115 | +0.015 | 1.05 | 0.20 | 52 | 53.8 | 1.15 | +1,615 | +0.021 | +0.082 | 0.061 | 0.39 | 3,118 | diag |
| C1 / all_cap10 (diag) | 490 | −0.028 | 0.95 | −0.30 | 176 | 52.8 | 0.74 | −11,420 | −0.076 | −0.011 | 0.066 | −1.22 | 14,959 | diag |
| C2 or30_fail_fade / base | 261 | −0.084 | 0.70 | −1.91 | 85 | 32.9 | 0.23 | −19,222 | −0.284 | −0.196 | 0.088 | −4.43 | 19,222 | FAIL |
| C2 / cap100 | 269 | −0.045 | 0.79 | −1.34 | 85 | 35.3 | 0.35 | −35,860 | −0.247 | −0.196 | 0.051 | −3.81 | 35,860 | FAIL |
| C3 orb30_short / base | 182 | −0.133 | 0.52 | −2.59 | 92 | 48.9 | 1.48 | +6,901 | +0.069 | +0.149 | 0.079 | 1.29 | 4,622 | ~~"STRONG PASS"~~ not pre-qualified; drift, see below |
| C3 / cap100 | 195 | −0.083 | 0.73 | −1.28 | 92 | 50.0 | 1.60 | +23,033 | +0.107 | +0.149 | 0.042 | 1.47 | 13,440 | ~~"STRONG PASS"~~ not pre-qualified; drift, see below |
| C3 / all_cap10 (diag) | 1,354 | −0.108 | 0.60 | −2.59 | 576 | 54.5 | 1.26 | +20,172 | +0.036 | +0.120 | 0.084 | 0.86 | 21,174 | diag |
| C4 orb30_long / base | 204 | −0.128 | 0.66 | −1.84 | 77 | 29.9 | 0.36 | −15,409 | −0.263 | −0.181 | 0.082 | −3.02 | 15,409 | FAIL |
| C4 / cap100 | 198 | −0.095 | 0.65 | −1.81 | 83 | 34.9 | 0.49 | −31,853 | −0.152 | −0.109 | 0.044 | −2.12 | 31,853 | FAIL |

OOS rule (Step 3): **pass** needs n ≥ 30, net R > 0 and PF > 1. **Strong pass** also needs R ≥ +0.05 and t ≥ 1.0. t is day-clustered over all 33 OOS sessions, with no-trade days counted as 0. OOS long/short split:
- C1 cap100: longs −0.075 R (37 trades), shorts −0.251 R (26).
- C5 cap100: longs −0.044 R (22), shorts +0.119 R (25).
- C3 has shorts only and C4 longs only.

**Why C3's OOS pass is not evidence of edge:**
- C3 and C4 are exact mirrors. C4 was pre-registered as the drift control for C3.
- In IS both were gross-negative by the same amount (−0.054 / −0.052 R at base). In OOS they swing by about the same amount in opposite directions: C3 +0.149, C4 −0.181 gross R.
- That symmetric swing is what a market move of −0.156%/stock-day produces when both rules hold to 15:14. A breakout skill would show up as both rules beating their drift-adjusted baseline, and neither does.
- C3 also failed IS on 6 of 7 criteria, so it is not eligible under any reading of the rules.

### Robustness (IS+OOS, 126 sessions) — context only

**Bootstrap 95% CI of net R/trade** (10,000 resamples; "trade" resamples trades, "day" resamples whole days):

| Config | Window | n | mean R | trade CI | day CI | P(mean ≤ 0), day |
|---|---|---|---|---|---|---|
| C5 cap100 | IS | 110 | +0.035 | [−0.063, +0.132] | [−0.062, +0.127] | 0.24 |
| C5 cap100 | OOS | 47 | +0.042 | [−0.113, +0.192] | [−0.108, +0.201] | 0.29 |
| C5 cap100 | IS+OOS | 157 | +0.037 | [−0.044, +0.118] | [−0.040, +0.117] | 0.18 |
| C5 base | IS+OOS | 158 | +0.015 | [−0.067, +0.096] | [−0.062, +0.095] | 0.36 |
| C1 cap100 | IS | 177 | +0.019 | [−0.060, +0.095] | [−0.079, +0.113] | 0.34 |
| C1 cap100 | OOS | 63 | −0.161 | [−0.299, −0.018] | [−0.308, −0.014] | 0.99 |
| C1 cap100 | IS+OOS | 240 | −0.028 | [−0.098, +0.040] | [−0.113, +0.050] | 0.75 |
| C1 base | IS+OOS | 240 | −0.060 | [−0.132, +0.008] | [−0.146, +0.019] | 0.93 |

(Bootstrap means are over taken trades. The OOS mean for C1 cap100 differs slightly from the table above, −0.161 vs −0.148, because of how cooldown state carries over in the continuous IS+OOS run.)

**Per-month net ₹ (R/trade)**, from the continuous IS+OOS run:

| Month | C5 cap100 | C5 base | C1 cap100 | C1 base |
|---|---|---|---|---|
| 2026-05 | −6,906 (−0.17) | −3,436 (−0.19) | −4,001 (−0.05) | −3,401 (−0.07) |
| 2026-06 | −768 (−0.01) | −379 (−0.03) | +7,788 (+0.10) | +3,519 (+0.08) |
| 2026-07 | −1,524 (−0.02) | −2,289 (−0.05) | −7,343 (−0.05) | −864 (−0.07) |
| 2026-08 | **+18,956 (+0.22)** | +6,723 (+0.19) | +10,884 (+0.09) | +2,268 (+0.05) |
| 2026-09 (OOS) | +3,457 (+0.03) | +884 (+0.01) | −15,489 (−0.15) | −8,414 (−0.22) |
| 2026-10 (OOS) | +2,175 (+0.06) | +1,066 (+0.06) | −7,086 (−0.19) | −3,862 (−0.20) |

C5 made money in 3 of 6 months. Without August, IS+OOS net at cap100 is −₹3.6k. C1 alternates between months and then loses every OOS month.

**Per-stock concentration (IS+OOS):**
- **C5 cap100:** 45 symbols, net +₹15,389. The top-5 stocks (TCS, POWERGRID, BAJFINANCE, HINDALCO, HDFCLIFE) earned +₹28,442; **the other 40 lost −₹13,053.** The result depends on a handful of names.
- **C1 cap100:** 48 symbols, net −₹15,246. Top-5 +₹21,864, rest −₹37,110.

**Long vs short (IS+OOS):**
- C5 cap100 is balanced: longs +0.037 R (74 trades, +₹6.4k) and shorts +0.037 R (83 trades, +₹9.0k). In IS longs carried it; in OOS shorts did, matching the market drift.
- C1 cap100: longs −0.042 R (105), shorts −0.018 R (135). Both are negative.

**±20% parameter perturbation, one parameter at a time** (IS+OOS; `base` / `cap100`):

| Variant | n | net R base | net R cap100 | t cap100 | IS net ₹ cap100 | OOS net ₹ cap100 |
|---|---|---|---|---|---|---|
| C5 frozen (z ≥ 1.0, stop 1.0 dATR) | 158 / 157 | +0.015 | +0.037 | 1.00 | +9,758 | +5,632 |
| C5 z ≥ 0.8 | 172 / 171 | +0.001 | +0.022 | 0.65 | +5,109 | +4,992 |
| C5 z ≥ 1.2 | 132 / 131 | +0.034 | +0.027 | 0.65 | +2,782 | +6,186 |
| C5 stop 0.8 dATR | 158 / 157 | +0.009 | +0.044 | 0.90 | +10,294 | +6,914 |
| C5 stop 1.2 dATR | 159 / 161 | +0.022 | +0.035 | 1.07 | +6,261 | +7,549 |
| C1 frozen (0.5 ≤ \|d\| ≤ 1.0, stop 1.0 dATR) | 240 / 240 | −0.060 | −0.028 | −0.65 | +7,328 | −22,574 |
| C1 band 0.4–0.8 | 269 / 276 | −0.055 | −0.013 | −0.64 | −3,072 | −11,430 |
| C1 band 0.6–1.2 | 202 / 202 | −0.068 | −0.055 | −1.05 | −6,749 | −17,913 |
| C1 stop 0.8 dATR | 240 / 243 | −0.065 | −0.027 | −0.59 | +14,029 | −30,208 |
| C1 stop 1.2 dATR | 240 / 240 | −0.053 | −0.028 | −0.79 | +4,118 | −19,441 |

- **C5 is stable in sign.** All 10 perturbed runs are ≥ 0 R, and every cap100 variant is positive in both IS and OOS. But it is uniformly small: +0.00 to +0.04 R and t ≤ 1.07. This is a flat plateau near zero, not a parameter spike, and the variants are highly correlated (mostly the same trades).
- **C1 is negative in every variant.**

**Larger-notional sensitivity:**
- `cap500` (MIS 5×) is **identical to `cap100`** for both C1 and C5. With stops ≥ 1 dATR, 0% of trades bind the notional cap at 100%; size is set by the 0.5% risk budget.
- So for these dATR-stop rules, the only relevant owner decision is lifting MAX_POSITION_SIZE_PCT from 10% to about 100%. That roughly halves the cost R, from ≈0.06 to ≈0.04. Above 100% nothing changes.
- At the live 10% cap (`base`), C5 IS+OOS is +0.015 R with t 0.36. That is break-even after costs.

### Verdicts

| Candidate | Status going in | OOS (frozen, once) | Robustness | Verdict |
|---|---|---|---|---|
| — (finalists) | none qualified in IS | not applicable | not applicable | **No finalist; nothing to validate. "No edge found in IS" stands.** |
| C5 xs_reversal_1d (cap100; base) | NOT pre-qualified (IS 4/7, 3/7) | PASS, not strong: +0.042 R, PF 1.28, t 0.65, n 47 (base +0.027 R, PF 1.20) | IS+OOS +0.037 R, day-CI [−0.040, +0.117]. Sign-stable under ±20%. Concentrated: top-5 names > 100% of P&L; Aug 2026 alone is more than the total | ~~**INCONCLUSIVE.**~~ **FAILED the pre-registered IS selection. Post-hoc lead only, not pre-qualified (Step 6).** The only unbiased estimate is OOS alone: +0.042 R, day-CI [−0.108, +0.201]. It is consistent with a small edge, but also with zero. It is not tradeable at the live 10% cap (≈ +0.015 R). Its motivation also pooled IS and OOS (Pivot Boss E), so this OOS is not fully clean either. |
| C1 pp_ext_fade (cap100; base) | NOT pre-qualified (IS 4/7) | FAIL: −0.148 / −0.201 R, PF ≈ 0.5, t −2.1 / −2.4 | OOS CI entirely below 0. Every ±20% variant is negative IS+OOS | **FAILED.** |
| C3 orb30_short | IS loser (t −2.59) | Nominal "strong pass" (+0.07 / +0.11 R) | The mirror C4 lost symmetrically; OOS drift was −0.156%/stock-day | **FAILED (OOS gain is drift; it was never eligible).** |
| C2 or30_fail_fade, C4 orb30_long | IS losers | FAIL (−0.15 to −0.28 R) | — | **FAILED.** |

**What would be needed to take C5 further.** This is not a recommendation to trade it.
1. Write a fresh pre-registration that fixes C5 exactly as frozen.
2. Paper-trade it forward on new sessions after 2026-10-19 at `cap100`-equivalent sizing. That sizing needs the owner's decision on MAX_POSITION_SIZE_PCT.
3. ~~Give it at least ~60 trades (about 2–3 months at ~1.3 trades/day), and judge it only on that forward sample.~~ **Corrected in Step 6:** 60 trades cannot confirm a +0.04 R edge; confirming it takes about 800 trades. 60 trades can only support a kill rule. C5 also has no target, so the live RiskEngine would reject every C5 intent today.

The roughly +0.04 R effect is too small to separate from zero on the sample sizes this data can provide (126 sessions): it would need about 1,000+ trades to reach t ≈ 2.

### Reproduce

From `backend/`: `nice -n 10 .venv/bin/python research_output/strategy_search/step5_oos/step5.py`. Phase 1 takes about 85 s and is cached in `oos_candidates.pkl` / `all_candidates.pkl`.

Outputs (gitignored):
- `research_output/strategy_search/step5_oos/step5_results.json` holds every metric, the bootstraps, per-month and top-5 data.
- `oos/<rule>/<config>/` and `all/<C1|C5>/<config>/` hold trades.csv, daily.csv and summary.
- `run.log`.

No code under `plugins/`, `apps/`, settings or the frontend was changed in this step. The perturbation variants live only in the research script.

---

## Step 6 — Audit fixes and corrected numbers (Engineer-fix role, 2026-10-21)

**Bottom line: no verdict improves, and two get worse.**
- With the live RiskEngine's R:R and target checks modelled, **@RiskGuard would reject every C3, C4 and C5 intent**, 99% of C1 and 98% of C2. The few C1/C2 trades that get through lose.
- **C5 is re-labelled as a FAILED pre-registered IS selection, kept only as a post-hoc lead.** Its small paper edge disappears at about 0.07% slippage per side (cap100) or 0.05% (live 10% cap).
- A 60-trade forward test can only *kill* C5. It cannot confirm it; confirmation needs about 800 trades.
- The 2026-09-01 → 10-19 OOS window is now **spent** for this whole family of rules. It cannot serve as validation for any future pre-registration.

### What was changed (code)

| Audit issue | Fix | Regression test |
|---|---|---|
| 1. Sim omitted live criterion 7 (round(rr, 2) ≥ 1.5) and the target-required field check | `SimConfig.live_rr_gate` (default off, so earlier steps reproduce) plus `sim.rr_gate_skip()`. Target-less intents are skipped as `no_target`; `round(rr, 2) < min_rr` (default 1.5) is skipped as `min_rr`, using the live engine's rounding. CLI: `strategy_search … --live-rr-gate`. | `TestLiveRRGate` (4 tests), including 500 random cases checked against the live `RiskEngine._rr_below_minimum` |
| 2. C5 lifted from an IS failure to "INCONCLUSIVE" | Doc only. The Step 5 wording is struck through in place and the verdicts are corrected below. | n/a (no code) |
| 3. A 60-trade forward test is under-powered | `plugins/strategy_search/stats.py`: `trades_for_t`, `p_mean_above`, `detectable_edge` | `TestStats` (power figures match the audit's: 812 trades, P = 0.705) |
| 4. No slippage sensitivity | Grid run below; `stats.zero_crossing` gives break-even slippage | `TestStats.test_zero_crossing` |

Other details:
- No change to `apps/trading`, `apps/news`, settings/urls, the frontend or `strategy_screener`. The new flag lives in `apps/market_data/management/commands/strategy_search.py`. Nothing was committed.
- Tests: `tests/unit/test_strategy_search.py` now has 40 tests, and all 358 tests in `tests/unit` pass.
- The audit's figures were checked and hold, with small differences:
  - It said C1 base had 4/177 trades at RR ≥ 1.5. I find 6 of 490 C1 signals pass the gate.
  - Its break-even slippage estimates were about 0.06% (cap100) and about 0.035% (base). The re-run gives about 0.07% and about 0.05%.

### Script, sanity checks and outputs

`research_output/strategy_search/step6_audit/step6.py`, run from `backend/`, takes 96 s.
- **R:R gate (6A):** re-sizes the frozen Step 4 IS candidates (`step4_candidates.pkl`) and the frozen Step 5 OOS candidates (`oos_candidates.pkl`).
- **Slippage grid (6B):** re-resolves phase 1 for C1 and C5 only, at 0.02 / 0.035 / 0.05 / 0.075 / 0.10% per side.
  - The signals and stop *prices* are the frozen ones, so only the fills get worse.
  - The IS pass uses an IS-capped CandleSource.
  - The OOS pass is the single permitted re-run of the closest configs. Nothing was re-selected or re-tuned.
- **Sanity checks:** the 0.02% row reproduces Steps 4 and 5 to the rupee (C1 base IS +₹1,522, C5 cap100 IS +₹9,758, C5 cap100 OOS +₹5,632, C1 base OOS −₹11,363).
- **Outputs:** `step6_audit/step6_results.json` and `run.log`.

### 6A. Live RiskEngine R:R gate (criterion 7 + target required) — before / after

**IS** (93 sessions):

| Rule / config | Before: n | net ₹ | R/trade | After: n | net ₹ | R/trade | Signals rejected by the gate |
|---|---|---|---|---|---|---|---|
| C1 pp_ext_fade / base | 177 | +1,522 | −0.006 | 6 | −1,788 | −0.331 | 484 of 490 (`min_rr`) |
| C1 / cap100 | 177 | +7,328 | +0.019 | 6 | −4,580 | −0.301 | 484 of 490 |
| C2 or30_fail_fade / base | 261 | −15,662 | −0.084 | 45 | −10,518 | −0.332 | 1,881 of 2,291 (`min_rr`) |
| C2 / cap100 | 269 | −30,729 | −0.045 | 45 | −28,788 | −0.293 | 1,931 of 2,291 |
| C3 orb30_short / base, cap100 | 182 / 195 | −19,919 / −26,855 | | **0** | 0 | — | all 1,354 (`no_target`) |
| C4 orb30_long / base, cap100 | 204 / 198 | −17,449 / −42,266 | | **0** | 0 | — | all 1,230 (`no_target`) |
| C5 xs_reversal_1d / base, cap100 | 110 / 110 | +620 / +9,758 | | **0** | 0 | — | all 115 (`no_target`) |

**OOS** (frozen closest configs only):

| Rule / config | Before: n | net ₹ | After: n | net ₹ |
|---|---|---|---|---|
| C1 / base | 63 | −11,363 | 4 | −2,557 |
| C1 / cap100 | 63 | −21,755 | 4 | −6,330 |
| C5 / base, cap100 | 48 / 47 | +1,950 / +5,632 | 0 | 0 (all 52 `no_target`) |

**Reading:**
- C1's reward-to-risk is about |d| ≤ 1.0 by construction (target PP, stop 1 dATR), so only gap-through fills reach 1.5.
- C3, C4 and C5 are trail-only, so the live engine cannot route them at all.
- The Step 2 screener strategies set 1.5R/2R targets, so they are not affected by the target check.
- Every "real RiskEngine" claim in Steps 2–5 should be read as covering **max open positions, position-%, max risk per trade and the daily-loss gate only**. Criterion 8 (confidence and regime) is also not modelled.
- Running C5 live would need an owner decision, pre-registered before any new data is looked at. There are two options:
  - (a) a synthetic target ≥ 1.5R, which changes the exits and needs a fresh IS look on new data; or
  - (b) a RiskEngine change that admits trail-only intents. That belongs to the `apps/trading` owner; research does not touch it.

### 6B. Slippage sensitivity (per side; signals and stop prices frozen)

R/trade is net. Pooled = the trade-weighted mean of the IS and OOS trades. The pooled figures carry winner's-curse bias, because IS was used to pick C5 and C1 as the closest.

| Rule / config | slip % | IS n | IS R | IS gross R | IS net ₹ | IS t | OOS n | OOS R | OOS net ₹ | pooled R |
|---|---|---|---|---|---|---|---|---|---|---|
| C5 / cap100 | 0.02 | 110 | +0.035 | +0.071 | +9,758 | 0.76 | 47 | +0.042 | +5,632 | +0.037 |
| C5 / cap100 | 0.035 | 110 | +0.022 | +0.058 | +6,304 | 0.49 | 47 | +0.028 | +3,997 | +0.024 |
| C5 / cap100 | 0.05 | 107 | +0.026 | +0.062 | +5,098 | 0.41 | 48 | +0.021 | +3,804 | +0.024 |
| C5 / cap100 | 0.075 | 103 | −0.009 | +0.027 | −4,184 | −0.35 | 48 | −0.001 | +1,244 | −0.007 |
| C5 / cap100 | 0.10 | 103 | −0.030 | +0.006 | −9,421 | −0.80 | 48 | −0.024 | −1,131 | −0.028 |
| C5 / base | 0.02 | 110 | +0.009 | +0.071 | +620 | 0.10 | 48 | +0.027 | +1,950 | +0.015 |
| C5 / base | 0.035 | 110 | +0.013 | +0.074 | +1,055 | 0.18 | 48 | +0.013 | +1,249 | +0.013 |
| C5 / base | 0.05 | 107 | +0.001 | +0.062 | −263 | −0.04 | 48 | −0.001 | +541 | 0.000 |
| C5 / base | 0.075 | 107 | −0.022 | +0.038 | −2,863 | −0.49 | 48 | −0.023 | −622 | −0.022 |
| C5 / base | 0.10 | 107 | −0.042 | +0.018 | −5,258 | −0.89 | 48 | −0.044 | −1,789 | −0.042 |
| C1 / cap100 | 0.02 | 177 | +0.019 | +0.056 | +7,328 | 0.35 | 63 | −0.148 | −21,755 | −0.025 |
| C1 / cap100 | 0.05 | 177 | −0.008 | +0.029 | −4,072 | −0.20 | 63 | −0.176 | −26,625 | −0.052 |
| C1 / cap100 | 0.10 | 177 | −0.054 | −0.018 | −24,184 | −1.19 | 63 | −0.219 | −32,576 | −0.097 |
| C1 / base | 0.02 | 177 | −0.006 | +0.056 | +1,522 | 0.16 | 63 | −0.201 | −11,363 | −0.057 |
| C1 / base | 0.05 | 177 | −0.032 | +0.029 | −3,366 | −0.35 | 63 | −0.198 | −11,733 | −0.076 |
| C1 / base | 0.10 | 177 | −0.089 | −0.029 | −13,628 | −1.42 | 63 | −0.240 | −14,834 | −0.129 |

The 0.035% and 0.075% rows for C1 are in `step6_results.json`.

**Break-even slippage** (linear interpolation of net R = 0):

| Config | IS | OOS | pooled |
|---|---|---|---|
| C5 / cap100 | 0.068% | 0.073% | **≈ 0.07%** |
| C5 / base (live 10% cap) | 0.051% | 0.049% | **≈ 0.05%** |
| C1 / cap100 | 0.042% | already < 0 at 0.02% | already < 0 |
| C1 / base | already < 0 at 0.02% | already < 0 | already < 0 |

**Reading:**
- The grid is not perfectly monotone. A worse fill shifts the +1R breakeven trigger and cooldown timing, and moves a few C5 trades in and out (n 110 → 103).
- **C5 needs real slippage at or below about 0.05–0.07% per side** on ₹1–2.5L market orders in the 09:16 minute, the widest-spread minute of the day, just to break even. Nobody has measured that. Any forward test must log the actual fill against the 09:16 1m open.
- **C1 is dead at any slippage.**

### 6C. Power of a forward test for C5

- Per-trade SD of net R is 0.527 (C5 cap100, 157 trades pooled).
- The only unbiased mean estimate is OOS-alone: +0.042 R, day-CI [−0.108, +0.201], P(≤ 0) = 0.29.

| Question | Answer |
|---|---|
| Trades for an expected t = 2 at +0.037 R (pooled) / +0.042 R (OOS) | **816 / 621.** At ~1.4 trades per tradable session that is roughly 2–2.5 years. |
| SE of the mean after 60 trades | 0.068 R |
| Smallest edge 60 trades flags half the time (one-sided 5%) | +0.11 R, three times the claimed edge |
| P(60-trade mean > 0): true +0.037 R vs true 0 | 0.71 vs 0.50, which barely discriminates |

The only decision rule that 60 trades can support is a **kill rule**: stop if the 60-trade mean is ≤ −0.10 R.

| True edge | P(kill) |
|---|---|
| +0.037 R | 2% |
| 0 | 7% |
| −0.15 R | 77% |

Passing the kill rule says nothing positive. A 60-trade forward test **cannot confirm** a +0.04 R edge.

### Corrected verdicts

| Candidate | Step 5 verdict | Corrected verdict (Step 6) |
|---|---|---|
| (finalists) | none | **none.** "No edge found in IS" stands. |
| C5 xs_reversal_1d | ~~INCONCLUSIVE~~; OOS "PASS (not strong)" | **FAILED the pre-registered IS selection** (base 3/7, cap100 4/7; t 0.76; H1 −₹7,674). **Post-hoc lead only, not pre-qualified.** The OOS rule does not apply to it, so the "PASS" labels are struck. C5 base and cap100 are the same trades, one coin flip with P(R > 0) ≈ 0.5 under the null. Unbiased estimate (OOS-alone) is +0.042 R, CI spanning zero. It breaks even at ≈ 0.07% slippage per side (cap100), ≈ 0.05% (live cap). **Not routable through the live RiskEngine** (no target). |
| C1 pp_ext_fade | FAILED | **FAILED.** It is also 99% rejected by the live R:R gate, and the trades that pass lose. It is negative at every slippage ≥ 0.02% outside IS-cap100. |
| C3 orb30_short | FAILED (OOS gain is drift) | **FAILED.** The OOS "STRONG PASS" label is struck, because C3 was not pre-qualified. Also 100% rejected live (no target). |
| C2, C4 | FAILED | **FAILED.** C2 is 98% rejected live and the remainder loses. C4 is 100% rejected live. |
| FB1–FB4 | FAILED (IS) | unchanged |

**OOS status:** the window 2026-09-01 → 2026-10-19 has now been read by Step 5 (13 configs and the perturbations) and by this re-run. **It is spent for the whole Pivot-Boss/strategy-search family.** Any future pre-registration, C5 included, must be validated on sessions after 2026-10-19 only.

**If the owner still wants to pursue C5,** each of the following needs a new pre-registration and an owner decision before any new data is looked at:
1. Choose the target handling: a synthetic target ≥ 1.5R, or a RiskEngine change for trail-only intents.
2. Choose MAX_POSITION_SIZE_PCT: 10%, or about 100% under MIS.
3. Log the actual 09:16 fill on every trade.
4. Use the kill rule above: mean ≤ −0.10 R after 60 trades → stop.
5. Accept that a confirmation (t ≈ 2) needs about 800 forward trades.

---

## THE PLAN (Strategist role, 2026-10-21)

### Verdict

**No configuration survived.** None of the 18 selection-eligible IS tests met the pre-registered finalist rule (Step 4). With the live RiskEngine's R:R and target checks applied, every candidate is either rejected outright or loses (Step 6). The OOS window cannot rescue anything: no rule went into it pre-qualified, and it is now spent.

**There is therefore no plugin spec and no paper-trading protocol for a strategy.** Writing one would be the best-of-the-losers move that Step 3 forbade. The plan below lists what we learned and the single experiment worth running next.

### Total configurations tried

| Stage | Distinct signal rules | Configuration runs | Notes |
|---|---|---|---|
| Pivot Boss study (prior) | 6 setups | 12 (IS + OOS each) | plus descriptive tables A–F (CPR width, Camarilla, gaps, two-day relations, VPOC) |
| Harness smoke | (2) | 3 | plumbing only, not counted |
| Step 1 — bias verification | 13 variants | 65 (13 × 5 configs) | plus about 60 descriptive table cells |
| Step 2 — screener baselines | 6 firing + 2 that never fire | 28 (ALL + 6 isolated × 4 configs) | 14,942 signals replayed through the real ScreenerEngine |
| Step 4 — pre-registered IS | 9 (FB1–4, C1–C5) | 27 (18 selection + 9 diagnostic) | 0 finalists |
| Step 5 — OOS + robustness | (C1–C5) | 35 (13 OOS + 20 perturbation + 2 cap500) | not pre-qualified, so context only |
| Step 6 — audit re-runs | (C1–C5) | 54 (14 R:R-gate re-sizes + 40 slippage grid) | same frozen signals |
| **Total** | **34 signal rules (36 incl. the 2 that never fire)** | **≈ 221 configuration runs** | plus about 60 descriptive cells, all on the same 126 sessions |

Of these, **one** rule–config was net-positive in both IS and OOS: C5 at about +0.03–0.04 R. Even that one is a post-hoc lead and not a result (Step 6).

### What we now know about NSE large caps (1m, Apr–Oct 2026, ₹5L, 0.5% risk)

**Doesn't work (measured, net of costs):**

- **Every live screener strategy**: Breakout/Breakdown, VWAP Bounce/Rejection, Pivot Rejection. They lose gross, before any fee (Step 2), and the losses run from −0.02 to −0.25 R raw. One-entry-per-day and gap-context filters do not rescue them (FB1–FB4).
- **BB Squeeze and EMA Crossover** cannot fire in the live engine at all. The 15m indicators lack warm-up because bootstrap seeds only prior-day 5m bars (Step 2). That is a screener bug to report to its owner, not a strategy result.
- **Pivot Boss setups**: floor breakaway, Camarilla H3/L3 fade and H4/L4 breakout, magnet, and Golden Pivot Zone. All lose in both IS and OOS.
- **Pivot "survivor" statistics as signals**: open side of PP, PP untouched → trend, open outside CPR, first-15/30-min side vs PP, and H3/L3 continuation. All of them are explained by mechanical nulls, drift or geometry (Step 1). They have no forward return from the moment of entry.
- **Narrow CPR / compression as a trend-day forecaster.** It does not forecast trend days.
- **Opening-range fades (C2) and ORB30 breaks either way (C3/C4).** These are gross-flat or negative. C3's OOS "gain" was the −0.156%/stock-day drift, and its mirror C4 lost the same amount.
- **PP-extension fade (C1).** IS was break-even, and OOS had a CI entirely below 0.

**Does hold (true, but not tradeable on its own):**

- **Cost arithmetic.** At the 10% notional cap, friction is about 0.17% of price round-trip, so cost R ≈ 0.17 / stop%. A stop of at least 0.75 × daily ATR brings cost down to 0.06 R (10% cap) or 0.04 R (100% cap). Above a 100% cap nothing changes for dATR stops. Pivot- and VWAP-tight stops pay 0.15–0.9 R, which is why everything earlier died.
- **Small-gap fills are real** (about 62% fill after 09:16 for gaps under 0.25%). But the move is smaller than the friction.
- **Gross edges available in this data are at most about +0.08 R.** Detecting +0.04 R at t = 2 needs about 800 trades (Step 6C). This 126-session dataset cannot produce that for any one-or-two-trades-a-day rule. **Intraday 1m history is the binding constraint, not ideas.**
- **The owner rules behave as intended.** The cooldown saved about ₹3.1L of shadow losses in Step 2. With dATR stops the trail rarely engages, and 75–90% of trades exit at 15:14.
- **The live RiskEngine requires a target with R:R ≥ 1.5.** That rules out every "disaster stop + hold to close" design, which is the only family that came close. Any future candidate must be designed against that gate from the start, or the owner must decide to change it.

### The single most promising next experiment

**E1 — Does prior-day cross-sectional reversal (C5) carry a positive open-to-close edge across 10 years of NSE large caps?**

Why this one:

- C5 is the only rule with positive gross R everywhere it was measured: IS +0.07, OOS +0.08, all ±20% variants ≥ 0, and both legs equal on IS+OOS.
- It needs only **daily** data, which NSE publishes free for decades. That turns an impossible power problem (800 trades from 126 sessions) into an easy one (several thousand trades).
- The pre-2026-04 history has **never been seen** by any role, so the test is clean.
- It is cheap: no broker API, no 1m bars, and about one evening of compute.

If it fails, we stop. If it passes, it becomes the first idea that deserves a plugin.

**Hypothesis (frozen; write it into a fresh pre-registration before any data is loaded):**

- H1: buying the most-oversold and shorting the most-overbought liquid large cap at the open earns a positive mean net R.
  - Oversold and overbought are measured by z = (C₋₁ − C₋₂) / dATR14.
  - The position is held to the close with a 1.0 × dATR stop.
  - Net R is measured after NSE costs and **0.05% slippage per side**, the conservative end of Step 6B.
- H0: the mean net R is ≤ 0.

**Data needed (none of it from the broker):**

1. **NSE CM bhavcopy, 2016-01-01 → 2026-04-17.** That is about 2,550 sessions.
   - `backfill_bhavcopy` only parses the UDiFF format, which NSE has published since about July 2024. Older days use the legacy `cm<DD><MON><YYYY>bhav.csv` format, so the job needs a legacy parser. Verify the archive path before writing it.
   - Store the data in a **separate research table or a SQLite/Parquet file under `research_output/`**, not in `Candle`. That keeps the live DB and every existing IS/OOS guard unchanged.
   - Bhavcopy rows are small (about 2k equities/day), but process them one year at a time.
2. **Point-in-time universe.** Each day, take the top 50 EQ-series names by trailing 60-session median turnover, using data before that day only. This avoids the survivorship bias of today's NIFTY 50 list.
3. **Corporate-action guard.** Compute C₋₁/C₋₂ from the file's own `PREVCLOSE` (NSE adjusts it on ex-dates). Drop any name whose day return is beyond ±20% or whose symbol changed.

**Test (frozen rules, one run, no tuning):**

- Signal: exactly as C5. That means the single highest z ≥ 1.0 → short and the single lowest z ≤ −1.0 → long, so at most 2 trades per day. dATR14 is the simple mean true range of the prior 14 sessions.
- Fill: the official open O, worsened by 0.05%. Exit: the close C, worsened by 0.05%.
  - Stop: if L ≤ O − dATR (long) or H ≥ O + dATR (short), the result is exactly −1R minus costs.
  - With entry at the open and no target, the stop-vs-close ordering problem does not arise.
- No trail and no cooldown. The test is about the signal. The daily bar cannot model the +1R breakeven move, and in the 1m data 85% of C5 trades ended at 15:14 anyway.
- Sizing: 0.5% risk. Report both the 10% cap and the 100% cap, using `costs.py` unchanged.
- Bracket the bhavcopy proxy (auction open and closing price) against the 1m execution (09:16 open, 15:14 close). Run the **same** daily-proxy code on 2026-04-20 → 10-19 and compare it trade by trade with the Step 4/5 C5 trades. The overlap is known data, so it is calibration only and does not count as evidence.

**Pre-registered decision (all must hold, at the 100% cap and 0.05% slippage):**

| # | Criterion |
|---|---|
| 1 | n ≥ 2,000 trades |
| 2 | Mean net R ≥ +0.03, day-clustered t ≥ 3.0 (strict, because this is the ~35th rule tried on NSE here) |
| 3 | 2016–2020 and 2021–2026-04 are **each** positive, at t ≥ 1.5 |
| 4 | Net > 0 in at least 7 of 10 calendar years |
| 5 | Long leg and short leg gross R are both > 0 (rules out a market-drift artefact) |
| 6 | At the **10% cap**, mean net R > 0. If this fails, the strategy needs the owner's MAX_POSITION_SIZE_PCT decision before anything else |

- **Pass** → write the plugin spec below and start forward paper trading.
- **Fail on any of 1–5** → C5 is dead. Stop single-stock intraday directional research at this capital, and record the conclusion in this doc.
- Do not re-run with other z thresholds, top-k, holding periods or stops. Those would be new hypotheses on now-seen data.

**Power check (from Step 6C, SD ≈ 0.53 R):** about 2,500 trades at a true +0.037 R give an expected t ≈ 3.5. At a true 0 they give t ≈ 0 ± 1. So the test can discriminate.

### Only if E1 passes: what the plugin would look like (spec sketch, not yet approved)

- **Plugin:** `backend/plugins/strategy_xs_reversal/`. The universe is NIFTY 50 (live list), with the same point-in-time turnover filter as E1.
- **Nightly scan (21:00 IST):**
  - Compute dATR14 and z from that day's bhavcopy (published by about 18:00 IST).
  - Pick at most one long and one short, and store them as next-day intents with their stop prices.
  - Apply the TRADING_RULES §3 news filter: skip on results days, and veto a long on strongly negative sentiment or a short on strongly positive sentiment.
- **Entry:** at the 09:16 open as a MARKET order. Log the fill against the 09:16 1m open every time.
- **Exit:** a 1.0 × dATR stop-loss order, the owner trail (+1R → breakeven, then 1.5 × ATR(5m)), and a square-off at 15:14.
- **Sizing:** qty = 0.5% × capital ÷ dATR, capped by MAX_POSITION_SIZE_PCT. It needs about a 45% cap to be risk-bound, not notional-bound.
- **RiskEngine blocker:** the strategy has no target, so it is rejected today. Before any code, the owner chooses one of:
  - (a) a synthetic 1.5R target, which changes the exits and needs its own E1 variant pre-registered *now*; or
  - (b) a RiskEngine exemption for trail-only intents. That is owned by `apps/trading`.
- **Expected performance:** +0.00 to +0.08 R/trade. That is about ₹0–200 per trade, or ₹0–₹60k/year at about 300 trades. The real figure is whatever E1 says, shrunk toward 0.
- **Forward paper protocol:** paper only, on sessions after the plugin ships.
  - Kill rule: stop if the mean is ≤ −0.10 R after 60 trades.
  - Also kill if the median 09:16 slippage is > 0.05% per side after 30 trades.
  - Confirmation needs about 800 forward trades. A 60-trade window cannot confirm the edge, it can only kill it.

### What not to do next

- Do not start new 1m intraday pattern searches on the same 126 sessions. More than 200 looks are already spent there.
- Do not reuse 2026-09-01 → 10-19 as a validation window.
- Do not trade any of C1–C5 or FB1–FB4 live or on paper on the strength of this doc.
- Do not raise MAX_POSITION_SIZE_PCT in anticipation. It only matters if E1 passes.
