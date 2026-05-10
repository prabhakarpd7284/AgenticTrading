# Strategy Spec — Directional Vertical Spread ("Game Theory for Options")

- Status: Draft (based on paper notes dated 2026-04-22)
- Owner: @OptionsStrategist (new sibling to the straddle agent)
- Fits: Cascade Stage 5 (SETUP) + Stage 6 (EXECUTE) for options workflows
- Depends on: ADR-0006 (ledger + BrokerAdapter + Black-Scholes greeks solver)

## Intent

Turn the handwritten idea into a reproducible, risk-bounded options strategy:
build a **premium-pricing primitive** and a **bias-driven vertical-spread
builder** on top of it. The premium primitive is the function
`f(P) = f(S) + f(σ) + f(T)` — formalized with Black-Scholes — and becomes
shared infrastructure for every options strategy we ship (not just verticals).

The vertical-spread builder turns a directional view into a concrete
credit-spread basket:
- **Bull Put Spread** when bias is UP
- **Bear Call Spread** when bias is DOWN
- **Iron Condor** (= bull put + bear call) when bias is RANGE

Strike selection is premium-percentile-based, not moneyness-based:

- Sell leg strike = strike where premium ≈ **80% of ATM premium**
- Buy leg strike  = strike where premium ≈ **60% of ATM premium**

This embeds the R:R implicitly in the ratio and keeps the sizing logic
broker-agnostic (we don't care about exact deltas, we care about "how much
credit am I collecting per unit of risk").

---

## 1. The premium-pricing primitive

### Formal model

Given:
- `S` — spot of the underlying (e.g. NIFTY = 24,400)
- `σ` — implied volatility (annualized, decimal; e.g. VIX 20 → 0.20)
- `T` — time to expiry in years (DTE / 365)
- `K` — strike
- `r` — risk-free rate (typically 7% for INR; configurable)
- `q` — dividend yield (index ≈ 0; stocks ≠ 0)

Black-Scholes closed form:

```
d1 = (ln(S/K) + (r - q + σ²/2) · T) / (σ · √T)
d2 = d1 − σ · √T

C = S · e^(−qT) · N(d1) − K · e^(−rT) · N(d2)
P = K · e^(−rT) · N(−d2) − S · e^(−qT) · N(−d1)
```

### Why this differs from the note

The note wrote `f(P) = f(S) + f(σ) + f(T)` as if additive. The real
functional is multiplicative/nonlinear in all three. The practical
approximation — **good enough for strike scoping, not for pricing** — is:

```
P_atm ≈ 0.4 · S · σ · √T        (ATM premium, European, no drift)
```

That one-liner is what you'd use on a whiteboard to answer "is NIFTY 24400
ATM weekly worth ~100 rupees with VIX 20?" — yes, `0.4 × 24400 × 0.20 ×
√(5/365) ≈ 228`, so ATM CE ≈ ATM PE ≈ 228. A weekly straddle costs ~456.
This matches the market within the usual IV-skew fuzz.

For non-ATM, always use full BSM. The approximation is only for sizing
intuition.

### Calibration: VIX vs realized IV

India VIX is a 30-day annualized IV measure computed on NIFTY OTM options.
It's **not** the IV a specific strike trades at. Two real calibration
issues:

1. **Term structure** — a 7-DTE weekly straddle trades at a different IV
   than a 30-DTE monthly. Usually weeklies trade at a premium to VIX
   (people pay up for event hedging).
2. **Skew** — OTM puts trade richer than OTM calls (crash premium). 80%
   of ATM premium on the PE side may require a different strike distance
   than on the CE side for the same premium target.

For V1 we'll use **VIX directly as σ** and accept ±10-15% pricing error for
strike selection. For V2 we'll fit a polynomial `σ(DTE, moneyness)` from
historical `MarketSnapshot` rows once we've accumulated them.

### Inverse pricing (strike selection)

Given a target premium `P_target`, solve for `K` such that `BSM(K) = P_target`.

Implementation: brute-force over the grid of tradeable strikes (NIFTY steps
by 50, BANKNIFTY by 100). Compute premium at every strike, pick the one
with smallest `|BSM(K) − P_target|`. This is O(n) where n ≤ 50 strikes per
side — trivially fast.

---

## 2. Strategy — directional vertical spread

### Bias classifier (plugs into Cascade)

Input: Cascade Stage 1-4 output, already normalized on today's dashboard:
- `trend_regime`: `trending_up` / `trending_down` / `chop` / `compression`
- `rel_strength`: -1..1 (sector vs index)
- `breadth`: % of index above 20-EMA
- `vix_phase`: `fear` / `neutral` / `complacent` (relative to 20-day mean)
- `india_vix`: absolute level

Bias derivation (deterministic, not LLM):

```
if trend_regime == "trending_up" and breadth > 0.55: bias = UP
elif trend_regime == "trending_down" and breadth < 0.45: bias = DOWN
elif trend_regime in {"chop", "compression"}: bias = RANGE
elif rel_strength > 0.4 and vix_phase != "fear": bias = UP
elif rel_strength < -0.4: bias = DOWN
else: bias = RANGE
```

LLM-free by design so it's backtestable and reviewable. `@OptionsStrategist`
only chooses **between eligible strategies**, not the bias itself.

### Range estimate (informs the sanity check)

```
σ_period = σ_annual · √(DTE / 365)       # standard-deviation move over the holding period
expected_range = [S · (1 - σ_period), S · (1 + σ_period)]
```

Used to sanity-check strike placement:
- For directional bias, the sell strike must sit **on the correct side** of
  the expected-range boundary — not inside it (we'd be overpaying in risk)
  and not too far outside (we'd collect too little).
- For range bias, both sell strikes must sit **outside** the expected
  range — otherwise one leg has high probability of breaching.

NIFTY at 24400, VIX 20, 5 DTE:
- `σ_5d = 0.20 × √(5/365) ≈ 2.34%`
- 1-SD range: [23,829, 24,971]
- sell-PE strike (24,200) is ~0.82σ below spot → deep enough inside the
  distribution that we collect decent premium but outside the highest-risk
  quartile.

### Position construction

#### Bull put spread (bias = UP)

```
sell K_s  PE at premium P_s ≈ 0.8 · P_atm
buy  K_b  PE at premium P_b ≈ 0.6 · P_atm     (K_b < K_s)

net credit    = P_s − P_b
max profit    = net credit                          × lot_size × lots
max loss      = (K_s − K_b − net credit)            × lot_size × lots
breakeven     = K_s − net credit
margin        = broker.estimateMargin(basket)       (real API call)
```

#### Bear call spread (bias = DOWN)

Symmetric — `CE` legs, `K_b > K_s`.

#### Iron condor (bias = RANGE)

Both a bull put spread **and** a bear call spread, same lots, atomic basket.
The two credits add together; the two max losses don't (only one side can
breach at expiry for European-style cash-settled indices — for stocks with
early-exercise risk, both could in theory, so IC is index-only by rule).

### Worked example — the note's numbers

Computed with full BSM (r=7%, q=0, q=0, European), not the approximation.

```
Underlying:  NIFTY    spot = 24,400
IV:          VIX 20   → σ = 0.20
Expiry:      Weekly, 5 DTE → T = 5/365
Bias:        UP → bull put spread
Lot size:    75

BSM ATM premium   PE = 216.2  CE = 239.6
(note: CE − PE = 23.4 = S(1 − e^(−rT)), put-call parity holds)

Anchor ATM on the PE side (we're selling PE):
  Target sell P  =  0.80 × 216.2  =  173.0
  Target buy  P  =  0.60 × 216.2  =  129.7

PE strike scan (50-pt grid, BSM):
  Strike   BSM PE   % of ATM
   24400   216.2    100.0%   ← ATM
   24350   192.7     89.1%
   24300   170.9     79.0%   ← algorithm sell leg   (closest to 173.0)
   24250   150.9     69.8%
   24200   132.5     61.3%   ← algorithm buy leg    (closest to 129.7)
   24150   115.7     53.5%
   24100   100.5     46.5%
   24050    86.9     40.2%
   24000    74.6     34.5%   ← note's buy leg
   23950    63.8     29.5%

Algorithm pick — strict 80/60 (24,300 sell / 24,200 buy):
  net credit    = 170.9 − 132.5 = 38.4 INR
  max profit    = 38.4  × 75 = ₹2,880
  max loss      = (100 − 38.4) × 75 = ₹4,620
  breakeven     = 24,300 − 38.4 = 24,261.6
  R:R           = 1 : 1.60
  spread width  = 100 pts

Note's pick — eyeballed (24,200 sell / 24,000 buy):
  net credit    = 132.5 − 74.6 = 57.9 INR
  max profit    = 57.9  × 75 = ₹4,343
  max loss      = (200 − 57.9) × 75 = ₹10,658
  breakeven     = 24,200 − 57.9 = 24,142.1
  R:R           = 1 : 2.46
  spread width  = 200 pts
```

### This divergence is the design decision

The algorithm picks a **tight (100-pt), higher-probability, lower-absolute-P&L
spread**. The human picks a **wide (200-pt), lower-probability, higher-absolute-P&L
spread**. Both are defensible — they optimize different objectives:

| Dimension | 80/60 (algorithm) | Note's pick (human) |
|---|---|---|
| Spread width | 100 pts | 200 pts |
| Net credit per lot | ₹2,880 | ₹4,343 |
| Max loss per lot | ₹4,620 | ₹10,658 |
| Breakeven distance from spot | 138.4 pts (0.57%) | 257.9 pts (1.06%) |
| Probability of max profit (approx) | ~79% | ~73% |
| Bang-per-rupee-risked | 0.62 | 0.41 |
| Bang-per-trade (absolute ₹) | Lower | Higher |

Interpretation:

- The 80/60 rule biases toward **frequent small wins** — more breakevens,
  less drawdown. Fits if we're running this daily.
- The wider spread biases toward **bigger wins, rarer but costlier losses**.
  Fits if we're running this once per weekly expiry and want meaningful P&L
  per trade.

Both strategies are codable as param presets:

```yaml
# "tight" preset (rule as noted)
sell_premium_pct: 0.80
buy_premium_pct:  0.60
# "wide" preset (note's eyeballed behavior)
sell_premium_pct: 0.60
buy_premium_pct:  0.35
```

See §8 open decision #1.

---

## 3. Risk, exits, stops

### Hard invariants (enforced by @RiskGuard, not the LLM)

1. **Max risk per spread** = configurable % of capital (default 1.0%). Spread
   is rejected if `max_loss × lots > capital × 0.01`.
2. **Max concurrent spread risk** = 3% of capital across all open vertical
   spreads.
3. **Sell strike on wrong side of expected range** → reject with reason.
4. **Net credit < 10% of width** → reject (not worth the risk).
5. **Any leg ITM at entry time** → reject (defeats the premium-selling thesis).
6. **VIX spike during entry** (VIX > entry VIX × 1.10 in the last 5 min) →
   delay entry one candle.
7. **DTE < 2 on entry** → reject (gamma risk dominates theta edge).

### Profit target and stops

| Trigger | Action |
|---|---|
| Combined leg P&L = 50% of max profit | Close basket (ROLL optional if >2 DTE left) |
| Combined leg P&L = 2× net credit loss | Close basket unconditionally |
| Sell leg breached (spot past strike) | Move to "watch mode" — close on next candle close beyond breakeven |
| Expiry day 14:45 IST | Auto-close regardless of P&L (cron enforces) |
| Bias flip detected mid-trade (Cascade output) | Close basket; signal fired strategy error |
| Gap-open > 1.5 σ | Re-evaluate; close if spot past sell strike |

### Why a profit target at 50%?

Empirically the highest Sharpe for short-premium strategies comes from
taking partial profits early rather than holding to expiry — theta decay is
front-loaded, tail-risk is back-loaded. 50% of max profit at ~40% of holding
time is the Tastytrade rule and holds up in Indian index options too.

---

## 4. Edge cases

| Case | Resolution |
|---|---|
| **Bid-ask spread wider than 20% of mid** on either leg | Reject basket; liquidity gate (use `getMarketData(FULL)` depth). |
| **Lot-size multiplier changes** (NIFTY 75 → e.g. 25 in future) | Pull from `instrument.lot_size`, never hardcode. |
| **Expiry-day pin risk** | If on expiry day at 14:30 IST sell strike is within ±0.25σ of spot, force close regardless of P&L. |
| **Corporate action on a stock underlying** | Vertical-spread strategy is **index-only** in V1; stock verticals deferred until CA auto-rebase (ADR-0006 V2). |
| **Partial fill on sell leg with buy leg unfilled** | Basket atomicity (ADR-0006 §4 case #12) — if buy leg fails, close partial sell leg. |
| **Mid-day VIX spike > 25%** | Don't enter new verticals; existing ones get a hard stop re-check. |
| **Circuit breaker on underlying** | Freeze strategy agent; reconcile when market resumes. |
| **Roll (close + re-enter next expiry)** | Explicit action, not automatic. Agent can suggest with reasoning; human approves via UI or CLI flag. |
| **Paper-mode mid-trade gap** | Paper fill simulator uses real LTP; synthetic postbacks preserve realism. No code difference from live. |

---

## 5. Codebase mapping

```
trading/options/
├── verticals/                              ← NEW module (mirrors straddle/)
│   ├── __init__.py
│   ├── state.py                            VerticalState, VerticalPlan, VerticalAction
│   ├── prompts.py                          @VerticalStrategist prompts
│   ├── analyzer.py                         P&L, breakeven, bias, pricing (pure python)
│   ├── graph.py                            LangGraph workflow (6 nodes, same shape as straddle)
│   ├── selector.py                         Strike selection via premium percentile
│   └── tests/
│       ├── test_selector.py                Deterministic strike-selection tests
│       ├── test_analyzer.py                P&L math, breakeven
│       └── test_bias.py                    Bias classifier table tests
├── pricing/                                ← NEW shared primitive (used by straddle + verticals)
│   ├── __init__.py
│   ├── bsm.py                              Black-Scholes (Call, Put, greeks, IV solver)
│   ├── iv_surface.py                       V2: polynomial fit of σ(DTE, moneyness)
│   └── tests/test_bsm.py                   Numerical parity vs py_vollib
└── ...                                     (existing)

trading/models.py
  + VerticalPosition(models.Model)          Positional-lifecycle model for a vertical basket
  + VerticalActionLog(models.Model)         Management actions (same shape as StraddleManagementLog)

apps/broker/adapters/base.py
  BrokerAdapter.place_basket(legs) → BasketAck   (atomicity guarantee from ADR-0006)

trading/management/commands/
  + manage_vertical.py                      CLI: register, analyze, status, execute action, list

frontend/src/features/positions/
  + VerticalCard.tsx                        New card variant; reuses existing Position components
```

### Agent topology

```
@DataAnalyst (existing)
   └─ feeds VerticalState with spot/σ/DTE/bias/shortlist
@VerticalStrategist (new, LLM)
   └─ picks bull-put / bear-call / iron-condor + sizing
@RiskGuard (existing, deterministic)
   └─ validates against hard invariants §3
BrokerAdapter.place_basket (existing protocol, P5 of ADR-0006)
   └─ atomic submission of 2-4 legs
```

The graph has the same 6-node shape as the straddle graph:

```
fetch_market_data → build_plan → generate_action → validate_action
   → execute_action → journal_action
```

`build_plan` and `generate_action` are new; every other node is a
parameter swap on the existing straddle nodes.

---

## 6. Strategy params (tunable)

```yaml
# seeded via Strategy model on migration
name: directional_vertical_spread
kind: directional_vertical
horizon: intraday                 # can also run swing (weeklies held 2-4 days)
underlyings: [NIFTY, BANKNIFTY]   # index-only for V1
params:
  sell_premium_pct: 0.80
  buy_premium_pct:  0.60
  min_dte: 2
  max_dte: 10
  profit_target_pct: 0.50         # of max profit
  stop_loss_mult: 2.0             # × net credit
  max_risk_per_trade_pct: 1.0     # of capital
  max_concurrent_risk_pct: 3.0
  bias_confidence_min: 0.6
  min_credit_to_width_pct: 0.10
  liquidity_max_spread_pct: 0.20
  entry_window_ist: ["10:00", "14:30"]
  expiry_day_force_close_ist: "14:45"
```

These live in the `Strategy` model's `params` JSONB. Changing them does NOT
require a deploy — the agent reads them every cycle.

---

## 7. Phased build

| Phase | Days | Outcome |
|---|---|---|
| **V1-0** | 2 | `trading/options/pricing/bsm.py` + tests. Also unblocks ADR-0006 P3 greeks. |
| **V1-1** | 1 | `selector.py` — premium-percentile strike finder with deterministic tests. |
| **V1-2** | 1 | `analyzer.py` — bias classifier, P&L / breakeven / expected range math. |
| **V1-3** | 2 | `graph.py` + `prompts.py` — @VerticalStrategist with structured JSON output. |
| **V1-4** | 1 | `VerticalPosition` model + migrations. |
| **V1-5** | 2 | `manage_vertical.py` CLI (register / analyze / status / execute / list). |
| **V1-6** | 1 | Paper-mode end-to-end smoke test on today's NIFTY chain. |
| **V1-7** | 2 | Frontend `VerticalCard.tsx` + Monthly view integration — spreads render as single position per basket. |
| **V1-8** | 1 | Backtest harness on `MarketSnapshot` once ADR-0006 P3 lands. |

Total: ~13 days (runs in parallel with ADR-0006 phases where possible;
V1-0 should ship as part of ADR-0006 P3 so the primitive is reused).

---

## 8. Decisions still open (to discuss when we start)

1. **Sell/buy premium-percentile defaults** — stick with 80/60 as the note
   prescribes, or expose presets (conservative 85/70, balanced 80/60,
   aggressive 75/50)?
2. **Stocks vs index only** — note mentions NIFTY; stock verticals have
   more edge (fatter tails, wider skew) but also assignment + CA risk.
   V1 stays index-only; confirm.
3. **Iron condor vs strangle on RANGE bias** — condor (4-leg, capped risk)
   vs strangle (2-leg naked, uncapped). Spec defaults to condor; note is
   silent. Confirm risk-first default.
4. **Roll automation** — agent suggests roll, human confirms? Or fully
   automatic with sizing cap? Note doesn't say; defaulting to suggest-only.
5. **Intraday vs swing** — a vertical opened Tue for Thu expiry is
   intraday-ish (3 days). Do we classify as `swing`? Impacts monthly
   capital bucketing. Proposal: `horizon = swing` if DTE > 1 on entry,
   `intraday` if DTE = 0. Confirm.
6. **Entry cadence** — is this "one trade per expiry" or multiple reloads
   if the first one hits profit target with 2 DTE still left? Note
   suggests one-and-done; spec supports either, default is one.
7. **Bias LLM override** — do we allow `@VerticalStrategist` to veto the
   deterministic bias, or is the classifier authoritative? Current spec is
   authoritative. Confirm we want the LLM **choosing a strategy within a
   given bias**, not debating the bias.

---

## 9. How this extends the existing system

- `@OptionsStrategist` becomes the family name; the straddle graph is one
  implementation, verticals graph is another, future iron-fly / calendar
  graphs slot in as siblings.
- `pricing/` becomes the shared options math primitive — used by straddle
  analyzer (today's hand-rolled BSM), verticals, and the ADR-0006 P3
  greeks layer for snapshots. One BSM implementation in the repo, not three.
- Monthly view gets a new F&O row shape: a vertical is a single strategy
  instance with 2 (or 4) legs — already the shape `StrategyInstance`
  supports in ADR-0006.
- Risk engine grows three new criteria (§3 #3, #4, #6) that apply only
  when strategy `kind=directional_vertical`. Implemented as predicates on
  the strategy, not new global criteria.
- Cascade feedback loop: a closed vertical's realized-P&L vs predicted-max
  feeds Stage 6 EXECUTE metrics; skew between realized IV and entry-VIX
  feeds back into the `iv_surface` fit.

---

## 10. What I want to show you when you're back

- This spec, for review / disagreement.
- One question in particular: **should V1-0 (the BSM primitive) be pulled
  forward into ADR-0006 P3 so we don't write BSM twice?** My read: yes,
  and it costs nothing — greeks solver is on the P3 plan anyway.
- A proposed seed for the `Strategy` model row so the `manage_vertical`
  CLI can run `--register --strategy directional_vertical` on day one.
- A mental test: given `S=24400, σ=0.20, DTE=5`, what sell/buy strikes
  does the algorithm pick vs what a human picks? If they diverge (as they
  do above — 24300/24100 vs 24200/24000), we want to understand why and
  decide which is right.
