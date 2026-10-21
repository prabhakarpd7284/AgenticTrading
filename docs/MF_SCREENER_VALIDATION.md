# MF Screener — S5 validation (point-in-time backtest)

Run on 2026-10-21 with real data (`manage.py mf_backtest_screen`; the code is in
`backend/apps/mutual_funds/services/backtest.py`). PERSONAL USE ONLY.

## Verdict

**The score has no measurable predictive value.** Within a category, the
composite rank did not tell the next 12 or 36 months' winners from the losers.
Pooled over 9 equity categories:

- Top quartile minus bottom quartile: -0.3 pp/yr (12m) and -0.4 pp/yr (36m).
- Mean rank IC: -0.01 at both horizons. Non-overlapping t-stats: -0.6 and +0.5.
- The top quartile beat the passive alternative on 52-55% of dates, a coin flip.

No single component did better. The best were trailing 3y CAGR (+0.6 pp, t
+0.96) and risk-adjusted return (+0.3 pp, t +0.69) at 12 months; neither is
significant. Drawdown-only was mildly *negative* (12m IC t = -2.0), which looks
like reversal. This matches the literature on weak fund-performance persistence.

What follows:

1. The screen is a **description of past behaviour** (consistency, risk,
   drawdown, age) for picking among funds you would be comfortable holding. It is
   **not a forecast**. The CLI and UI say so.
2. Default weights were changed (see "Weights decision" below) as the plan's
   weak-result rule says: drawdown and age down, cost up.
3. In categories where low-cost index funds have a track record, they are the
   honest default. Large Cap shows this most clearly: the median active fund
   beat the Nifty 50 index-fund median in 65% of rolling windows, but the gap was
   only +0.8 pp, and the screen could not pick the winners in advance. The
   planner (S6) should lean passive wherever the index-vs-active check is not a
   clear "yes".

## Method

- **Dates.** Quarter-ends from 2016-03-31 (Direct plans started in Jan 2013, so
  this is the first date with 3 years of history) to the last date whose forward
  window is complete. That is 39 dates for 12m (to 2025-09) and 31 dates for 36m
  (to 2023-09).
- **Eligible at t.** Universe schemes of the category, **live and dead**, that
  had a NAV within 7 days of t and at least 3 years of history at t.
- **Point-in-time.** Every metric is computed from NAV ≤ t only. The peer
  medians for "% of rolling windows above the category median" also use only
  NAV ≤ t, and include funds that later died. The scoring code (`score_frame`)
  is the same one the live screener uses. A NAV jump known at t (face-value
  change, etc.) cuts the history at the jump. A fund-date whose *forward* window
  contains a jump is skipped.
- **Forward return.** Simple return from t to t+h, annualised for 36m.
  **Dead funds stay in.** A fund that dies inside the horizon gets its return up
  to its last NAV, chained with the category's median forward return for the rest
  of the horizon. We do not have the scheme-merger mapping, so the category median
  is the neutral assumption. "Dead chained" in the tables counts such fund-dates.
- **Groups.** Rank-percentile within the date: top quartile (>75th), middle half,
  bottom quartile (≤25th). The top-minus-bottom spread, the Spearman rank IC, and
  the share of dates where the top quartile beat the passive alternative are
  averaged over dates. Forward windows overlap between quarterly dates, so the IC
  t-stat uses only non-overlapping dates (every 4th date for 12m, every 12th for
  36m).
- **Passive alternative.** The median of the category's plain index funds (NAV,
  after their costs) that existed over the whole window. Before index funds
  existed (mid/small/500 index funds are mostly 2019+), the gross Nifty TRI is
  used. Gross TRI flatters passive by about one index-fund TER.
- **Variants.** `composite_default` (the final weights below), `plan_weights`
  (the plan's 30/25/20/15/10), and single components: consistency,
  risk-adjusted, drawdown, age, risk+drawdown, and trailing 3y CAGR (the "just
  buy last 3 years' winner" baseline). TER is not sourced, so cost is dropped and
  weights are renormalised in every variant, exactly as in the live screener.
- **Risk-free rate.** A constant 5.52% (see the data-sources doc).

## Results — pooled over categories

Equal weight per category. "IC t" uses only non-overlapping dates pooled across
categories (n = the number of category-dates).

### 12-month forward

| Variant | Top Q | Middle | Bottom Q | Top-Bottom (pp) | Categories T>B | Mean IC | IC t | Top Q beats passive (% of dates) |
|---|---|---|---|---|---|---|---|---|
| composite_default (final) | 16.9% | 17.0% | 17.2% | -0.29 | 4/9 | -0.008 | -0.58 (n=88) | 55% |
| plan_weights (30/25/20/15/10) | 16.8% | 17.1% | 17.2% | -0.43 | 3/9 | -0.020 | -0.89 (n=88) | 54% |
| consistency_only | 16.9% | 17.1% | 17.0% | -0.06 | 4/9 | -0.006 | -0.46 (n=88) | 55% |
| risk_adjusted_only | 17.2% | 16.9% | 17.0% | +0.27 | 6/9 | +0.028 | +0.69 (n=88) | 60% |
| drawdown_only | 16.4% | 17.2% | 17.4% | -1.04 | 2/9 | -0.067 | -2.03 (n=88) | 49% |
| age_only | n/a* | 16.8% | n/a* | n/a* | 3/9 | -0.064 | -1.48 (n=88) | 49% |
| risk_plus_drawdown | 16.9% | 17.0% | 17.3% | -0.40 | 3/9 | -0.019 | -0.82 (n=88) | 56% |
| trailing_3y_cagr | 17.3% | 17.0% | 16.7% | +0.60 | 6/9 | +0.039 | +0.96 (n=88) | 61% |

### 36-month forward (annualised)

| Variant | Top Q | Middle | Bottom Q | Top-Bottom (pp) | Categories T>B | Mean IC | IC t | Top Q beats passive (% of dates) |
|---|---|---|---|---|---|---|---|---|
| composite_default (final) | 16.3% | 16.7% | 16.7% | -0.38 | 4/9 | -0.013 | +0.47 (n=27) | 52% |
| plan_weights | 16.4% | 16.6% | 16.8% | -0.42 | 3/9 | -0.024 | +0.35 (n=27) | 53% |
| consistency_only | 16.7% | 16.4% | 16.7% | -0.08 | 6/9 | +0.009 | +1.18 (n=27) | 58% |
| risk_adjusted_only | 16.3% | 16.7% | 16.6% | -0.30 | 5/9 | -0.013 | +0.50 (n=27) | 57% |
| drawdown_only | 16.6% | 16.5% | 16.6% | +0.10 | 3/9 | -0.024 | -0.32 (n=27) | 60% |
| age_only | n/a* | 16.3% | n/a* | n/a* | 3/9 | -0.141 | -1.20 (n=27) | 43% |
| risk_plus_drawdown | 16.2% | 16.7% | 16.6% | -0.40 | 3/9 | -0.030 | -0.01 (n=27) | 52% |
| trailing_3y_cagr | 16.4% | 16.7% | 16.6% | -0.22 | 6/9 | -0.009 | +0.91 (n=27) | 55% |

\* Age ties (many funds share the 2013 Direct-plan start) leave the top/bottom
quartile empty on some dates, so the group mean is undefined there. The IC is
still computed.

## Results — per category (final composite)

### 12-month forward

| Category | Dates | Funds/date | Dead chained | Passive basis | Passive | Top Q | Middle | Bottom Q | IC | Trailing-3y T-B (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| Large Cap | 39 (2016-03..2025-09) | 26.6 | 20 | index fund | 13.8% | 14.2% | 14.4% | 14.5% | -0.020 | -0.92 |
| Large & Mid Cap | 39 | 21.9 | 7 | index fund / TRI | 17.5% | 17.4% | 17.3% | 17.2% | +0.046 | +1.00 |
| Mid Cap | 39 | 20.2 | 16 | index fund / TRI | 20.5% | 18.7% | 19.9% | 19.6% | -0.023 | +0.85 |
| Small Cap | 39 | 16.8 | 12 | index fund / TRI | 19.4% | 21.2% | 22.1% | 19.8% | +0.140 | +3.81 |
| Flexi Cap | 39 | 20.8 | 0 | index fund / TRI | 15.3% | 16.3% | 16.3% | 16.0% | -0.006 | +1.45 |
| Multi Cap | 32 | 9.2 | 13 | index fund / TRI | 13.5% | 16.5% | 15.1% | 15.2% | -0.004 | +0.61 |
| Focused | 39 | 16.6 | 14 | index fund / TRI | 15.3% | 15.3% | 15.3% | 18.5% | -0.155 | -1.81 |
| Value | 39 | 12.8 | 8 | index fund / TRI | 15.3% | 16.8% | 16.8% | 17.7% | +0.009 | -0.40 |
| ELSS | 39 | 30.3 | 26 | index fund / TRI | 15.3% | 15.7% | 16.1% | 16.5% | -0.057 | +0.84 |
| Contra, Dividend Yield | 0 | | | | | | | | | |

### 36-month forward (annualised)

| Category | Dates | Funds/date | Dead chained | Passive | Top Q | Middle | Bottom Q | IC | Trailing-3y T-B (pp) |
|---|---|---|---|---|---|---|---|---|---|
| Large Cap | 31 (2016-03..2023-09) | 26.5 | 60 | 13.7% | 14.2% | 14.3% | 14.0% | +0.010 | -0.90 |
| Large & Mid Cap | 31 | 20.9 | 23 | 16.7% | 16.7% | 16.7% | 16.2% | +0.126 | +0.89 |
| Mid Cap | 31 | 19.9 | 48 | 19.1% | 17.5% | 18.7% | 20.2% | -0.235 | -2.52 |
| Small Cap | 31 | 15.6 | 36 | 16.8% | 19.4% | 19.8% | 16.7% | +0.149 | +2.78 |
| Flexi Cap | 31 | 19.3 | 0 | 14.9% | 15.4% | 16.0% | 15.4% | -0.012 | +0.35 |
| Multi Cap | 26 (to 2022-09) | 8.7 | 45 | 15.3% | 18.1% | 17.9% | 17.6% | +0.092 | +0.21 |
| Focused | 31 | 15.2 | 31 | 14.9% | 14.0% | 15.3% | 17.7% | -0.260 | -3.14 |
| Value | 31 | 12.1 | 20 | 14.9% | 16.2% | 15.8% | 16.4% | +0.092 | +0.04 |
| ELSS | 31 | 30.0 | 66 | 14.9% | 15.5% | 15.2% | 16.2% | -0.076 | +0.33 |

Contra (5 live funds) and Dividend Yield (12) never had the 8 eligible funds a
date needs, so they are not backtested.

Signs flip between categories and horizons. Small Cap is mildly positive (IC
+0.14/+0.15). Mid Cap and Focused are clearly negative at 36m: last period's
best mid-cap and focused funds lagged. Nothing here is stable enough to tune
weights on. Doing so would be overfitting 9 noisy series.

## Index-vs-active check (live screener, as of 2026-10-16)

Rolling 3-year windows, month-end sampled. The median active fund of the
category (dead funds included) is compared with the passive reference. The
reference is index funds when their history covers at least 60% of the windows;
otherwise it is the gross TRI. "Alt" is the result against the other basis.

| Category | Basis | Windows | Range | Median active beat ref (% windows) | Last 3y (%) | Median gap (pp/yr) | Funds beating ref most of the time | Verdict | Alt basis (windows, %) |
|---|---|---|---|---|---|---|---|---|---|
| Large Cap | Nifty 50 index funds | 130 | 2016-01..2026-10 | 65.4 | 100.0 | +0.78 | 28/35 | yes | TRI Nifty 100 (130, 56.2) |
| Large & Mid Cap | TRI LargeMidcap 250 | 130 | 2016-01..2026-10 | 44.6 | 62.2 | -0.08 | 14/28 | mixed | index funds (23, 100.0) |
| Mid Cap | TRI Midcap 150 | 130 | 2016-01..2026-10 | 40.8 | 13.5 | -0.79 | 13/32 | mixed | index funds (50, **10.0**) |
| Small Cap | TRI Smallcap 250 | 130 | 2016-01..2026-10 | 78.5 | 27.0 | +3.19 | 20/27 | yes | index funds (50, 68.0) |
| Flexi Cap | TRI Nifty 500 | 130 | 2016-01..2026-10 | 56.2 | 62.2 | +0.34 | 21/36 | mixed | index funds (50, 60.0) |
| Multi Cap | TRI Nifty 500 | 130 | 2016-01..2026-10 | 71.5 | 100.0 | +2.82 | 24/27 | yes | index funds (50, 100.0) |
| Focused | TRI Nifty 500 | 130 | 2016-01..2026-10 | 68.5 | 89.2 | +0.57 | 17/31 | yes | index funds (50, 96.0) |
| Value | TRI Nifty 500 | 130 | 2016-01..2026-10 | 64.6 | 100.0 | +1.82 | 17/22 | yes | index funds (50, 100.0) |
| Contra | TRI Nifty 500 | 130 | 2016-01..2026-10 | 100.0 | 100.0 | +2.57 | 3/3 | yes (3 funds) | index funds (50, 100.0) |
| Dividend Yield | TRI Nifty 500 | 130 | 2016-01..2026-10 | 58.5 | 100.0 | +0.70 | 7/9 | mixed | index funds (50, 100.0) |
| ELSS | TRI Nifty 500 | 130 | 2016-01..2026-10 | 65.4 | 94.6 | +0.52 | 26/45 | yes | index funds (50, 96.0) |

How to read this, honestly:

- **Mid Cap: passive wins.** The median active mid-cap fund lost to the Midcap
  150 TRI in 59% of windows, and in 87% of the last 3 years' windows. It lost to
  Midcap 150 index funds in 90% of their 50 windows.
- **Large Cap.** Active edged out the index funds historically (+0.8 pp/yr
  median gap). The backtest shows that edge could not be captured by picking
  funds in advance (top-quartile IC ≈ 0).
- **Multi / Value / Contra / Focused / ELSS vs Nifty 500.** The "yes" here is
  mostly a **style mismatch**: these funds hold more mid and small caps than the
  Nifty 500, and mid/small caps rallied in 2020-24. For Multi Cap the SEBI
  benchmark is Nifty 500 Multicap 50:25:25, which niftyindices returned empty for
  2014. A "yes" means the median fund beat a plain broad index after costs. It
  does **not** mean a fund chosen by this screen will.
- **Small Cap** is the one category where both the check ("yes", +3.2 pp) and
  the backtest (IC +0.14) lean the same way. With 9 categories tested, one
  agreeing pair is what chance alone would give, so do not lean on it.

## Weights decision

| Component | Plan's suggestion | **Final default** | Why |
|---|---|---|---|
| Consistency (rolling 3y: % windows above category median, worst window) | 30 | **30** | Neutral in S5. Kept, because it describes the "steady fund" behaviour the owner wants, not because it predicts |
| Risk-adjusted (Sortino, Sharpe) | 25 | **25** | Slightly positive at 12m (not significant) |
| Drawdown (max DD, worst 1y) | 20 | **10** | Negative at 12m (t = -2.0); cut |
| Cost (TER) | 15 | **30** | The only factor with strong prior evidence (costs are certain, returns are not). **Inactive until TER is sourced**; then it dominates |
| Size / age (age only; AUM not sourced) | 10 | **5** | Age was negative in S5; kept small as a sanity tie-breaker |

While TER is missing, the renormalised weights are about consistency 42.9,
risk-adjusted 35.7, drawdown 14.3, age 7.1. Every output prints the weights it
actually used. The consistency share is also shrunk toward 50% with a 24-window
prior. Without that, a 3-year-old fund with a handful of good windows scored
"100% above median" (seen in the first live Flexi Cap run).

## Caveats (read before trusting any number above)

1. **Category relabelling.** Each fund's *current* SEBI/AMFI category is applied
   to its whole history. Flexi Cap only exists since Nov 2020: earlier, those
   funds were "Multi Cap", which is why Flexi Cap has 0 dead chained funds and
   Multi Cap has few funds per date.
2. **Survivorship is reduced, not removed.** 436 dead Direct-Growth schemes were
   recovered from mfapi.in. Dead schemes that carry pre-2018 category labels
   ("Income" 231, "Growth" 11) cannot be placed in a SEBI category, so they are
   absent from these equity backtests. The dead-fund "chain with the category
   median" rule is an assumption.
3. **Passive basis is mixed over time.** Index funds are used where they existed,
   the gross TRI otherwise. Gross TRI makes passive look slightly *better*, so it
   is conservative toward active funds.
4. **Short sample.** About 10 years, one market regime dominated by a mid/small
   cap rally. 36m results rest on 31 overlapping dates per category (27
   non-overlapping category-dates pooled).
5. **TER and AUM are not sourced**, so the cost and size components could not be
   tested at all.
6. The risk-free rate is a single constant (5.52%). That changes Sharpe levels
   only, not ranks.

Reproduce:

```bash
cd backend
nice -n 10 .venv/bin/python manage.py mf_backtest_screen --out /tmp/mf_backtest.json   # ~4 min
```
