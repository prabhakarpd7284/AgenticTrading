# Pivot Boss on NSE — measured edges and backtest results

Run date: 2026-10-21. Universe: NIFTY 50 constituents (49 names in
`apps.market_data.constants.NIFTY_50_SYMBOLS`), 2026-04-20 → 2026-10-19
(125 sessions, **6,111 stock-days**; the first session of each stock only
supplies pivot inputs). Spec: `docs/strategies/PIVOT_BOSS_SPEC.md`.

**Bottom line:** most of the book's descriptive statistics carry over to NSE
large caps in rough shape. The predictive claims mostly don't: CPR width does
not predict trend days, the two-day relationships carry no directional bias,
and Camarilla H3/L3 is broken through more often than it is faded. **Every
setup we backtested lost money after costs, in-sample and out-of-sample.**
Before slippage and costs each one was at roughly 0 R per trade, so there was
no raw edge for costs to eat into.

Reproduce:

```bash
cd backend
nice -n 10 .venv/bin/python manage.py pivot_research --out-dir research_output   # ~1 min, flat memory
```

Raw outputs (gitignored) are in `backend/research_output/`:
`pivot_research_days.csv` (one row per stock-day), `pivot_backtest_trades.csv`
(every trade), `pivot_backtest_summary.json`, and `summary.md` (every table,
including the IS/OOS splits that are left out below). Code is in
`backend/plugins/strategy_pivot_boss/` and
`backend/apps/market_data/management/commands/pivot_research.py`. Tests are in
`backend/tests/unit/test_pivot_boss.py`.

Notation: `n` = pooled stock-days or events. `*` marks a cell with n < 30 (low
confidence). IS = 2026-04-20 → 08-31 (92 sessions). OOS = 09-01 → 10-19 (33
sessions).

---

## Data and method choices

| Item | Choice |
|---|---|
| Pivot inputs | NSE bhavcopy daily H/L/C of the prior session. Touches and closes in A/B/E come from the bhavcopy (official close = last-30-min VWAP). |
| Intraday paths | Angel 1-minute bars. **The last tradable bar is 15:14**: 15:15–15:27 and 15:29 have been missing every day since 2026-08-03. "By the close" in C/D means by 15:14. |
| Initial balance | 09:15–10:14 1m bars. "IB broken" = any 1m high/low after 10:15 beyond the IB. |
| ATR | Daily ATR(14) is a simple mean of true range over the 14 sessions *before* the day. Days without 14 sessions of history are excluded from ATR-based cells. The trailing-stop ATR is ATR(14) on 15-min bars, published only when a 15-min bar closes. |
| Trend day | **def1**: the open is within 20% of one extreme of the day's range and the close within 20% of the other. **def2**: the close is within 20% of an extreme *and* range ≥ ATR14. |
| Money Zone | Volume-at-price from 1m bars. Each bar's volume is spread equally over its H–L. Bin = 0.05% of price, rounded to a 0.05 tick. VPOC = the max-volume bin. VA = 70%, expanding from the POC and adding the larger adjacent row (both on a tie). TPO POC uses 30-min periods. |
| Two-day "unchanged" | Both band edges within **0.02% of price** of yesterday's edges. The book gives no number. With this tolerance "unchanged" is almost never seen (≤ 0.1%). |
| Appendix-A "touch" | A level counts as reached when high ≥ R*n* or low ≤ S*n*, so a gap through the level counts. |
| Narrow/wide CPR | Calibrated on IS only: **narrow < 0.116%** (p20), **wide > 0.564%** (p80). |

---

## A. Appendix-A replication (floor pivots)

The book's table is internally ambiguous. It reports "close beyond L1" =
24.8%, yet also says the close lands inside R1–S1 on 53.4% of days, which
implies 46.6% beyond on either side. On our data the book's touch rows match
**either-side** rates, and its close-beyond rows match **per-side** averages
(the mean of the R-side and S-side rates). We show both.

| Statistic | Book (YM 08–09) | NSE, either side | NSE, per-side avg (R / S) | n |
|---|---|---|---|---|
| PP touched | 63.0% | 69.8% | — | 6111 |
| CPR band (BC–TC) touched | — | 78.2% | — | 6111 |
| L1 reached | 73.3% | 81.2% | 45.7% (43.9 / 47.4) | 6111 |
| L2 reached | 38.2% | 36.5% | 18.7% | 6111 |
| L3 reached | 17.6% | 15.3% | 7.7% | 6111 |
| L4 reached | 5.5% | 6.9% | 3.4% | 6111 |
| Close beyond L1 | 24.8% | 44.8% | **22.4%** | 6111 |
| Close beyond L2 | 12.1% | 18.9% | **9.5%** | 6111 |
| Close beyond L3 | 6.7% | 7.6% | 3.8% | 6111 |
| Close beyond L4 | 3.0% | 3.4% | 1.7% | 6111 |
| PP untouched days | 37.0% | 30.2% | — | 6111 |
| L1 / L2 / L3 / L4 reached, PP untouched | 67.2 / 63.9 / 34.4 / 11.5% | 90.4 / 57.1 / 27.7 / 14.4% | — | 1845 |
| Close beyond L1, PP untouched | 73.8% (prose) | **65.3%** | 32.6% | 1845 |
| Close beyond L1, PP touched | 46.6% (prose) | **36.0%** | — | 4266 |
| Close within S1–R1 | 53.4% | 55.2% | — | 6111 |
| Open > PP → close > PP | 53.4% | **63.6%** (IS 66.6, OOS 54.3) | — | 2961 |
| Open < PP → close < PP | 48.8% | **68.5%** (IS 68.1, OOS 69.3) | — | 3150 |

Takeaways:
- The touch and close distributions are close to the book's. R1/S1 is the
  realistic default target and R3/S3 is rare (closes beyond it on ~4% of days
  per side).
- A PP that stays untouched does flag a directional day: 65% close beyond L1,
  against 36% when the PP is touched. That is a 29-point gap, compared with the
  book's 27.
- Open-side-of-PP persistence is **stronger** on NSE than in the book (64–69%,
  against 49–53%), and it holds out-of-sample. Part of this is mechanical: the
  open is already on that side of the PP. It is still the cleanest
  "context" fact in this study.

## B. CPR width vs day type

CPR width percentiles (% of PP):

| sample | p5 | p10 | p20 | p25 | p40 | p50 | p60 | p75 | p80 | p90 | p95 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.032 | 0.061 | 0.119 | 0.150 | 0.243 | 0.308 | 0.382 | 0.521 | 0.584 | 0.785 | 0.992 | 6111 |
| IS | 0.031 | 0.059 | 0.116 | 0.145 | 0.236 | 0.301 | 0.371 | 0.507 | 0.564 | 0.751 | 0.942 | 4508 |

On NSE large caps, the book's 0.25% "narrow" cut-off covers 42.5% of IS
stock-days and its 0.75% "wide" cut-off covers 10.1%.

| CPR width (IS quintile edges), all days | n | mean range/ATR14 | trend day def1 | trend day def2 | IB broken (either) | IB broken both |
|---|---|---|---|---|---|---|
| Q1 (< 0.116%) | 1070 | 0.94 | **14.9%** | **16.0%** | 86.4% | 12.7% |
| Q2 (0.116–0.236) | 1063 | 0.95 | 17.7% | 20.2% | 88.1% | 13.5% |
| Q3 (0.236–0.371) | 1058 | 0.95 | 19.3% | 19.7% | 87.9% | 13.9% |
| Q4 (0.371–0.564) | 1062 | 0.95 | 17.6% | 19.3% | 87.8% | 13.7% |
| Q5 (≥ 0.564%) | 1172 | 1.00 | **19.2%** | **22.0%** | 84.2% | 12.9% |
| Book: < 0.25 | 2264 | 0.94 | 16.7% | 18.5% | 87.3% | 13.1% |
| Book: > 0.75 | 607 | 1.02 | 18.3% | 21.9% | 84.5% | 13.3% |

Unconditional trend-day rate: 17.8% (def1) / 19.5% (def2), n = 5425.

**The book's central forecasting claim fails on this data.** Narrow-CPR days
do not have larger ranges and are not more often trend days. If anything the
narrowest quintile has the *fewest* trend days, in both IS (14.0%) and OOS
(17.1%). The IB is broken on 84–88% of days regardless of width. The
narrow/wide thresholds above are recorded as calibrated values, but they carry
no day-type information here.

## C. Camarilla

| Level reached (daily H/L) | H3 | L3 | H4 | L4 | H5 | L5 |
|---|---|---|---|---|---|---|
| % of 6,111 days | 64.7% | 64.9% | 40.7% | 43.1% | 17.9% | 18.6% |

3,984 days opened strictly inside L3–H3. The first band touched that day:

| Event | n | → opposite band first | → H4/L4 first | neither by 15:14 |
|---|---|---|---|---|
| First touch H3 | 1910 | 23.2% | **54.2%** | 22.6% |
| First touch L3 | 1895 | 20.9% | **58.2%** | 20.9% |

First close beyond H4/L4, crossing from inside, then a race between the 5
level and the 3 level:

| Event | n | → H5/L5 first | → back to H3/L3 first | neither by 15:14 |
|---|---|---|---|---|
| 1m close > H4 | 2083 | 36.1% | 45.7% | 18.3% |
| 5m close > H4 | 1902 | 41.1% | 38.0% | 20.9% |
| 1m close < L4 | 2255 | 34.2% | 40.2% | 25.5% |
| 5m close < L4 | 2122 | 37.8% | 34.3% | 27.9% |

H3/L3 is **not** a reversal level on NSE large caps. Once touched, price runs
on to H4/L4 more than twice as often as it returns to the opposite band. A
5-minute close beyond H4/L4 is roughly a coin flip between H5/L5 and the
retest. Because the H5/L5 target is about 2× further away than the H3/L3
stop, that looks like a positive raw expectancy. The backtest below shows
that the 5m-close entry fill and the session-end exits remove it.

## D. Gaps and the magnet effect

Gap = 09:15 open vs the prior bhavcopy close. CPR "reached" is measured only
on days that opened outside the CPR.

| Gap in daily ATR14 units (all months) | n | prior close filled by 10:15 | … by 15:14 | n (open outside CPR) | CPR reached by 10:15 | … by 15:14 |
|---|---|---|---|---|---|---|
| 0–0.25 | 3617 | 61.0% | 68.6% | 3013 | 75.9% | 84.0% |
| 0.25–0.5 | 1256 | 42.8% | 59.1% | 1038 | 48.3% | 63.4% |
| 0.5–1 | 471 | 19.7% | 34.8% | 440 | 32.7% | 44.8% |
| ≥ 1 | 81 | 8.6% | 17.3% | 79 | 17.7% | 24.1% |

| Gap in % of prior close (all months) | n | prior close filled by 10:15 | … by 15:14 | CPR reached by 10:15 | … by 15:14 |
|---|---|---|---|---|---|
| 0–0.25% | 2197 | 60.9% | 65.4% | 82.2% | 88.3% |
| 0.25–0.5% | 1290 | 60.6% | 73.1% | 65.5% | 76.0% |
| 0.5–1% | 1259 | 46.3% | 62.4% | 50.7% | 66.0% |
| 1–2% | 568 | 22.2% | 38.0% | 31.5% | 46.2% |
| ≥ 2% | 111 | 11.7% | 18.9% | 23.4% | 29.0% |

Earnings months (Apr 20–30, Jul, Oct) vs the rest, 0.5–1 ATR gaps: the prior
close was filled by 15:14 on 29.7% of days (n = 145) vs 37.1% (n = 326). The
CPR was reached on 37.2% vs 48.2%. Results-season gaps fill less often, as
the book says, though the samples are modest. Across the other buckets the
earnings split differs by only a few points (full tables are in `summary.md`).

The magnet effect is real for small gaps and fades quickly above ~0.5 ATR or
~1%. Fill odds within the first hour are about half the full-day odds, which
supports the book's "abandon after 30–60 min" rule.

## E. Two-day relationships → that day's outcome

CPR (BC–TC). The direction rate is close > open using bhavcopy prices. The
whole sample has a slightly negative O→C drift (mean around −0.05%).

| Relationship | n | freq | close > open | mean O→C | trend day (def1) |
|---|---|---|---|---|---|
| higher (book: bullish) | 2153 | 35.5% | 43.4% | −0.104% | 16.3% |
| overlapping higher | 419 | 6.9% | 47.3% | −0.007% | 20.0% |
| lower (book: bearish) | 2425 | 40.0% | 48.1% | −0.031% | 18.6% |
| overlapping lower | 452 | 7.5% | 47.1% | −0.020% | 16.6% |
| inside (book: breakout) | 395 | 6.5% | 43.0% | −0.080% | 16.5% |
| outside (book: sideways) | 213 | 3.5% | 45.1% | −0.021% | 19.7% |
| unchanged | 5 | 0.1% | 60.0% * | +0.008% | 0.0% * |

The bias goes the wrong way: "higher" days have the *lowest* close > open rate
(43%) and "lower" days the highest (48%), a mild mean reversion rather than
continuation. "Inside" CPR days are not more often trend days (16.5% vs 17.8%
base). The Camarilla and value-area relationship tables in `summary.md` show
the same thing: every bucket falls within a few points of the base rate.

Open location (Table 4.1, prior range plus prior value area): out of range
and out of value gives a trend day on 19.7% of days (n = 1427). In range and in
value gives 16.4% (n = 2524). The direction is the book's, but the effect is
small. Mean range/ATR is 1.03 vs 0.93.

## F. Virgin VPOC

The prior session's VPOC was untouched the next day on 30.2% of days
(n = 6097). Of those, **54.8%** were filled within the next 4 sessions
(n = 1797). The book claims about 80%.

---

## Backtest

### Rules (deterministic, book defaults, no OOS tuning)

- Each setup is run separately on each stock. There is at most one trade per
  setup per stock-day, and at most one open position per stock.
- Entry is at the **next bar's open** after the trigger bar. There are no new
  entries after 14:30, and every position is flat at the close of the 15:14
  bar.
- Stops and targets are checked on 1m bars. When one bar touches both, the
  **stop is assumed first**. A bar that opens through a level fills at its
  open. Slippage is 0.02% per side on every fill.
- Sizing: ₹5,00,000 capital, 0.5% risk (₹2,500), qty = risk / |entry − stop|,
  capped at 20% notional. Capital is fixed (no compounding).
- Costs per executed order: brokerage min(₹20, 0.1%); STT 0.025% on the sell
  side; NSE transaction charge 0.00297%; SEBI ₹10/crore; stamp duty 0.003% on
  the buy side; GST 18% on brokerage + transaction + SEBI.

How each setup was operationalised (book wording → our rule):

| Setup | Rule used |
|---|---|
| Floor breakaway | CPR width < 0.116% (IS p20). The 09:15 open is above the prior high, above the prior VAH and in (R1, R2] (mirror for shorts). The first 15m bar's low comes within 0.10% of R1, it closes above R1 and in the upper half of its range ("retest and hold"; a strict 2.5× wick was not required). Entry 09:30. Stop is 1 tick beyond the bar's extreme. Target R3, plus a 1.5 × ATR(15m) trail. `_any_cpr` is the same rule without the narrow-CPR filter. |
| Camarilla H3/L3 reversal | Open inside the prior range. A 5m bar has high > H3 and close < H3. The next 5m bar has high < H3 and close < the prior close. Enter at the next bar. Stop H4 + 1 tick, target L3 (mirror at L3). |
| Camarilla H4/L4 breakout | First 5m close above H4 that crosses it (the previous 5m close or the open was ≤ H4). Enter at the next bar. Stop H3, target H5 (mirror). |
| Magnet | Gap of 0.25–1.0 daily ATR. Open beyond the CPR. The near CPR edge is within 0.5 × gap of the prior close. No R1/S1 between the open and the CPR. The first 15m bar is a reversal candle: opposite colour with its close in the gap-fill half, or any Appendix-B trigger. Entry 09:30. Stop beyond the bar extreme. Target the near CPR edge. Time exit 60 min after entry. |
| Golden Pivot Zone | [BC, TC] contains H3, L3, or the prior VAH/VAL/POC/VPOC. Trend: the prior day closed beyond its own CPR, and today's prior close and open are both beyond today's CPR on the same side. A 15m bar pulls back into the zone and fires any of the 4 triggers, closing on the trend side of the far edge. Enter at the next bar. Stop beyond the zone or the bar. Target the nearest of R1/H3 (S1/L3). |

### Results (net of costs)

| Setup | Sample | Trades | Win % | Avg win ₹ | Avg loss ₹ | PF | Gross ₹ | Costs ₹ | **Net ₹** | Exp ₹/trade | Exp R (net) | Max DD ₹ | Trades/day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Floor breakaway | IS | 31 | 19.4% | 505 | −469 | 0.26 | −6,150 | 2,546 | **−8,696** | −281 | −0.34 | 9,531 | 0.34 |
| | OOS | 10 * | 50.0% | 562 | −429 | 1.31 | 1,486 | 822 | **+664** | +66 | −0.01 | 929 | 0.30 |
| Floor breakaway (any CPR) | IS | 110 | 26.4% | 538 | −462 | 0.42 | −12,846 | 9,007 | **−21,853** | −199 | −0.27 | 21,853 | 1.20 |
| | OOS | 31 | 38.7% | 416 | −392 | 0.67 | 85 | 2,545 | **−2,460** | −79 | −0.18 | 5,735 | 0.94 |
| Cam H3/L3 reversal | IS | 1989 | 47.8% | 421 | −655 | 0.59 | −118,280 | 162,985 | **−281,265** | −141 | −0.17 | 282,448 | 21.6 |
| | OOS | 744 | 44.6% | 428 | −694 | 0.50 | −83,002 | 60,938 | **−143,940** | −193 | −0.24 | 144,711 | 22.6 |
| Cam H4/L4 breakout | IS | 2567 | 44.5% | 449 | −605 | 0.59 | −139,780 | 210,403 | **−350,183** | −136 | −0.23 | 350,792 | 27.9 |
| | OOS | 950 | 47.8% | 435 | −591 | 0.67 | −17,821 | 77,874 | **−95,695** | −101 | −0.18 | 101,038 | 28.8 |
| Magnet | IS | 122 | 44.3% | 218 | −334 | 0.52 | −965 | 9,991 | **−10,956** | −90 | −0.17 | 12,388 | 1.33 |
| | OOS | 21 * | 61.9% | 262 | −443 | 0.96 | 1,580 | 1,726 | **−146** | −7 | +0.07 | 1,671 | 0.64 |
| Golden Pivot Zone | IS | 607 | 41.8% | 401 | −509 | 0.57 | −27,923 | 49,722 | **−77,645** | −128 | −0.25 | 78,144 | 6.6 |
| | OOS | 224 | 40.6% | 430 | −582 | 0.51 | −19,969 | 18,350 | **−38,319** | −171 | −0.37 | 42,816 | 6.8 |

(Win/loss are classified on net P&L. Trades/day is pooled across all 49
stocks. Exit-reason mixes are in `summary.md`.)

Diagnostics:

| Setup | Trades | Notional-capped | Avg risk ₹ | Costs per trade | Costs in R | Gross before slippage (R/trade) | Max concurrent positions |
|---|---|---|---|---|---|---|---|
| Floor breakaway | 41 | 100% | 812 | ₹82 | 0.10 R | −0.08 | 4 |
| Cam H3/L3 reversal | 2733 | 99.8% | 898 | ₹82 | 0.09 R | −0.04 | 22 |
| Cam H4/L4 breakout | 3517 | 99.9% | 673 | ₹82 | 0.12 R | **−0.00** | 24 |
| Magnet | 143 | 100% | 742 | ₹82 | 0.11 R | +0.06 | 6 |
| Golden Pivot Zone | 831 | 99.6% | 727 | ₹82 | 0.11 R | −0.04 | 13 |

- **Every setup loses money net, in IS and in OOS.** The two small positive
  OOS cells (floor breakaway, n = 10; magnet expectancy in R, n = 21) are
  low-n noise. Their IS counterparts are clearly negative.
- Pivot stops are tight relative to price (median H4−H3 is 0.5% of
  price). The 20% notional cap therefore binds on nearly every trade, and the
  real risk per trade is about ₹700–900, not ₹2,500. Fixed costs of about ₹82
  per round trip (mostly STT, brokerage and GST on a ₹1 lakh ticket), plus
  0.04% of round-trip slippage, come to about **0.2 R per trade**. To survive
  that, a setup would need a gross edge of at least 0.2 R. None comes close:
  before slippage the best is magnet at +0.06 R (n = 143). Camarilla H4/L4
  breakout is exactly 0 R.
- The Camarilla setups fire on almost every stock-day (22–28 trades/day
  across 49 stocks, up to 24 at once). Unfiltered, they are mostly noise
  trades. Up to 24 concurrent positions at 20% notional each is about 4.8×
  capital, at the edge of MIS leverage.

---

## Five most important findings

1. **CPR width does not forecast trend days on NSE large caps.** Trend-day
   rate, range/ATR and IB-break rate are flat across width quintiles, and
   the narrowest quintile is slightly *less* trendy. This is the book's
   headline forecasting tool and it failed both IS and OOS.
2. **Camarilla H3/L3 are continuation levels, not fade levels.** After a first
   touch from inside, H4/L4 comes next 54–58% of the time and the opposite
   band 21–23%. The H3/L3 reversal setup lost money in both samples.
3. **Floor-pivot statistics replicate:** touch rates, close distribution, and
   "PP untouched → directional day" (65% vs 36%). Open-side-of-PP persistence
   is stronger here (64–69%). These work as *context and targets* (R1/S1 as
   the default target, R3/S3 rare), not as entry signals.
4. **The magnet effect is real but only for small gaps.** Below 0.25 ATR, 76%
   of days reach the CPR by 10:15. Between 0.5 and 1 ATR it is 33%. The
   tradable "moderate gap" zone is thin. The magnet setup was the only one
   with positive gross R before slippage (+0.06 R, n = 143), which is still
   far below costs.
5. **Costs and sizing kill intraday pivot trading at this capital level.** With
   a 20% notional cap and pivot-tight stops, round-trip friction is about 0.2 R
   per trade. Under the book's defaults no setup produced a gross edge anywhere
   near that.

## Caveats

- **Short history:** 6 months (125 sessions), one market regime with a mild
  negative intraday drift. OOS is only 33 sessions, so several OOS setup cells
  have n < 30.
- **Survivorship:** we used today's NIFTY 50 list for the whole window. The
  bias is small over 6 months but not zero.
- **1-minute granularity:** the order of events inside a bar is unknown. Stops
  are assumed first, which is conservative for the backtest. Sequence stats
  count same-bar double hits separately (none occurred at these level
  spacings).
- **Missing last 15 minutes:** since 2026-08-03 there are no bars between
  15:15 and 15:29, so every intraday statistic and exit stops at 15:14. The
  bhavcopy close (last-30-min VWAP) is used for the daily close stats, so
  "close vs PP" and "close vs open" refer to that official close, not to the
  15:29 print.
- **Costs** are the stated assumptions (a discount broker's MIS schedule).
  Exchange-fee and stamp-duty revisions, DP charges (not applicable to MIS) and
  impact cost beyond the 0.02% slippage are not modelled.
- **Interpretation choices** are listed in the method table. The ones that
  matter most: the breakaway "retest and hold" (stricter versions fire even
  less), the magnet's "moderate gap" (0.25–1 ATR) and "CPR near prior close"
  (≤ 0.5 × gap), and the golden-zone trend definition. None were tuned. The
  only calibrated values are the IS CPR-width percentiles, which the B-section
  results show carry no signal anyway.
- **Not covered:** the Camarilla breakaway, pivot trend pullback, CPR trend,
  hidden layers, virgin-level fade, Inside Value breakout, and
  multi-timeframe hot zones. Weekly and monthly pivot helpers and the virgin
  tracker are built (`levels.py`) but were not backtested.
