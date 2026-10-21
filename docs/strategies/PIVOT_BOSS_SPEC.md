# Pivot Boss — implementation reference

Extracted from *Secrets of a Pivot Boss* (Frank Ochoa, 2010), all 11 chapters +
Appendix A (statistics) + Appendix B (the author's 9 scripts). This is the spec
for building these tools in AlphaDesk; the book's statistics are from US futures
(2008–09) and **must be re-verified on NSE data before they are trusted**.

Session mapping for NSE: regular session 09:15–15:30 IST; initial balance (IB) =
first 60 min (09:15–10:15). Our Angel 1-min data has no bars 15:15–15:29 (except
15:28) since 2026-08-03 — use NSE bhavcopy daily H/L/C for pivot inputs and treat
15:15 as the practical session end (the book's "exit 15 min before close").

## 1. Formulas (inputs = prior period's H, L, C)

**Floor pivots (standard + expanded)**
- PP = (H+L+C)/3 · R1 = 2PP−L · S1 = 2PP−H · R2 = PP+(H−L) · S2 = PP−(H−L)
- R3 = R1+(H−L) · S3 = S1−(H−L) · R4 = R3+(R2−R1) · S4 = S3−(S1−S2)

**Central Pivot Range (CPR)**
- BC = (H+L)/2 · TC = 2PP−BC. TC may come out below BC → top = max, bottom = min.
- Width % = |TC−BC| / PP × 100. Book thresholds: **< 0.25 narrow → trend day; > 0.75 wide → range day**; 0.5 midline. (Futures-calibrated — recalibrate as NSE percentiles.)
- Developing CPR: same formulas on today's running high/low and current close; at the close it equals tomorrow's CPR.

**Camarilla (standard + expanded)**, RANGE = H−L
- H4/L4 = C ± RANGE·1.1/2 · H3/L3 = C ± RANGE·1.1/4 · H2/L2 = C ± RANGE·1.1/6 · H1/L1 = C ± RANGE·1.1/12
- H5 = (H/L)·C · L5 = C − (H5 − C). No H6/L6 in the book.
- Camarilla "value" band = L3–H3; width = H3−L3 = RANGE·0.55. No numeric narrow/wide threshold — use percentiles.
- Developing Camarilla 3 (DC3): H3/L3 from running high/low + current close.

**Money Zone (Market Profile)**
- TPO profile: 30-min periods. POC = price with most TPOs. Value area = 70% of activity (VAH/VAL). The book gives no VA algorithm → standard: expand from POC adding the larger adjacent row until 70%.
- Volume at Price: spread each bar's volume equally over its H–L. VPOC = max-volume price. HVA = volume peaks, LVA = valleys (price moves fast through LVAs).
- Virgin level = a POC/VAH/VAL/VPOC never touched in its own session's aftermath; "~80% filled within a week"; relevance fades by ~3 months.

**Higher timeframes**: same formulas on prior week (Mon–Fri), month, calendar year. Weekly → swing, monthly → position, yearly → investors.

## 2. Context / forecasting

**Market-day types** (Ch 1): Trend, Double-Distribution Trend, Typical, Expanded Typical, Trading Range, Sideways. Narrow IB → breakout likely; wide IB → IB extremes usually hold. Markets trend ~30% of the time.

**Open vs prior range/value** (Table 4.1): in range & value → range day, fade extremes; in range, out of value → moderate; out of range & value → trend day unless price re-enters the prior range (then failed breakout).

**Two-day relationships** (same 7 for CPR [BC–TC], Camarilla [L3–H3], Money Zone [VAL–VAH]); "today" = band computed from yesterday:
| Relationship | Rule | Bias |
|---|---|---|
| Higher | today bottom > yesterday top | Bullish |
| Overlapping higher | today higher, overlapping | Mod. bullish |
| Lower | today top < yesterday bottom | Bearish |
| Overlapping lower | today lower, overlapping | Mod. bearish |
| Unchanged | ~same (no tolerance given) | Sideways → breakout (3rd unchanged day) |
| Outside | today engulfs yesterday | Sideways |
| Inside | today inside yesterday | Breakout (best with open beyond prior range) |
Prior close vs CPR sets bias; the open confirms or rejects it.

**Value/CPR width**: narrow → trend/double-distribution/expanded typical (hold longer, 1.5 ATR trail, target beyond R2); wide → range day (smaller size, fade IB extremes, target R1/S1). Wide and narrow alternate.

## 3. Entry triggers (Appendix B code, defaults)

- **Wick reversal** (`sysWickRev`): wick ≥ 2.5 × body; close in top 25% of range (long) / bottom 25% (short). Plus full-range bar at extreme with range ≥ 50-bar avg.
- **Extreme reversal** (`sysExtremeRev`): prev bar body ≥ 52.5% of its range, range > 2 × 50-bar avg range, body > avg body; current bar opposite colour → fade prev bar. Best in first 30 min of session / on trend pullbacks.
- **Outside reversal** (`sysOutsideRev`): long L < L[1] and C > H[1]; short H > H[1] and C < L[1]; range ≥ 1.25 × 50-bar avg.
- **Doji reversal** (`sysDojiRev`): prior bar body ≤ 10% of range; short if doji entirely above SMA(10) and a bar within 2 closes below doji low (red); long mirror below SMA(10). Best on ≥15-min bars.

## 4. Setups

| Setup | Conditions | Entry | Stop | Target / management |
|---|---|---|---|---|
| **Breakaway (floor)** | narrow CPR; gap beyond prior range & value and beyond R1/S1 (not far past R2) | first 15-min bar retests the surpassed pivot and holds (wick reversal); next bar open, new 15-min high, or limit at pivot | beyond day's low/high | R3/R4; 1.5 ATR trail (15-min); exit 15 min before close |
| **Camarilla breakaway** | narrow width + Inside Value; gap beyond prior range and H3/L3 (prefer beyond H4/L4), not beyond H5/L5 | test of nearest Camarilla pivot / wick on first 15-min bar | beyond day's extreme | 1.5 ATR trail; partial at H5/L5; session exit |
| **H3/L3 reversal** | open inside prior range; prior trend day favours reversals | wick through H3 closing back below + confirm bar (high < H3, close < prev close); mirror at L3 | just beyond H4/L4 | opposite L3/H3 |
| **H4/L4 breakout** | prior range/sideways day favours breakouts | bar CLOSE beyond H4/L4 (not first touch) | H3/L3 | H5/L5 |
| **Magnet trade** | gap away from prior close; CPR near prior close; moderate gap; no pivots in between | reversal candle on first 15-min bar | beyond gap extreme | any CPR line; fill usually within 30–60 min — abandon if not |
| **Pivot trend pullback** | uptrend (price > S1; daily close < S1 flips it); open between S1 and R1 | pullback to S1/CPR + wick reversal | new low | R1/R2 (mirror for downtrend) |
| **CPR trend** | 2 consecutive days holding above CPR | pullback to CPR, early in session | below BC (trailing centrals stop) | prior S/R or new high |
| **Camarilla trend** | 2-day move = short-term trend; open above L3, below H4 | limit at L3 or wick reversal | below wick low | H3/H4/new high |
| **Hidden layers** | prior day excessively wide; open inside L3–H3 | fade H1/H2 short, L1/L2 long | tight | quick small profit |
| **Virgin-level fade** | short-term trend (≥2 days); test of virgin level in first 30–60 min | exhaustion candle | beyond session extreme | opposite day extreme (late in day: virgin level = target only) |
| **Inside Value breakout** | Inside Value / very narrow VA; open beyond prior range & value | wait 5–15 min, 2 consecutive closes in gap direction | beyond day's low | 2 ATR trail; exit last 15 min |
| **Golden Pivot Zone** | H3/L3/VAH/VAL/POC/VPOC inside [BC,TC]; established trend; open & prior close on trend side of CPR | pullback into zone + any of the 4 candle triggers | beyond zone | next pivot (R1/H3 or S1/L3) |
| **Double/MTF hot zone** | ≥2 uncorrelated pivots (different family or timeframe) within a tolerance (book gives none — pick % or ATR fraction) | candle trigger at zone, with trend | beyond zone | next confluence zone; size up when "stars align" |

Exits used throughout: fixed stop beyond day extreme; 1.5 ATR trail on trend days (2 ATR on Money Zone breakout; ATR length unspecified); session-boundary exit 15 min before close; trailing centrals stop (long exits when low ≤ BC, short when high ≥ TC — code exits on touch, prose on close).

## 5. Appendix A statistics (YM futures, Nov 2008–Jun 2009, 165 days)

| | CP | L1 (R1/S1) | L2 | L3 | L4 |
|---|---|---|---|---|---|
| Touched (all days) | 63.0% | 73.3% | 38.2% | 17.6% | 5.5% |
| Close beyond (all days) | — | 24.8% | 12.1% | 6.7% | 3.0% |
| Touched, CP-untouched days (37%) | 0 | 67.2% | 63.9% | 34.4% | 11.5% |
| Close beyond, CP-untouched days | 0 | 27.9% | 23.0% | 16.4% | 6.6% |

Close within R1–S1: 53.4%. Open > PP → close > PP 53.4%; open < PP → close < PP 48.8%. CP untouched → close beyond L1 73.8% (vs 46.6%) → trend day; hold for R3/S3. Default target R1/S1; scale out at R2/S2.

## 6. Daily Flight Plan (Ch 11) — pre-market checklist

1. Most pertinent levels from Floor/CPR, Camarilla, Money Zone.
2. Two-day relationships → bias (higher/lower/overlapping) or regime (inside/outside/unchanged).
3. Pivot width → trend day vs range day.
4. Virgin Money Zone levels.
5. Hot zones (double, golden, multi-timeframe).
6. Prior day high/low = breakout confirmation levels.
Then confirm with the opening print — if the open doesn't confirm, the bias changes. Only trade instruments that historically respect pivots.

## 7. Open calibration questions for NSE

- Narrow/wide thresholds for CPR (0.25/0.75%), Camarilla, value area → percentiles over our history.
- "Unchanged" relationship tolerance; hot-zone distance tolerance; gap-size limits for Magnet/Breakaway.
- ATR period for trailing stops.
- Re-measure Appendix A touch rates on NIFTY 50 stocks before using them as targets.
