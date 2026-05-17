/**
 * Catalog of indicators the strategies use.
 *
 * Drives:
 *   - Info popovers on the Indicators tab (`/stock/:symbol`)
 *   - Tooltips on indicator cells in run-detail cards
 *
 * Add an entry whenever a strategy emits a new field into
 * `state["indicators"]` so the operator can self-serve "what is this?".
 */

export type IndicatorKind = "trend" | "momentum" | "volatility" | "structure" | "options-greek" | "regime";

export interface IndicatorDef {
  key: string;                 // matches the key in state["indicators"]
  label: string;               // human title
  kind: IndicatorKind;
  unit?: "pts" | "%" | "INR" | "ratio";
  oneLiner: string;            // what it measures
  formula?: string;            // plain-language formula
  usage: string;               // how the strategy reads it
  goodRange?: string;          // "30-70 = neutral RSI"
  strategies: string[];        // which strategies surface it
}

export const INDICATORS: IndicatorDef[] = [
  /* ── Directional / trend ── */
  {
    key: "sma_5",
    label: "SMA-5",
    kind: "trend",
    unit: "pts",
    oneLiner: "5-bar simple moving average — fastest read on intraday trend.",
    formula: "avg(close, last 5 bars)",
    usage: "Price above SMA-5 = short-term momentum up; reclaim of SMA-5 after a dip is a continuation trigger.",
    strategies: ["directional"],
  },
  {
    key: "sma_20",
    label: "SMA-20",
    kind: "trend",
    unit: "pts",
    oneLiner: "20-bar moving average — the intraday ‘floor of intent’.",
    formula: "avg(close, last 20 bars)",
    usage: "Plans built on the long side prefer entries that hold above SMA-20; shorts prefer rejections from it.",
    strategies: ["directional"],
  },
  {
    key: "sma_50",
    label: "SMA-50",
    kind: "trend",
    unit: "pts",
    oneLiner: "50-bar moving average — multi-day trend reference.",
    formula: "avg(close, last 50 bars)",
    usage: "Used as the broader trend filter — only take trades aligned with this slope unless reasoning says otherwise.",
    strategies: ["directional"],
  },
  {
    key: "vwap",
    label: "VWAP",
    kind: "structure",
    unit: "pts",
    oneLiner: "Volume-weighted average price — the day’s ‘fair value’.",
    formula: "Σ(typical_price × volume) / Σ(volume)",
    usage: "Distance from VWAP is the institutional read on whether buyers or sellers are in control; VWAP reclaim/fade is a common trigger.",
    strategies: ["directional", "pyramid"],
  },
  {
    key: "rsi_14",
    label: "RSI-14",
    kind: "momentum",
    unit: "ratio",
    oneLiner: "Relative Strength Index — 0–100 momentum oscillator.",
    formula: "100 − 100 / (1 + avg_gain/avg_loss over 14 bars)",
    usage: "70+ = stretched up, possible exhaustion. 30− = stretched down, possible bounce. 40-60 = no extreme.",
    goodRange: "30–70 = neutral",
    strategies: ["directional", "pyramid"],
  },
  {
    key: "atr_14",
    label: "ATR-14",
    kind: "volatility",
    unit: "pts",
    oneLiner: "Average True Range — typical bar move over 14 bars.",
    formula: "avg(max(H−L, |H−prevC|, |L−prevC|), 14)",
    usage: "Stop distance ≈ 1.0–1.5 × ATR keeps you outside normal noise. Bigger ATR = wider stops needed.",
    strategies: ["directional"],
  },
  {
    key: "prev_close",
    label: "Prev close",
    kind: "structure",
    unit: "pts",
    oneLiner: "Yesterday’s close — universal reference for gap and overnight bias.",
    usage: "Gap up/down to it changes the whole intraday narrative; opening drives that fail to reclaim it often retest the level.",
    strategies: ["directional"],
  },
  {
    key: "day_high",
    label: "Day high",
    kind: "structure",
    unit: "pts",
    oneLiner: "Session-to-date high.",
    usage: "Break-and-hold of day high is a momentum continuation; failure at it is a fade signal.",
    strategies: ["directional"],
  },
  {
    key: "day_low",
    label: "Day low",
    kind: "structure",
    unit: "pts",
    oneLiner: "Session-to-date low.",
    usage: "Same logic as day high but inverted — break and hold = down continuation, false break = bounce.",
    strategies: ["directional"],
  },

  /* ── Options-Greeks / regime (short straddle) ── */
  {
    key: "ce_delta",
    label: "CE Δ",
    kind: "options-greek",
    unit: "ratio",
    oneLiner: "Delta of the short call leg.",
    formula: "∂C/∂S (computed via Black-Scholes approximation)",
    usage: "More-negative net delta (because we're short the call) means upside in spot hurts more.",
    strategies: ["short_straddle"],
  },
  {
    key: "pe_delta",
    label: "PE Δ",
    kind: "options-greek",
    unit: "ratio",
    oneLiner: "Delta of the short put leg.",
    formula: "∂P/∂S",
    usage: "Positive PE delta on a short put → downside in spot hurts the position.",
    strategies: ["short_straddle"],
  },
  {
    key: "net_delta",
    label: "Net Δ",
    kind: "options-greek",
    unit: "ratio",
    oneLiner: "Total delta exposure of the straddle.",
    formula: "(−ce_delta) + (−pe_delta)  // both legs short",
    usage: "0 = neutral. |Net Δ| > 0.5 = directionally exposed → consider hedging or rolling the tested leg.",
    goodRange: "−0.25 to +0.25 = neutral",
    strategies: ["short_straddle"],
  },
  {
    key: "delta_bias",
    label: "Delta bias",
    kind: "options-greek",
    oneLiner: "Plain-English label for net delta direction.",
    usage: "LONG = spot fall hurts; SHORT = spot rise hurts; NEUTRAL = roughly balanced.",
    strategies: ["short_straddle"],
  },
  {
    key: "vix_phase",
    label: "VIX phase",
    kind: "regime",
    oneLiner: "Categorised India VIX level — CALM / ELEVATED / SPIKE.",
    formula: "<15 calm · 15-22 elevated · >22 spike",
    usage: "SPIKE = wider stops, larger drawdowns expected on shorts; CALM = premium-collection regime favourable.",
    strategies: ["short_straddle"],
  },
  {
    key: "vix_current",
    label: "VIX",
    kind: "regime",
    oneLiner: "Current India VIX value.",
    usage: "Above prior ~22 = stressful regime; below 14 = complacent regime.",
    strategies: ["short_straddle"],
  },
  {
    key: "market_phase",
    label: "Market phase",
    kind: "regime",
    oneLiner: "Categorised intraday regime — TREND_UP / TREND_DOWN / CHOP / EXPANSION / SQUEEZE.",
    usage: "CHOP favours premium decay; TREND_* and EXPANSION threaten short option positions.",
    strategies: ["short_straddle"],
  },
  {
    key: "premium_decayed_pct",
    label: "Premium decayed",
    kind: "options-greek",
    unit: "%",
    oneLiner: "Fraction of total premium already captured by theta.",
    formula: "(combined_sold − combined_current) / combined_sold × 100",
    usage: "Many desks book at 50%+. 0% = no decay yet, 100% = full premium captured.",
    strategies: ["short_straddle"],
  },
  {
    key: "days_to_expiry",
    label: "DTE",
    kind: "regime",
    unit: "pts",
    oneLiner: "Days until the option expires.",
    usage: "Gamma risk explodes < 2 DTE on ATM straddles. Most desks close before final day.",
    strategies: ["short_straddle"],
  },
  {
    key: "is_underwater",
    label: "Underwater?",
    kind: "regime",
    oneLiner: "Combined premium-now > premium-sold.",
    usage: "Hard rule: if true, the legacy lifecycle.decide forces CLOSE_BOTH at 1.3× severity.",
    strategies: ["short_straddle"],
  },
  {
    key: "nearest_itm_leg",
    label: "Nearest ITM leg",
    kind: "options-greek",
    oneLiner: "Which leg (CE / PE / BOTH_OTM) is closest to going in-the-money.",
    usage: "Tells the operator which leg to hedge / roll first.",
    strategies: ["short_straddle"],
  },
];

export function getIndicator(key: string): IndicatorDef | undefined {
  return INDICATORS.find((i) => i.key === key);
}

export function indicatorsByStrategy(name: string): IndicatorDef[] {
  return INDICATORS.filter((i) => i.strategies.includes(name));
}
