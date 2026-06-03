/**
 * Monthly performance tracking — mock-first data layer.
 *
 * The monthly view is the *post-trade feedback loop* complement to The Cascade
 * (pre-trade stages).  It groups every position by month → asset class → underlying
 * so the trader can see "what did I earn / risk / deploy this month?" at a glance
 * and drill into individual legs only when a roll-up number looks off.
 *
 * Why mock-first
 * ───────────────
 * The UI needs to be designed and iterated on before the aggregation service is
 * even spec'd.  Rather than push premature schema into the Django side, we drive
 * the whole page from realistic mock data that matches the exact shape we'd
 * expect from a future `/api/v1/monthly/` endpoint.
 *
 * Toggling to live data later is a one-line flip: change `USE_MOCK` to false
 * (or wire to `import.meta.env.VITE_MONTHLY_LIVE`) and the hook hits the real
 * endpoint via the shared `api` client.  No component touches mock data
 * directly — the hook is the seam.
 *
 * Paper-mode
 * ──────────
 * Every mock position sets `paper_mode: true` at the payload level.  The UI
 * shows a prominent PAPER badge so the trader is never confused about whether
 * these numbers came from real broker fills.
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

/* ================================================================== */
/* Types — intentionally mirrors the future backend contract            */
/* ================================================================== */

export type AssetClass = "cash" | "fno" | "commodity";

export const ASSET_CLASS_LABEL: Record<AssetClass, string> = {
  cash: "Cash (equity)",
  fno: "F&O (derivatives)",
  commodity: "Commodity",
};

export type PositionStatus = "OPEN" | "CLOSED";

export interface PositionLeg {
  id: string;
  symbol: string;             // "RELIANCE" or "NIFTY 24200 PE"
  side: "BUY" | "SELL";
  quantity: number;
  entry_price: number;
  exit_price: number | null;  // null if still open
  entry_date: string;         // ISO date
  exit_date: string | null;
  target_price: number | null;
  stop_price: number | null;
  pnl: number;                // realized for closed, running for open
  status: PositionStatus;
  lot_size?: number;          // F&O only
  notes?: string;
  close_reason?: string;      // SL_HIT | TARGET_HIT | EOD | …
  source?: string;            // "intraday" | "swing" | ""
}

/** Per-underlying roll-up inside a month + asset-class bucket. */
export interface UnderlyingRoll {
  underlying: string;         // "RELIANCE" / "NIFTY" / "CRUDE"
  asset_class: AssetClass;
  capital_deployed: number;   // ₹ — margin or cash actually put up
  exposure: number;           // ₹ — notional size (qty × entry)
  running_pnl: number;        // ₹ — realized + unrealized
  realized_pnl: number;
  unrealized_pnl: number;
  target_total: number;       // ₹ — sum of (target − entry) × qty across legs
  risk_total: number;         // ₹ — sum of (entry − stop) × qty across legs
  trade_count: number;
  winning_trades: number;
  losing_trades: number;
  avg_rr: number;             // realized avg reward:risk
  days_in_position: number;   // cumulative across legs
  pct_of_month: number;       // 0..1 — share of trading days in a position
  legs: PositionLeg[];
}

export interface MonthGroup {
  month: string;              // "2026-04"
  month_label: string;        // "Apr 2026"
  total_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
  capital_deployed: number;
  exposure: number;
  trade_count: number;
  win_rate: number;           // 0..1
  by_asset_class: Record<AssetClass, UnderlyingRoll[]>;
}

export interface YtdMonthBar {
  month: string;              // "2026-04"
  month_label: string;        // "Apr"
  pnl: number;
}

export interface YtdSummary {
  capital_base: number;
  total_pnl: number;
  best_month: string;
  worst_month: string;
  win_rate: number;
  trade_count: number;
  months: YtdMonthBar[];      // 12 most-recent months, oldest → newest
}

/* ── New sections: capture matrix, signal audit, rejections, lessons ── */

export interface StockCapture {
  symbol: string;
  month_move_pct: number;
  signals_fired: number;
  trades_taken: number;
  trades_skipped: number;
  captured_pnl: number;
  potential_pnl: number;
  capture_rate_pct: number;
  best_signal: { strategy: string; rr: number; outcome: string } | null;
  worst_miss: { strategy: string; rr: number; potential_pnl: number } | null;
}

export interface SignalAudit {
  total_signals: number;
  by_outcome: Record<string, number>;
  by_source: Record<string, number>;
  by_strategy: Record<string, { count: number; win_rate: number; avg_rr: number }>;
  profitable_if_taken: number;
  loss_avoided: number;
}

export interface RejectionReview {
  symbol: string;
  date: string;
  reason: string;
  would_have_profited: boolean;
  hypothetical_pnl: number;
}

/* ── Equity curve + drawdown ── */

export interface EquityCurvePoint {
  date: string;
  pnl: number;
  cumulative: number;
  trades: number;
  drawdown: number;          // negative or zero
}

export interface EquityCurve {
  points: EquityCurvePoint[];
  max_drawdown: number;
  max_drawdown_date: string;
  peak_equity: number;
  final_equity: number;
}

/* ── Analytics breakdowns ── */

export interface HourBucket {
  hour: number;
  label: string;
  trades: number;
  wins: number;
  losses: number;
  pnl: number;
  win_rate: number;
}

export interface DayOfWeekBucket {
  day: number;
  label: string;
  trades: number;
  wins: number;
  pnl: number;
  win_rate: number;
}

export interface SectorBucket {
  sector: string;
  trades: number;
  pnl: number;
  win_rate: number;
  symbols: string[];
}

export interface Analytics {
  by_hour: HourBucket[];
  by_day_of_week: DayOfWeekBucket[];
  by_sector: SectorBucket[];
}

/* ── Benchmark comparison ── */

export interface BenchmarkComparison {
  portfolio_return_pct: number;
  nifty_return_pct: number;
  alpha_pct: number;
  trading_days: number;
  nifty_start: number;
  nifty_end: number;
}

export interface DataFreshness {
  latest_signal_date: string | null;
  latest_trade_date: string | null;
  trades_stale: boolean;      // signals are newer than the last derived trade
}

export interface MonthlyPayload {
  paper_mode: boolean;
  current_month: string;      // "2026-04"
  generated_at: string;       // ISO
  ytd: YtdSummary;
  months: MonthGroup[];       // desc, current first

  /* Feedback report sections */
  capture_matrix: StockCapture[];
  signal_audit: SignalAudit;
  rejections: RejectionReview[];
  lessons: string[];
  equity_curve: EquityCurve;
  analytics: Analytics;
  benchmark: BenchmarkComparison;
  data_freshness?: DataFreshness;
}

/* ================================================================== */
/* Toggle seam                                                          */
/* ================================================================== */

/**
 * Mock vs. live toggle.  Set VITE_MONTHLY_LIVE=1 in .env.local once the
 * backend is running and serving /api/v1/portfolios/monthly/.
 * Defaults to mock so the page always works without a backend.
 */
const USE_MOCK = import.meta.env.VITE_MONTHLY_LIVE !== "1";

const LIVE_ENDPOINT = "portfolios/monthly/";

/* ================================================================== */
/* Mock fixture — April 2026, seeded with realistic paper trades        */
/* ================================================================== */

/** Helper: build a fully-computed leg from the minimum fields. */
function leg(init: {
  id: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  entry_price: number;
  entry_date: string;
  target_price?: number | null;
  stop_price?: number | null;
  exit_price?: number | null;
  exit_date?: string | null;
  lot_size?: number;
  notes?: string;
}): PositionLeg {
  const {
    exit_price = null, exit_date = null,
    target_price = null, stop_price = null,
    lot_size, notes,
    ...rest
  } = init;
  const closed = exit_price != null;
  const signedFill = (exit_price ?? rest.entry_price) - rest.entry_price;
  const pnl =
    (closed ? signedFill : 0) * rest.quantity * (rest.side === "BUY" ? 1 : -1);
  return {
    ...rest,
    exit_price,
    exit_date,
    target_price,
    stop_price,
    lot_size,
    notes,
    pnl,
    status: closed ? "CLOSED" : "OPEN",
  };
}

/** Roll up leg-level facts into an underlying row. */
function rollUp(
  underlying: string,
  asset_class: AssetClass,
  legs: PositionLeg[],
  opts: {
    /** Additional unrealized P&L injected for still-open legs (LTP-derived). */
    unrealized_override?: number;
    days_in_position?: number;
    pct_of_month?: number;
  } = {},
): UnderlyingRoll {
  const realized = legs
    .filter((l) => l.status === "CLOSED")
    .reduce((s, l) => s + l.pnl, 0);
  const unrealized =
    opts.unrealized_override ??
    legs.filter((l) => l.status === "OPEN").reduce((s, l) => s + l.pnl, 0);
  const exposure = legs.reduce(
    (s, l) => s + l.entry_price * l.quantity,
    0,
  );
  // Capital deployed ≈ exposure for cash, 20% margin proxy for F&O/commodity.
  const margin_factor = asset_class === "cash" ? 1 : 0.2;
  const capital_deployed = exposure * margin_factor;

  const target_total = legs.reduce((s, l) => {
    if (l.target_price == null) return s;
    const per_share = (l.target_price - l.entry_price) * (l.side === "BUY" ? 1 : -1);
    return s + Math.max(0, per_share) * l.quantity;
  }, 0);
  const risk_total = legs.reduce((s, l) => {
    if (l.stop_price == null) return s;
    const per_share = (l.entry_price - l.stop_price) * (l.side === "BUY" ? 1 : -1);
    return s + Math.max(0, per_share) * l.quantity;
  }, 0);

  const closed = legs.filter((l) => l.status === "CLOSED");
  const winners = closed.filter((l) => l.pnl > 0).length;
  const losers = closed.filter((l) => l.pnl < 0).length;
  const wins_r = closed
    .filter((l) => l.pnl > 0)
    .reduce((s, l) => s + Math.abs(l.pnl) / l.quantity, 0);
  const losses_r = closed
    .filter((l) => l.pnl < 0)
    .reduce((s, l) => s + Math.abs(l.pnl) / l.quantity, 0);
  const avg_rr = losses_r > 0 ? (wins_r / Math.max(1, winners)) / (losses_r / Math.max(1, losers)) : 0;

  return {
    underlying,
    asset_class,
    capital_deployed,
    exposure,
    running_pnl: realized + unrealized,
    realized_pnl: realized,
    unrealized_pnl: unrealized,
    target_total,
    risk_total,
    trade_count: legs.length,
    winning_trades: winners,
    losing_trades: losers,
    avg_rr: isFinite(avg_rr) ? avg_rr : 0,
    days_in_position: opts.days_in_position ?? 0,
    pct_of_month: opts.pct_of_month ?? 0,
    legs,
  };
}

/* --------------- current month (April 2026) rolls ------------------ */

const CASH_RELIANCE = rollUp(
  "RELIANCE",
  "cash",
  [
    leg({
      id: "rel-1", symbol: "RELIANCE", side: "BUY", quantity: 45,
      entry_price: 2842, entry_date: "2026-04-03",
      target_price: 2950, stop_price: 2788,
      exit_price: 2934, exit_date: "2026-04-08",
      notes: "Sector leader breakout — Stage 4 shortlist pick.",
    }),
    leg({
      id: "rel-2", symbol: "RELIANCE", side: "BUY", quantity: 30,
      entry_price: 2912, entry_date: "2026-04-15",
      target_price: 3020, stop_price: 2858,
      exit_price: null, exit_date: null,
      notes: "Retest of prior breakout.",
    }),
  ],
  { unrealized_override: 1140, days_in_position: 7, pct_of_month: 0.47 },
);

const CASH_TCS = rollUp(
  "TCS",
  "cash",
  [
    leg({
      id: "tcs-1", symbol: "TCS", side: "BUY", quantity: 20,
      entry_price: 3820, entry_date: "2026-04-06",
      target_price: 3960, stop_price: 3751,
      exit_price: 3748, exit_date: "2026-04-09",
      notes: "Stopped out — IT-index weakness confirmed.",
    }),
  ],
  { days_in_position: 3, pct_of_month: 0.20 },
);

const CASH_HDFCBANK = rollUp(
  "HDFCBANK",
  "cash",
  [
    leg({
      id: "hdfc-1", symbol: "HDFCBANK", side: "BUY", quantity: 60,
      entry_price: 1612, entry_date: "2026-04-10",
      target_price: 1688, stop_price: 1578,
      exit_price: 1655, exit_date: "2026-04-14",
    }),
    leg({
      id: "hdfc-2", symbol: "HDFCBANK", side: "BUY", quantity: 40,
      entry_price: 1648, entry_date: "2026-04-16",
      target_price: 1720, stop_price: 1612,
      exit_price: null, exit_date: null,
    }),
  ],
  { unrealized_override: 820, days_in_position: 10, pct_of_month: 0.67 },
);

const FNO_NIFTY = rollUp(
  "NIFTY",
  "fno",
  [
    leg({
      id: "nifty-ce", symbol: "NIFTY 24200 CE", side: "SELL", quantity: 75,
      lot_size: 75,
      entry_price: 394.85, entry_date: "2026-04-07",
      target_price: 295, stop_price: 494,
      exit_price: null, exit_date: null,
      notes: "Short straddle — @OptionsStrategist managed.",
    }),
    leg({
      id: "nifty-pe", symbol: "NIFTY 24200 PE", side: "SELL", quantity: 75,
      lot_size: 75,
      entry_price: 138.35, entry_date: "2026-04-07",
      target_price: 95, stop_price: 215,
      exit_price: null, exit_date: null,
    }),
    leg({
      id: "nifty-hist-1", symbol: "NIFTY 23800 CE", side: "BUY", quantity: 75,
      lot_size: 75,
      entry_price: 156.2, entry_date: "2026-04-01",
      target_price: 245, stop_price: 118,
      exit_price: 224.6, exit_date: "2026-04-03",
      notes: "Event-trade around RBI policy.",
    }),
  ],
  { unrealized_override: 3740, days_in_position: 11, pct_of_month: 0.73 },
);

const FNO_BANKNIFTY = rollUp(
  "BANKNIFTY",
  "fno",
  [
    leg({
      id: "bn-1", symbol: "BANKNIFTY 50800 CE", side: "BUY", quantity: 30,
      lot_size: 15,
      entry_price: 412, entry_date: "2026-04-11",
      target_price: 560, stop_price: 338,
      exit_price: 348, exit_date: "2026-04-12",
      notes: "Quick reversal after PSU-bank downgrade.",
    }),
    leg({
      id: "bn-2", symbol: "BANKNIFTY 51200 CE", side: "BUY", quantity: 15,
      lot_size: 15,
      entry_price: 298, entry_date: "2026-04-17",
      target_price: 420, stop_price: 236,
      exit_price: null, exit_date: null,
    }),
  ],
  { unrealized_override: -1170, days_in_position: 4, pct_of_month: 0.27 },
);

const COMMODITY_CRUDE = rollUp(
  "CRUDEOIL",
  "commodity",
  [
    leg({
      id: "crude-1", symbol: "CRUDEOIL 6100 PE", side: "BUY", quantity: 100,
      lot_size: 100,
      entry_price: 78.5, entry_date: "2026-04-13",
      target_price: 128, stop_price: 54,
      exit_price: 62, exit_date: "2026-04-16",
      notes: "Geopolitical hedge — trailed out for -ve pnl.",
    }),
  ],
  { days_in_position: 3, pct_of_month: 0.20 },
);

/* --------------- month aggregation helper -------------------------- */

function toMonthGroup(
  month: string,
  month_label: string,
  by_asset_class: Record<AssetClass, UnderlyingRoll[]>,
): MonthGroup {
  const flat = [
    ...by_asset_class.cash,
    ...by_asset_class.fno,
    ...by_asset_class.commodity,
  ];
  const total_pnl = flat.reduce((s, r) => s + r.running_pnl, 0);
  const realized_pnl = flat.reduce((s, r) => s + r.realized_pnl, 0);
  const unrealized_pnl = flat.reduce((s, r) => s + r.unrealized_pnl, 0);
  const capital_deployed = flat.reduce((s, r) => s + r.capital_deployed, 0);
  const exposure = flat.reduce((s, r) => s + r.exposure, 0);
  const trade_count = flat.reduce((s, r) => s + r.trade_count, 0);
  const wins = flat.reduce((s, r) => s + r.winning_trades, 0);
  const losses = flat.reduce((s, r) => s + r.losing_trades, 0);
  const win_rate = wins + losses === 0 ? 0 : wins / (wins + losses);
  return {
    month, month_label,
    total_pnl, realized_pnl, unrealized_pnl,
    capital_deployed, exposure, trade_count, win_rate,
    by_asset_class,
  };
}

const APR_2026 = toMonthGroup("2026-04", "Apr 2026", {
  cash: [CASH_RELIANCE, CASH_TCS, CASH_HDFCBANK],
  fno: [FNO_NIFTY, FNO_BANKNIFTY],
  commodity: [COMMODITY_CRUDE],
});

/* --------------- prior months — summary-only --------------------- */
/* One representative closed roll-up per asset-class so drill-in still
   works for past months without inventing 100 fake legs. */

function priorMonth(
  month: string,
  label: string,
  cash_pnl: number,
  fno_pnl: number,
  commodity_pnl: number,
  trades: number,
  wins: number,
  losses: number,
): MonthGroup {
  const capital_total = 480_000;
  const exposure_total = 1_620_000;
  return {
    month,
    month_label: label,
    total_pnl: cash_pnl + fno_pnl + commodity_pnl,
    realized_pnl: cash_pnl + fno_pnl + commodity_pnl,
    unrealized_pnl: 0,
    capital_deployed: capital_total,
    exposure: exposure_total,
    trade_count: trades,
    win_rate: wins + losses === 0 ? 0 : wins / (wins + losses),
    by_asset_class: {
      cash: [{
        underlying: "Aggregate", asset_class: "cash",
        capital_deployed: capital_total * 0.55, exposure: exposure_total * 0.4,
        running_pnl: cash_pnl, realized_pnl: cash_pnl, unrealized_pnl: 0,
        target_total: 0, risk_total: 0,
        trade_count: Math.round(trades * 0.55),
        winning_trades: Math.round(wins * 0.55),
        losing_trades: Math.round(losses * 0.55),
        avg_rr: 1.8, days_in_position: 0, pct_of_month: 0, legs: [],
      }],
      fno: [{
        underlying: "Aggregate", asset_class: "fno",
        capital_deployed: capital_total * 0.4, exposure: exposure_total * 0.55,
        running_pnl: fno_pnl, realized_pnl: fno_pnl, unrealized_pnl: 0,
        target_total: 0, risk_total: 0,
        trade_count: Math.round(trades * 0.35),
        winning_trades: Math.round(wins * 0.35),
        losing_trades: Math.round(losses * 0.35),
        avg_rr: 1.6, days_in_position: 0, pct_of_month: 0, legs: [],
      }],
      commodity: [{
        underlying: "Aggregate", asset_class: "commodity",
        capital_deployed: capital_total * 0.05, exposure: exposure_total * 0.05,
        running_pnl: commodity_pnl, realized_pnl: commodity_pnl, unrealized_pnl: 0,
        target_total: 0, risk_total: 0,
        trade_count: Math.max(0, trades - Math.round(trades * 0.55) - Math.round(trades * 0.35)),
        winning_trades: 0, losing_trades: 0,
        avg_rr: 1.2, days_in_position: 0, pct_of_month: 0, legs: [],
      }],
    },
  };
}

const PRIOR_MONTHS: MonthGroup[] = [
  priorMonth("2026-03", "Mar 2026",  8_420,  6_110,    -820, 18, 11, 7),
  priorMonth("2026-02", "Feb 2026",  4_800, -2_310,   1_120, 14,  8, 6),
  priorMonth("2026-01", "Jan 2026", 11_320,  9_540,     640, 22, 14, 8),
  priorMonth("2025-12", "Dec 2025",  3_210,  2_480,    -310, 11,  6, 5),
  priorMonth("2025-11", "Nov 2025", -2_140, -4_620,  -1_020, 16,  5, 11),
  priorMonth("2025-10", "Oct 2025",  9_710,  7_820,     940, 19, 13, 6),
  priorMonth("2025-09", "Sep 2025",  6_240,  3_110,     210, 17, 10, 7),
  priorMonth("2025-08", "Aug 2025",  2_420,  1_140,    -420, 12,  6, 6),
  priorMonth("2025-07", "Jul 2025",  7_910,  5_220,     330, 20, 12, 8),
  priorMonth("2025-06", "Jun 2025", -1_210, -1_840,     120, 13,  4, 9),
  priorMonth("2025-05", "May 2025",  5_540,  2_410,     710, 16,  9, 7),
];

/* --------------- assemble the payload ----------------------------- */

function buildMockPayload(): MonthlyPayload {
  // Mirror the live backend: only months in the current financial year
  // (Apr→Mar) are shown. The mock's "today" is Apr 2026, so the FY starts
  // Apr 2026 and prior-FY months (Mar 2026 and earlier) drop out.
  const FY_START = "2026-04";
  const months = [APR_2026, ...PRIOR_MONTHS].filter((m) => m.month >= FY_START);

  const ytd: YtdSummary = {
    capital_base: 500_000,
    total_pnl: months.reduce((s, m) => s + m.total_pnl, 0),
    best_month: months.reduce((b, m) => (m.total_pnl > b.total_pnl ? m : b)).month,
    worst_month: months.reduce((w, m) => (m.total_pnl < w.total_pnl ? m : w)).month,
    trade_count: months.reduce((s, m) => s + m.trade_count, 0),
    win_rate:
      months.reduce((s, m) => s + m.win_rate * m.trade_count, 0) /
      Math.max(1, months.reduce((s, m) => s + m.trade_count, 0)),
    // 12 most recent, oldest → newest (left-to-right bar chart)
    months: [...months]
      .slice(0, 12)
      .reverse()
      .map((m) => ({
        month: m.month,
        month_label: m.month_label.split(" ")[0],
        pnl: m.total_pnl,
      })),
  };

  return {
    paper_mode: true,
    current_month: APR_2026.month,
    generated_at: new Date().toISOString(),
    ytd,
    months,

    /* Feedback report sections */
    capture_matrix: [
      {
        symbol: "RELIANCE", month_move_pct: 3.2, signals_fired: 5,
        trades_taken: 2, trades_skipped: 2, captured_pnl: 4_140,
        potential_pnl: 8_420, capture_rate_pct: 49.2,
        best_signal: { strategy: "ORB_LONG", rr: 2.4, outcome: "TRADED" },
        worst_miss: { strategy: "VWAP_RECLAIM", rr: 1.8, potential_pnl: 2_300 },
      },
      {
        symbol: "HDFCBANK", month_move_pct: 2.7, signals_fired: 4,
        trades_taken: 2, trades_skipped: 1, captured_pnl: 3_400,
        potential_pnl: 5_100, capture_rate_pct: 66.7,
        best_signal: { strategy: "PDH_BREAK", rr: 2.1, outcome: "TRADED" },
        worst_miss: null,
      },
      {
        symbol: "TCS", month_move_pct: -2.1, signals_fired: 3,
        trades_taken: 1, trades_skipped: 2, captured_pnl: -1_440,
        potential_pnl: 1_800, capture_rate_pct: 0,
        best_signal: { strategy: "GAP_AND_GO", rr: 1.5, outcome: "EXPIRED" },
        worst_miss: { strategy: "GAP_AND_GO", rr: 1.5, potential_pnl: 1_200 },
      },
      {
        symbol: "NIFTY", month_move_pct: 1.4, signals_fired: 8,
        trades_taken: 3, trades_skipped: 3, captured_pnl: 8_880,
        potential_pnl: 16_200, capture_rate_pct: 54.8,
        best_signal: { strategy: "WEDGE_POP", rr: 3.1, outcome: "TRADED" },
        worst_miss: { strategy: "ORB_LONG", rr: 2.6, potential_pnl: 4_100 },
      },
    ],
    signal_audit: {
      total_signals: 28,
      by_outcome: { TRADED: 12, REJECTED: 5, SKIPPED: 3, EXPIRED: 8 },
      by_source: { SCREENER: 18, OK_SCANNER: 7, PREMARKET: 3 },
      by_strategy: {
        ORB_LONG: { count: 8, win_rate: 0.625, avg_rr: 2.1 },
        VWAP_RECLAIM: { count: 6, win_rate: 0.5, avg_rr: 1.7 },
        PDH_BREAK: { count: 5, win_rate: 0.6, avg_rr: 1.9 },
        WEDGE_POP: { count: 4, win_rate: 0.75, avg_rr: 2.8 },
        GAP_AND_GO: { count: 3, win_rate: 0.33, avg_rr: 1.4 },
        BASIN_BREAK: { count: 2, win_rate: 0.5, avg_rr: 1.6 },
      },
      profitable_if_taken: 4,
      loss_avoided: 3,
    },
    rejections: [
      { symbol: "BAJFINANCE", date: "2026-04-08", reason: "Position size exceeds 10% of capital", would_have_profited: true, hypothetical_pnl: 3_200 },
      { symbol: "INFY", date: "2026-04-11", reason: "Daily loss limit 2.8% near threshold", would_have_profited: false, hypothetical_pnl: 0 },
      { symbol: "TATAMOTORS", date: "2026-04-14", reason: "R:R ratio 0.9 below minimum 1.5", would_have_profited: true, hypothetical_pnl: 1_800 },
      { symbol: "ADANIENT", date: "2026-04-17", reason: "Max open positions (5) reached", would_have_profited: false, hypothetical_pnl: 0 },
      { symbol: "SBIN", date: "2026-04-22", reason: "Confidence 0.42 below threshold 0.6", would_have_profited: true, hypothetical_pnl: 2_100 },
    ],
    lessons: [
      "Average capture rate is 43% — over half the signal potential is left on the table. Consider lowering confidence thresholds for high-R:R setups.",
      "4 of 8 skipped/rejected signals (50%) would have been profitable. Review @RiskGuard position size gate — it blocked 3 winners.",
      "Best strategy: WEDGE_POP — 75% win rate across 4 signals with 2.8x avg R:R. Lean into this setup.",
      "Weakest strategy: GAP_AND_GO — 33% win rate. Consider disabling or requiring higher confidence.",
      "Only 43% of signals converted to trades. 8 expired without action — review if alerting latency is causing missed windows.",
    ],

    equity_curve: {
      points: [
        { date: "2026-04-01", pnl: 1_840, cumulative: 1_840, trades: 3, drawdown: 0 },
        { date: "2026-04-02", pnl: 2_210, cumulative: 4_050, trades: 4, drawdown: 0 },
        { date: "2026-04-03", pnl: -620, cumulative: 3_430, trades: 2, drawdown: -620 },
        { date: "2026-04-06", pnl: 940, cumulative: 4_370, trades: 3, drawdown: 0 },
        { date: "2026-04-07", pnl: 1_580, cumulative: 5_950, trades: 2, drawdown: 0 },
        { date: "2026-04-08", pnl: -1_420, cumulative: 4_530, trades: 4, drawdown: -1_420 },
        { date: "2026-04-09", pnl: 310, cumulative: 4_840, trades: 2, drawdown: -1_110 },
        { date: "2026-04-10", pnl: -780, cumulative: 4_060, trades: 3, drawdown: -1_890 },
        { date: "2026-04-13", pnl: 1_120, cumulative: 5_180, trades: 2, drawdown: -770 },
        { date: "2026-04-14", pnl: 650, cumulative: 5_830, trades: 3, drawdown: -120 },
        { date: "2026-04-15", pnl: 420, cumulative: 6_250, trades: 2, drawdown: 0 },
        { date: "2026-04-16", pnl: -1_650, cumulative: 4_600, trades: 3, drawdown: -1_650 },
        { date: "2026-04-17", pnl: 880, cumulative: 5_480, trades: 2, drawdown: -770 },
      ],
      max_drawdown: -1_890,
      max_drawdown_date: "2026-04-10",
      peak_equity: 6_250,
      final_equity: 5_480,
    },

    analytics: {
      by_hour: [
        { hour: 9,  label: "09:00", trades: 8,  wins: 5, losses: 3, pnl: 3_210,  win_rate: 0.625 },
        { hour: 10, label: "10:00", trades: 6,  wins: 4, losses: 2, pnl: 1_840,  win_rate: 0.667 },
        { hour: 11, label: "11:00", trades: 4,  wins: 2, losses: 2, pnl: -320,   win_rate: 0.5 },
        { hour: 12, label: "12:00", trades: 3,  wins: 1, losses: 2, pnl: -890,   win_rate: 0.333 },
        { hour: 13, label: "13:00", trades: 4,  wins: 2, losses: 2, pnl: 640,    win_rate: 0.5 },
        { hour: 14, label: "14:00", trades: 5,  wins: 2, losses: 3, pnl: -1_210, win_rate: 0.4 },
        { hour: 15, label: "15:00", trades: 2,  wins: 1, losses: 1, pnl: 210,    win_rate: 0.5 },
      ],
      by_day_of_week: [
        { day: 0, label: "Mon", trades: 6, wins: 4, pnl: 2_180, win_rate: 0.667 },
        { day: 1, label: "Tue", trades: 7, wins: 3, pnl: -410,  win_rate: 0.429 },
        { day: 2, label: "Wed", trades: 8, wins: 5, pnl: 1_920, win_rate: 0.625 },
        { day: 3, label: "Thu", trades: 6, wins: 3, pnl: 1_340, win_rate: 0.5 },
        { day: 4, label: "Fri", trades: 5, wins: 2, pnl: 450,   win_rate: 0.4 },
      ],
      by_sector: [
        { sector: "Banking", trades: 10, pnl: 2_840, win_rate: 0.6, symbols: ["HDFCBANK", "KOTAKBANK", "AXISBANK"] },
        { sector: "IT",      trades: 7,  pnl: 1_420, win_rate: 0.57, symbols: ["INFY", "WIPRO", "TCS"] },
        { sector: "Energy",  trades: 5,  pnl: 890,   win_rate: 0.6, symbols: ["RELIANCE"] },
        { sector: "Auto",    trades: 3,  pnl: 640,   win_rate: 0.667, symbols: ["MARUTI"] },
        { sector: "Metals",  trades: 4,  pnl: -510,  win_rate: 0.25, symbols: ["HINDALCO", "TATASTEEL"] },
        { sector: "FMCG",    trades: 3,  pnl: 200,   win_rate: 0.667, symbols: ["ITC", "HINDUNILVR"] },
      ],
    },

    benchmark: {
      portfolio_return_pct: 1.1,
      nifty_return_pct: 0.6,
      alpha_pct: 0.5,
      trading_days: 21,
      nifty_start: 24_120,
      nifty_end: 24_265,
    },
  };
}

/* ================================================================== */
/* React Query hook — same surface mock or live                         */
/* ================================================================== */

/**
 * Runtime data-source state.  Persisted in localStorage so the choice
 * survives page reloads.  The UI toggle flips this and invalidates the
 * React Query cache so the hook refetches from the new source.
 */
const LS_KEY = "monthly_data_source";

function readSource(): "mock" | "live" {
  if (typeof window === "undefined") return USE_MOCK ? "mock" : "live";
  const stored = localStorage.getItem(LS_KEY);
  if (stored === "mock" || stored === "live") return stored;
  return USE_MOCK ? "mock" : "live";
}

let _source: "mock" | "live" = readSource();

export function getMonthlySource(): "mock" | "live" { return _source; }

export function setMonthlySource(s: "mock" | "live") {
  _source = s;
  localStorage.setItem(LS_KEY, s);
}

/**
 * @param month  Focused month "YYYY-MM" (from the YTD bar-chart / URL). When
 *   omitted the backend picks the current month (falling back to the most
 *   recent month with activity). Passing it re-scopes the feedback sections
 *   (capture matrix / signal audit / rejections / equity / analytics) to that
 *   month — the YTD strip + month list stay full regardless.
 */
export function useMonthlyView(month?: string | null) {
  return useQuery<MonthlyPayload>({
    queryKey: ["monthly-view", _source, month ?? "current"],
    queryFn: async () => {
      if (_source === "mock") {
        await new Promise((r) => setTimeout(r, 120));
        return buildMockPayload();
      }
      const url = month
        ? `${LIVE_ENDPOINT}?month=${encodeURIComponent(month)}`
        : LIVE_ENDPOINT;
      const { data } = await api.get<MonthlyPayload>(url);
      return data;
    },
    staleTime: 60_000,
    retry: _source === "live" ? 1 : false,
  });
}

/** Exported for tests & the prominent "paper mode" badge. */
export const MONTHLY_USE_MOCK = USE_MOCK;
