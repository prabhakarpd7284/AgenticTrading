/**
 * Market Pulse — live briefing for the "What's Happening Today" homepage.
 *
 * This is Stage 1+2 of The Cascade trader framework:
 *   Stage 1 REGIME   → is it a day to trade? (vol, trend, global tone)
 *   Stage 2 CONTEXT  → what's driving flows? (FX, commodities, rates)
 *
 * Backend: backend/apps/market_data/services/pulse_service.py
 * Endpoint: GET /api/v1/market-data/pulse/  (cached 30s server-side)
 *
 * Every numeric field is nullable — data-source hiccups render as "—" rather
 * than 500-ing the whole screen.
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

/* ------------------------------------------------------------------ */
/* Types — mirror backend dataclasses (pulse_service.py)               */
/* ------------------------------------------------------------------ */

export type QuoteGroup =
  | "indices_in"
  | "indices_global"
  | "vol"
  | "fx"
  | "commodities"
  | "rates";

export interface Quote {
  symbol: string;
  label: string;
  last: number | null;
  change: number | null;
  change_pct: number | null;
  prev_close: number | null;
  day_high: number | null;
  day_low: number | null;
  as_of: string | null;
  stale: boolean;
  source: string;
}

export interface Sector {
  key: string;
  label: string;
  change_pct: number | null;
  last: number | null;
  rank: number;
}

export type VolTier =
  | "complacent" | "low" | "normal" | "elevated" | "high" | "extreme" | "unknown";

export type TrendTier = "up" | "down" | "range" | "unknown";

export type GlobalTone = "risk_on" | "risk_off" | "neutral" | "unknown";

export type AgentVerdict = "favored" | "neutral" | "avoid";

export interface Regime {
  vol: VolTier;
  trend: TrendTier;
  global_tone: GlobalTone;
  vix: number | null;
  nifty_gap_pct: number | null;
  sp500_change_pct: number | null;
  tradeable: boolean;
  summary: string;
}

export interface Guidance {
  directional: AgentVerdict;
  straddle: AgentVerdict;
  reasons: string[];
}

export type SessionPhase = "pre-open" | "open" | "post-close" | "weekend";

export interface PulsePayload {
  as_of: string;
  session_phase: SessionPhase;
  is_market_open: boolean;
  regime: Regime;
  quotes: Partial<Record<QuoteGroup, Quote[]>>;
  sectors: Sector[];
  guidance: Guidance;
  errors: string[];
}

/* ------------------------------------------------------------------ */
/* Hook — lightweight wrapper that plugs into TanStack Query            */
/* ------------------------------------------------------------------ */

/** Default refetch cadence during market hours. Backend caches for 30s, so
 *  15s keeps the UI fresh without ever actually thrashing yfinance. */
const REFRESH_MS_OPEN = 15_000;
/** Slower refetch off-hours — nothing is moving, save the bandwidth. */
const REFRESH_MS_CLOSED = 60_000;

export function useMarketPulse(options?: { force?: boolean }) {
  return useQuery<PulsePayload>({
    queryKey: ["market-pulse", options?.force ?? false],
    queryFn: async () => {
      const r = await api.get<PulsePayload>("/market-data/pulse/", {
        params: options?.force ? { force: 1 } : undefined,
      });
      return r.data;
    },
    refetchInterval: (query) => {
      const d = query.state.data as PulsePayload | undefined;
      return d?.is_market_open ? REFRESH_MS_OPEN : REFRESH_MS_CLOSED;
    },
    refetchIntervalInBackground: false,
    staleTime: 10_000,
  });
}

/* ------------------------------------------------------------------ */
/* Presentational helpers — shared by cards / banner / heatmap          */
/* ------------------------------------------------------------------ */

export const VOL_LABEL: Record<VolTier, string> = {
  complacent: "Complacent",
  low: "Low",
  normal: "Normal",
  elevated: "Elevated",
  high: "High",
  extreme: "Extreme",
  unknown: "Unknown",
};

export const TREND_LABEL: Record<TrendTier, string> = {
  up: "Trending up",
  down: "Trending down",
  range: "Range-bound",
  unknown: "—",
};

export const TONE_LABEL: Record<GlobalTone, string> = {
  risk_on: "Risk-on",
  risk_off: "Risk-off",
  neutral: "Neutral",
  unknown: "—",
};

/** Map regime.vol to a Badge tone. Matches the @RiskGuard / straddle gate:
 *  VIX > 20  → straddle avoid
 *  VIX > 35  → all options avoid */
export function volTone(v: VolTier): "neutral" | "success" | "warning" | "danger" {
  if (v === "extreme" || v === "high") return "danger";
  if (v === "elevated") return "warning";
  if (v === "low" || v === "normal") return "success";
  return "neutral";
}

export function trendTone(t: TrendTier): "success" | "danger" | "neutral" {
  if (t === "up") return "success";
  if (t === "down") return "danger";
  return "neutral";
}

export function toneColor(g: GlobalTone): "success" | "danger" | "neutral" {
  if (g === "risk_on") return "success";
  if (g === "risk_off") return "danger";
  return "neutral";
}

export function verdictTone(
  v: AgentVerdict,
): "success" | "warning" | "danger" | "neutral" {
  if (v === "favored") return "success";
  if (v === "neutral") return "warning";
  if (v === "avoid") return "danger";
  return "neutral";
}

export function phaseLabel(p: SessionPhase): string {
  return {
    "pre-open": "Pre-open",
    open: "Market open",
    "post-close": "Market closed",
    weekend: "Weekend",
  }[p];
}

/* ================================================================== */
/* Stage 3 — Sector Rotation                                          */
/*                                                                    */
/* Drill-in from the pulse heatmap: per-sector top-3 leaders + bottom- */
/* 3 laggards + breadth.  Kept co-located with Stage 1/2 types because  */
/* they share helpers (tone colours, percent formatting).               */
/* ================================================================== */

export interface StockMove {
  symbol: string;
  last: number | null;
  change_pct: number | null;
  change: number | null;
  prev_close: number | null;
}

export interface SectorRotation {
  key: string;
  label: string;
  rank: number;
  change_pct: number | null;
  last: number | null;
  leaders: StockMove[];
  laggards: StockMove[];
  breadth: { up: number; down: number; flat: number };
}

export interface RotationPayload {
  as_of: string;
  sectors: SectorRotation[];
  errors: string[];
}

/** Backend caches for 60s.  Poll at 30s while open, 2 min off-hours. */
const ROTATION_MS_OPEN = 30_000;
const ROTATION_MS_CLOSED = 120_000;

export function useSectorRotation(options?: { force?: boolean; isOpen?: boolean }) {
  return useQuery<RotationPayload>({
    queryKey: ["sector-rotation", options?.force ?? false],
    queryFn: async () => {
      const r = await api.get<RotationPayload>("/market-data/rotation/", {
        params: options?.force ? { force: 1 } : undefined,
      });
      return r.data;
    },
    refetchInterval: options?.isOpen ? ROTATION_MS_OPEN : ROTATION_MS_CLOSED,
    refetchIntervalInBackground: false,
    staleTime: 20_000,
  });
}

/* ================================================================== */
/* Stage 4 — Shortlist                                                 */
/*                                                                    */
/* Turn Stage 3 sector rotation into a ranked list of 10-15 tradeable  */
/* names, each carrying a 0-100 confluence score and human-readable    */
/* reasons pills.  Hard-rejected candidates come back separately so    */
/* the operator can audit the gates.                                    */
/* ================================================================== */

export interface StockCandidate {
  symbol: string;
  sector_key: string;
  sector_label: string;
  sector_rank: number;
  change_pct: number | null;
  last: number | null;
  atr_pct: number | null;
  turnover_cr: number | null;
  rel_volume: number | null;
  range_52w_pos: number | null;
  score: number;
  reasons: string[];
  is_leader: boolean;
}

export interface RejectedCandidate extends StockCandidate {
  reject_reasons: string[];
}

export interface ShortlistPayload {
  as_of: string;
  hot_sectors: string[];
  candidates: StockCandidate[];
  filtered_out: RejectedCandidate[];
  errors: string[];
}

/** Backend caches for 300s.  No need to hit harder than 60s while open. */
const SHORTLIST_MS_OPEN = 60_000;
const SHORTLIST_MS_CLOSED = 300_000;

export function useShortlist(options?: { force?: boolean; isOpen?: boolean }) {
  return useQuery<ShortlistPayload>({
    queryKey: ["shortlist", options?.force ?? false],
    queryFn: async () => {
      const r = await api.get<ShortlistPayload>("/market-data/shortlist/", {
        params: options?.force ? { force: 1 } : undefined,
      });
      return r.data;
    },
    refetchInterval: options?.isOpen ? SHORTLIST_MS_OPEN : SHORTLIST_MS_CLOSED,
    refetchIntervalInBackground: false,
    staleTime: 30_000,
  });
}

/** Map 0-100 score to a semantic Badge tone so cards self-describe. */
export function scoreTone(
  score: number,
): "success" | "warning" | "danger" | "neutral" {
  if (score >= 70) return "success";
  if (score >= 45) return "warning";
  if (score > 0) return "neutral";
  return "danger";
}

/* ================================================================== */
/* Stage 5 — Setup Preview                                              */
/*                                                                    */
/*  Operator clicks a shortlist row → we synthesise a deterministic    */
/*  BUY/SELL plan from the last candles and run the full 10-criterion  */
/*  @RiskGuard gate.  The UI renders the plan side-by-side with a row  */
/*  per criterion so the "would this trade clear?" question has a      */
/*  visible, auditable answer — not a hidden veto.                      */
/* ================================================================== */

export type CriterionSeverity = "info" | "warning" | "danger";

export interface SetupCriterion {
  key: string;
  label: string;
  passed: boolean;
  detail: string;
  severity: CriterionSeverity;
}

export interface SetupPlan {
  symbol: string;
  side: "BUY" | "SELL";
  entry_price: number;
  stop_loss: number;
  target: number;
  quantity: number;
  confidence: number;
  risk_per_share: number;
  risk_amount: number;
  reward_amount: number;
  risk_reward_ratio: number;
  notes: string[];
}

export interface SetupMarket {
  last: number | null;
  atr: number | null;
  atr_pct: number | null;
  change_pct: number | null;
  candle_count: number;
}

export interface SetupRegime {
  tradeable: boolean;
  vol: string | null;
  trend: string | null;
  summary: string;
  cached: boolean;
}

export interface SetupRisk {
  approved: boolean;
  reason: string;
  criteria: SetupCriterion[];
}

export interface SetupPayload {
  as_of: string;
  symbol: string;
  side: "BUY" | "SELL";
  market: SetupMarket;
  plan: SetupPlan | null;
  regime: SetupRegime;
  risk: SetupRisk;
  errors: string[];
}

/** Backend is uncached (plan is capital-sensitive per tenant) — poll every
 *  30s while the market is open so the breakdown stays honest as the LTP
 *  drifts.  Off-hours, nothing is moving. */
const SETUP_MS_OPEN = 30_000;
const SETUP_MS_CLOSED = 600_000;

export function useSetupPreview(
  symbol: string | undefined,
  options?: {
    side?: "BUY" | "SELL";
    capital?: number;
    isOpen?: boolean;
    enabled?: boolean;
  },
) {
  const side = options?.side ?? "BUY";
  return useQuery<SetupPayload>({
    queryKey: ["setup-preview", symbol, side, options?.capital ?? null],
    enabled: !!symbol && (options?.enabled ?? true),
    queryFn: async () => {
      const r = await api.get<SetupPayload>("/market-data/setup/", {
        params: {
          symbol,
          side,
          ...(options?.capital ? { capital: options.capital } : {}),
        },
      });
      return r.data;
    },
    refetchInterval: options?.isOpen ? SETUP_MS_OPEN : SETUP_MS_CLOSED,
    refetchIntervalInBackground: false,
    staleTime: 15_000,
  });
}

/** Map a criterion severity to a Badge tone for the breakdown row. */
export function criterionTone(
  severity: CriterionSeverity,
  passed: boolean,
): "success" | "warning" | "danger" | "neutral" {
  if (passed) return "success";
  if (severity === "danger") return "danger";
  if (severity === "warning") return "warning";
  return "neutral";
}

/* ================================================================== */
/* Swing Scanner — Oliver Kell Cycle of Price Action                    */
/*                                                                    */
/* Daily/weekly cycle phase detection across NIFTY 100.  Each stock    */
/* carries a phase code (RE/WP/EC/BB/EX/WD/EC_BEAR/BB_BEAR), trend    */
/* state for both daily & weekly timeframes, and a confidence score.    */
/* Best setups: BUY phases (WP/EC/BB) with aligned bullish trends.     */
/* ================================================================== */

export type CyclePhase =
  | "RE" | "WP" | "EC" | "BB"
  | "EX" | "WD" | "EC_BEAR" | "BB_BEAR"
  | "NONE";

export type TrendState = "bullish" | "bearish" | "neutral";

export interface SwingStock {
  symbol: string;
  phase: CyclePhase;
  phase_label: string;
  action: string;
  trend_daily: TrendState;
  trend_weekly: TrendState;
  aligned: boolean;
  confidence: number;
  close: number;
  ema10: number;
  ema20: number;
  ema50: number;
  upper_ext: number;
  lower_ext: number;
  volume_ratio: number;
  error: string;
}

export interface SwingScanPayload {
  as_of: string;
  scan_date: string;
  total: number;
  active: number;
  buy_aligned: number;
  short_aligned: number;
  watch: number;
  stocks: SwingStock[];
  phase_distribution: Record<string, number>;
  errors: string[];
}

/** Cycle phases are daily — 5 min poll while open, 10 min off-hours. */
const SWING_MS_OPEN = 300_000;
const SWING_MS_CLOSED = 600_000;

export function useSwingScanner(options?: { force?: boolean; isOpen?: boolean }) {
  return useQuery<SwingScanPayload>({
    queryKey: ["swing-scanner", options?.force ?? false],
    queryFn: async () => {
      const r = await api.get<SwingScanPayload>("/market-data/swing-scanner/", {
        params: options?.force ? { force: 1 } : undefined,
      });
      return r.data;
    },
    refetchInterval: options?.isOpen ? SWING_MS_OPEN : SWING_MS_CLOSED,
    refetchIntervalInBackground: false,
    staleTime: 60_000,
  });
}

/** Phase metadata — shared across Scanner, Backtester, Strategies pages. */
export const PHASE_INFO: Record<
  string,
  { label: string; action: string; color: string; description: string }
> = {
  RE:      { label: "Reversal Extension", action: "WATCH", color: "bg-info/20 text-info border-info/30",
             description: "Potential bottom — price extended below EMAs with volume spike" },
  WP:      { label: "Wedge Pop", action: "BUY", color: "bg-success/20 text-success border-success/30",
             description: "Momentum entry — price crosses above EMAs with volume" },
  EC:      { label: "EMA Crossback", action: "BUY", color: "bg-accent/20 text-accent border-accent/30",
             description: "Low-risk pullback entry — price tests EMA support and bounces" },
  BB:      { label: "Basin Break", action: "BUY", color: "bg-success/20 text-success border-success/30",
             description: "Continuation — breakout from consolidation near EMAs" },
  EX:      { label: "Exhaustion Extension", action: "SELL", color: "bg-warning/20 text-warning border-warning/30",
             description: "Potential top — price extended above EMAs with volume spike" },
  WD:      { label: "Wedge Drop", action: "AVOID", color: "bg-danger/20 text-danger border-danger/30",
             description: "Breakdown — price drops below EMAs" },
  EC_BEAR: { label: "Bear Crossback", action: "SHORT", color: "bg-danger/20 text-danger border-danger/30",
             description: "Failed bounce at EMA resistance" },
  BB_BEAR: { label: "Bear Break", action: "SHORT", color: "bg-danger/20 text-danger border-danger/30",
             description: "Continuation down from consolidation" },
  GAR:     { label: "Green After Red", action: "BUY", color: "bg-success/20 text-success border-success/30",
             description: "Buy green candle after red retracement post first bounce of the day" },
  EB3:     { label: "EMA Bounce", action: "BUY", color: "bg-accent/20 text-accent border-accent/30",
             description: "Buy on bounce from EMA line(s) in bullish trend" },
};

/** Map cycle phase to a semantic Badge tone. */
export function phaseTone(
  phase: CyclePhase | string,
): "success" | "warning" | "danger" | "info" | "neutral" {
  if (phase === "WP" || phase === "EC" || phase === "BB") return "success";
  if (phase === "GAR" || phase === "EB3") return "success";
  if (phase === "RE") return "info";
  if (phase === "EX") return "warning";
  if (phase === "WD" || phase === "EC_BEAR" || phase === "BB_BEAR") return "danger";
  return "neutral";
}

/** Map trend state to a Badge tone. */
export function trendStateTone(
  t: TrendState,
): "success" | "danger" | "neutral" {
  if (t === "bullish") return "success";
  if (t === "bearish") return "danger";
  return "neutral";
}

/* ================================================================== */
/* OK Backtest — daily swing + intraday multi-TF grid                  */
/* ================================================================== */

export interface OKBacktestTrade {
  symbol: string;
  side: string;
  phase: string;
  entry_date: string;
  entry: number;
  sl: number;
  target: number;
  qty: number;
  exit_date: string;
  exit: number;
  exit_reason: string;
  pnl: number;
  pnl_pct: number;
  rr: number;
  bars_held: number;
  won: boolean;
}

export interface TFGridRow {
  tf: string;
  sl_atr: number;
  rr: number;
  trades: number;
  win_rate: number;
  pf: number;
  pnl: number;
  pnl_pct: number;
  max_dd: number;
  avg_bars: number;
  avg_win: number;
  avg_loss: number;
  phase_stats: Record<string, { trades: number; win_rate: number; pnl: number }>;
}

export interface OKBacktestPayload {
  as_of: string;
  mode: "daily" | "intraday";
  from_date: string;
  to_date: string;
  capital: number;
  symbols_count: number;

  // Smart universe info
  scanned_universe: number;
  active_phases: number;
  universe_symbols: string[];

  total_trades: number;
  winners: number;
  win_rate: number;
  total_pnl: number;
  total_pnl_pct: number;
  profit_factor: number;
  avg_rr: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  avg_win: number;
  avg_loss: number;
  best_trade: number;
  worst_trade: number;
  avg_bars_held: number;

  phase_stats: Record<string, { trades: number; win_rate: number; pnl: number }>;
  weekly_pnl: Record<string, number>;

  trades: OKBacktestTrade[];
  equity_curve: { t: string; v: number }[];

  tf_grid: TFGridRow[];
  best_config: { tf?: string; sl_atr?: number; rr?: number; trades?: number; win_rate?: number; pf?: number; pnl?: number };

  errors: string[];
}

export function useOKBacktest(params: {
  mode: "daily" | "intraday" | "basket";
  from_date: string;
  to_date: string;
  enabled?: boolean;
}) {
  return useQuery<OKBacktestPayload>({
    queryKey: ["ok-backtest", params.mode, params.from_date, params.to_date],
    queryFn: async () => {
      // Cold-cache backtests are slow — daily ~30s, intraday ~200s+ (98-stock
      // NIFTY 100 scan + multi-TF grid). Result is cached server-side for 1h,
      // so the second call is sub-second. 300s gives the cold path headroom.
      const r = await api.get<OKBacktestPayload>("/market-data/ok-backtest/", {
        params: { mode: params.mode, from_date: params.from_date, to_date: params.to_date },
        timeout: 300_000,
      });
      return r.data;
    },
    enabled: params.enabled ?? true,
    staleTime: 300_000,
    retry: false,
  });
}

/* ================================================================== */
/* Morning Basket                                                       */
/* ================================================================== */

export interface BasketSignal {
  symbol: string;
  side: string;
  leg_type: "equity" | "option";
  entry_price: number;
  stoploss: number;
  risk_points: number;
  phase: string;
  confluence: number;
  option_type?: string;
  strike?: number;
  expiry?: string;
  option_symbol?: string;
}

export interface BasketPayload {
  as_of: string;
  mood: string;
  mood_details: {
    mood: string;
    confidence: number;
    advance: number;
    decline: number;
    ad_ratio: number;
    nifty_spot: number;
    gap_pct: number;
    vix: number;
    vix_tier: string;
    reasons: string[];
  };
  signals: BasketSignal[];
  errors: string[];
}

export function useBasketStatus(options?: { isOpen?: boolean }) {
  return useQuery<BasketPayload>({
    queryKey: ["basket-status"],
    queryFn: async () => {
      const r = await api.get<BasketPayload>("/market-data/basket/", {
        timeout: 60_000,
      });
      return r.data;
    },
    refetchInterval: options?.isOpen ? 60_000 : 300_000,
    staleTime: 30_000,
  });
}
