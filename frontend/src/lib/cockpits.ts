import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

// ---------------------------------------------------------------------------
// Cockpit response shapes — mirror apps.portfolio.services.cockpits.*
// Field sets verified against a live `python manage.py shell` call.
// ---------------------------------------------------------------------------
export interface CapitalCockpit {
  total_capital: number;
  deployed_capital: number;
  free_margin: number;
  margin_used: number;
  notional_exposure: number;
  delta_adjusted_exposure: number;
  premium_received: number;
  leverage_ratio: number;
  buckets: { name: string; margin: number; notional: number }[];
}

export interface PlanVsActualRow {
  trade_id: number | string;
  symbol: string;
  side: string;
  strategy?: string;
  planned_entry: number;
  actual_entry: number;
  slippage_bps: number;
  planned_sl?: number;
  actual_sl?: number;
  sl_hit: boolean;
  status: string;
  timestamp?: string;
}

export interface PlanVsActual {
  count: number;
  avg_abs_slippage_bps: number;
  rows: PlanVsActualRow[];
}

export interface GreeksRow {
  underlying: string;
  expiry: string;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  positions: number;
}

export interface GreeksHeatmap {
  count: number;
  note?: string;
  rows: GreeksRow[];
}

export interface SignalFunnel {
  totals: { fired: number; risk_passed: number; executed: number; profitable: number };
  by_strategy: { strategy: string; fired: number; risk_passed: number; executed: number; profitable: number }[];
  rejection_reasons: { reason: string; count: number }[];
}

export interface RiskBudget {
  capital: number;
  daily_pnl: number;
  daily_loss: number;
  max_risk_per_trade_pct: number;
  allowed_risk_pct: number;
  used_risk_pct: number;
  open_risk_at_stop: number;
  open_risk_pct_of_capital: number;
  drawdown_waterfall: { date: string; pnl: number; cum_pnl: number; drawdown: number }[];
}

export interface ExpiryClose {
  symbol: string;
  ce_symbol?: string;
  pe_symbol?: string;
  status: string;
  pnl?: number;
}

export interface ExpiryCockpit {
  underlying: string;
  is_expiry_day: boolean;
  countdown_seconds: number;
  pin_strike?: number;
  active_count: number;
  close_list: ExpiryClose[];
  gamma_by_strike: { strike: number; gamma: number }[];
}

export interface BrokerReconRow {
  symbol: string;
  side?: string;
  qty?: number;
  pnl?: number;
}

export interface BrokerRecon {
  on_date: string;
  broker_count: number;
  journal_count: number;
  broker_pnl: number;
  journal_pnl: number;
  pnl_delta: number;
  mismatched_count: number;
  only_in_broker: BrokerReconRow[];
  only_in_journal: BrokerReconRow[];
  broker_note?: string;
}

export interface EdgeDecayPoint {
  as_of?: string;
  trade_idx?: number;
  expectancy: number;
  win_rate: number;
  avg_r?: number;
  n?: number;
}

export interface EdgeDecay {
  window: number;
  // Backend returns {strategy_name: [points]}; older shape was an array.
  series: Record<string, EdgeDecayPoint[]> | { strategy: string; points: EdgeDecayPoint[] }[];
}

export interface ThetaForecast {
  count: number;
  positions: {
    position_id: number;
    underlying: string;
    expiry: string;
    minutes_to_expiry: number;
    premium_remaining: number;
    theta_per_minute: number;
    projection: { minute: number; premium: number }[];
  }[];
}

export interface RegimeHeatmap {
  current_regime: string;
  regimes: string[];
  strategies: string[];
  cells: { strategy: string; regime: string; expectancy: number; trades: number }[];
}

// ---------------------------------------------------------------------------
// React Query hooks — staleTime keeps cockpit nav snappy without thrashing
// the broker call paths the legacy services hit underneath.
// ---------------------------------------------------------------------------
const COMMON = { staleTime: 30_000, refetchOnWindowFocus: false } as const;

export const useCapitalCockpit = () =>
  useQuery({
    queryKey: ["cockpits", "capital"],
    queryFn: () => api.get<CapitalCockpit>("/portfolios/capital-cockpit/").then((r) => r.data),
    ...COMMON,
  });

export const usePlanVsActual = () =>
  useQuery({
    queryKey: ["cockpits", "plan-vs-actual"],
    queryFn: () => api.get<PlanVsActual>("/portfolios/plan-vs-actual/").then((r) => r.data),
    ...COMMON,
  });

export const useGreeksHeatmap = () =>
  useQuery({
    queryKey: ["cockpits", "greeks-heatmap"],
    queryFn: () => api.get<GreeksHeatmap>("/portfolios/greeks-heatmap/").then((r) => r.data),
    ...COMMON,
  });

export const useSignalFunnel = () =>
  useQuery({
    queryKey: ["cockpits", "signal-funnel"],
    queryFn: () => api.get<SignalFunnel>("/portfolios/signal-funnel/").then((r) => r.data),
    ...COMMON,
  });

export const useRiskBudget = () =>
  useQuery({
    queryKey: ["cockpits", "risk-budget"],
    queryFn: () => api.get<RiskBudget>("/portfolios/risk-budget/").then((r) => r.data),
    ...COMMON,
  });

export const useExpiryCockpit = (underlying: string = "NIFTY") =>
  useQuery({
    queryKey: ["cockpits", "expiry", underlying],
    queryFn: () =>
      api
        .get<ExpiryCockpit>(`/portfolios/expiry-cockpit/?underlying=${encodeURIComponent(underlying)}`)
        .then((r) => r.data),
    ...COMMON,
  });

export const useBrokerRecon = () =>
  useQuery({
    queryKey: ["cockpits", "broker-recon"],
    queryFn: () => api.get<BrokerRecon>("/portfolios/broker-recon/").then((r) => r.data),
    ...COMMON,
  });

export const useEdgeDecay = (window: number = 20) =>
  useQuery({
    queryKey: ["cockpits", "edge-decay", window],
    queryFn: () => api.get<EdgeDecay>(`/portfolios/edge-decay/?window=${window}`).then((r) => r.data),
    ...COMMON,
  });

export const useThetaForecast = () =>
  useQuery({
    queryKey: ["cockpits", "theta-forecast"],
    queryFn: () => api.get<ThetaForecast>("/portfolios/theta-forecast/").then((r) => r.data),
    ...COMMON,
  });

export const useRegimeHeatmap = () =>
  useQuery({
    queryKey: ["cockpits", "regime-heatmap"],
    queryFn: () => api.get<RegimeHeatmap>("/portfolios/regime-heatmap/").then((r) => r.data),
    ...COMMON,
  });

// ---------------------------------------------------------------------------
// Cycle-2 cockpits — added after the second AI team planning cycle.
// ---------------------------------------------------------------------------
export interface CorrelationReport {
  symbols: string[];
  matrix: number[][];
  independent_bets: number;
  sector_weights: Record<string, number>;
  factor_weights: Record<string, number>;
  as_of: string;
}

export interface PostMortemRow {
  trade_id: number;
  symbol: string;
  side: string;
  entry: number;
  exit: number;
  pnl: number;
  cause: string;
  evidence: string;
  timestamp: string | null;
}

export interface PostMortemReport {
  month: string | null;
  count: number;
  by_cause: Record<string, number>;
  rows: PostMortemRow[];
  taxonomy: string[];
}

export interface GapRiskPosition {
  symbol: string;
  side: string;
  qty: number;
  entry: number;
  kind: string;
  pnl_at_gap: Record<string, number>;
}

export interface GapRiskReport {
  implied_gap_pct: number;
  implied_gap_source: string;
  positions: GapRiskPosition[];
  hedge_checklist: string[];
  as_of: string;
}

export interface LiquidityRow {
  symbol: string;
  bid: number;
  ask: number;
  mid: number;
  spread_bps: number;
  depth_imbalance: number;
  avg_historical_slippage_bps: number;
}

export interface LiquidityMap {
  count: number;
  rows: LiquidityRow[];
  note?: string;
}

export interface SizerRequest {
  symbol: string; qty: number; side: "BUY" | "SELL";
  stop?: number; entry?: number; product?: string;
}

export interface SizerResponse {
  symbol: string; qty: number; side: string; entry: number; stop: number;
  post_trade_delta: { margin_added: number; notional_added: number };
  capital: number;
  margin_used_before: number; margin_used: number;
  free_cash: number; leverage_ratio: number;
  worst_case_loss_inr: number;
  distance_to_daily_loss_cap_pct: number;
  daily_loss_cap_inr: number;
  realised_pnl_today: number;
  error?: string;
}

export const useCorrelationMatrix = () =>
  useQuery({
    queryKey: ["cockpits", "correlation"],
    queryFn: () => api.get<CorrelationReport>("/portfolios/correlation/").then((r) => r.data),
    ...COMMON,
  });

export const usePostMortem = (month?: string) =>
  useQuery({
    queryKey: ["cockpits", "post-mortem", month],
    queryFn: () =>
      api
        .get<PostMortemReport>(`/portfolios/post-mortem/${month ? `?month=${encodeURIComponent(month)}` : ""}`)
        .then((r) => r.data),
    ...COMMON,
  });

export const useGapRisk = () =>
  useQuery({
    queryKey: ["cockpits", "gap-risk"],
    queryFn: () => api.get<GapRiskReport>("/portfolios/gap-risk/").then((r) => r.data),
    ...COMMON,
  });

export const useLiquidityMap = () =>
  useQuery({
    queryKey: ["cockpits", "liquidity"],
    queryFn: () => api.get<LiquidityMap>("/market-data/liquidity/").then((r) => r.data),
    ...COMMON,
  });

export async function simulateSizer(payload: SizerRequest): Promise<SizerResponse> {
  const { data } = await api.post<SizerResponse>("/portfolios/sizer/simulate/", payload);
  return data;
}

export interface SetCapitalResponse {
  capital: number;
  invested: number;
  available_cash: number;
  snapshot_date: string;
  error?: string;
}

export async function setCapital(capital: number): Promise<SetCapitalResponse> {
  const { data } = await api.post<SetCapitalResponse>("/portfolios/capital/", { capital });
  return data;
}

// ---------------------------------------------------------------------------
// Cycle-3 cockpits — structural stops, forced-flat, slippage-edge, ORB.
// ---------------------------------------------------------------------------
export interface StructuralStopRow {
  position_id: number; symbol: string; side?: string; entry: number; qty: number;
  swing_low: number; ten_wma: number; atr_trail: number;
  recommended: string | null; recommended_value?: number;
  r_distance: number; pct_loss: number; loss_at_stop_inr?: number;
  note?: string;
}
export interface StructuralStops { count: number; rows: StructuralStopRow[]; as_of: string; note: string; }

export interface ForcedFlatRow {
  trade_id: number; symbol: string; side: string; qty: number;
  entry: number; ltp: number; pnl: number;
  est_slippage_bps: number; status: string;
}
export interface ForcedFlat {
  now_ist: string; deadline: string; countdown_seconds: number;
  active: boolean; count: number; total_pnl: number;
  rows: ForcedFlatRow[]; note: string;
}

export interface SlippageEdgeRequest {
  symbol: string; qty: number; setup_avg_r_inr: number;
}
export interface SlippageEdgeResponse {
  symbol: string; qty: number;
  bid: number; ask: number; mid: number;
  half_spread_inr: number; impact_inr: number; brokerage_inr: number;
  total_cost_inr: number; expected_edge_inr: number; net_edge_inr: number;
  edge_to_cost_ratio: number;
  verdict: "green" | "amber" | "red";
  note?: string; error?: string;
}

export interface ORBRow {
  symbol: string; or_high: number; or_low: number; or_width: number;
  or_width_atr: number; atr14: number;
  state: "pre_open" | "inside" | "breakout_up" | "breakout_down" | "failed_breakout";
  breakout_time: string | null; retests: number;
}
export interface ORB { count: number; rows: ORBRow[]; note?: string; }

export const useStructuralStops = () =>
  useQuery({
    queryKey: ["cockpits", "structural-stops"],
    queryFn: () => api.get<StructuralStops>("/portfolios/structural-stops/").then((r) => r.data),
    ...COMMON,
  });

export const useForcedFlat = () =>
  useQuery({
    queryKey: ["cockpits", "forced-flat"],
    queryFn: () => api.get<ForcedFlat>("/portfolios/forced-flat/").then((r) => r.data),
    refetchInterval: 30_000,    // countdown ticks
    staleTime: 10_000,
    refetchOnWindowFocus: false,
  });

export async function flattenAll() {
  const { data } = await api.post<{ flattened: number; trades: { trade_id: number; symbol: string; ltp: number; pnl: number }[] }>(
    "/portfolios/forced-flat/flatten/", {},
  );
  return data;
}

export async function checkSlippageEdge(payload: SlippageEdgeRequest): Promise<SlippageEdgeResponse> {
  const { data } = await api.post<SlippageEdgeResponse>("/portfolios/slippage-edge/", payload);
  return data;
}

export const useORB = () =>
  useQuery({
    queryKey: ["cockpits", "orb"],
    queryFn: () => api.get<ORB>("/market-data/orb/").then((r) => r.data),
    ...COMMON,
  });

// ---------------------------------------------------------------------------
// Cycle-4 cockpits — VWAP bands, first-5-min profile, base-quality score.
// ---------------------------------------------------------------------------
export interface VWAPSeries {
  t: string; c: number; vwap: number;
  sigma1_up: number; sigma1_dn: number;
  sigma2_up: number; sigma2_dn: number; sd: number;
}
export interface VWAPBands {
  symbol: string;
  vwap: number;
  sigma1_up: number; sigma1_dn: number;
  sigma2_up: number; sigma2_dn: number;
  dist_sigma: number;
  state: "neutral" | "stretched_up" | "stretched_down" | "no_data";
  last_close?: number;
  series: VWAPSeries[];
  bar_count?: number;
  note?: string;
  error?: string;
}

export const useVWAPBands = (symbol: string) =>
  useQuery({
    queryKey: ["cockpits", "vwap-bands", symbol],
    queryFn: () =>
      api.get<VWAPBands>(`/market-data/vwap-bands/?symbol=${encodeURIComponent(symbol)}`).then((r) => r.data),
    enabled: !!symbol,
    ...COMMON,
  });

export interface First5MinRow {
  symbol: string;
  classification: string;
  day_type_tag: string;
  open?: number; high?: number; low?: number; close?: number;
  gap_pct: number; body_pct: number; vol: number;
  range_atr?: number;
}
export interface First5Min { count: number; rows: First5MinRow[]; note?: string; }

export const useFirst5Min = () =>
  useQuery({
    queryKey: ["cockpits", "first-5min"],
    queryFn: () => api.get<First5Min>("/market-data/first-5min/").then((r) => r.data),
    ...COMMON,
  });

export interface BaseQualityRow {
  symbol: string;
  score: number;
  pattern_tag: string;
  pivot: number;
  depth_pct: number;
  length_weeks: number;
  tightness_pct: number;
  volume_dryup: number;
  last_close: number;
  pct_from_pivot: number;
}
export interface BaseQuality { count: number; rows: BaseQualityRow[]; note?: string; }

export const useBaseQuality = (symbols?: string) =>
  useQuery({
    queryKey: ["cockpits", "base-quality", symbols ?? ""],
    queryFn: () => {
      const q = symbols ? `?symbols=${encodeURIComponent(symbols)}` : "";
      return api.get<BaseQuality>(`/strategies/base-quality/${q}`).then((r) => r.data);
    },
    ...COMMON,
  });

// ---------------------------------------------------------------------------
// Cycle-5 cockpits — MTF stage scanner, OR failure, fresh-breakout, edge ledger
// ---------------------------------------------------------------------------
export interface StagePhase {
  stage: "STAGE_1" | "STAGE_2" | "STAGE_3" | "STAGE_4" | "UNKNOWN";
  ma: number; slope_pct: number; close: number; gap_pct: number;
}
export interface MTFStageRow {
  symbol: string;
  daily: StagePhase | null;
  weekly: StagePhase | null;
  monthly: StagePhase | null;
  alignment: "long_aligned" | "short_aligned" | "conflict" | "mixed" | "no_data";
  stage2_aligned?: boolean;
}
export interface MTFStage { count: number; rows: MTFStageRow[]; note?: string; }

export const useMTFStage = (symbols?: string) =>
  useQuery({
    queryKey: ["cockpits", "mtf-stage", symbols ?? ""],
    queryFn: () => {
      const q = symbols ? `?symbols=${encodeURIComponent(symbols)}` : "";
      return api.get<MTFStage>(`/strategies/mtf-stage/${q}`).then((r) => r.data);
    },
    ...COMMON,
  });

export interface BreakoutRow {
  symbol: string; close?: number; pivot?: number; sma20?: number;
  pct_from_pivot: number; pct_from_20dma: number; base_depth_pct: number;
  state: "fresh" | "extended" | "consolidating" | "base_too_shallow" | "neutral" | "no_data";
}
export interface BreakoutClassifier { count: number; rows: BreakoutRow[]; note?: string; }

export const useBreakoutClassifier = (symbols?: string) =>
  useQuery({
    queryKey: ["cockpits", "breakout-classifier", symbols ?? ""],
    queryFn: () => {
      const q = symbols ? `?symbols=${encodeURIComponent(symbols)}` : "";
      return api.get<BreakoutClassifier>(`/strategies/breakout-classifier/${q}`).then((r) => r.data);
    },
    ...COMMON,
  });

export interface ORBFailureRow {
  symbol: string;
  or_high: number; or_low: number;
  state: string;
  breakout_time: string | null;
  retest_count_after_break: number;
  failure_flag: boolean;
  reversal_target: number;
  reversal_p: number;
}
export interface ORBFailure { count: number; rows: ORBFailureRow[]; note?: string; }

export const useORBFailure = () =>
  useQuery({
    queryKey: ["cockpits", "orb-failure"],
    queryFn: () => api.get<ORBFailure>("/market-data/orb-failure/").then((r) => r.data),
    ...COMMON,
  });

export interface EdgeLedgerRow {
  trade_id: number; symbol: string; side: string; qty: number;
  entry: number; fill: number; strategy: string;
  gross_pnl_inr: number; spread_cost_inr: number; brokerage_inr: number;
  total_cost_inr: number; net_edge_inr: number;
  edge_to_cost: number; edge_bps: number; created_at: string | null;
}
export interface EdgeBucket {
  trades: number; gross_pnl_inr?: number; cost_inr?: number;
  net_edge_inr?: number; cost_drag_pct?: number;
  win_rate?: number; avg_net_edge_inr?: number;
}
export interface EdgeLedger {
  count: number;
  totals: EdgeBucket;
  by_strategy: Record<string, EdgeBucket>;
  by_symbol: Record<string, EdgeBucket>;
  rows: EdgeLedgerRow[];
  note?: string;
}

export const useEdgeLedger = () =>
  useQuery({
    queryKey: ["cockpits", "edge-ledger"],
    queryFn: () => api.get<EdgeLedger>("/portfolios/edge-ledger/").then((r) => r.data),
    ...COMMON,
  });

// ---------------------------------------------------------------------------
// Cycle-6 cockpits — 11 new tabs covering pure-compute, sector, and stubbed
// external-feed views. Stubs return the same shape they will once a feed is
// wired, so the panels never need a rewrite.
// ---------------------------------------------------------------------------
export interface IntradayBucket {
  slot: string; deployed: number; gross: number;
  realised_pnl: number; trades_open: number; idle_pct: number;
  sector_breakdown: Record<string, number>;
}
export interface IntradayRotation {
  date: string; capital: number; peak_deployed: number;
  peak_utilisation_pct: number; total_realised_pnl: number;
  buckets: IntradayBucket[]; note?: string;
}

export const useIntradayRotation = (on?: string) =>
  useQuery({
    queryKey: ["cockpits", "intraday-rotation", on ?? ""],
    queryFn: () => {
      const q = on ? `?date=${encodeURIComponent(on)}` : "";
      return api.get<IntradayRotation>(`/portfolios/intraday-rotation/${q}`).then((r) => r.data);
    },
    ...COMMON,
  });

export interface GapFillRow {
  symbol: string; prev_close?: number; open?: number; high?: number; low?: number; close?: number;
  gap_pct: number; status: "open" | "filled" | "no_gap" | "no_data";
  filled_today: boolean; historical_fill_p: number;
}
export interface GapFill { count: number; rows: GapFillRow[]; note?: string; }

export const useGapFill = () =>
  useQuery({
    queryKey: ["cockpits", "gap-fill"],
    queryFn: () => api.get<GapFill>("/market-data/gap-fill/").then((r) => r.data),
    ...COMMON,
  });

export interface Second5MinRow {
  symbol: string;
  classification: "continuation" | "reversal" | "consolidation" | "weak" | "no_data";
  day_type_tag: string;
  first_bar: { o: number; h: number; l: number; c: number; v: number } | null;
  second_bar: { o: number; h: number; l: number; c: number; v: number } | null;
  vol_ratio: number;
}
export interface Second5Min { count: number; rows: Second5MinRow[]; note?: string; }

export const useSecond5Min = () =>
  useQuery({
    queryKey: ["cockpits", "second-5min"],
    queryFn: () => api.get<Second5Min>("/market-data/second-5min/").then((r) => r.data),
    ...COMMON,
  });

export interface VolRegimePoint {
  t: string; c: number; realised_vol_ann: number; trades_per_sec: number;
  regime: "TREND" | "CHOP" | "DEAD" | "SHOCK" | "no_data";
}
export interface VolRegime {
  symbol: string; bar_count: number;
  current_regime: string; current_vol_ann?: number; current_tps?: number;
  series: VolRegimePoint[]; note?: string;
}

export const useVolRegime = (symbol: string) =>
  useQuery({
    queryKey: ["cockpits", "vol-regime", symbol],
    queryFn: () =>
      api.get<VolRegime>(`/market-data/vol-regime/?symbol=${encodeURIComponent(symbol)}`)
         .then((r) => r.data),
    enabled: !!symbol,
    ...COMMON,
  });

export interface RRGTailPoint { rs_ratio: number; rs_mom: number; }
export interface RRGRow {
  sector: string; yf_symbol?: string;
  rs_ratio: number; rs_momentum: number;
  quadrant: "LEADING" | "WEAKENING" | "LAGGING" | "IMPROVING" | "no_data";
  tail: RRGTailPoint[];
}
export interface SectorRRG { count: number; rows: RRGRow[]; note?: string; }

export const useSectorRRG = () =>
  useQuery({
    queryKey: ["cockpits", "sector-rrg"],
    queryFn: () => api.get<SectorRRG>("/market-data/sector-rrg/").then((r) => r.data),
    ...COMMON,
  });

export interface SectorDispRow {
  sector: string; cohort_size: number;
  median_pct: number; dispersion_pct: number;
  leaders: { symbol: string; pct: number }[];
  laggards: { symbol: string; pct: number }[];
}
export interface SectorDispersion { count: number; rows: SectorDispRow[]; note?: string; }

export const useSectorDispersion = () =>
  useQuery({
    queryKey: ["cockpits", "sector-dispersion"],
    queryFn: () => api.get<SectorDispersion>("/market-data/sector-dispersion/").then((r) => r.data),
    ...COMMON,
  });

export interface TapeSpeedPoint {
  t: string; c: number; trades_per_sec: number; rupees_per_min: number;
  ratio_to_baseline: number; realised_vol_pm: number;
  state: "cold" | "normal" | "hot" | "shock";
}
export interface TapeSpeed {
  symbol: string; bar_count: number; baseline_rupees_per_min: number;
  current_state: string; current_tps?: number;
  current_rupees_per_min?: number; current_realised_vol_pm?: number;
  series: TapeSpeedPoint[]; note?: string;
}

export const useTapeSpeed = (symbol: string) =>
  useQuery({
    queryKey: ["cockpits", "tape-speed", symbol],
    queryFn: () =>
      api.get<TapeSpeed>(`/market-data/tape-speed/?symbol=${encodeURIComponent(symbol)}`)
         .then((r) => r.data),
    enabled: !!symbol,
    ...COMMON,
  });

export interface NewsShockEvent {
  symbol: string; severity: "info" | "warning" | "critical";
  source?: string; headline?: string; ts?: string;
  flatten_recommendation?: boolean;
}
export interface NewsShock {
  events: NewsShockEvent[]; coverage_symbols: string[];
  paused_symbols?: PauseRecord[];
  active_pause_count?: number;
  default_cooldown_min?: number;
  as_of: string; data_source: string; note?: string;
}

export const useNewsShock = () =>
  useQuery({
    queryKey: ["cockpits", "news-shock"],
    queryFn: () => api.get<NewsShock>("/market-data/news-shocks/").then((r) => r.data),
    ...COMMON,
  });

export interface FIIDIIPoint { date: string; value?: number; close?: number; }
export interface FIIDIIFlow {
  days: number;
  fii_cash: FIIDIIPoint[]; dii_cash: FIIDIIPoint[];
  fii_futures_oi: FIIDIIPoint[]; fii_options_premium: FIIDIIPoint[];
  nifty_close: FIIDIIPoint[];
  regimes?: Record<string, string>;
  data_source: string; note?: string;
}

export const useFIIDIIFlow = () =>
  useQuery({
    queryKey: ["cockpits", "fii-dii-flow"],
    queryFn: () => api.get<FIIDIIFlow>("/market-data/fii-dii-flow/").then((r) => r.data),
    ...COMMON,
  });

export interface DepthRow {
  symbol: string; today_volume: number; baseline_volume: number;
  volume_ratio: number; depth_imbalance: number; iceberg_flag: boolean;
}
export interface DepthImbalance {
  count: number; rows: DepthRow[]; data_source: string; note?: string;
}

export const useDepthImbalance = () =>
  useQuery({
    queryKey: ["cockpits", "depth-imbalance"],
    queryFn: () => api.get<DepthImbalance>("/market-data/depth-imbalance/").then((r) => r.data),
    ...COMMON,
  });

export interface EarningsRow {
  trade_id: number; symbol: string; side: string; qty: number;
  earnings_date: string | null; ex_div_date: string | null;
  consensus_eps: number | null; avg_post_earn_gap_pct: number | null;
  days_to_event: number | null;
}
export interface EarningsOverlay {
  count: number; rows: EarningsRow[]; as_of: string;
  data_source: string; note?: string;
}

export const useEarningsOverlay = () =>
  useQuery({
    queryKey: ["cockpits", "earnings-overlay"],
    queryFn: () => api.get<EarningsOverlay>("/portfolios/earnings-overlay/").then((r) => r.data),
    ...COMMON,
  });

// ---------------------------------------------------------------------------
// Reset trading data — wipes journal / straddles / runs / cache / etc.
// ---------------------------------------------------------------------------
export interface ResetRequest {
  flags: string[];            // e.g. ["all"] or ["journal","cache"] or ["nuke"]
  keep_watchlist?: boolean;
  no_reseed?: boolean;
  capital?: number;
}
export interface ResetResponse {
  ok: boolean;
  summary: string;
  flags: string[];
  capital?: number;
  reseeded?: boolean;
  error?: string;
}

export async function resetTradingData(payload: ResetRequest): Promise<ResetResponse> {
  const { data } = await api.post<ResetResponse>("/portfolios/reset/", payload);
  return data;
}

// ---------------------------------------------------------------------------
// Cycle-8 new endpoints
// ---------------------------------------------------------------------------
export interface StockRRGRow {
  symbol: string;
  rs_ratio: number;
  rs_momentum: number;
  quadrant: "LEADING" | "WEAKENING" | "LAGGING" | "IMPROVING" | "no_data";
  tail: { rs_ratio: number; rs_mom: number }[];
}
export interface StockRRG {
  count: number;
  mode: string;
  tail_length: number;
  rows: StockRRGRow[];
  note?: string;
}

export const useStockRRG = (symbols?: string) =>
  useQuery({
    queryKey: ["cockpits", "stock-rrg", symbols ?? ""],
    queryFn: () => {
      const q = symbols ? `?symbols=${encodeURIComponent(symbols)}` : "";
      return api.get<StockRRG>(`/market-data/stock-rrg/${q}`).then((r) => r.data);
    },
    ...COMMON,
  });

export interface PartialFillRow {
  trade_id: number; symbol: string; side: string;
  qty_requested: number; qty_filled: number; fill_ratio: number;
  entry: number; fill: number; slippage_bps: number;
  queue_score: number; cost_per_lot_inr: number;
  strategy: string; flags: string[]; created_at: string | null;
}
export interface PartialFillBucket {
  count: number;
  avg_queue_score?: number;
  median_queue_score?: number;
}
export interface PartialFillReport {
  count: number;
  totals: {
    trades: number; avg_queue_score: number; avg_fill_ratio: number;
    high_slippage_count: number; low_fill_count: number; chased_count: number;
  };
  by_strategy: Record<string, PartialFillBucket>;
  by_symbol: Record<string, PartialFillBucket>;
  rows: PartialFillRow[];
  note?: string;
}

export const usePartialFill = () =>
  useQuery({
    queryKey: ["cockpits", "partial-fill"],
    queryFn: () => api.get<PartialFillReport>("/portfolios/partial-fill/").then((r) => r.data),
    ...COMMON,
  });

export interface SectorHeatmapCell {
  slot: string; n: number; median: number;
  dispersion: number; leader: number; laggard: number;
}
export interface SectorHeatmapRow {
  sector: string;
  cells: SectorHeatmapCell[];
}
export interface IntradaySectorHeatmap {
  slots: string[];
  rows: SectorHeatmapRow[];
  sector_count: number;
  slot_count: number;
  note?: string;
}

export const useIntradaySectorHeatmap = () =>
  useQuery({
    queryKey: ["cockpits", "intraday-sector-heatmap"],
    queryFn: () => api.get<IntradaySectorHeatmap>("/market-data/intraday-sector-heatmap/").then((r) => r.data),
    ...COMMON,
  });

// News-shock pause/unpause mutations (the GET hook already exists).
export interface PauseRequest { symbol: string; minutes?: number; reason?: string; }
export interface PauseRecord {
  symbol: string; paused_at: string; re_entry_at: string;
  minutes: number; reason: string; error?: string;
}
export async function pauseSymbol(payload: PauseRequest): Promise<PauseRecord> {
  const { data } = await api.post<PauseRecord>("/market-data/news-shocks/pause/", payload);
  return data;
}
export async function unpauseSymbol(symbol: string): Promise<{ symbol: string; was_paused: boolean }> {
  const { data } = await api.post<{ symbol: string; was_paused: boolean }>(
    "/market-data/news-shocks/unpause/", { symbol },
  );
  return data;
}
