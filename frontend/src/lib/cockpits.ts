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

export interface EdgeDecay {
  window: number;
  series: { strategy: string; points: { trade_idx: number; expectancy: number; win_rate: number }[] }[];
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
