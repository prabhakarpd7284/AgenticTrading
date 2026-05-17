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
