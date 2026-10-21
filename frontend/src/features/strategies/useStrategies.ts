/**
 * Strategy Studio data layer (R4): strategies with versions, templates,
 * spec validation / preview and real backtests (`/api/v1/strategies/...`).
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export type Stage = "draft" | "backtested" | "paper" | "live";
export type Backtestability = "backtest" | "forward_paper_only" | "unavailable";

export interface JsonSchema {
  type?: string | string[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: (string | number)[];
  items?: JsonSchema;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  description?: string;
  format?: string;
}

export interface LiveGate { passes: boolean | null; min_rr: number; message: string }
export interface FieldError { path: string; message: string }

export interface Template {
  key: string;
  kind: "spec" | "plugin";
  strategy_name: string;
  title: string;
  summary: string;
  backtest: Backtestability;
  timeframe: string | null;
  spec: Spec | null;
  schema: JsonSchema | null;
  defaults: Record<string, unknown> | null;
  notes: string[];
  broken: Record<string, string>;
  reason: string;
  live_gate: LiveGate | null;
  asset_class?: string;
  version?: string;
}

export interface Metrics {
  trades: number;
  wins?: number;
  losses?: number;
  win_rate: number | null;
  avg_r: number | null;
  expectancy_r: number | null;
  expectancy_inr?: number | null;
  profit_factor: number | null;
  net_pnl: number;
  gross_pnl?: number;
  costs: number;
  return_pct?: number;
  max_drawdown: number;
  max_drawdown_pct?: number;
  total_r?: number;
  too_few_trades: boolean;
  min_trades?: number;
  avg_bars_held?: number | null;
  long_trades?: number;
  short_trades?: number;
}

export interface Provenance {
  data_source: string;
  synthetic: boolean;
  timeframe: string;
  requested: { from: string; to: string };
  actual: { first: string | null; last: string | null };
  sessions: number;
  symbols_requested: number;
  symbols_loaded: string[];
  symbols_missing: string[];
  bars: number;
  gap_sessions: number;
  split: { method: string; is: [string | null, string | null]; oos: [string | null, string | null] | null };
  label: string;
  research_oos_overlap: boolean;
  research_note: string;
  sample: { trades: number; oos_trades: number; threshold: number; too_few: boolean };
  costs: { model: string; description: string; as_of: string; slippage_pct_per_side: number };
  engine: string;
  definition: { kind: string; strategy: string; version: number | null; hash: string };
  signals: { detected: number; taken_to_fill: number; rejected: Record<string, number>; dropped: Record<string, number> };
  live_gate: LiveGate | null;
  notes: string[];
  warnings: string[];
}

export type BacktestStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface BacktestSummary {
  id: string;
  status: BacktestStatus;
  created_at: string;
  version: number | null;
  from_date: string;
  to_date: string;
  progress: number;
  progress_note: string;
  error: string;
  metrics: Partial<Metrics>;
  provenance: Partial<Provenance> | null;
  instance_id?: string;
  instance_name?: string;
}

export interface Backtest extends Omit<BacktestSummary, "metrics" | "provenance"> {
  instance_id: string;
  instance_name: string;
  config: { symbols: string[] | null; capital: number; slippage_pct: number | null; oos_fraction: number };
  metrics: Metrics | Record<string, never>;
  split_metrics: { is?: Metrics; oos?: Metrics; oos_start?: string | null; oos_fraction?: number };
  provenance: Provenance | Record<string, never>;
  equity_curve: { t: string; equity: number; drawdown: number }[];
  started_at: string | null;
  finished_at: string | null;
  cancel_requested: boolean;
  trade_count: number;
}

export interface BacktestTrade {
  seq: number;
  symbol: string;
  side: "BUY" | "SHORT";
  segment: "is" | "oos";
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  stop: number;
  target: number | null;
  qty: number;
  gross_pnl: number;
  costs: number;
  net_pnl: number;
  r_multiple: number;
  exit_reason: string;
  bars_held: number;
  meta: Record<string, unknown>;
}

export interface NextStep { code: string; title: string; detail: string; action: string | null }

export interface StrategyRow {
  id: string;
  name: string;
  description: string;
  kind: "spec" | "plugin";
  strategy_name: string;
  strategy_version: string;
  template_key: string;
  stage: Stage;
  current_version: number;
  archived: boolean;
  archived_at: string | null;
  created_at: string;
  updated_at: string;
  backtest: Backtestability;
  backtest_reason: string;
  last_backtest: BacktestSummary | null;
  last_result: BacktestSummary | null;
  next_step: NextStep;
  can_delete: boolean;
  delete_blocker: string | null;
  live_gate: LiveGate | null;
}

export interface StrategyDetail extends StrategyRow {
  definition: Spec | Record<string, unknown>;
  version_hash: string | null;
  explain: string[];
  schema?: JsonSchema | null;
  template_notes?: string[];
  broken?: Record<string, string>;
  versions: number;
}

export interface StrategyVersion {
  id: string;
  version: number;
  spec_hash: string;
  note: string;
  created_at: string;
  created_by: string | null;
  definition: Record<string, unknown>;
  backtests: number;
}

/* ---- StrategySpec (mirror of apps/strategies/services/spec.py) ---- */
export type PriceField = "open" | "high" | "low" | "close" | "volume";
export interface Operand {
  price?: PriceField;
  value?: number;
  ind?: string;
  ref?: string;
  source?: PriceField;
  period?: number;
  std?: number;
  minutes?: number;
  bars_ago?: number;
  mult?: number;
}
export interface Condition {
  all?: Condition[];
  any?: Condition[];
  not?: Condition;
  left?: Operand;
  op?: string;
  right?: Operand;
  bars?: number;
}
export interface Spec {
  schema_version?: number;
  meta: { name: string; description?: string; tags?: string[] };
  universe: { symbols: string[] };
  timeframe: "1m" | "5m" | "15m" | "60m" | "1d";
  indicators: ({ id: string; kind: string } & Operand)[];
  entry: {
    long: Condition | null;
    short: Condition | null;
    time_from: string;
    time_to: string;
    weekdays: number[];
    cooldown_bars: number;
  };
  exits: {
    stop: { kind: "atr" | "pct"; value: number; atr_period?: number };
    target: { kind: "r" | "pct" | "none"; value?: number };
    breakeven_r: number | null;
    trail: { kind: "owner" | "atr" | "none"; trigger_r?: number; atr_mult?: number };
    eod_time: string | null;
    max_hold_bars: number | null;
  };
  sizing: { risk_pct: number; max_position_pct: number; max_positions: number };
  limits: { daily_loss_pct: number; max_trades_per_day: number; owner_cooldown: boolean };
  costs: { slippage_pct: number; model: string };
}

export interface ValidateResult {
  valid: boolean;
  errors: FieldError[];
  warnings: FieldError[];
  spec: Spec;
  spec_hash: string | null;
  explain: string[];
  live_gate: LiveGate | null;
}

export interface PreviewResult {
  symbol: string;
  timeframe?: string;
  bars: { t: string; o: number; h: number; l: number; c: number }[];
  signals: { t: string; fill_t: string | null; side: "BUY" | "SHORT"; price: number; stop: number; target: number | null }[];
  stats: Record<string, unknown> | null;
  downsampled?: boolean;
  source?: string;
  message?: string;
}

export interface Universe {
  intraday: { symbols: string[]; indices: string[]; first: string | null; last: string | null; source: string };
  daily: { symbol_count: number; first: string | null; last: string | null; source: string };
  research_oos: { from: string; to: string };
}

const BASE = "/strategies";
export const TERMINAL_BT: BacktestStatus[] = ["done", "failed", "cancelled"];

export function apiErrors(err: unknown): { detail: string; errors: FieldError[] } {
  const e = err as { response?: { data?: { detail?: string; errors?: FieldError[] } }; message?: string };
  return {
    detail: e?.response?.data?.detail ?? e?.message ?? "request failed",
    errors: e?.response?.data?.errors ?? [],
  };
}

export const useTemplates = () =>
  useQuery({
    queryKey: ["strategy-templates"],
    queryFn: () => api.get<Template[]>(`${BASE}/templates/`).then((r) => r.data),
    staleTime: 5 * 60_000,
  });

export const useStrategies = (archived = false) =>
  useQuery({
    queryKey: ["strategies", archived ? "archived" : "active"],
    queryFn: () =>
      api.get<StrategyRow[]>(`${BASE}/instances/${archived ? "?archived=1" : ""}`).then((r) => r.data),
  });

export const useStrategy = (id: string | undefined) =>
  useQuery({
    queryKey: ["strategy", id],
    enabled: !!id,
    queryFn: () => api.get<StrategyDetail>(`${BASE}/instances/${id}/`).then((r) => r.data),
  });

export const useVersions = (id: string | undefined) =>
  useQuery({
    queryKey: ["strategy-versions", id],
    enabled: !!id,
    queryFn: () => api.get<StrategyVersion[]>(`${BASE}/instances/${id}/versions/`).then((r) => r.data),
  });

export const useBacktests = (instanceId: string | undefined) =>
  useQuery({
    queryKey: ["backtests", instanceId],
    enabled: !!instanceId,
    queryFn: () => api.get<BacktestSummary[]>(`${BASE}/backtests/?instance=${instanceId}`).then((r) => r.data),
    refetchInterval: (q) =>
      (q.state.data ?? []).some((b) => !TERMINAL_BT.includes(b.status)) ? 3000 : false,
  });

export const useBacktest = (id: string | null | undefined) =>
  useQuery({
    queryKey: ["backtest", id],
    enabled: !!id,
    queryFn: () => api.get<Backtest>(`${BASE}/backtests/${id}/`).then((r) => r.data),
    refetchInterval: (q) => (q.state.data && !TERMINAL_BT.includes(q.state.data.status) ? 2000 : false),
  });

export const useBacktestTrades = (id: string | null | undefined, enabled = true) =>
  useQuery({
    queryKey: ["backtest-trades", id],
    enabled: !!id && enabled,
    queryFn: () => api.get<BacktestTrade[]>(`${BASE}/backtests/${id}/trades/`).then((r) => r.data),
  });

export const useUniverse = () =>
  useQuery({
    queryKey: ["strategy-universe"],
    queryFn: () => api.get<Universe>(`${BASE}/universe/`).then((r) => r.data),
    staleTime: 30 * 60_000,
  });

function useInvalidate() {
  const qc = useQueryClient();
  return (id?: string) => {
    qc.invalidateQueries({ queryKey: ["strategies"] });
    if (id) {
      qc.invalidateQueries({ queryKey: ["strategy", id] });
      qc.invalidateQueries({ queryKey: ["strategy-versions", id] });
      qc.invalidateQueries({ queryKey: ["backtests", id] });
    }
  };
}

export interface CreateBody {
  name: string;
  description?: string;
  template_key?: string;
  spec?: Spec;
  strategy_name?: string;
  params?: Record<string, unknown>;
}

export function useCreateStrategy() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: CreateBody) => api.post<StrategyDetail>(`${BASE}/instances/`, body).then((r) => r.data),
    onSuccess: (d) => inv(d.id),
  });
}

export function useRenameStrategy(id: string) {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: { name?: string; description?: string }) =>
      api.patch<StrategyDetail>(`${BASE}/instances/${id}/`, body).then((r) => r.data),
    onSuccess: () => inv(id),
  });
}

export function useSaveVersion(id: string) {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (body: { spec?: Spec; params?: Record<string, unknown>; note?: string }) =>
      api
        .post<{ created: boolean; instance: StrategyDetail; version: StrategyVersion }>(
          `${BASE}/instances/${id}/versions/`, body)
        .then((r) => r.data),
    onSuccess: () => inv(id),
  });
}

export function useStrategyAction() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: ({ id, action }: { id: string; action: "duplicate" | "archive" | "unarchive" | "delete" }) =>
      action === "delete"
        ? api.delete(`${BASE}/instances/${id}/`).then(() => null)
        : api.post<StrategyDetail>(`${BASE}/instances/${id}/${action}/`, {}).then((r) => r.data),
    onSuccess: (_d, v) => inv(v.id),
  });
}

export interface RunBody {
  instance: string;
  from_date: string;
  to_date: string;
  symbols?: string[] | null;
  capital?: number;
  slippage_pct?: number | null;
  version?: number;
}

export function useRunBacktest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RunBody) => api.post<Backtest>(`${BASE}/backtests/`, body).then((r) => r.data),
    onSuccess: (bt) => {
      qc.invalidateQueries({ queryKey: ["backtests", bt.instance_id] });
      qc.invalidateQueries({ queryKey: ["strategy", bt.instance_id] });
      qc.invalidateQueries({ queryKey: ["strategies"] });
    },
  });
}

export function useCancelBacktest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.post<Backtest>(`${BASE}/backtests/${id}/cancel/`, {}).then((r) => r.data),
    onSuccess: (bt) => qc.invalidateQueries({ queryKey: ["backtest", bt.id] }),
  });
}

export function useValidateSpec() {
  return useMutation({
    mutationFn: (spec: Spec) => api.post<ValidateResult>(`${BASE}/spec/validate/`, { spec }).then((r) => r.data),
  });
}

export function usePreviewSignals() {
  return useMutation({
    mutationFn: (body: { spec: Spec; symbol: string; from_date: string; to_date: string }) =>
      api.post<PreviewResult>(`${BASE}/spec/preview-signals/`, body).then((r) => r.data),
  });
}

export const STAGE_TONE: Record<Stage, "neutral" | "info" | "warning" | "success"> = {
  draft: "neutral",
  backtested: "info",
  paper: "warning",
  live: "success",
};

export const BACKTEST_LABEL: Record<Backtestability, string> = {
  backtest: "Backtestable",
  forward_paper_only: "Forward paper test only",
  unavailable: "Not backtestable",
};
