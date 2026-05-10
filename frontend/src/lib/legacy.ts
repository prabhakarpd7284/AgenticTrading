/**
 * Typed client + React-Query hooks for the `/api/v1/legacy/` bridge
 * endpoints exposed by `apps.legacy.api`.
 *
 * These mirror what `dashboard_utils.data_layer` returns and let the
 * React UI render the *real* sqlite data (113 trades, 14 straddles,
 * 15 portfolio snapshots, 398 audit-log entries) right now, while the
 * v2 schema migration is still in progress.
 *
 * Once the native v2 endpoints (`/portfolios/`, `/orders/`, …) are
 * fully populated, individual hooks here can be deprecated in favor
 * of the v2 ones — call sites only need to swap the hook name.
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { legacyApi as api } from "./api";

/* --------------------------------------------------------------- */
/* Types — match the JSON shapes from data_layer.py                */
/* --------------------------------------------------------------- */
export interface LegacyPortfolio {
  capital: number;
  invested: number;
  available_cash: number;
  daily_pnl: number;
  total_pnl: number;
  daily_loss: number;
  open_positions: number;
  snapshot_date: string;
  straddle_count: number;
  straddle_pnl: number;
  straddle_premium_sold: number;
  today_trades: number;
  today_wins: number;
  today_losses: number;
  combined_pnl: number;
  combined: { equity_pnl: number; options_pnl: number; total_pnl: number };
}

export interface LegacyEquityPosition {
  id: number;
  symbol: string;
  side: "BUY" | "SELL";
  entry_price: number;
  stop_loss: number;
  target: number;
  quantity: number;
  pnl: number | null;
  status: string;
  confidence: number;
  fill_price: number | null;
}

export interface LegacyOptionPosition {
  id: number;
  underlying: string;
  strike: number;
  ce_strike: number;
  pe_strike: number;
  expiry: string;
  lots: number;
  lot_size: number;
  ce_sell: number;
  pe_sell: number;
  ce_current: number;
  pe_current: number;
  net_delta: number;
  pnl_inr: number;
  realized_pnl: number;
  unrealized_pnl: number;
  status: string;
  dte: number;
}

export interface LegacyPositions {
  equity: LegacyEquityPosition[];
  options: LegacyOptionPosition[];
}

export interface LegacyTrade {
  id: number;
  trade_date: string;
  symbol: string;
  side: "BUY" | "SELL";
  status: string;
  entry_price: number;
  stop_loss: number;
  target: number;
  quantity: number;
  fill_price: number | null;
  pnl: number | null;
  confidence: number;
  reasoning: string;
  exit_reason: string;
  created_at: string;
}

export interface LegacyAuditEntry {
  time: string;
  type: string;
  symbol: string;
  detail: string;
}

export interface LegacyRisk {
  capital: number;
  daily_loss: number;
  daily_loss_pct: number;
  max_daily_loss: number;
  daily_loss_limit_pct: number;
  capital_deployed: number;
  capital_deployed_pct: number;
  max_position_value: number;
  open_positions: number;
  max_open_positions: number;
  underwater_options: number;
  active_straddles: number;
  options_margin_exposure: number;
  total_exposure: number;
  total_exposure_pct: number;
  status: "GREEN" | "YELLOW" | "RED";
}

export interface LegacyAlert {
  severity: "info" | "warning" | "critical";
  message: string;
  action: string;
}

export interface LegacySystem {
  ai_paused: boolean;
  is_market_open: boolean;
  trading_mode: "paper" | "live";
  session: {
    is_open: boolean;
    is_weekday: boolean;
    current_time: string;
    market_open: string;
    market_close: string;
    elapsed_minutes: number;
    remaining_minutes: number;
    progress_pct: number;
    session_phase: string;
  };
}

export interface LegacyStrategy {
  id: number;
  name: string;
  category: string;
  description: string;
  rules: string;
  created_at: string;
}

export interface LegacyStraddle {
  id: number;
  underlying: string;
  strike: number;
  expiry: string;
  trade_date: string;
  status: string;
  lots: number;
  lot_size: number;
  ce_symbol: string;
  pe_symbol: string;
  ce_sell: number;
  pe_sell: number;
  ce_current: number;
  pe_current: number;
  premium_sold: number;
  pnl_inr: number;
  net_delta: number;
  action_taken: string;
}

/* --------------------------------------------------------------- */
/* Hooks                                                           */
/* --------------------------------------------------------------- */
const REFETCH_MS = 15_000; // refresh sqlite-backed views every 15s

export function useLegacyPortfolio() {
  return useQuery({
    queryKey: ["legacy", "portfolio"],
    queryFn: () => api.get<LegacyPortfolio>("/legacy/portfolio/").then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function useLegacyPositions() {
  return useQuery({
    queryKey: ["legacy", "positions"],
    queryFn: () => api.get<LegacyPositions>("/legacy/positions/").then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function useLegacyTrades(opts?: { limit?: number; symbol?: string }) {
  const params = new URLSearchParams();
  if (opts?.limit) params.set("limit", String(opts.limit));
  if (opts?.symbol) params.set("symbol", opts.symbol);
  const qs = params.toString();
  return useQuery({
    queryKey: ["legacy", "trades", opts],
    queryFn: () =>
      api
        .get<{ count: number; results: LegacyTrade[] }>(`/legacy/trades/${qs ? "?" + qs : ""}`)
        .then((r) => r.data),
  });
}

export function useLegacyAudit(limit = 25) {
  return useQuery({
    queryKey: ["legacy", "audit", limit],
    queryFn: () =>
      api
        .get<{ results: LegacyAuditEntry[] }>(`/legacy/audit/?limit=${limit}`)
        .then((r) => r.data.results),
    refetchInterval: REFETCH_MS,
  });
}

export function useLegacyRisk() {
  return useQuery({
    queryKey: ["legacy", "risk"],
    queryFn: () => api.get<LegacyRisk>("/legacy/risk/").then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function useLegacyAlerts() {
  return useQuery({
    queryKey: ["legacy", "alerts"],
    queryFn: () => api.get<{ results: LegacyAlert[] }>("/legacy/alerts/").then((r) => r.data.results),
    refetchInterval: REFETCH_MS,
  });
}

export function useLegacySystem() {
  return useQuery({
    queryKey: ["legacy", "system"],
    queryFn: () => api.get<LegacySystem>("/legacy/system/").then((r) => r.data),
    refetchInterval: 30_000,
  });
}

export function useLegacyStrategies() {
  return useQuery({
    queryKey: ["legacy", "strategies"],
    queryFn: () =>
      api
        .get<{ count: number; results: LegacyStrategy[] }>("/legacy/strategies/")
        .then((r) => r.data.results),
  });
}

export function useLegacyStraddles(status?: string) {
  return useQuery({
    queryKey: ["legacy", "straddles", status],
    queryFn: () =>
      api
        .get<{ count: number; results: LegacyStraddle[] }>(
          `/legacy/straddles/${status ? "?status=" + status : ""}`,
        )
        .then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

/* --------------------------------------------------------------- */
/* Mutations                                                       */
/* --------------------------------------------------------------- */
export function usePauseAi() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/legacy/ai/pause/").then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["legacy", "system"] }),
  });
}

export function useResumeAi() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/legacy/ai/resume/").then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["legacy", "system"] }),
  });
}
