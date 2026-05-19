/**
 * Typed React-Query client for the v2-native top-level endpoints that
 * absorbed the old /api/v1/legacy/* surface in the v1→v2 migration.
 *
 * Hook names dropped the "Legacy" prefix; shapes stay identical because
 * the underlying view functions are the same (apps.trading.api.
 * legacy_compat_views) — only the URL prefix changed.
 *
 * This file replaces the old `lib/legacy.ts`; once Wave 3 deletes that
 * file, the only thing left to do is to move these hooks into per-domain
 * files (lib/portfolios.ts, lib/system.ts, …) if the organisation grows.
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";

/* --------------------------------------------------------------- */
/* Types — match the JSON shapes from legacy_compat_views.py        */
/* --------------------------------------------------------------- */
export interface PortfolioSummary {
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

export interface EquityPosition {
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

export interface OptionPosition {
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

export interface PositionsOverview {
  equity: EquityPosition[];
  options: OptionPosition[];
}

export interface Trade {
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

export interface AuditEntry {
  /** Underlying Event PK — present in v2; absent on rows produced by older
   *  bridge code. Drives click-through to /events/{id}/ for detail. */
  id?: number;
  time: string;
  type: string;
  symbol: string;
  detail: string;
}

/** Full Event row as returned by /api/v1/events/{id}/. See backend
 *  EventSerializer (apps/events/api/views.py) for canonical field list. */
export interface EventDetail {
  id: number;
  ts: string;
  type: string;
  severity: "info" | "warn" | "error";
  actor_kind: string;
  actor_user: number | null;
  workflow_run: string | null;   // UUID of the AgentRun, if any
  step_name: string;
  trade_id: string | null;
  order: string | null;
  signal_id: number | null;
  payload: Record<string, unknown> | null;
  text: string;
  request_id: string;
}

export interface RiskOverview {
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

export interface RiskAlert {
  severity: "info" | "warning" | "critical";
  message: string;
  action: string;
}

export interface SystemStatus {
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

export interface KnowledgeDoc {
  id: number;
  title: string;       // (legacy.ts incorrectly called this "name")
  category: string;
  content: string;     // (legacy.ts incorrectly called this "description")
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface OptionsPositionRow {
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
const REFETCH_MS = 15_000;

export function usePortfolioSummary() {
  return useQuery({
    queryKey: ["portfolio-summary"],
    queryFn: () => api.get<PortfolioSummary>("/portfolios/summary/").then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function usePositions() {
  return useQuery({
    queryKey: ["positions"],
    queryFn: () => api.get<PositionsOverview>("/positions/").then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function useTrades(opts?: { limit?: number; symbol?: string }) {
  const params = new URLSearchParams();
  if (opts?.limit) params.set("limit", String(opts.limit));
  if (opts?.symbol) params.set("symbol", opts.symbol);
  const qs = params.toString();
  return useQuery({
    queryKey: ["trades", opts],
    queryFn: () =>
      api
        .get<{ count: number; results: Trade[] }>(`/trades/${qs ? "?" + qs : ""}`)
        .then((r) => r.data),
  });
}

export function useAuditFeed(limit = 25) {
  return useQuery({
    queryKey: ["audit", limit],
    queryFn: () =>
      api
        .get<{ results: AuditEntry[] }>(`/events/audit/?limit=${limit}`)
        .then((r) => r.data.results),
    refetchInterval: REFETCH_MS,
  });
}

/** Fetch a single Event row from /api/v1/events/{id}/.
 *  Enabled only when `id` is set — call sites pass `undefined` to suspend the
 *  fetch (e.g. while the detail dialog is closed). */
export function useEvent(id: number | undefined) {
  return useQuery({
    queryKey: ["event", id],
    queryFn: () => api.get<EventDetail>(`/events/${id}/`).then((r) => r.data),
    enabled: id != null,
    staleTime: 60_000,   // Event rows are append-only; cache aggressively.
  });
}

export function useRiskOverview() {
  return useQuery({
    queryKey: ["risk"],
    queryFn: () => api.get<RiskOverview>("/risk/").then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function useRiskAlerts() {
  return useQuery({
    queryKey: ["risk-alerts"],
    queryFn: () => api.get<{ results: RiskAlert[] }>("/risk/alerts/").then((r) => r.data.results),
    refetchInterval: REFETCH_MS,
  });
}

export function useSystemStatus() {
  return useQuery({
    queryKey: ["system-status"],
    queryFn: () => api.get<SystemStatus>("/system/").then((r) => r.data),
    refetchInterval: 30_000,
  });
}

export function useKnowledgeDocs() {
  return useQuery({
    queryKey: ["knowledge"],
    queryFn: () =>
      api
        .get<{ count: number; results: KnowledgeDoc[] }>("/rag/knowledge/")
        .then((r) => r.data.results),
  });
}

export function useOptionsPositions(status?: string) {
  return useQuery({
    queryKey: ["options-positions", status],
    queryFn: () =>
      api
        .get<{ count: number; results: OptionsPositionRow[] }>(
          `/options-positions/${status ? "?status=" + status : ""}`,
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
    mutationFn: () => api.post("/system/pause/").then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["system-status"] }),
  });
}

export function useResumeAi() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post("/system/resume/").then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["system-status"] }),
  });
}

/* ================================================================== */
/* TradingView webhook integration                                     */
/* ================================================================== */

export interface TradingViewLink {
  id: string;
  display_name: string;
  is_active: boolean;
  autofire_enabled: boolean;
  default_strategy_name: string;
  portfolio: string | null;
  allowed_actions: string[];
  /** Optional symbol allowlist gate — autofire only fires when the alert's
   *  symbol appears in this watchlist's resolved symbols. Orthogonal to
   *  allowed_actions (which gates BUY/SELL). */
  watchlist: string | null;
  webhook_secret: string;
  webhook_url: string;
  last_received_at: string | null;
  receive_count: number;
  last_error: string;
  created_at: string;
  updated_at: string;
}

export interface TradingViewSignalRow {
  id: number;
  received_at: string;
  parsed: Record<string, unknown>;
  parse_error: string;
  raw_payload: string;
  signal: number | null;
  workflow_run: string | null;
}

export interface TradingViewLinkUpsert {
  display_name?: string;
  is_active?: boolean;
  autofire_enabled?: boolean;
  default_strategy_name?: string;
  portfolio?: string | null;
  allowed_actions?: string[];
  watchlist?: string | null;
}

export function useTradingViewLinks() {
  return useQuery({
    queryKey: ["tradingview-links"],
    // NOTE: lib/api.ts has a global response interceptor that strips the DRF
    // pagination envelope ({next, previous, results}) down to a bare array
    // when both `next` and `previous` are present. So `r.data` IS the array.
    queryFn: () => api
      .get<TradingViewLink[]>("/notifications/tradingview/")
      .then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function useTradingViewRecent(id: string | undefined) {
  return useQuery({
    queryKey: ["tradingview-recent", id],
    queryFn: () => api
      .get<TradingViewSignalRow[]>(`/notifications/tradingview/${id}/recent/`)
      .then((r) => r.data),
    enabled: !!id,
    refetchInterval: REFETCH_MS,
  });
}

export function useCreateTradingViewLink() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: TradingViewLinkUpsert) =>
      api.post<TradingViewLink>("/notifications/tradingview/", body).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["tradingview-links"] }),
  });
}

export function useUpdateTradingViewLink() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: TradingViewLinkUpsert & { id: string }) =>
      api.patch<TradingViewLink>(`/notifications/tradingview/${id}/`, body).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["tradingview-links"] }),
  });
}

export function useRotateTradingViewSecret() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api.post<TradingViewLink>(`/notifications/tradingview/${id}/rotate-secret/`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["tradingview-links"] }),
  });
}

export function useDeleteTradingViewLink() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api.delete(`/notifications/tradingview/${id}/`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["tradingview-links"] }),
  });
}

/* ── Watchlists ───────────────────────────────────────────────────────── */

export type WatchlistKind =
  | "MANUAL"
  | "SIGNAL_RANK"
  | "SOURCE_HOT"
  | "RECENT_ACTIVE"
  | "TRADED_RECENTLY"
  | "SHORTLIST_TODAY";

/** UI-only metadata for each kind: labels and trader-facing blurbs. Default
 *  config values are NOT here — they come from /api/v1/watchlists/kinds/
 *  (see useWatchlistKinds) so backend resolvers and frontend forms can't
 *  drift on numerics like window_days / top_n. */
export const WATCHLIST_KIND_META: Record<WatchlistKind, {
  label: string;
  blurb: string;
  isAuto: boolean;
}> = {
  MANUAL: {
    label: "Manual",
    blurb: "You type the symbols. The list never changes unless you edit it.",
    isAuto: false,
  },
  SIGNAL_RANK: {
    label: "Top-N by signal count",
    blurb: "Most-active symbols across every signal source in the window.",
    isAuto: true,
  },
  SOURCE_HOT: {
    label: "Top-N for one source",
    blurb: "Same as Top-N, but pinned to one source (e.g. TradingView only).",
    isAuto: true,
  },
  RECENT_ACTIVE: {
    label: "Active in last N hours",
    blurb: "Every symbol that fired any signal recently.",
    isAuto: true,
  },
  TRADED_RECENTLY: {
    label: "Recently traded",
    blurb: "Symbols on real-money Trade rows in the last N days.",
    isAuto: true,
  },
  SHORTLIST_TODAY: {
    label: "Today's premarket shortlist",
    blurb: "The premarket scanner's output for today (Cascade Stage 4).",
    isAuto: true,
  },
};

/** Canonical default config per kind — fetched from the backend so resolver
 *  fallbacks and form initial values stay in sync. Long staleTime because
 *  defaults only change on backend deploy. */
export interface WatchlistKindMeta {
  kind: WatchlistKind;
  label: string;
  defaults: Record<string, unknown>;
}
export function useWatchlistKinds() {
  return useQuery({
    queryKey: ["watchlist-kinds"],
    queryFn: () => api.get<WatchlistKindMeta[]>("/watchlists/kinds/").then((r) => r.data),
    staleTime: 5 * 60_000,
  });
}

export interface Watchlist {
  id: string;
  name: string;
  description: string;
  kind: WatchlistKind;
  config: Record<string, unknown>;
  is_auto: boolean;
  symbols: string[];
  symbol_count: number;
  symbols_refreshed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface WatchlistUpsert {
  name?: string;
  description?: string;
  kind?: WatchlistKind;
  config?: Record<string, unknown>;
  symbols?: string[];
}

export function useWatchlists() {
  return useQuery({
    queryKey: ["watchlists"],
    // lib/api.ts strips DRF's {next, previous, results} envelope down to a
    // bare array — same shape contract as useTradingViewLinks.
    queryFn: () => api
      .get<Watchlist[]>("/watchlists/")
      .then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}

export function useCreateWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: WatchlistUpsert) =>
      api.post<Watchlist>("/watchlists/", body)
        .then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlists"] }),
  });
}

export function useUpdateWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: WatchlistUpsert & { id: string }) =>
      api.patch<Watchlist>(`/watchlists/${id}/`, body)
        .then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlists"] }),
  });
}

export function useDeleteWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api.delete(`/watchlists/${id}/`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlists"] }),
  });
}

export function useAddSymbolsToWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, symbols }: { id: string; symbols: string[] }) =>
      api.post<Watchlist>(
        `/watchlists/${id}/add-symbols/`,
        { symbols },
      ).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlists"] }),
  });
}

export function useRemoveSymbolsFromWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, symbols }: { id: string; symbols: string[] }) =>
      api.post<Watchlist>(
        `/watchlists/${id}/remove-symbols/`,
        { symbols },
      ).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlists"] }),
  });
}

/** On-demand re-resolve of an auto-kind watchlist. 400s for MANUAL kind. */
export function useRefreshWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api.post<Watchlist>(
        `/watchlists/${id}/refresh/`,
      ).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlists"] }),
  });
}

/** Which of the operator's watchlists contain a given symbol. Used by the
 *  Setup page to render "In: <list>, <list>" badges, and by any future
 *  surface that wants a "what am I tracking this for" backlink. */
export function useWatchlistsBySymbol(symbol: string | undefined) {
  return useQuery({
    queryKey: ["watchlists-by-symbol", symbol],
    queryFn: () => api
      .get<Watchlist[]>(
        `/watchlists/by-symbol/?symbol=${encodeURIComponent(symbol || "")}`,
      )
      .then((r) => r.data),
    enabled: !!symbol,
    staleTime: 30_000,
  });
}

/* ── Grouped signals (TradingView Manager view) ───────────────────────── */

export type SignalGroupBy = "symbol" | "strategy" | "source" | "day";

export interface GroupedSignalRow {
  key: string;
  count: number;
  buys: number;
  sells: number;
  latest_at: string | null;
  latest_action: string;
}

export interface GroupedSignalsResponse {
  by: SignalGroupBy;
  rows: GroupedSignalRow[];
  window_days: number;
}

export interface GroupedSignalDetailRow {
  id: number;
  signal_time: string;
  symbol: string;
  side: string;
  source: string;
  strategy: string;
  entry_price: number;
  stoploss: number;
  target: number;
  reasons: string[];
  indicators: Record<string, unknown>;
  trade_id: string | null;
  outcome: string;
}

export interface GroupedSignalDetailResponse {
  by: SignalGroupBy;
  key: string;
  rows: GroupedSignalDetailRow[];
}

/** Drill-in for a single bucket in the grouped-signals view. Gated on
 *  `key`/`by` being set so the underlying network call only fires when the
 *  detail panel is actually open. */
export function useGroupedSignalsDetail(params: {
  by?: SignalGroupBy;
  key?: string;
  days?: number;
  source?: string;
  watchlist?: string;
}) {
  const qs = new URLSearchParams();
  if (params.by)        qs.set("by", params.by);
  if (params.key)       qs.set("key", params.key);
  if (params.days)      qs.set("days", String(params.days));
  if (params.source)    qs.set("source", params.source);
  if (params.watchlist) qs.set("watchlist", params.watchlist);
  return useQuery({
    queryKey: ["tradingview-grouped-detail", params],
    queryFn: () => api
      .get<GroupedSignalDetailResponse>(
        `/notifications/tradingview/signals/detail/?${qs.toString()}`,
      )
      .then((r) => r.data),
    enabled: !!params.key && !!params.by,
    staleTime: 15_000,
  });
}

export function useGroupedSignals(params: {
  by?: SignalGroupBy;
  days?: number;
  source?: string;
  symbol?: string;
  watchlist?: string;
} = {}) {
  const qs = new URLSearchParams();
  if (params.by)        qs.set("by", params.by);
  if (params.days)      qs.set("days", String(params.days));
  if (params.source)    qs.set("source", params.source);
  if (params.symbol)    qs.set("symbol", params.symbol);
  if (params.watchlist) qs.set("watchlist", params.watchlist);
  const query = qs.toString();
  return useQuery({
    queryKey: ["tradingview-grouped", params],
    queryFn: () => api
      .get<GroupedSignalsResponse>(
        `/notifications/tradingview/signals/${query ? "?" + query : ""}`,
      )
      .then((r) => r.data),
    refetchInterval: REFETCH_MS,
  });
}
