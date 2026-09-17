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
  id: string;                    // Trade UUID (string) — keys the chart endpoint
  trade_date: string;
  symbol: string;
  side: "BUY" | "SELL";
  status: string;
  entry_price: number;
  stop_loss: number;
  target: number;
  quantity: number;
  fill_price: number | null;
  exit_price: number | null;     // null while still open
  exit_quantity: number | null;
  closed_at: string | null;      // ISO; null while open
  close_reason: string;          // SL_HIT | TARGET_HIT | EOD | TRAIL | MANUAL | ""
  pnl: number | null;
  confidence: number;
  reasoning: string;
  source: string;                // "swing" | "intraday" — drives chart interval
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

/** Count + most-recent of the `signal.fired` events for one symbol.
 *  Backs the Setup track-record "Screener fired N signals" line.  Reads
 *  `count` straight off DRF pagination, so the page size is irrelevant. */
export interface SymbolSignals {
  count: number;
  latest: { ts: string; side?: string; source?: string } | null;
}

export function useSymbolSignals(symbol: string | undefined) {
  return useQuery({
    queryKey: ["symbol-signals", symbol],
    enabled: !!symbol,
    queryFn: async () => {
      // The list endpoint is cursor-paginated (no `count`), so use the
      // dedicated count action which returns total + the latest match.
      const { data } = await api.get<{
        count: number;
        latest: { ts: string; type: string; payload: Record<string, unknown> | null } | null;
      }>(`/events/count/?type=signal.fired&symbol=${encodeURIComponent(symbol!)}`);
      const latest = data.latest
        ? {
            ts: data.latest.ts,
            side: (data.latest.payload?.side as string) ?? undefined,
            source: (data.latest.payload?.source as string) ?? undefined,
          }
        : null;
      return { count: data.count ?? 0, latest } as SymbolSignals;
    },
  });
}

/* ── Saved setups (Setup snapshots → setup.saved events) ──────────────── */

/** Body for POST /market-data/setup/ — snapshots the on-screen plan. */
export interface SetupSnapshotBody {
  symbol: string;
  side: "BUY" | "SELL";
  entry_price: number | null;
  stop_loss: number | null;
  target: number | null;
  quantity: number;
  confidence: number | null;
  risk_reward_ratio: number | null;
  risk_approved: boolean;
  generated_at: string;
}

/** A persisted setup.saved snapshot, flattened from the Event payload. */
export interface SavedSetup {
  id: number;
  saved_at: string;       // Event ts — when the operator clicked save
  generated_at: string;   // when the plan itself was computed (as_of)
  symbol: string;
  side: "BUY" | "SELL";
  entry_price: number | null;
  stop_loss: number | null;
  target: number | null;
  quantity: number;
  confidence: number | null;
  risk_reward_ratio: number | null;
  risk_approved: boolean;
}

export function useSaveSetupSnapshot() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: SetupSnapshotBody) =>
      api.post<{ id: number | null }>("/market-data/setup/", body).then((r) => r.data),
    onSuccess: (_d, body) =>
      qc.invalidateQueries({ queryKey: ["saved-setups", body.symbol] }),
  });
}

export function useSavedSetups(symbol: string | undefined) {
  return useQuery({
    queryKey: ["saved-setups", symbol],
    enabled: !!symbol,
    queryFn: async () => {
      // api.ts unwraps the cursor envelope → a bare EventDetail[].
      const { data } = await api.get<EventDetail[]>(
        `/events/?type=setup.saved&symbol=${encodeURIComponent(symbol!)}`,
      );
      return (data ?? []).map((e): SavedSetup => {
        const p = (e.payload ?? {}) as Record<string, unknown>;
        const num = (k: string) =>
          typeof p[k] === "number" ? (p[k] as number) : null;
        return {
          id: e.id,
          saved_at: e.ts,
          generated_at: (p.generated_at as string) || e.ts,
          symbol: (p.symbol as string) || symbol!,
          side: (p.side as "BUY" | "SELL") || "BUY",
          entry_price: num("entry_price"),
          stop_loss: num("stop_loss"),
          target: num("target"),
          quantity: typeof p.quantity === "number" ? (p.quantity as number) : 0,
          confidence: num("confidence"),
          risk_reward_ratio: num("risk_reward_ratio"),
          risk_approved: Boolean(p.risk_approved),
        };
      });
    },
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
/* Options chain + agent run trigger                                    */
/* ================================================================== */

export interface OptionQuote {
  token: string;
  symbol: string;
  strike: number;
  opt: "CE" | "PE";
  ltp: number;
  bid: number;
  ask: number;
  bid_qty: number;
  ask_qty: number;
  volume: number;
  oi: number;
  oi_change: number;
  iv: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  mid: number;
  spread_bps: number;
}

export interface OptionsChainRow {
  strike: number;
  ce: OptionQuote | null;
  pe: OptionQuote | null;
}

export interface AttemptedSource {
  name: string;            // "angel_one" | "fyers" | "zerodha" | "paper"
  ok: boolean;
  error?: string;
}

export interface OptionsChainSnapshot {
  underlying: string;
  spot: number;
  expiry: string;
  fetched_at: string | null;
  source: string;
  vix: number | null;
  pcr_oi: number | null;
  pcr_volume: number | null;
  atm_strike: number | null;
  rows: OptionsChainRow[];
  /** Audit trail of which adapters were tried, in order. The last `ok=true`
   *  entry is the actual data source. */
  attempted_sources: AttemptedSource[];
  /** True when no live broker chain was available and the response is
   *  synthesised by the PaperBrokerAdapter. Drives the "Connect a broker"
   *  call-to-action on the Options Desk. */
  is_fallback?: boolean;
}

export interface ExpiryRow {
  expiry: string;          // canonical DDMMMYYYY, e.g. "28MAY2026"
  dte: number;             // days to expiry, 0 on expiry day
  is_weekly: boolean;
  is_monthly: boolean;
  weekday: string;         // "Tuesday" / "Thursday" / etc
}

export interface ExpiriesPayload {
  underlying: string;
  count: number;
  expiries: ExpiryRow[];
}

export function useExpiries(underlying: string, limit = 20, enabled = true) {
  return useQuery({
    queryKey: ["expiries", underlying, limit],
    queryFn: () =>
      api.get<ExpiriesPayload>(`/market-data/expiries/?underlying=${encodeURIComponent(underlying)}&limit=${limit}`)
        .then((r) => r.data),
    enabled,
    staleTime: 5 * 60_000,   // master refreshes daily; cache for 5min
  });
}

export function useOptionsChain(params?: {
  underlying?: string;
  expiry?: string;
  strikes_window?: number;
  source?: "broker" | "paper";
  enabled?: boolean;
}) {
  const q = new URLSearchParams();
  q.set("underlying", params?.underlying ?? "NIFTY");
  if (params?.expiry) q.set("expiry", params.expiry);
  if (params?.strikes_window) q.set("strikes_window", String(params.strikes_window));
  if (params?.source) q.set("source", params.source);
  return useQuery({
    queryKey: ["options-chain", q.toString()],
    queryFn: () =>
      api.get<OptionsChainSnapshot>(`/market-data/options-chain/?${q.toString()}`)
        .then((r) => r.data),
    enabled: params?.enabled ?? true,
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

export interface AgentRunCreatePayload {
  strategy_name: string;
  portfolio: string;
  config: Record<string, unknown>;
}

export interface AgentRunRow {
  id: string;
  strategy_name: string;
  strategy_version: string;
  status: string;
  config: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export function useStartAgentRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: AgentRunCreatePayload) =>
      api.post<AgentRunRow>("/agents/runs/", payload).then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent-runs"] });
      qc.invalidateQueries({ queryKey: ["audit"] });
    },
  });
}

/** Mirror of apps.trading.models.Portfolio. No `is_default` flag exists
 *  in the schema — pick the first portfolio (the default created at
 *  onboarding) or one matching the requested `mode`. */
export interface PortfolioRow {
  id: string;
  name: string;
  capital: number;
  used_capital: number;
  realized_pnl: number;
  day_pnl: number;
  mode: "paper" | "live";
  broker_link: string | null;
}

export function usePortfolios() {
  return useQuery({
    queryKey: ["portfolios"],
    /* Axios response interceptor (lib/api.ts) already unwraps DRF's
     * paginated envelope `{results: [...]}` into a plain array, so the
     * shape here is always `PortfolioRow[]`. */
    queryFn: () =>
      api.get<PortfolioRow[]>("/portfolios/").then((r) =>
        Array.isArray(r.data) ? r.data : [],
      ),
    staleTime: 60_000,
  });
}

/** Pick the portfolio to use for a new trade. Strategy:
 *   1. prefer paper-mode portfolio (default for new traders)
 *   2. fall back to the first portfolio
 *   3. return null if none exist
 */
export function pickDefaultPortfolio(
  portfolios: PortfolioRow[] | undefined,
  prefer: "paper" | "live" = "paper",
): PortfolioRow | null {
  if (!portfolios || portfolios.length === 0) return null;
  return portfolios.find((p) => p.mode === prefer) ?? portfolios[0];
}

export interface BrokerLinkRow {
  id: string;
  broker_name: string;     // "angel_one" | "zerodha" | "fyers"
  display_name: string;
  is_default: boolean;
  status: "active" | "expired" | "disabled" | "errored";
  last_refreshed_at: string | null;
  last_error: string;
  credential_meta?: Record<string, unknown>;
}

export function useBrokerLinks() {
  return useQuery({
    queryKey: ["broker-links"],
    queryFn: () =>
      api.get<BrokerLinkRow[]>("/brokers/").then((r) =>
        Array.isArray(r.data) ? r.data : [],
      ),
    staleTime: 60_000,
  });
}

/** Parse a DRF error payload into a human-readable string + per-field map.
 *
 * Handles every shape we see from `/agents/runs/` POST:
 *   - `"detail"` (string)
 *   - `{"field": ["msg", ...]}` (validation errors)
 *   - `{"config": [{"path": [...], "message": "..."}]}` (JSONSchema validator)
 *   - bare string body
 *   - axios network/timeout errors (no response)
 *
 * Returns `{message, fields}` where `message` is always non-empty and
 * `fields` maps `dotted.path` → `array of human-readable strings`.
 */
export function parseDrfError(
  err: unknown,
): { message: string; fields: Record<string, string[]> } {
  // Network / timeout — axios put the error on err.message
  const ax = err as { response?: { data?: unknown; status?: number }; message?: string; code?: string };
  if (ax.code === "ERR_NETWORK") {
    return { message: "Network error — backend unreachable", fields: {} };
  }
  if (ax.code === "ECONNABORTED") {
    return { message: "Request timed out", fields: {} };
  }
  const body = ax.response?.data;
  if (body == null) {
    return { message: ax.message ?? "Unknown error", fields: {} };
  }
  if (typeof body === "string") {
    return { message: body, fields: {} };
  }
  if (typeof body !== "object") {
    return { message: String(body), fields: {} };
  }
  const obj = body as Record<string, unknown>;
  // Plain DRF `{"detail": "..."}` envelope
  if (typeof obj.detail === "string") {
    return { message: obj.detail, fields: {} };
  }
  // JSONSchema-validator envelope: {"config": [{path:[], message:""}, ...]}
  const fields: Record<string, string[]> = {};
  const messages: string[] = [];
  for (const [k, v] of Object.entries(obj)) {
    if (Array.isArray(v)) {
      for (const item of v) {
        if (typeof item === "string") {
          fields[k] = [...(fields[k] ?? []), item];
          messages.push(`${k}: ${item}`);
        } else if (item && typeof item === "object") {
          const ent = item as { path?: unknown[]; message?: string };
          const path = [k, ...(ent.path ?? [])].filter(Boolean).join(".");
          const msg = ent.message ?? JSON.stringify(item);
          fields[path] = [...(fields[path] ?? []), msg];
          messages.push(`${path}: ${msg}`);
        }
      }
    } else if (typeof v === "string") {
      fields[k] = [v];
      messages.push(`${k}: ${v}`);
    }
  }
  return {
    message: messages.length ? messages.join(" · ") : "Request rejected",
    fields,
  };
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

/* ── Pine Script export (generated from screener strategies) ──────────── */

export interface PineStrategy {
  key: string;
  label: string;
  description: string;
  side: string;
}

/** Enabled screener strategies that can be exported as a TradingView Pine v5
 *  indicator. Long staleTime — the set only changes on backend deploy. */
export function usePineStrategies() {
  return useQuery({
    queryKey: ["pine-strategies"],
    queryFn: () => api
      .get<PineStrategy[]>("/notifications/tradingview/pine-strategies/")
      .then((r) => (Array.isArray(r.data) ? r.data : [])),
    staleTime: 5 * 60_000,
  });
}

/** Generated Pine v5 source for one strategy. When `linkId` is supplied the
 *  script's comment header carries that link's webhook URL (owner-scoped on
 *  the backend — another tenant's secret is never embedded). */
export function usePineScript(strategy: string | undefined, linkId?: string) {
  return useQuery({
    queryKey: ["pine-script", strategy, linkId],
    queryFn: () => {
      const qs = new URLSearchParams({ strategy: strategy! });
      if (linkId) qs.set("link", linkId);
      return api
        .get<{ strategy: string; code: string }>(
          `/notifications/tradingview/pine/?${qs.toString()}`,
        )
        .then((r) => r.data);
    },
    enabled: !!strategy,
    staleTime: 5 * 60_000,
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
