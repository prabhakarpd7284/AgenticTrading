/**
 * Mutual funds — data hooks for /mutual-funds.
 *
 * Backend: /api/v1/mutual-funds/ (apps.mutual_funds.api.views)
 *   GET  holdings/            tenant holdings valued at latest NAV + summary
 *   GET  schemes/?q=          AMFI catalog search (limit/offset paginated)
 *   GET  schemes/<code>/      scheme + NAV history (?days=)
 *   GET  advisor/findings/    latest advisor run's findings
 *   POST advisor/run/         run the advisor now (owner only)
 *   GET  status/              NAV freshness + stale count for held schemes
 *   GET  screener/?category=  within-category ranking + component scores (MF-S)
 *   GET  schemes/<code>/metrics/      latest metric snapshot + data issues
 *   GET  schemes/<code>/nav-history/  NAV series (+ benchmark TRI, rebased)
 *
 * NAVs only change once a day, so nothing here polls aggressively.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Types — mirror the backend payloads                                 */
/* ------------------------------------------------------------------ */

export interface MFHoldingRow {
  id: number;
  portfolio_id: string;
  portfolio_name: string;
  folio_number: string;
  source: string;
  scheme_code: string;
  scheme_name: string;
  amc_name: string;
  plan: string;
  option: string;
  category: string;
  is_direct: boolean;
  units: number;
  avg_cost_per_unit: number | null;
  invested: number | null;
  latest_nav: number | null;
  nav_date: string | null;
  nav_is_stale: boolean;
  current_value: number | null;
  gain: number | null;
  gain_pct: number | null;
}

export interface MFAllocationRow {
  category: string;
  value: number;
  pct: number;
}

export interface MFHoldingsSummary {
  holding_count: number;
  invested: number;
  current_value: number;
  gain: number;
  gain_pct: number | null;
  gain_basis_invested: number;
  unvalued_count: number;
  missing_cost_count: number;
  allocation: MFAllocationRow[];
}

export interface MFHoldingsPayload {
  results: MFHoldingRow[];
  summary: MFHoldingsSummary;
  include_closed: boolean;
}

export interface MFStatus {
  /** Date most schemes are priced at (liquid funds forward-date weekend NAVs). */
  latest_nav_date: string | null;
  /** Raw MAX(nav_date) — can be in the future for liquid/overnight funds. */
  max_nav_date: string | null;
  last_pull_at: string | null;
  stale_cutoff_date: string;
  stale_after_trading_days: number;
  catalog_is_stale: boolean;
  scheme_count: number;
  held_scheme_count: number;
  stale_holdings_count: number;
  stale_holdings: { scheme_code: string; scheme_name: string; latest_nav_date: string | null }[];
}

export interface MFFinding {
  id?: number;
  ts?: string;
  kind: string;
  severity: string;
  text: string;
  is_estimate: boolean;
  payload: Record<string, unknown>;
}

export interface MFFindingsPayload {
  latest_run_at: string | null;
  count: number;
  results: MFFinding[];
}

export interface MFScheme {
  scheme_code: string;
  scheme_name: string;
  amc_name: string;
  plan: string;
  option: string;
  category: string;
  category_label: string;
  is_direct: boolean;
  isin_growth: string;
  isin_div_reinvestment: string;
  latest_nav: string | null;
  latest_nav_date: string | null;
}

export interface MFSchemeSearchPayload {
  count: number;
  next: string | null;
  previous: string | null;
  results: MFScheme[];
}

export interface MFSchemeDetail {
  scheme: MFScheme;
  days: number;
  navs: { date: string; nav: number }[];
  held_units: number;
}

/* ------------------------------------------------------------------ */
/* Hooks                                                               */
/* ------------------------------------------------------------------ */

const BASE = "/mutual-funds";

export function useMFHoldings(includeClosed = false) {
  return useQuery<MFHoldingsPayload>({
    queryKey: ["mf", "holdings", includeClosed],
    queryFn: async () =>
      (await api.get<MFHoldingsPayload>(`${BASE}/holdings/`, {
        params: includeClosed ? { include_closed: 1 } : undefined,
      })).data,
    staleTime: 60_000,
  });
}

export function useMFStatus() {
  return useQuery<MFStatus>({
    queryKey: ["mf", "status"],
    queryFn: async () => (await api.get<MFStatus>(`${BASE}/status/`)).data,
    staleTime: 60_000,
  });
}

export function useMFFindings() {
  return useQuery<MFFindingsPayload>({
    queryKey: ["mf", "findings"],
    queryFn: async () => (await api.get<MFFindingsPayload>(`${BASE}/advisor/findings/`)).data,
    staleTime: 60_000,
  });
}

export function useRunMFAdvisor() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => (await api.post(`${BASE}/advisor/run/`, {})).data as { count: number; notes: string[] },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["mf", "findings"] }),
  });
}

export function useMFSchemeSearch(q: string) {
  const term = q.trim();
  return useQuery<MFSchemeSearchPayload>({
    queryKey: ["mf", "schemes", term],
    queryFn: async () =>
      (await api.get<MFSchemeSearchPayload>(`${BASE}/schemes/`, { params: { q: term, limit: 20 } })).data,
    enabled: term.length >= 2,
    staleTime: 5 * 60_000,
  });
}

export function useMFSchemeDetail(code: string | null, days = 365) {
  return useQuery<MFSchemeDetail>({
    queryKey: ["mf", "scheme", code, days],
    queryFn: async () =>
      (await api.get<MFSchemeDetail>(`${BASE}/schemes/${encodeURIComponent(code ?? "")}/`, {
        params: { days },
      })).data,
    enabled: !!code,
    staleTime: 5 * 60_000,
  });
}

/* ------------------------------------------------------------------ */
/* Screener (MF-S)                                                     */
/* ------------------------------------------------------------------ */

export type ScoreComponent = "consistency" | "risk_adjusted" | "drawdown" | "cost" | "size_age";

export interface MFScreenRow {
  rank: number;
  scheme_code: string;
  scheme_name: string;
  amc_name: string;
  score: number | null;
  partial_score: boolean;
  components: Record<ScoreComponent, number | null>;
  cagr_1y: number | null;
  cagr_3y: number | null;
  cagr_5y: number | null;
  rolling_3y_median: number | null;
  rolling_3y_min: number | null;
  rolling_3y_pct_above_cat_median: number | null;
  rolling_3y_windows: number;
  volatility: number | null;
  max_drawdown: number | null;
  worst_1y: number | null;
  sharpe: number | null;
  sortino: number | null;
  excess_cagr_3y_vs_benchmark: number | null;
  benchmark_name: string;
  history_years: number | null;
  last_nav_date: string | null;
  quality_flags: string[];
  expense_ratio_pct: number | null;
  aum_cr: number | null;
}

export interface MFIndexCheck {
  category: string;
  basis: "index_fund" | "tri";
  reference: string;
  windows: number;
  verdict: "yes" | "no" | "mixed" | "unknown";
  text: string;
  pct_windows_median_active_beat?: number;
  pct_windows_recent_3y?: number | null;
  median_gap_pp?: number;
  funds_beating_reference_majority?: string;
  first_window_end?: string;
  last_window_end?: string;
  alternative?: { basis: "index_fund" | "tri"; windows: number; pct_windows_median_active_beat: number } | null;
}

export interface MFScreenCategoryResult {
  category: string;
  as_of: string | null;
  rows: MFScreenRow[];
  universe_count: number;
  eligible_count: number;
  excluded: Record<string, number>;
  weights_used: Partial<Record<ScoreComponent, number>>;
  notes: string[];
  index_check: MFIndexCheck | null;
}

export interface MFScreenerPayload {
  available_categories: { category: string; count: number; as_of: string }[];
  personal_use_note: string;
  metrics_as_of: string | null;
  result: MFScreenCategoryResult | null;
}

export interface MFScreenerParams {
  category: string | null;
  minYears: number;
  excludeDataIssues: boolean;
}

export function useMFScreener(p: MFScreenerParams) {
  return useQuery<MFScreenerPayload>({
    queryKey: ["mf", "screener", p.category, p.minYears, p.excludeDataIssues],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (p.category) params.set("category", p.category);
      params.set("min_years", String(p.minYears));
      if (p.excludeDataIssues) params.append("exclude_flag", "data_issue_in_3y_window");
      return (await api.get<MFScreenerPayload>(`${BASE}/screener/?${params.toString()}`)).data;
    },
    staleTime: 10 * 60_000,
  });
}

export interface MFNavHistory {
  scheme_code: string;
  scheme_name: string;
  category: string;
  is_active: boolean;
  count: number;
  first_date: string | null;
  last_date: string | null;
  navs: { date: string; nav: number }[];
  benchmark: { index_name: string; is_tri: boolean; points: { date: string; value: number }[] } | null;
}

export function useMFNavHistory(code: string | null, days = 1825) {
  return useQuery<MFNavHistory>({
    queryKey: ["mf", "nav-history", code, days],
    queryFn: async () =>
      (await api.get<MFNavHistory>(`${BASE}/schemes/${encodeURIComponent(code ?? "")}/nav-history/`, {
        params: { days, benchmark: 1 },
      })).data,
    enabled: !!code,
    staleTime: 10 * 60_000,
  });
}

export const COMPONENT_LABEL: Record<ScoreComponent, string> = {
  consistency: "Consistency",
  risk_adjusted: "Risk-adjusted",
  drawdown: "Drawdown",
  cost: "Cost (TER)",
  size_age: "Size / age",
};

export const FLAG_LABEL: Record<string, string> = {
  history_lt_3y: "<3y history",
  history_lt_5y: "<5y history",
  stale_nav: "Stale NAV",
  data_issue_in_3y_window: "Data issue (3y)",
  data_issue_in_history: "Data issue (old)",
  drawdown_not_recovered: "Not recovered",
  no_benchmark: "No benchmark",
  history_cut_at_nav_jump: "History cut at NAV jump",
};

/** Human label for an advisor finding kind. */
export const FINDING_KIND_LABEL: Record<string, string> = {
  direct_vs_regular: "Direct vs regular",
  allocation: "Allocation",
  overlap: "Overlap",
  sip_continuity: "SIP continuity",
  note: "Note",
};

/** The shell command shown in the empty state (working tenant / MF portfolio). */
export const CAS_IMPORT_COMMAND =
  'cd backend && CAS_PASSWORD=<PAN+DOB> .venv/bin/python manage.py import_cas_statement ' +
  '--file <cas.pdf> --tenant istpduckymomo --portfolio "Mutual Funds" --dry-run';
