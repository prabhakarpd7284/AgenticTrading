/**
 * Nightly scan — data hooks for /nightly-scan.
 *
 * Backend (read-only):
 *   GET /api/v1/nightly-scan/latest/          newest finished scan + per-stock rows
 *   GET /api/v1/nightly-scan/<YYYY-MM-DD>/    scan governing that trading day
 *   GET /api/v1/nightly-scan/                 recent scans
 *   GET /api/v1/trading-rules/cooldown/       losing-streak cooldown per portfolio
 *
 * The scan runs once a night (21:00 IST), so nothing here polls hard.
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export type SentimentLabel =
  | "strong_positive" | "positive" | "neutral" | "negative" | "strong_negative" | "unscored";
export type Allowed = "both" | "long" | "short" | "none";

export interface ScanHeadline {
  id: number;
  title: string;
  url: string;
  source: string;
  publisher: string;
  kind: string;
  published_at: string;
  score: number | null;
  materiality: string | null;
  confidence?: number | null;
}

export interface ScanLevels {
  source?: string;
  data_date?: string;
  stale?: boolean;
  floor?: Record<string, number>;
  cpr?: { pp: number; tc: number; bc: number; width_pct: number; class: string };
  camarilla?: Record<string, number>;
}

export interface ScanRow {
  symbol: string;
  data_date: string | null;
  levels: ScanLevels;
  cpr_width_pct: number | null;
  cpr_class: string;
  two_day_relationship: string;
  camarilla_relationship: string;
  bias: string;
  sentiment_score: number | null;
  sentiment_label: SentimentLabel;
  sentiment_weight: number | null;
  headline_count: number;
  scored_count: number;
  event_tomorrow: boolean;
  event_purpose: string;
  allow_long: boolean;
  allow_short: boolean;
  allowed: Allowed;
  reasons: string[];
  headlines: ScanHeadline[];
}

export interface SourceStatus {
  ok?: boolean;
  seen?: number;
  new?: number;
  requests?: number;
  failures?: number;
  errors?: string[];
  secs?: number;
  error?: string;
  [k: string]: unknown;
}

export interface ScanSummary {
  universe?: number;
  pivot_source?: string;
  market?: { score: number | null; label: string; headlines: number; scored: number; breadth: number };
  indices?: Record<string, { cpr_class: string; cpr_width_pct: number; relationship: string; bias: string }>;
  headlines_total?: number;
  headlines_scored?: number;
  sentiment_available?: boolean;
  skip?: { symbol: string; allowed: Allowed; reasons: string[] }[];
  unscored?: string[];
  degraded?: string[];
}

export interface NightlyScan {
  id: number;
  for_date: string;
  session_date: string;
  status: "running" | "success" | "partial" | "failed";
  started_at: string | null;
  finished_at: string | null;
  timings: Record<string, number>;
  sources: Record<string, SourceStatus>;
  summary: ScanSummary;
  telegram_sent: boolean;
  error: string;
  rows: ScanRow[];
}

export interface ScanResponse {
  scan: NightlyScan | null;
  running: boolean;
  thresholds: Record<string, number>;
}

export interface CooldownRow {
  portfolio_id: string;
  portfolio: string;
  mode: string;
  as_of: string;
  status: "normal" | "cooldown" | "reduced";
  loss_streak: number;
  trigger_date: string | null;
  cooldown_until: string | null;
  risk_multiplier: number;
  reason: string;
  history: { date: string; net_pnl: number; kind: string; in_cooldown: boolean }[];
}

export interface CooldownResponse {
  as_of: string;
  results: CooldownRow[];
  rules: { loss_days: number; cooldown_trading_days: number; reduced_risk_multiplier: number; costs_deducted: boolean };
}

export const RUN_SCAN_COMMAND = "cd backend && .venv/bin/python manage.py run_nightly_scan --no-telegram";

export function useNightlyScan(forDate?: string) {
  return useQuery({
    queryKey: ["nightly-scan", forDate ?? "latest"],
    queryFn: async () =>
      (await api.get<ScanResponse>(`/nightly-scan/${forDate ? `${forDate}/` : "latest/"}`)).data,
    staleTime: 60_000,
    refetchInterval: (q) => (q.state.data?.running ? 30_000 : false),
  });
}

export function useCooldownStatus() {
  return useQuery({
    queryKey: ["trading-rules", "cooldown"],
    queryFn: async () => (await api.get<CooldownResponse>("/trading-rules/cooldown/")).data,
    staleTime: 60_000,
  });
}
