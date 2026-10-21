/**
 * Run receipts — data hooks (UX redesign R1, plan §8).
 *
 *   GET  /api/v1/runs/receipts/            list (kind, status, name, date, celery_task_id, limit)
 *   GET  /api/v1/runs/receipts/<id>/       one receipt + previous run of the same job
 *   GET  /api/v1/runs/schedule/            today's expected-schedule slots + states
 *   GET  /api/v1/runs/status/              mode · data status · job health (top bar)
 *   POST /api/v1/runs/rerun/<name>/        owner-only re-trigger (returns celery_task_id)
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export type RunStatus =
  | "queued" | "running" | "success" | "partial" | "failed" | "not_run" | "skipped";
export type CheckStatus = "pass" | "warn" | "fail" | "info";

export interface RunCheck {
  name: string;
  status: CheckStatus;
  message: string;
  [k: string]: unknown;
}

export interface RunLink { label: string; href: string }

export interface RunReceipt {
  id: string;
  kind: string;
  name: string;
  scope: "system" | "tenant";
  task_name: string;
  schedule_key: string;
  expected_at: string | null;
  trigger: "beat" | "manual" | "api" | "watcher";
  status: RunStatus;
  started_at: string;
  finished_at: string | null;
  duration_seconds: number | null;
  outputs: Record<string, unknown> & { links?: RunLink[] };
  warnings: string[];
  error: string;
  check_counts: Record<CheckStatus, number>;
  detail_kind: string;
  detail_id: string;
  rerunnable: boolean;
  celery_task_id: string;
}

export interface RunReceiptDetail extends RunReceipt {
  inputs: Record<string, unknown>;
  checks: RunCheck[];
  data_provenance: Record<string, unknown>;
  previous: RunReceipt | null;
  detail: { kind: string; id: number | string; status: string; href: string; [k: string]: unknown } | null;
  rerun_note?: string | null;
}

export type SlotState = "ok" | "warn" | "failed" | "missed" | "running" | "pending" | "untracked";

export interface ScheduleSlot {
  key: string;
  label: string;
  task: string;
  receipt_name: string | null;
  due_at: string;
  window_end: string;
  schedule: string;
  rerunnable: boolean;
  state: SlotState;
  reason?: string;
  receipt: {
    id: string; status: RunStatus; started_at: string; finished_at: string | null;
    duration_seconds: number | null; trigger: string; warn_checks: number; error: string;
  } | null;
}

export interface JobSummary {
  ok: number; total: number; warn: number; failed: number; missed: number;
  pending: number; untracked: number; health: "green" | "amber" | "red" | "grey"; label: string;
}

export interface ScheduleResponse {
  date: string;
  slots: ScheduleSlot[];
  summary: JobSummary;
  jobs: { key: string; label: string; task: string; schedule: string; tracked: boolean;
          receipt_name: string | null; grace_minutes: number; rerunnable: boolean }[];
}

export interface CandleCheck extends RunCheck { latest?: string; sessions_behind?: number }

export interface RunsStatus {
  as_of: string;
  mode: { mode: "PAPER" | "LIVE" | "HALT"; reason: string; env_mode: string };
  data: {
    state: "live" | "stale" | "synthetic";
    reason: string;
    market_open: boolean;
    brokers: { broker: string; status: string; last_refreshed_at: string | null; last_error: string }[];
    candles: { "1m": CandleCheck; "1d": CandleCheck };
  };
  jobs: JobSummary & {
    problems: { key: string; label: string; state: SlotState; due_at: string; receipt_id: string | null }[];
  };
}

export interface ReceiptFilters {
  kind?: string;
  status?: string;
  name?: string;
  date?: string;
  celery_task_id?: string;
  limit?: number;
}

export const TERMINAL: RunStatus[] = ["success", "partial", "failed", "not_run", "skipped"];

export function useReceipts(filters: ReceiptFilters, opts: { refetchInterval?: number | false } = {}) {
  const params = Object.fromEntries(
    Object.entries(filters).filter(([, v]) => v !== undefined && v !== "" && v !== "all"),
  );
  return useQuery({
    queryKey: ["runs", "receipts", params],
    queryFn: async () =>
      (await api.get<{ results: RunReceipt[]; count: number; names: string[] }>(
        "/runs/receipts/", { params },
      )).data,
    refetchInterval: opts.refetchInterval ?? 30_000,
  });
}

export function useReceipt(id: string | null | undefined) {
  return useQuery({
    queryKey: ["runs", "receipt", id],
    enabled: !!id,
    queryFn: async () => (await api.get<RunReceiptDetail>(`/runs/receipts/${id}/`)).data,
    // Keep polling while the run is still in flight.
    refetchInterval: (q) =>
      q.state.data && !TERMINAL.includes(q.state.data.status) ? 3_000 : false,
  });
}

export function useSchedule(date?: string) {
  return useQuery({
    queryKey: ["runs", "schedule", date ?? "today"],
    queryFn: async () =>
      (await api.get<ScheduleResponse>("/runs/schedule/", { params: date ? { date } : {} })).data,
    refetchInterval: 60_000,
  });
}

export function useRunsStatus() {
  return useQuery({
    queryKey: ["runs", "status"],
    queryFn: async () => (await api.get<RunsStatus>("/runs/status/")).data,
    refetchInterval: 60_000,
    retry: 1,
  });
}

export function useRerun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (name: string) =>
      (await api.post<{ name: string; dispatched: boolean; celery_task_id: string; note?: string }>(
        `/runs/rerun/${encodeURIComponent(name)}/`,
      )).data,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

/** Find the receipt a dispatched Celery task produced (polls until it appears + finishes). */
export function useReceiptByTask(celeryTaskId: string | null) {
  return useQuery({
    queryKey: ["runs", "by-task", celeryTaskId],
    enabled: !!celeryTaskId,
    queryFn: async () =>
      (await api.get<{ results: RunReceipt[] }>("/runs/receipts/", {
        params: { celery_task_id: celeryTaskId, limit: 1 },
      })).data.results[0] ?? null,
    refetchInterval: (q) => {
      const r = q.state.data;
      return !r || !TERMINAL.includes(r.status) ? 2_000 : false;
    },
  });
}

export const STATUS_TONE: Record<RunStatus, "success" | "warning" | "danger" | "info" | "neutral"> = {
  success: "success",
  skipped: "neutral",
  partial: "warning",
  failed: "danger",
  not_run: "danger",
  running: "info",
  queued: "info",
};

export const SLOT_TONE: Record<SlotState, "success" | "warning" | "danger" | "info" | "neutral"> = {
  ok: "success",
  warn: "warning",
  failed: "danger",
  missed: "danger",
  running: "info",
  pending: "neutral",
  untracked: "neutral",
};

export const CHECK_TONE: Record<CheckStatus, "success" | "warning" | "danger" | "neutral"> = {
  pass: "success",
  warn: "warning",
  fail: "danger",
  info: "neutral",
};

export function fmtDuration(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${Math.round(s % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function fmtIstTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString("en-IN", {
    hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Kolkata",
  });
}

export function fmtIstDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-IN", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false,
    timeZone: "Asia/Kolkata",
  });
}
