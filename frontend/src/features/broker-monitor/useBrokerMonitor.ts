/**
 * Broker Monitor — live broker-API telemetry hook.
 *
 * Backend: GET /api/v1/brokers/monitor/  (auth required)
 *
 * Surfaces the four early-warning signals an operator watches when the
 * Angel One SmartAPI is under stress:
 *   1. the rate-limit breaker (open = we've stopped hammering the broker),
 *   2. per-minute call volume over the last 30 minutes,
 *   3. Celery queue depths (a growing backlog was the canary in the
 *      2026-06-24 incident),
 *   4. the health of each linked broker account.
 *
 * Polls every 5s — fast enough to watch a breaker trip in near-real-time.
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Types — mirror the backend monitor payload                          */
/* ------------------------------------------------------------------ */

export interface BreakerState {
  open: boolean;
  cooldown_remaining_s: number;
  trips: number;
}

/** One bucket of the call-rate series. `minute` is an epoch in SECONDS. */
export interface CallRatePoint {
  minute: number;
  calls: number;
}

export interface QueueDepths {
  celery: number;
  orders: number;
  agents: number;
  backtests: number;
}

export type LinkStatus = "ACTIVE" | "ERRORED" | "EXPIRED" | "DISABLED";

export interface BrokerLink {
  id: string;
  broker: string;
  display_name: string;
  status: LinkStatus;
  last_refreshed_at: string | null;
  last_error: string;
  last_snapshot_ok: boolean | null;
}

export interface BrokerMonitorPayload {
  breaker: BreakerState;
  call_rate: CallRatePoint[];
  queues: QueueDepths;
  links: BrokerLink[];
  ts: string;
}

/* ------------------------------------------------------------------ */
/* Hook                                                                */
/* ------------------------------------------------------------------ */

/** Poll cadence — the breaker countdown wants a tight loop. */
const REFRESH_MS = 5_000;

export function useBrokerMonitor() {
  return useQuery<BrokerMonitorPayload>({
    queryKey: ["broker-monitor"],
    queryFn: async () => {
      const r = await api.get<BrokerMonitorPayload>("/brokers/monitor/");
      return r.data;
    },
    refetchInterval: REFRESH_MS,
    refetchIntervalInBackground: false,
    staleTime: 2_000,
  });
}

/* ------------------------------------------------------------------ */
/* Presentational helpers                                              */
/* ------------------------------------------------------------------ */

/** A Celery queue this deep is the early-warning signal of a stuck worker. */
export const QUEUE_WARN_THRESHOLD = 100;

export const QUEUE_LABEL: Record<keyof QueueDepths, string> = {
  celery: "Celery",
  orders: "Orders",
  agents: "Agents",
  backtests: "Backtests",
};

/** Map a broker-link status to a Badge tone. */
export function linkStatusTone(
  s: LinkStatus,
): "success" | "warning" | "danger" | "neutral" {
  if (s === "ACTIVE") return "success";
  if (s === "ERRORED") return "danger";
  if (s === "EXPIRED") return "warning";
  return "neutral"; // DISABLED
}

/** epoch SECONDS → "HH:MM" clock label (24h, IST market timezone). */
export function epochToClock(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Kolkata",
  });
}

/** Seconds → "M:SS" countdown, e.g. 95 → "1:35". */
export function fmtCountdown(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${String(rem).padStart(2, "0")}`;
}
