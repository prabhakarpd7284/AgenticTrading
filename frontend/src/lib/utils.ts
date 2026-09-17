import { clsx, ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/* ------------------------------------------------------------------ */
/* Numeric formatters                                                  */
/* ------------------------------------------------------------------ */

const inrCompact = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
  notation: "compact",
});
const inrFull = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

export function fmtInr(n: number | string | null | undefined, opts?: { compact?: boolean }) {
  if (n == null) return "—";
  const v = typeof n === "string" ? parseFloat(n) : n;
  if (!isFinite(v)) return "—";
  return (opts?.compact ? inrCompact : inrFull).format(v);
}

export function fmtNum(n: number | string | null | undefined, digits = 0) {
  if (n == null) return "—";
  const v = typeof n === "string" ? parseFloat(n) : n;
  if (!isFinite(v)) return "—";
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(v);
}

export function fmtPct(n: number | null | undefined, digits = 2) {
  if (n == null || !isFinite(n)) return "—";
  const s = n.toFixed(digits);
  return `${n > 0 ? "+" : ""}${s}%`;
}

/** Short relative time for journal/agent feeds: "32s", "4m", "2h", "3d". */
export function fmtRel(iso: string | Date | null | undefined, nowMs = Date.now()) {
  if (!iso) return "—";
  const t = iso instanceof Date ? iso.getTime() : new Date(iso).getTime();
  const diff = Math.max(0, Math.floor((nowMs - t) / 1000));
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86_400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86_400)}d`;
}

/** Absolute IST timestamp for anchored events: "24 Jun, 14:32". Always IST
 *  (the market timezone) regardless of the viewer's locale, so a saved
 *  setup's generation time reads the same for everyone. */
export function fmtDateTime(iso: string | Date | null | undefined): string {
  if (!iso) return "—";
  const d = iso instanceof Date ? iso : new Date(iso);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-IN", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
    hour12: false, timeZone: "Asia/Kolkata",
  });
}

/** Semantic P&L colour class — never relies on sign alone (paired with +/− glyph). */
export function clsPnl(n: number | string | null | undefined): string {
  const v = typeof n === "string" ? parseFloat(n) : (n ?? 0);
  if (v > 0) return "text-pnl-up";
  if (v < 0) return "text-pnl-down";
  return "text-fg-muted";
}

/** Pretty-print arbitrary JSON-shaped data for inline display. Falls back to
 *  String() if the value contains cycles (JSON.stringify throws on those). */
export function safeStringify(v: unknown): string {
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}

/** Forward-looking duration ("how long did this take") with sub-second
 *  precision: 423ms / 3.2s / 1m 4s. Distinct from fmtRel which floors to
 *  seconds and reads as "X ago". */
export function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}
