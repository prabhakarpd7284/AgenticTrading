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

/** Semantic P&L colour class — never relies on sign alone (paired with +/− glyph). */
export function clsPnl(n: number | string | null | undefined): string {
  const v = typeof n === "string" ? parseFloat(n) : (n ?? 0);
  if (v > 0) return "text-pnl-up";
  if (v < 0) return "text-pnl-down";
  return "text-fg-muted";
}
