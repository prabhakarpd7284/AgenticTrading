/**
 * Backtester run history — localStorage-backed log of the last 20 runs.
 *
 * Why client-side: the OK Backtester is one-shot on the server (no DB
 * persistence layer for runs). localStorage gives operators a "last N runs"
 * timeline they can scroll without re-paying the 30-200s cold-cache cost.
 * One operator = one device, so server-side persistence isn't urgent yet.
 *
 * Capped at 20 entries to keep payloads under the ~5 MB localStorage budget.
 * Each entry stores the full OKBacktestPayload so "re-display" loads
 * instantly without hitting the API.
 */
import type { OKBacktestPayload } from "@/lib/market-pulse";

const STORAGE_KEY = "alphadesk:backtest:history:v1";
const MAX_RUNS = 20;

export interface BacktestHistoryEntry {
  id: string;
  ran_at: string;             // ISO timestamp
  mode: "daily" | "intraday" | "basket";
  from_date: string;
  to_date: string;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  total_pnl: number;
  symbols_count: number;
  universe_symbols: string[];
  payload: OKBacktestPayload; // full result for re-display without re-fetch
}

function load(): BacktestHistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function save(entries: BacktestHistoryEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_RUNS)));
  } catch {
    // QuotaExceeded — drop oldest half and retry once
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, Math.floor(MAX_RUNS / 2))));
    } catch {
      // give up silently — history is best-effort
    }
  }
}

export function listHistory(): BacktestHistoryEntry[] {
  return load();
}

export function recordRun(
  mode: "daily" | "intraday" | "basket",
  payload: OKBacktestPayload,
): BacktestHistoryEntry {
  const entry: BacktestHistoryEntry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    ran_at: new Date().toISOString(),
    mode,
    from_date: payload.from_date,
    to_date: payload.to_date,
    total_trades: payload.total_trades,
    win_rate: payload.win_rate,
    profit_factor: payload.profit_factor,
    total_pnl: payload.total_pnl,
    symbols_count: payload.symbols_count,
    universe_symbols: payload.universe_symbols ?? [],
    payload,
  };
  // De-dup: identical mode + from + to + total_trades within last 10s replaces
  // the existing entry instead of duplicating (covers React Query refetches).
  const existing = load();
  const tenSecondsAgo = Date.now() - 10_000;
  const filtered = existing.filter((e) => {
    if (e.mode !== mode || e.from_date !== payload.from_date || e.to_date !== payload.to_date) return true;
    if (e.total_trades !== payload.total_trades) return true;
    return new Date(e.ran_at).getTime() < tenSecondsAgo;
  });
  save([entry, ...filtered]);
  return entry;
}

export function getHistoryEntry(id: string): BacktestHistoryEntry | undefined {
  return load().find((e) => e.id === id);
}

export function deleteHistoryEntry(id: string): void {
  save(load().filter((e) => e.id !== id));
}

export function clearHistory(): void {
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
}
