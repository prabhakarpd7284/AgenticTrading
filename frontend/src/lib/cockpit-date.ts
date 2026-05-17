/**
 * Cockpit time-travel — selected session date threaded into every
 * /market-data/ + /strategies/ request via the axios interceptor in api.ts.
 *
 * Default: null = live behaviour (backend picks intraday_session_date()).
 * When set: every cockpit panel call gets ?date=YYYY-MM-DD appended.
 *
 * The Cockpits page reads/sets this; other pages don't touch it. The axios
 * interceptor is URL-scoped so even if the value leaks, only cockpit-style
 * endpoints get the override.
 */
import { create } from "zustand";

interface CockpitDateState {
  /** YYYY-MM-DD, or null to use live/default. */
  selectedDate: string | null;
  setSelectedDate: (d: string | null) => void;
}

export const useCockpitDateStore = create<CockpitDateState>((set) => ({
  selectedDate: null,
  setSelectedDate: (d) => set({ selectedDate: d }),
}));

/** True when the URL is a cockpit-style endpoint that supports as-of replay. */
export function isCockpitUrl(url: string | undefined): boolean {
  if (!url) return false;
  return url.includes("/market-data/") || url.includes("/strategies/");
}
