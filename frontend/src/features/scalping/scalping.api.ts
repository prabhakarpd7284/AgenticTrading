import { useMutation, useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { ScalpConfig } from "./scalping.types";

/** Create an interactive scalp sim run → returns the run_id the WS connects to. */
export function useCreateScalpRun() {
  return useMutation({
    mutationFn: (cfg: Partial<ScalpConfig>) =>
      api.post<{ run_id: string }>("/strategies/scalp/runs/", cfg).then((r) => r.data),
  });
}

/** Resolved default Date (last trading day) + Expiry (nearest weekly/monthly). */
export function useScalpDefaults(underlying: string) {
  return useQuery({
    queryKey: ["scalp-defaults", underlying],
    queryFn: () =>
      api
        .get<{ date: string; expiry: string; expiries: string[]; strike: number | null; strike_step: number }>(
          "/strategies/scalp/defaults/", { params: { underlying } },
        )
        .then((r) => r.data),
    staleTime: 5 * 60_000,
  });
}
