/**
 * TrackSetupButton — the Setup page's non-execute action.
 *
 * One click does two things, neither of which sends an order:
 *   1. Adds the symbol to a dedicated manual "Tracked Setups" watchlist
 *      (auto-created the first time), so the name lands in your daily list.
 *   2. Snapshots the exact on-screen plan + its generation time as a
 *      `setup.saved` event, so the setup can later be correlated with
 *      whatever trade (if any) was actually taken — see TrackRecordCard.
 *
 * Idempotent: re-clicking re-snapshots (a fresh re-evaluation) and is a
 * no-op on the watchlist if the symbol is already there.
 */
import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { BookmarkPlus, Check, Loader2 } from "lucide-react";

import {
  useWatchlists, useCreateWatchlist, useAddSymbolsToWatchlist,
  useSaveSetupSnapshot, useWatchlistsBySymbol,
} from "@/lib/v2";
import type { SetupPayload } from "@/lib/market-pulse";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/Button";

const TRACKED_LIST = "Tracked Setups";

export function TrackSetupButton({ data }: { data: SetupPayload }) {
  const symbol = data.symbol;
  const plan = data.plan;

  const { data: lists = [] } = useWatchlists();
  const { data: bySymbol = [] } = useWatchlistsBySymbol(symbol);
  const createList = useCreateWatchlist();
  const addSymbols = useAddSymbolsToWatchlist();
  const saveSnapshot = useSaveSetupSnapshot();
  const qc = useQueryClient();

  const [justSaved, setJustSaved] = React.useState(false);
  const alreadyTracked = bySymbol.some(
    (w) => w.name.toLowerCase() === TRACKED_LIST.toLowerCase(),
  );
  const busy = createList.isPending || addSymbols.isPending || saveSnapshot.isPending;

  async function handleTrack() {
    if (!plan) {
      toast.error("No plan to track — insufficient market data for this name.");
      return;
    }
    try {
      // 1. Ensure the dedicated manual list exists, then add the symbol.
      const existing = lists.find(
        (w) => w.name.toLowerCase() === TRACKED_LIST.toLowerCase(),
      );
      if (existing) {
        await addSymbols.mutateAsync({ id: existing.id, symbols: [symbol] });
      } else {
        await createList.mutateAsync({
          name: TRACKED_LIST,
          kind: "MANUAL",
          description: "Setups you saved from Stage 5 to watch live.",
          symbols: [symbol],
        });
      }

      // 2. Snapshot the exact plan + generation time.
      await saveSnapshot.mutateAsync({
        symbol,
        side: plan.side,
        entry_price: plan.entry_price,
        stop_loss: plan.stop_loss,
        target: plan.target,
        quantity: plan.quantity,
        confidence: plan.confidence,
        risk_reward_ratio: plan.risk_reward_ratio,
        risk_approved: data.risk.approved,
        generated_at: data.as_of,
      });

      // Refresh the "In: …" badges + the saved-setups section.
      qc.invalidateQueries({ queryKey: ["watchlists-by-symbol", symbol] });
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 2500);
      toast.success(`Tracking ${symbol} · added to ${TRACKED_LIST}`);
    } catch (e) {
      toast.error(`Couldn't track setup: ${(e as Error)?.message ?? "unknown error"}`);
    }
  }

  return (
    <Button
      variant={alreadyTracked ? "secondary" : "primary"}
      size="sm"
      onClick={handleTrack}
      disabled={busy || !plan}
      title={`Add ${symbol} to "${TRACKED_LIST}" and snapshot this plan — nothing executes`}
    >
      {busy ? (
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      ) : justSaved || alreadyTracked ? (
        <Check className={cn("h-4 w-4", justSaved && "text-pnl-up")} aria-hidden />
      ) : (
        <BookmarkPlus className="h-4 w-4" aria-hidden />
      )}
      {justSaved ? "Saved" : alreadyTracked ? "Tracking" : "Track setup"}
    </Button>
  );
}
