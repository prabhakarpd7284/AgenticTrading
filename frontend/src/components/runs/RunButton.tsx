import * as React from "react";
import { Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { RunReceiptPanel } from "@/components/runs/RunReceiptPanel";
import { TERMINAL, useReceiptByTask, useRerun } from "@/features/runs/useRuns";

/**
 * RunButton — the one way a page starts a tracked job (UX redesign plan §8.4):
 *   1. confirm the inputs (what will run, and its caveat note),
 *   2. live progress while the receipt is queued / running,
 *   3. the receipt itself when it finishes (RunReceiptPanel).
 * Only jobs with a safe manual trigger on the backend (``RERUNNABLE``) can be
 * started here; the endpoint 404s anything else.
 */
export function RunButton({
  name, label = "Run now", note, size = "sm",
}: { name: string; label?: string; note?: string | null; size?: "sm" | "md" }) {
  const [phase, setPhase] = React.useState<"idle" | "confirm" | "started">("idle");
  const rerun = useRerun();
  const taskId = rerun.data?.celery_task_id ?? null;
  const receipt = useReceiptByTask(phase === "started" ? taskId : null);

  if (phase === "idle") {
    return (
      <Button size={size} variant="secondary" leading={<Play className="h-3.5 w-3.5" />}
              onClick={() => setPhase("confirm")}>
        {label}
      </Button>
    );
  }

  if (phase === "confirm") {
    return (
      <div className="rounded-sm border border-border bg-surface-2 p-3 space-y-2" role="group"
           aria-label={`Confirm run of ${name}`}>
        <p className="text-body-sm text-fg">
          Run <span className="font-mono">{name}</span> now (manual trigger)?
        </p>
        {note && <p className="text-caption text-fg-muted">{note}</p>}
        {rerun.isError && (
          <p className="text-caption text-pnl-down">
            Could not start: {(rerun.error as Error)?.message ?? "error"}
          </p>
        )}
        <div className="flex gap-2">
          <Button size="sm" loading={rerun.isPending}
                  onClick={() => rerun.mutate(name, { onSuccess: () => setPhase("started") })}>
            Confirm
          </Button>
          <Button size="sm" variant="ghost" onClick={() => { rerun.reset(); setPhase("idle"); }}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  const r = receipt.data;
  const done = r && TERMINAL.includes(r.status);
  return (
    <div className="space-y-2">
      {!done && (
        <p className="flex items-center gap-2 text-body-sm text-fg-muted" aria-live="polite">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          {r ? `Running ${name}…` : `Queued ${name} — waiting for a worker to pick it up…`}
        </p>
      )}
      {done && r && <RunReceiptPanel receiptId={r.id} />}
      {done && (
        <Button size="sm" variant="ghost" onClick={() => { rerun.reset(); setPhase("idle"); }}>
          Done
        </Button>
      )}
    </div>
  );
}
