import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/Badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/Tooltip";
import { useRunsStatus } from "@/features/runs/useRuns";

/**
 * Persistent status strip (UX redesign plan §2): trading mode, market-data
 * status (live / stale / synthetic) and today's job health. Answers "can I
 * trust what I'm looking at?" on every page without choosing one.
 */
export function StatusStrip() {
  const { data, isError } = useRunsStatus();

  if (isError) {
    return (
      <Badge tone="neutral" aria-label="Status unavailable">status unavailable</Badge>
    );
  }
  if (!data) return <span className="text-caption text-fg-subtle">…</span>;

  const modeTone = data.mode.mode === "LIVE" ? "danger" : data.mode.mode === "HALT" ? "warning" : "info";
  const dataTone = data.data.state === "live" ? "success" : data.data.state === "stale" ? "warning" : "danger";
  const jobsTone = ({ green: "success", amber: "warning", red: "danger", grey: "neutral" } as const)[data.jobs.health];
  const jobsLabel = data.jobs.total ? `${data.jobs.ok}/${data.jobs.total} jobs OK` : "jobs: none due";

  return (
    <div className="flex items-center gap-1.5" data-testid="status-strip">
      <Tip text={`Mode ${data.mode.mode} — ${data.mode.reason}`}>
        <Badge tone={modeTone} aria-label={`Trading mode ${data.mode.mode}`}>{data.mode.mode}</Badge>
      </Tip>
      <Tip text={data.data.reason}>
        <Link to="/brokers" className="inline-flex">
          <Badge tone={dataTone} dot aria-label={`Market data ${data.data.state}`}>
            data {data.data.state}
          </Badge>
        </Link>
      </Tip>
      <Tip text={
        data.jobs.label +
        (data.jobs.problems.length ? ` — problems: ${data.jobs.problems.map((p) => `${p.label} (${p.state})`).join(", ")}` : "") +
        (data.jobs.pending ? ` · ${data.jobs.pending} still to come today` : "")
      }>
        <Link to="/runs" className="inline-flex">
          <Badge tone={jobsTone} dot aria-label={data.jobs.label}>{jobsLabel}</Badge>
        </Link>
      </Tip>
    </div>
  );
}

function Tip({ text, children }: { text: string; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex">{children}</span>
      </TooltipTrigger>
      <TooltipContent className="max-w-sm">{text}</TooltipContent>
    </Tooltip>
  );
}
