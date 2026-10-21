import { Link } from "react-router-dom";
import { AlertTriangle, ArrowRight, CheckCircle2, CircleDashed, Info, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/utils";
import {
  CHECK_TONE, STATUS_TONE, fmtDuration, fmtIstDateTime, useReceipt,
  type RunCheck, type RunReceipt, type RunReceiptDetail,
} from "@/features/runs/useRuns";

/**
 * RunReceiptPanel — the uniform "what happened" view for any run
 * (UX redesign plan §8.4): status, duration, what it produced, warnings,
 * checks, links to the result, and a comparison with the previous run.
 * Raw logs are never the answer; they stay one click away on their own page.
 */
export function RunReceiptPanel({ receiptId, className }: { receiptId: string; className?: string }) {
  const { data, isLoading, isError } = useReceipt(receiptId);
  if (isLoading) return <Skeleton className={cn("h-40 w-full", className)} />;
  if (isError || !data) {
    return <p className={cn("text-body-sm text-pnl-down", className)}>Could not load this receipt.</p>;
  }
  return <ReceiptBody r={data} className={className} />;
}

export function ReceiptBody({ r, className }: { r: RunReceiptDetail; className?: string }) {
  const links = (r.outputs.links ?? []) as { label: string; href: string }[];
  const outputs = Object.entries(r.outputs).filter(([k]) => k !== "links");
  return (
    <div className={cn("space-y-4", className)} data-testid="run-receipt">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={STATUS_TONE[r.status]} dot>{r.status.replace("_", " ")}</Badge>
        <span className="font-mono text-body-sm text-fg">{r.name}</span>
        <span className="text-caption text-fg-subtle">
          {r.trigger} · {r.scope} · started {fmtIstDateTime(r.started_at)} IST · took {fmtDuration(r.duration_seconds)}
        </span>
      </div>

      {r.error && (
        <div className="rounded-sm border border-pnl-down/30 bg-pnl-down/10 px-3 py-2 text-body-sm text-pnl-down">
          {r.error}
        </div>
      )}

      {outputs.length > 0 && (
        <section>
          <h4 className="text-caption uppercase tracking-wider text-fg-subtle mb-1.5">What it produced</h4>
          <dl className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-x-4 gap-y-1.5">
            {outputs.map(([k, v]) => (
              <div key={k} className="min-w-0">
                <dt className="text-caption text-fg-subtle">{k.replace(/_/g, " ")}</dt>
                <dd className="text-body-sm text-fg font-mono truncate" title={fmtValue(v)}>
                  {fmtValue(v)}
                  <PrevDelta prev={r.previous} k={k} v={v} />
                </dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {r.warnings.length > 0 && (
        <section>
          <h4 className="text-caption uppercase tracking-wider text-fg-subtle mb-1.5">Warnings</h4>
          <ul className="space-y-1">
            {r.warnings.map((w, i) => (
              <li key={i} className="flex gap-2 text-body-sm text-warn">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden />{w}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h4 className="text-caption uppercase tracking-wider text-fg-subtle mb-1.5">
          Checks ({r.checks.length})
        </h4>
        {r.checks.length === 0 ? (
          <p className="text-body-sm text-fg-muted">No checks recorded for this run.</p>
        ) : (
          <ul className="divide-y divide-border/60 rounded-sm border border-border">
            {r.checks.map((c, i) => <CheckRow key={`${c.name}-${i}`} c={c} />)}
          </ul>
        )}
      </section>

      {(links.length > 0 || r.detail) && (
        <section className="flex flex-wrap gap-2">
          {links.map((l) => (
            <Link key={l.href + l.label} to={l.href}
                  className="inline-flex items-center gap-1 text-body-sm text-accent hover:underline">
              Open {l.label} <ArrowRight className="h-3.5 w-3.5" aria-hidden />
            </Link>
          ))}
          {r.detail && (
            <span className="text-caption text-fg-subtle self-center">
              detail: {r.detail.kind} #{r.detail.id} ({r.detail.status})
            </span>
          )}
        </section>
      )}

      {r.previous && (
        <p className="text-caption text-fg-subtle">
          Previous run: {r.previous.status} at {fmtIstDateTime(r.previous.started_at)} IST,
          took {fmtDuration(r.previous.duration_seconds)}.
        </p>
      )}

      {(Object.keys(r.inputs ?? {}).length > 0 || Object.keys(r.data_provenance ?? {}).length > 0) && (
        <details className="text-caption text-fg-muted">
          <summary className="cursor-pointer select-none">Inputs and data provenance</summary>
          <pre className="mt-1 whitespace-pre-wrap font-mono">
            {JSON.stringify({ inputs: r.inputs, provenance: r.data_provenance }, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

function CheckRow({ c }: { c: RunCheck }) {
  const Icon = { pass: CheckCircle2, warn: AlertTriangle, fail: XCircle, info: Info }[c.status] ?? CircleDashed;
  const color = { pass: "text-pnl-up", warn: "text-warn", fail: "text-pnl-down", info: "text-fg-subtle" }[c.status];
  return (
    <li className="flex items-start gap-2 px-3 py-2">
      <Icon className={cn("h-4 w-4 mt-0.5 shrink-0", color)} aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-mono text-body-sm text-fg">{c.name}</span>
          <Badge tone={CHECK_TONE[c.status]}>{c.status}</Badge>
        </div>
        <p className="text-body-sm text-fg-muted">{c.message}</p>
      </div>
    </li>
  );
}

function PrevDelta({ prev, k, v }: { prev: RunReceipt | null; k: string; v: unknown }) {
  if (!prev || typeof v !== "number") return null;
  const p = prev.outputs?.[k];
  if (typeof p !== "number" || p === v) return null;
  const d = v - p;
  return (
    <span className={cn("ml-1 text-caption", d > 0 ? "text-pnl-up" : "text-pnl-down")}>
      ({d > 0 ? "+" : ""}{d} vs prev)
    </span>
  );
}

function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") {
    return Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => `${k}: ${typeof x === "object" ? JSON.stringify(x) : String(x)}`)
      .join(", ");
  }
  return String(v);
}
