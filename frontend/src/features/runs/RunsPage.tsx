import * as React from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Bot, History, Terminal, Workflow } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { RunButton } from "@/components/runs/RunButton";
import { RunReceiptPanel } from "@/components/runs/RunReceiptPanel";
import { cn } from "@/lib/utils";
import {
  SLOT_TONE, STATUS_TONE, fmtDuration, fmtIstDateTime, fmtIstTime, useReceipt, useReceipts, useSchedule,
  type ReceiptFilters, type ScheduleSlot,
} from "./useRuns";

const STATUSES = ["all", "success", "partial", "failed", "not_run", "skipped", "running"];
const KINDS = ["all", "job", "pipeline", "scan", "import", "backtest", "agent", "analysis"];

const SLOT_LABEL: Record<ScheduleSlot["state"], string> = {
  ok: "OK", warn: "warnings", failed: "failed", missed: "missed", running: "running",
  pending: "pending", untracked: "not tracked",
};

/**
 * /runs — every job/scan/import with a receipt (UX redesign R1, plan §8).
 * Today's expected schedule (green / amber / red / grey) on top, the receipt
 * log with filters below, and a receipt drawer for the selected run.
 */
export function RunsPage() {
  const { receiptId } = useParams<{ receiptId?: string }>();
  const navigate = useNavigate();
  const [filters, setFilters] = React.useState<ReceiptFilters>({ status: "all", kind: "all", limit: 100 });
  const schedule = useSchedule();
  const receipts = useReceipts(filters);

  const set = (k: keyof ReceiptFilters) => (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) =>
    setFilters((f) => ({ ...f, [k]: e.target.value || undefined }));

  return (
    <div className="p-6 space-y-6 max-w-[1400px]">
      <header className="flex flex-wrap items-end gap-4">
        <div className="flex-1 min-w-[260px]">
          <h2 className="text-h2 text-fg">Runs</h2>
          <p className="text-body-sm text-fg-muted">
            Did it run, did it run correctly, and what did it produce? Every tracked job writes a receipt
            with checks; scheduled jobs that never fired show up as <em>missed</em>.
          </p>
        </div>
        <nav className="flex gap-3 text-body-sm" aria-label="Other run views">
          <Link to="/pipeline" className="inline-flex items-center gap-1 text-fg-muted hover:text-fg">
            <Workflow className="h-4 w-4" aria-hidden />Daily Pipeline
          </Link>
          <Link to="/agents" className="inline-flex items-center gap-1 text-fg-muted hover:text-fg">
            <Bot className="h-4 w-4" aria-hidden />Agent Console
          </Link>
          <Link to="/ops" className="inline-flex items-center gap-1 text-fg-muted hover:text-fg">
            <Terminal className="h-4 w-4" aria-hidden />Ops Console
          </Link>
        </nav>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Today's schedule</CardTitle>
          <CardDescription>
            {schedule.data
              ? `${schedule.data.date} · ${schedule.data.summary.label}` +
                (schedule.data.summary.warn ? ` · ${schedule.data.summary.warn} with warnings` : "")
              : "Expected beat schedule and what actually ran."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {schedule.isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : schedule.isError || !schedule.data ? (
            <p className="text-body-sm text-pnl-down">Could not load the schedule.</p>
          ) : (
            <ul className="grid gap-2 md:grid-cols-2 xl:grid-cols-3" aria-label="Scheduled jobs today">
              {schedule.data.slots.map((s) => (
                <li key={`${s.key}-${s.due_at}`} data-testid={`slot-${s.key}`}
                    className={cn("rounded-sm border px-3 py-2 space-y-1", slotBorder(s.state))}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-body-sm text-fg-muted">{fmtIstTime(s.due_at)}</span>
                    <span className="flex-1 text-body-sm text-fg truncate" title={s.task}>{s.label}</span>
                    <Badge tone={SLOT_TONE[s.state]} dot>{SLOT_LABEL[s.state]}</Badge>
                  </div>
                  <div className="flex items-center gap-2 text-caption text-fg-subtle">
                    <span className="flex-1 truncate">
                      {s.receipt
                        ? `${s.receipt.status} · ${fmtDuration(s.receipt.duration_seconds)}`
                          + (s.receipt.warn_checks ? ` · ${s.receipt.warn_checks} warn` : "")
                        : s.reason ?? s.schedule}
                    </span>
                    {s.receipt && (
                      <button type="button" className="text-accent hover:underline"
                              onClick={() => navigate(`/runs/${s.receipt!.id}`)}>
                        receipt
                      </button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card className="min-w-0">
          <CardHeader>
            <CardTitle>Receipts</CardTitle>
            <div className="flex flex-wrap gap-2 pt-2">
              <FilterSelect label="Status" value={filters.status ?? "all"} options={STATUSES} onChange={set("status")} />
              <FilterSelect label="Kind" value={filters.kind ?? "all"} options={KINDS} onChange={set("kind")} />
              <FilterSelect label="Job" value={filters.name ?? "all"}
                            options={["all", ...(receipts.data?.names ?? [])]} onChange={set("name")} />
              <label className="flex items-center gap-1 text-caption text-fg-subtle">
                Date
                <input type="date" aria-label="Date" value={filters.date ?? ""} onChange={set("date")}
                       className="h-8 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg" />
              </label>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {receipts.isLoading ? (
              <div className="p-5"><Skeleton className="h-40 w-full" /></div>
            ) : !receipts.data?.results.length ? (
              <div className="p-5">
                <EmptyState icon={<History />} title="No receipts match"
                            description="Receipts appear as tracked jobs run (beat or manual)." />
              </div>
            ) : (
              <table className="w-full text-body-sm">
                <thead>
                  <tr className="text-left text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
                    <th className="px-4 py-2 font-medium">Started (IST)</th>
                    <th className="px-2 py-2 font-medium">Job</th>
                    <th className="px-2 py-2 font-medium">Status</th>
                    <th className="px-2 py-2 font-medium">Took</th>
                    <th className="px-2 py-2 font-medium">Checks</th>
                  </tr>
                </thead>
                <tbody>
                  {receipts.data.results.map((r) => (
                    <tr key={r.id} onClick={() => navigate(`/runs/${r.id}`)}
                        className={cn("border-b border-border/60 cursor-pointer hover:bg-surface-2",
                                      r.id === receiptId && "bg-accent/10")}>
                      <td className="px-4 py-2 font-mono text-fg-muted whitespace-nowrap">{fmtIstDateTime(r.started_at)}</td>
                      <td className="px-2 py-2">
                        <span className="font-mono text-fg">{r.name}</span>
                        <span className="ml-1 text-caption text-fg-subtle">{r.kind} · {r.trigger}</span>
                      </td>
                      <td className="px-2 py-2"><Badge tone={STATUS_TONE[r.status]}>{r.status.replace("_", " ")}</Badge></td>
                      <td className="px-2 py-2 font-mono text-fg-muted">{fmtDuration(r.duration_seconds)}</td>
                      <td className="px-2 py-2 text-caption whitespace-nowrap">
                        {r.check_counts.fail > 0 && <span className="text-pnl-down mr-2">{r.check_counts.fail} fail</span>}
                        {r.check_counts.warn > 0 && <span className="text-warn mr-2">{r.check_counts.warn} warn</span>}
                        <span className="text-fg-subtle">{r.check_counts.pass} pass</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0">
          <CardHeader>
            <CardTitle>Receipt</CardTitle>
            <CardDescription>{receiptId ? "Selected run" : "Select a run to see its receipt."}</CardDescription>
          </CardHeader>
          <CardContent>
            {receiptId ? <SelectedReceipt id={receiptId} /> : (
              <p className="text-body-sm text-fg-muted">
                Each receipt shows status, duration, what the run produced, warnings, checks and links to its output.
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function SelectedReceipt({ id }: { id: string }) {
  const { data } = useReceipt(id);
  return (
    <div className="space-y-4">
      <RunReceiptPanel receiptId={id} />
      {data?.rerunnable && (
        <div className="border-t border-border pt-3">
          <RunButton key={id} name={data.name} label="Run again" note={data.rerun_note} />
        </div>
      )}
    </div>
  );
}

function FilterSelect({ label, value, options, onChange }: {
  label: string; value: string; options: string[];
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}) {
  return (
    <label className="flex items-center gap-1 text-caption text-fg-subtle">
      {label}
      <select aria-label={label} value={value} onChange={onChange}
              className="h-8 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg">
        {options.map((o) => <option key={o} value={o}>{o.replace("_", " ")}</option>)}
      </select>
    </label>
  );
}

function slotBorder(state: ScheduleSlot["state"]): string {
  switch (state) {
    case "ok": return "border-pnl-up/40";
    case "warn": return "border-warn/50";
    case "failed":
    case "missed": return "border-pnl-down/50 bg-pnl-down/5";
    case "running": return "border-info/50";
    default: return "border-border";
  }
}
