/**
 * Daily Pipeline — flow observability & manager.
 *
 * Shows the full daily data flow for any trading day: a pipeline diagram
 * with per-stage metrics, a per-strategy breakdown, a chronological
 * activity feed of every signal/run, and run history. Each runnable
 * stage can be force-triggered for debugging.
 *
 * Backend: GET /system/pipeline/?date=  +  POST /system/pipeline/<task>/run/.
 */
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  Activity, ArrowDown, CheckCircle2, ChevronRight, Database, Layers, Loader2,
  Play, Radio, ScanSearch, ShoppingBag, Sparkles, TrendingUp, Workflow, XCircle,
} from "lucide-react";

import { api } from "@/lib/api";
import { cn, fmtRel } from "@/lib/utils";

import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

// ─── API types ────────────────────────────────────────────────────────

type RunStatus = "running" | "success" | "failed";
type NodeStatus = RunStatus | "idle";

interface PipelineRun {
  id: number;
  task: string;
  status: RunStatus;
  trigger: "beat" | "manual";
  started_at: string;
  finished_at: string | null;
  duration_seconds: number | null;
  summary: Record<string, unknown>;
  error: string;
}

interface FlowNode {
  key: string;
  label: string;
  role: "source" | "hub" | "stage" | "sink";
  status: NodeStatus;
  live?: boolean;
  metric: number;
  metric_label: string;
}

interface StrategyStat {
  strategy: string;
  source: string;
  count: number;
  traded: number;
  buy: number;
  sell: number;
  avg_confidence: number;
}

interface StrategyInfo {
  name: string;
  label: string;
  category: string;
  automation: string;          // "daily" | "on-demand"
  pipeline_task: string | null;
  signal_source: string | null;
  wired: boolean;
  signals: number;
  description: string;
}

interface FeedItem {
  ts: string;
  kind: "signal" | "run";
  stage: string;
  title: string;
  detail: string;
  side?: string;
  source?: string;
  outcome?: string;
  status?: RunStatus;
  trigger?: string;
}

interface PipelineStatus {
  today: {
    date: string;
    is_trading_day: boolean;
    session_phase: string;
    next_trading_day: string;
  };
  selected_date: string;
  viewing_today: boolean;
  date_options: string[];
  flow: FlowNode[];
  signals: {
    total: number; enriched: number; traded: number;
    by_source: Record<string, number>;
  };
  by_strategy: StrategyStat[];
  strategies: StrategyInfo[];
  feed: FeedItem[];
  unenriched_total: number;
  recent_runs: PipelineRun[];
  tasks?: PipelineTask[];
  auto_execute_enabled?: boolean;
}

interface PipelineTask {
  key: string;
  label: string;
  schedule: string;
  description: string;
  category: "data" | "execution";
}

/** Manual-trigger payload. ``date``/``backfill`` only apply to EOD enrichment. */
interface TriggerVars {
  task: string;
  date?: string;
  backfill?: boolean;
}

// ─── Constants ────────────────────────────────────────────────────────

const RUNNABLE = new Set([
  "swing_scan", "premarket_basket", "screener_session", "eod_enrichment",
  "intraday_agent",
]);

const NODE_ICON: Record<string, typeof Activity> = {
  swing_scan: ScanSearch,
  premarket_basket: ShoppingBag,
  screener_session: Radio,
  ledger: Database,
  eod_enrichment: Sparkles,
  feedback: TrendingUp,
};

const STAGE_LABEL: Record<string, string> = {
  swing_scan: "Swing",
  premarket_basket: "Basket",
  screener_session: "Screener",
  eod_enrichment: "Enrichment",
  premarket: "Basket",
  tradingview: "TradingView",
  other: "Signal",
};

// ─── Helpers ──────────────────────────────────────────────────────────

function fmtDuration(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s === 0 ? `${m}m` : `${m}m ${s}s`;
}

function fmtDate(iso: string): string {
  return new Date(iso + "T00:00:00").toLocaleDateString("en-IN", {
    weekday: "short", day: "2-digit", month: "short",
  });
}

function nodeAccent(status: NodeStatus): { dot: string; text: string; ring: string } {
  switch (status) {
    case "running":
      return { dot: "bg-brand", text: "text-brand", ring: "border-brand/50" };
    case "success":
      return { dot: "bg-success", text: "text-success", ring: "border-success/40" };
    case "failed":
      return { dot: "bg-danger", text: "text-danger", ring: "border-danger/50" };
    default:
      return { dot: "bg-fg-subtle", text: "text-fg-subtle", ring: "border-border" };
  }
}

// ─── Page ─────────────────────────────────────────────────────────────

export function PipelinePage() {
  const qc = useQueryClient();
  // null → backend picks the latest day with data.
  const [dateParam, setDateParam] = useState<string | null>(null);

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["pipeline-status", dateParam],
    queryFn: () =>
      api
        .get<PipelineStatus>(
          "/system/pipeline/" + (dateParam ? `?date=${dateParam}` : ""),
        )
        .then((r) => r.data),
    // Only live-poll when we're looking at today; past days are static.
    refetchInterval: (q) => (q.state.data?.viewing_today ? 6_000 : false),
    staleTime: 4_000,
  });

  const trigger = useMutation({
    mutationFn: (v: TriggerVars) =>
      api
        .post(`/system/pipeline/${v.task}/run/`, {
          date: v.date, backfill: v.backfill,
        })
        .then((r) => r.data),
    onSuccess: (_d, v) => {
      toast.success(
        v.backfill
          ? "Backfilling enrichment across all dates…"
          : v.date
            ? `Running ${v.task.replace(/_/g, " ")} for ${v.date}…`
            : `Triggered ${v.task.replace(/_/g, " ")}`,
      );
      qc.invalidateQueries({ queryKey: ["pipeline-status"] });
    },
    onError: (e: any) =>
      toast.error(e?.response?.data?.detail ?? "Could not trigger task"),
  });

  // Toggle the auto-derive-trades opt-in (keeps the beat pipeline scan-only
  // until the operator explicitly turns execution on).
  const configMut = useMutation({
    mutationFn: (enabled: boolean) =>
      api
        .post("/system/pipeline/config/", { auto_execute_enabled: enabled })
        .then((r) => r.data),
    onSuccess: (_d, enabled) => {
      toast.success(enabled ? "Auto-derive trades enabled" : "Auto-derive trades disabled");
      qc.invalidateQueries({ queryKey: ["pipeline-status"] });
    },
    onError: () => toast.error("Could not update setting"),
  });

  // EOD enrichment + trade derivation target the day you're viewing; everything
  // else just runs.
  const runTask = (taskKey: string) => {
    if ((taskKey === "eod_enrichment" || taskKey === "intraday_agent") && data) {
      trigger.mutate({ task: taskKey, date: data.selected_date });
    } else {
      trigger.mutate({ task: taskKey });
    }
  };
  const pendingTask = trigger.isPending
    ? (trigger.variables as TriggerVars).task
    : null;
  const backfillPending =
    trigger.isPending && (trigger.variables as TriggerVars).backfill === true;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1280px] mx-auto">
      <header className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">System</p>
          <h1 className="text-h1 text-fg flex items-center gap-2">
            <Workflow className="h-6 w-6" aria-hidden /> Daily Pipeline
          </h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            The full daily data flow — premarket scan and live screener feed the
            signal ledger, enrichment scores it, the feedback loop learns from it.
          </p>
        </div>
        {data && (
          <div className="flex flex-col items-end gap-2">
            <DatePicker
              value={data.selected_date}
              options={data.date_options}
              onChange={(d) => setDateParam(d)}
            />
            <TradingDayBadge today={data.today} />
          </div>
        )}
      </header>

      {isLoading && (
        <Card><CardContent className="py-10 text-center text-body-sm text-fg-muted">
          Loading pipeline flow…
        </CardContent></Card>
      )}

      {isError && (
        <Card><CardContent className="py-6 text-center text-body-sm text-danger">
          {(error as any)?.response?.status === 403
            ? "Owner access required to view the pipeline."
            : "Could not load pipeline status."}
        </CardContent></Card>
      )}

      {data && (
        <>
          <ViewingBanner data={data} />
          {data.unenriched_total > 0 && (
            <BackfillBanner
              count={data.unenriched_total}
              pending={backfillPending}
              onBackfill={() =>
                trigger.mutate({ task: "eod_enrichment", backfill: true })}
            />
          )}
          <FlowDiagram
            flow={data.flow}
            onRun={runTask}
            pendingKey={pendingTask}
          />
          <ExecutionCard
            data={data}
            onRun={() => runTask("intraday_agent")}
            onToggle={(e) => configMut.mutate(e)}
            runPending={pendingTask === "intraday_agent"}
            togglePending={configMut.isPending}
          />
          <StrategyRoster strategies={data.strategies} />
          <StrategyActivity rows={data.by_strategy} signals={data.signals} />
          <ActivityFeed feed={data.feed} data={data} />
          <RunHistory runs={data.recent_runs} />
        </>
      )}
    </div>
  );
}

// ─── Date picker + banners ────────────────────────────────────────────

function DatePicker({
  value, options, onChange,
}: {
  value: string;
  options: string[];
  onChange: (d: string) => void;
}) {
  // Make sure the current value is selectable even if it's not in options
  // (e.g. today, when today has no signals yet).
  const opts = useMemo(
    () => (options.includes(value) ? options : [value, ...options]),
    [options, value],
  );
  return (
    <label className="flex items-center gap-2 text-body-sm">
      <span className="text-fg-muted">Trading day</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-sm border border-border bg-surface px-2 py-1 text-fg outline-none focus:border-brand"
      >
        {opts.map((d) => (
          <option key={d} value={d}>{fmtDate(d)} · {d}</option>
        ))}
      </select>
    </label>
  );
}

function TradingDayBadge({ today }: { today: PipelineStatus["today"] }) {
  return (
    <div className="flex items-center gap-2">
      {today.is_trading_day
        ? <Badge tone="success" dot>Today: trading day</Badge>
        : <Badge tone="neutral">Today: market closed</Badge>}
      <Badge tone="brand">{today.session_phase}</Badge>
    </div>
  );
}

/** Explains what day is on screen and why — especially when today is empty. */
function ViewingBanner({ data }: { data: PipelineStatus }) {
  if (data.viewing_today) {
    if (data.signals.total > 0) return null;
    return (
      <div className="rounded-md border border-border bg-surface-2 px-4 py-2.5 text-body-sm text-fg-muted">
        Today has no signals yet — the screener session runs 09:15–15:30 IST.
        Pick an earlier day above to review past activity, or force-run a stage below.
      </div>
    );
  }
  return (
    <div className="rounded-md border border-brand/40 bg-brand/5 px-4 py-2.5 text-body-sm">
      <span className="text-fg">Viewing {fmtDate(data.selected_date)}</span>
      <span className="text-fg-muted">
        {" "}— a past trading day. Today ({data.today.date}) has no signal data yet.
      </span>
    </div>
  );
}

/** Shown when signals exist whose EOD outcome hasn't been computed yet. */
function BackfillBanner({
  count, pending, onBackfill,
}: {
  count: number;
  pending: boolean;
  onBackfill: () => void;
}) {
  return (
    <div className="rounded-md border border-warning/40 bg-warning/5 px-4 py-3 flex items-center justify-between gap-3 flex-wrap">
      <div className="flex items-start gap-2">
        <Sparkles className="h-5 w-5 text-warning shrink-0 mt-0.5" aria-hidden />
        <div>
          <div className="text-body-sm font-medium text-fg">
            {count} signal{count === 1 ? "" : "s"} awaiting enrichment
          </div>
          <p className="text-caption text-fg-muted">
            EOD price, max favorable / adverse move and outcome aren't computed
            yet. Backfill fills them in across every date.
          </p>
        </div>
      </div>
      <Button
        size="sm" variant="primary"
        onClick={onBackfill} loading={pending}
        leading={<Sparkles className="h-3.5 w-3.5" />}
      >
        Backfill all
      </Button>
    </div>
  );
}

// ─── Flow diagram ─────────────────────────────────────────────────────

function FlowDiagram({
  flow, onRun, pendingKey,
}: {
  flow: FlowNode[];
  onRun: (key: string) => void;
  pendingKey: string | null;
}) {
  const byKey = useMemo(() => {
    const m: Record<string, FlowNode> = {};
    for (const n of flow) m[n.key] = n;
    return m;
  }, [flow]);

  const sources = flow.filter((n) => n.role === "source");
  const ledger = byKey["ledger"];
  const enrichment = byKey["eod_enrichment"];
  const feedback = byKey["feedback"];
  const ledgerActive = (ledger?.metric ?? 0) > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-h3">Pipeline flow</CardTitle>
        <CardDescription>
          Data flow for the selected day. Stages with a play button can be
          force-run for debugging.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col lg:flex-row lg:items-stretch gap-3">
          <div className="flex flex-col gap-3 lg:w-[230px] shrink-0">
            {sources.map((n) => (
              <FlowNodeCard
                key={n.key} node={n}
                onRun={onRun} pending={pendingKey === n.key}
              />
            ))}
          </div>

          <Connector active={sources.some((s) => s.metric > 0 || s.status === "running")} />

          {ledger && (
            <FlowNodeCard
              node={ledger} onRun={onRun} pending={false}
              className="lg:w-[200px] shrink-0"
            />
          )}

          <Connector active={ledgerActive} />

          {enrichment && (
            <FlowNodeCard
              node={enrichment} onRun={onRun}
              pending={pendingKey === enrichment.key}
              className="lg:w-[200px] shrink-0"
            />
          )}

          <Connector active={(enrichment?.metric ?? 0) > 0} />

          {feedback && (
            <FlowNodeCard
              node={feedback} onRun={onRun} pending={false}
              className="lg:w-[200px] shrink-0"
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Connector({ active }: { active: boolean }) {
  return (
    <div className="flex lg:flex-col items-center justify-center text-fg-subtle">
      <ChevronRight
        className={cn(
          "h-5 w-5 hidden lg:block transition-colors",
          active && "text-brand animate-pulse",
        )}
        aria-hidden
      />
      <ArrowDown
        className={cn(
          "h-5 w-5 lg:hidden transition-colors",
          active && "text-brand animate-pulse",
        )}
        aria-hidden
      />
    </div>
  );
}

function FlowNodeCard({
  node, onRun, pending, className,
}: {
  node: FlowNode;
  onRun: (key: string) => void;
  pending: boolean;
  className?: string;
}) {
  const Icon = NODE_ICON[node.key] ?? Activity;
  const accent = nodeAccent(node.status);
  const runnable = RUNNABLE.has(node.key);
  const screenerLive = node.key === "screener_session" && node.live;

  return (
    <div
      className={cn(
        "rounded-md border bg-surface p-3 flex flex-col gap-2 relative",
        accent.ring,
        node.status === "running" && "shadow-[0_0_0_3px] shadow-brand/10",
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Icon className={cn("h-4 w-4 shrink-0", accent.text)} aria-hidden />
          <span className="text-body-sm font-medium text-fg truncate">{node.label}</span>
        </div>
        <span className={cn(
          "h-2 w-2 rounded-full shrink-0", accent.dot,
          node.status === "running" && "animate-pulse",
        )} />
      </div>

      <div>
        <div className={cn("text-h2 tabular-nums leading-none", accent.text)}>
          {node.metric}
        </div>
        <div className="text-caption text-fg-muted mt-0.5">{node.metric_label}</div>
      </div>

      <div className="flex items-center justify-between gap-2 mt-auto pt-1">
        <span className="text-caption text-fg-subtle capitalize">
          {screenerLive ? "live session" : node.status}
        </span>
        {runnable && (
          <Button
            size="sm"
            variant={node.status === "running" || screenerLive ? "secondary" : "ghost"}
            onClick={() => onRun(node.key)}
            loading={pending}
            disabled={pending || screenerLive}
            leading={<Play className="h-3 w-3" />}
          >
            {screenerLive ? "Live" : "Run"}
          </Button>
        )}
      </div>
    </div>
  );
}

// ─── Strategy roster ──────────────────────────────────────────────────

/**
 * Execution card — the one pipeline stage that creates trades (not just
 * signals). Kept visually distinct from the scan-only data stages and gated
 * behind an explicit opt-in toggle, so the pipeline stays scan-only by default.
 */
function ExecutionCard({
  data, onRun, onToggle, runPending, togglePending,
}: {
  data: PipelineStatus;
  onRun: () => void;
  onToggle: (enabled: boolean) => void;
  runPending: boolean;
  togglePending: boolean;
}) {
  const task = data.tasks?.find((t) => t.key === "intraday_agent");
  if (!task) return null;
  const lastRun = data.recent_runs.find((r) => r.task === "intraday_agent");
  const enabled = !!data.auto_execute_enabled;
  const closed = lastRun?.summary?.["trades_closed"] as number | undefined;

  return (
    <Card className="border-warning/30">
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div className="min-w-0">
          <CardTitle className="flex items-center gap-2">
            {task.label}
            <Badge tone="warning">Execution</Badge>
            <Badge tone="neutral">Paper</Badge>
          </CardTitle>
          <CardDescription>{task.description}</CardDescription>
        </div>
        <div className="text-caption text-fg-subtle text-right shrink-0">{task.schedule}</div>
      </CardHeader>
      <CardContent className="flex items-center justify-between gap-4 flex-wrap">
        <div className="text-caption text-fg-muted">
          {lastRun
            ? <>Last run <span className="text-fg">{lastRun.status}</span>
                {closed != null && <> · {closed} trades closed</>}
                {" · "}{new Date(lastRun.started_at).toLocaleString()}</>
            : "Never run — turn on auto-derive or run a day on demand."}
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => onToggle(!enabled)}
            disabled={togglePending}
            className={cn(
              "flex items-center gap-1.5 rounded-sm border px-2.5 py-1.5 text-caption transition-colors",
              enabled
                ? "border-pnl-up/40 bg-pnl-up/10 text-pnl-up"
                : "border-border/60 bg-surface-2 text-fg-subtle hover:text-fg",
            )}
            title={enabled
              ? "Auto-derive trades each trading day at 16:30 IST — click to disable"
              : "Enable automatic daily trade derivation (paper)"}
          >
            <span className={cn("h-1.5 w-1.5 rounded-full",
              enabled ? "bg-pnl-up animate-pulse" : "bg-fg-subtle")} />
            Auto-derive {enabled ? "On" : "Off"}
          </button>
          <Button size="sm" variant="secondary" onClick={onRun} disabled={runPending}>
            {runPending ? "Running…" : `Run for ${data.selected_date}`}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

const CATEGORY_TONE: Record<string, "brand" | "warning" | "neutral"> = {
  Scanner: "brand",
  Execution: "warning",
  Options: "neutral",
  Tooling: "neutral",
};

function StrategyRoster({ strategies }: { strategies: StrategyInfo[] }) {
  const wired = strategies.filter((s) => s.wired).length;
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <div>
          <CardTitle className="text-h3 flex items-center gap-2">
            <Layers className="h-4 w-4" aria-hidden /> Strategy roster
          </CardTitle>
          <CardDescription>
            Every registered strategy plugin. Scanners feed the daily pipeline;
            execution strategies run on-demand from their own desks.
          </CardDescription>
        </div>
        <Badge tone="neutral">{wired}/{strategies.length} in pipeline</Badge>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-body-sm">
            <thead>
              <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
                <th className="text-left font-medium px-4 py-2">Strategy</th>
                <th className="text-left font-medium px-4 py-2">Category</th>
                <th className="text-left font-medium px-4 py-2">Automation</th>
                <th className="text-right font-medium px-4 py-2">Signals (day)</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s) => (
                <tr key={s.name} className="border-b border-border last:border-0">
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-fg">{s.label}</span>
                      {s.wired
                        ? <Badge tone="success" dot>In pipeline</Badge>
                        : <Badge tone="neutral">On-demand</Badge>}
                    </div>
                    {s.description && (
                      <div className="text-caption text-fg-muted">{s.description}</div>
                    )}
                  </td>
                  <td className="px-4 py-2">
                    <Badge tone={CATEGORY_TONE[s.category] ?? "neutral"}>
                      {s.category}
                    </Badge>
                  </td>
                  <td className="px-4 py-2 text-fg-muted">
                    {s.automation === "daily" ? (
                      <span className="text-success">Auto · daily</span>
                    ) : (
                      <span>Manual trigger</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums">
                    {s.signal_source ? (
                      <span className={s.signals > 0 ? "text-fg" : "text-fg-subtle"}>
                        {s.signals}
                      </span>
                    ) : (
                      <span className="text-fg-subtle">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Strategy activity ────────────────────────────────────────────────

function StrategyActivity({
  rows, signals,
}: {
  rows: StrategyStat[];
  signals: PipelineStatus["signals"];
}) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <div>
          <CardTitle className="text-h3">Strategy activity</CardTitle>
          <CardDescription>
            Which strategies fired on this day, and how their signals resolved.
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone="neutral">{signals.total} signals</Badge>
          {signals.traded > 0 && <Badge tone="success">{signals.traded} traded</Badge>}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {rows.length === 0 ? (
          <div className="py-8 text-center text-body-sm text-fg-muted">
            No strategy fired a signal on this day.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-body-sm">
              <thead>
                <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
                  <th className="text-left font-medium px-4 py-2">Strategy</th>
                  <th className="text-left font-medium px-4 py-2">Source</th>
                  <th className="text-right font-medium px-4 py-2">Signals</th>
                  <th className="text-right font-medium px-4 py-2">Buy / Sell</th>
                  <th className="text-right font-medium px-4 py-2">Traded</th>
                  <th className="text-right font-medium px-4 py-2">Avg conf.</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={`${r.strategy}:${r.source}`}
                    className="border-b border-border last:border-0"
                  >
                    <td className="px-4 py-2 text-fg font-medium">{r.strategy}</td>
                    <td className="px-4 py-2">
                      <Badge tone="brand">{STAGE_LABEL[
                        r.source === "SCREENER" ? "screener_session"
                        : r.source === "OK_SCANNER" ? "swing_scan"
                        : r.source.toLowerCase()
                      ] ?? r.source}</Badge>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-fg">{r.count}</td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      <span className="text-success">{r.buy}</span>
                      <span className="text-fg-subtle"> / </span>
                      <span className="text-danger">{r.sell}</span>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-fg-muted">
                      {r.traded || "—"}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-fg-muted">
                      {r.avg_confidence ? `${(r.avg_confidence * 100).toFixed(0)}%` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Activity feed ────────────────────────────────────────────────────

function ActivityFeed({ feed, data }: { feed: FeedItem[]; data: PipelineStatus }) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <div>
          <CardTitle className="text-h3 flex items-center gap-2">
            <Activity className="h-4 w-4" aria-hidden /> Activity feed
          </CardTitle>
          <CardDescription>
            Every signal and run on {fmtDate(data.selected_date)} — newest first
            {data.viewing_today && " · live, refreshes every 6s"}.
          </CardDescription>
        </div>
        <Badge tone="neutral">{feed.length} events</Badge>
      </CardHeader>
      <CardContent className="p-0">
        {feed.length === 0 ? (
          <div className="py-10 text-center text-body-sm text-fg-muted">
            No activity recorded on this day.
          </div>
        ) : (
          <ul className="max-h-[460px] overflow-y-auto divide-y divide-border">
            {feed.map((item, i) => (
              <FeedRow key={`${item.ts}:${i}`} item={item} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function FeedRow({ item }: { item: FeedItem }) {
  const time = new Date(item.ts).toLocaleTimeString("en-IN", {
    hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit",
  });

  if (item.kind === "run") {
    const ok = item.status === "success";
    const failed = item.status === "failed";
    return (
      <li className="flex items-center gap-3 px-4 py-2">
        <span className="text-caption tabular-nums text-fg-subtle w-16 shrink-0">{time}</span>
        {ok && <CheckCircle2 className="h-4 w-4 text-success shrink-0" aria-hidden />}
        {failed && <XCircle className="h-4 w-4 text-danger shrink-0" aria-hidden />}
        {item.status === "running" && (
          <Loader2 className="h-4 w-4 text-brand animate-spin shrink-0" aria-hidden />
        )}
        <Badge tone="neutral">{STAGE_LABEL[item.stage] ?? item.stage}</Badge>
        <div className="min-w-0 flex-1">
          <div className="text-body-sm text-fg truncate">{item.title}</div>
          {item.detail && (
            <div className={cn(
              "text-caption truncate",
              failed ? "text-danger" : "text-fg-muted",
            )}>{item.detail}</div>
          )}
        </div>
        {item.trigger && (
          <span className="text-caption text-fg-subtle shrink-0">{item.trigger}</span>
        )}
      </li>
    );
  }

  const buy = item.side === "BUY";
  return (
    <li className="flex items-center gap-3 px-4 py-2">
      <span className="text-caption tabular-nums text-fg-subtle w-16 shrink-0">{time}</span>
      <span
        className={cn("h-2 w-2 rounded-full shrink-0", buy ? "bg-success" : "bg-danger")}
        aria-hidden
      />
      <Badge tone="brand">{STAGE_LABEL[item.stage] ?? item.stage}</Badge>
      <div className="min-w-0 flex-1">
        <div className="text-body-sm text-fg truncate">
          <span className={cn("font-medium", buy ? "text-success" : "text-danger")}>
            {item.side}
          </span>{" "}
          {item.title.replace(/^(BUY|SELL)\s/, "")}
        </div>
        {item.detail && (
          <div className="text-caption text-fg-muted truncate">{item.detail}</div>
        )}
      </div>
      {item.outcome && item.outcome !== "PENDING" && (
        <Badge tone={item.outcome === "TRADED" ? "success" : "neutral"}>
          {item.outcome}
        </Badge>
      )}
    </li>
  );
}

// ─── Run history ──────────────────────────────────────────────────────

const TASK_LABELS: Record<string, string> = {
  swing_scan: "Swing Scan",
  screener_session: "Screener Session",
  eod_enrichment: "EOD Enrichment",
};

function RunHistory({ runs }: { runs: PipelineRun[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-h3">Run history</CardTitle>
        <CardDescription>Last {runs.length} pipeline executions (all dates).</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        {runs.length === 0 ? (
          <div className="py-8 text-center text-body-sm text-fg-muted">
            No pipeline runs recorded yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-body-sm">
              <thead>
                <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
                  <th className="text-left font-medium px-4 py-2">Task</th>
                  <th className="text-left font-medium px-4 py-2">Status</th>
                  <th className="text-left font-medium px-4 py-2">Trigger</th>
                  <th className="text-left font-medium px-4 py-2">Started</th>
                  <th className="text-right font-medium px-4 py-2">Duration</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id} className="border-b border-border last:border-0">
                    <td className="px-4 py-2 text-fg">
                      {TASK_LABELS[run.task] ?? run.task}
                    </td>
                    <td className="px-4 py-2">
                      {run.status === "success" && <Badge tone="success" dot>Success</Badge>}
                      {run.status === "failed" && <Badge tone="danger" dot>Failed</Badge>}
                      {run.status === "running" && <Badge tone="brand" dot>Running</Badge>}
                    </td>
                    <td className="px-4 py-2 text-fg-muted">{run.trigger}</td>
                    <td className="px-4 py-2 text-fg-muted">{fmtRel(run.started_at)}</td>
                    <td className="px-4 py-2 text-right tabular-nums text-fg-muted">
                      {fmtDuration(run.duration_seconds)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
