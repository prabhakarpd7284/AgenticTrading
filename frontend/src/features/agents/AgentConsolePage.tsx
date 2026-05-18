import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle, Bot, Brain, ChevronRight, CircleDot, Clock, Database,
  Play, Radio, Send, ShieldCheck, Sparkles, Terminal, Wifi, WifiOff, Wrench,
  type LucideIcon,
} from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { connect } from "@/lib/ws";
import type { AgentEvent, AgentRun, Portfolio, StrategySchema } from "@/types";
import { cn, fmtRel } from "@/lib/utils";
import { useAuditFeed } from "@/lib/v2";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";
import {
  Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger,
} from "@/components/ui/Dialog";
import { OpButton } from "@/features/ops/OpButton";

export function AgentConsolePage() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const { runId } = useParams<{ runId?: string }>();
  const [params, setParams] = useSearchParams();

  const openNew = params.get("new") === "1";
  const [newOpen, setNewOpen] = React.useState(openNew);
  React.useEffect(() => setNewOpen(openNew), [openNew]);

  /* ---------- queries ---------- */
  const { data: catalog = [] } = useQuery({
    queryKey: ["strategy-catalog"],
    queryFn: () => api.get<StrategySchema[]>("/agents/catalog/").then((r) => r.data),
  });
  const { data: portfolios = [] } = useQuery({
    queryKey: ["portfolios"],
    queryFn: () => api.get<Portfolio[]>("/portfolios/").then((r) => r.data),
  });
  const { data: runs = [] } = useQuery({
    queryKey: ["agent-runs"],
    queryFn: () => api.get<AgentRun[]>("/agents/runs/").then((r) => r.data),
    refetchInterval: 8_000,
  });

  // Legacy audit feed — shown alongside v2 runs so the console has real
  // content on day one (the legacy DB has 398 AuditLog rows from the
  // Streamlit-era pipelines).
  const { data: legacyAudit = [] } = useAuditFeed(50);

  /* ---------- selected run + stream ---------- */
  const [events, setEvents] = React.useState<AgentEvent[]>([]);
  // Connection status drives the inline banner. "connecting" covers both the
  // initial handshake and exponential-backoff reconnects (lib/ws auto-retries
  // up to 30s). "closed_auth" is a terminal state — the server rejected the
  // JWT and the lib won't retry. "live" means the socket is open right now.
  type WsState = "connecting" | "live" | "reconnecting" | "closed_auth";
  const [wsState, setWsState] = React.useState<WsState>("connecting");
  const feedRef = React.useRef<HTMLDivElement>(null);
  const wsRef = React.useRef<ReturnType<typeof connect>>();

  const selected = runs.find((r) => r.id === runId) ?? runs[0];

  React.useEffect(() => {
    setEvents([]);
    wsRef.current?.close();
    if (!selected) return;
    setWsState("connecting");
    wsRef.current = connect(
      `/ws/agents/${selected.id}/`,
      (msg) => setEvents((prev) => [...prev, msg as unknown as AgentEvent]),
      {
        onOpen: () => setWsState("live"),
        onClose: (ev) => {
          // 4401/4403 = auth failure, lib stops retrying.
          if (ev.code === 4401 || ev.code === 4403) setWsState("closed_auth");
          else setWsState("reconnecting");
        },
      },
    );
    return () => wsRef.current?.close();
  }, [selected?.id]);

  // autoscroll to newest event
  React.useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);

  /* ---------- new run mutation ---------- */
  const startMut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api.post<AgentRun>("/agents/runs/", body).then((r) => r.data),
    onSuccess: (data) => {
      toast.success("Run queued");
      qc.invalidateQueries({ queryKey: ["agent-runs"] });
      setNewOpen(false);
      setParams({});
      nav(`/agents/${data.id}`);
    },
    onError: () => toast.error("Failed to start run"),
  });

  return (
    <div className="grid grid-cols-[320px_1fr] min-h-[calc(100vh-3.5rem)]">
      {/* ----- run list rail ----- */}
      <aside className="border-r border-border bg-surface/50 flex flex-col min-h-0">
        <div className="h-12 flex items-center justify-between gap-2 px-4 border-b border-border sticky top-0 bg-surface/80 backdrop-blur z-sticky">
          <h2 className="text-body-sm font-semibold text-fg">Runs</h2>
          <Dialog open={newOpen} onOpenChange={(o) => { setNewOpen(o); if (!o) setParams({}); }}>
            <DialogTrigger asChild>
              <Button size="sm" leading={<Play className="h-3.5 w-3.5" />}>New</Button>
            </DialogTrigger>
            <NewRunDialog
              catalog={catalog}
              portfolios={portfolios}
              pending={startMut.isPending}
              onSubmit={(v) => startMut.mutate(v)}
            />
          </Dialog>
        </div>

        <div className="flex-1 overflow-auto">
          {runs.length === 0 && legacyAudit.length === 0 ? (
            <EmptyState
              className="m-3"
              icon={<Sparkles />}
              title="No runs yet"
              description="Start your first run to see the desk think."
            />
          ) : (
            <>
              {runs.length > 0 && (
                <ul className="divide-y divide-border">
                  {runs.map((r) => (
                    <li key={r.id}>
                      <button
                        onClick={() => nav(`/agents/${r.id}`)}
                        className={cn(
                          "w-full text-left px-4 py-3 flex gap-2 hover:bg-surface-2",
                          selected?.id === r.id && "bg-surface-2 border-l-2 border-l-accent",
                        )}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <RunDot status={r.status} />
                            <span className="text-body-sm text-fg truncate">{r.strategy_name}</span>
                          </div>
                          <div className="text-caption text-fg-subtle mt-0.5 font-mono">
                            {r.id.slice(0, 8)} · {fmtRel(r.created_at)} ago
                          </div>
                        </div>
                        <ChevronRight className="h-4 w-4 text-fg-subtle self-center" aria-hidden />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {legacyAudit.length > 0 && (
                <div className="border-t border-border">
                  <div className="px-4 py-2 text-caption uppercase tracking-wider text-fg-subtle bg-surface-2/40">
                    Legacy audit log · {legacyAudit.length}
                  </div>
                  <ul className="divide-y divide-border">
                    {legacyAudit.map((e, i) => (
                      <li key={i} className="px-4 py-2.5 flex items-start gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="text-body-sm text-fg truncate">{e.detail}</div>
                          <div className="text-caption text-fg-subtle font-mono">
                            {e.time}{e.symbol && ` · ${e.symbol}`}
                          </div>
                        </div>
                        <Badge tone="neutral" className="shrink-0">{e.type.split("_")[0]}</Badge>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      </aside>

      {/* ----- stream pane ----- */}
      <section className="flex flex-col min-w-0">
        {!selected ? (
          <div className="flex-1 flex items-center justify-center p-6">
            <EmptyState
              icon={<Bot />}
              title="Select or start a run"
              description="The left rail lists recent runs; press New to start one."
              action={<Button onClick={() => setNewOpen(true)} leading={<Play className="h-4 w-4" />}>New run</Button>}
            />
          </div>
        ) : (
          <RunDetail run={selected} events={events} feedRef={feedRef} wsState={wsState} />
        )}
      </section>
    </div>
  );
}

/* =================================================================== */
/* Run detail                                                           */
/* =================================================================== */
function RunDetail({
  run, events, feedRef, wsState,
}: {
  run: AgentRun;
  events: AgentEvent[];
  feedRef: React.RefObject<HTMLDivElement>;
  wsState: "connecting" | "live" | "reconnecting" | "closed_auth";
}) {
  const planEvt = events.find((e) => e.node === "planner" && e.type === "result");
  const riskEvt = events.find((e) => e.node === "risk"    && e.type === "result");
  const execEvt = events.find((e) => e.node === "execute" && e.type === "result");

  const lastTs = events.length ? events[events.length - 1].ts : undefined;
  const kpis = computeKpis(events);
  const cli = cliForStrategy(run.strategy_name);

  return (
    <>
      <header className="h-12 px-5 border-b border-border flex items-center gap-3 sticky top-0 bg-bg/80 backdrop-blur z-sticky">
        <RunStatusPill status={run.status} />
        <div className="flex-1 min-w-0">
          <div className="text-body-sm text-fg truncate">
            {run.strategy_name} <span className="text-fg-subtle">v{run.strategy_version}</span>
          </div>
          <div className="text-caption text-fg-subtle font-mono">
            run {run.id} · started {fmtRel(run.started_at)} ago
          </div>
        </div>
        {/* Live freshness pill — reads the timestamp the backend stamps on each
            WS event. Operators glance here to confirm the stream is healthy. */}
        <FreshnessIndicator
          label="Last event"
          timestamp={lastTs}
          freshMs={3_000}
          staleMs={30_000}
        />
        {cli && (
          <OpButton
            command={cli.command}
            defaultArgs={cli.args}
            label="Run via CLI"
            description={cli.description}
            icon={<Terminal className="mr-1.5 size-4" />}
            variant="secondary"
            size="sm"
          />
        )}
        <Badge tone="neutral">
          <Clock className="h-3 w-3 mr-1" aria-hidden /> {fmtRel(run.created_at)}
        </Badge>
      </header>

      <WsStatusBanner state={wsState} />

      {/* KPI strip — read at a glance: how many events, distinct steps, LLM
          calls so far, time elapsed since the first event. */}
      <KpiStrip kpis={kpis} />

      {/* Risk breach banner */}
      {riskEvt && (riskEvt.payload as any)?.approved === false && (
        <div role="alert" className="mx-5 mt-3 rounded-md border border-danger/40 bg-pnl-down/5 p-4 flex gap-3 items-start">
          <AlertTriangle className="h-5 w-5 text-danger shrink-0 mt-0.5" aria-hidden />
          <div className="flex-1">
            <div className="text-body-sm font-semibold text-fg">@RiskGuard blocked this plan</div>
            <p className="text-body-sm text-fg-muted mt-0.5">
              {(riskEvt.payload as any)?.reason ??
                "The plan violates a deterministic risk rule. No trade was placed."}
            </p>
          </div>
          <Button variant="secondary" size="sm" disabled>Execute (blocked)</Button>
        </div>
      )}

      <Tabs defaultValue="stream" className="flex-1 min-h-0 flex flex-col">
        <TabsList className="px-5">
          <TabsTrigger value="stream">Stream</TabsTrigger>
          <TabsTrigger value="plan">Plan</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="execution">Execution</TabsTrigger>
        </TabsList>

        <TabsContent value="stream" className="flex-1 min-h-0 px-5 pb-5">
          <div
            ref={feedRef}
            className="h-full overflow-auto rounded-md border border-border bg-surface p-4 space-y-3"
            aria-live="polite"
            aria-label="Agent event stream"
          >
            {events.length === 0 ? (
              <StreamSkeleton wsState={wsState} />
            ) : (
              events.map((e, i) => (
                <EventBubble key={e.seq} ev={e} prev={i > 0 ? events[i - 1] : undefined} />
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="plan" className="px-5 pb-5">
          <JsonCard title="Plan" payload={planEvt?.payload} emptyHint="Planner hasn't run yet." />
        </TabsContent>
        <TabsContent value="risk" className="px-5 pb-5">
          <JsonCard title="Risk check" payload={riskEvt?.payload} emptyHint="No risk result yet." />
        </TabsContent>
        <TabsContent value="execution" className="px-5 pb-5">
          <JsonCard title="Execution result" payload={execEvt?.payload} emptyHint="Not executed." />
        </TabsContent>
      </Tabs>
    </>
  );
}

function EventBubble({ ev, prev }: { ev: AgentEvent; prev?: AgentEvent }) {
  const a = agentForNode(ev.node);
  const kind = inferStepKind(ev.node);
  const KindIcon = kind.Icon;
  const ts = ev.ts ? new Date(ev.ts) : null;
  const delta = ev.ts && prev?.ts
    ? new Date(ev.ts).getTime() - new Date(prev.ts).getTime()
    : null;
  const clock = ts ? ts.toLocaleTimeString("en-IN", { hour12: false }) : null;
  const deltaStr = delta != null
    ? delta < 1000 ? `+${delta}ms` : `+${(delta / 1000).toFixed(1)}s`
    : null;

  if (ev.type === "token") {
    return (
      <div className="flex gap-2">
        <KindIcon className={cn("h-4 w-4 shrink-0 mt-0.5", kind.colorCls)} aria-hidden />
        <Badge tone={a.tone}>{a.label}</Badge>
        <span className="text-body-sm text-fg whitespace-pre-wrap">{String((ev.payload as any)?.text ?? "")}</span>
      </div>
    );
  }

  const isError = ev.type === "error";
  return (
    <div className={cn(
      "rounded-sm border p-3",
      isError ? "border-danger/40 bg-pnl-down/5" : "border-border bg-surface-2",
    )}>
      <div className="flex items-center gap-2 mb-1">
        <KindIcon
          className={cn("h-4 w-4 shrink-0", isError ? "text-danger" : kind.colorCls)}
          aria-label={kind.label}
        />
        <Badge tone={isError ? "danger" : a.tone}>{a.label}</Badge>
        <span className="text-caption text-fg-subtle font-mono uppercase tracking-wider">{ev.type}</span>
        {clock && (
          <span className="text-caption text-fg-subtle font-mono tabular ml-auto" title={ts?.toISOString()}>
            {clock}{deltaStr && <span className="text-fg-subtle/70"> · {deltaStr}</span>}
          </span>
        )}
        <span className={cn("text-caption text-fg-subtle font-mono", !clock && "ml-auto")}>#{ev.seq}</span>
      </div>
      <pre className="text-caption font-mono text-fg-muted whitespace-pre-wrap break-all max-h-64 overflow-auto">
        {safeStringify(ev.payload)}
      </pre>
    </div>
  );
}

/* ---------- KPI strip ---------- */
function KpiStrip({ kpis }: { kpis: KpiSummary }) {
  return (
    <dl className="mx-5 mt-5 grid grid-cols-4 gap-2 rounded-md border border-border bg-surface-2/40 p-2">
      <KpiCell label="Events" value={String(kpis.eventCount)} />
      <KpiCell label="Steps" value={String(kpis.distinctNodes)} />
      <KpiCell label="LLM calls" value={String(kpis.llmCalls)} />
      <KpiCell
        label="Elapsed"
        value={kpis.elapsedMs == null ? "—" : formatElapsed(kpis.elapsedMs)}
      />
    </dl>
  );
}

function KpiCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-3 py-1.5">
      <dt className="text-caption text-fg-subtle uppercase tracking-wider">{label}</dt>
      <dd className="text-body-sm text-fg font-mono tabular mt-0.5">{value}</dd>
    </div>
  );
}

/* ---------- WS status banner ---------- */
function WsStatusBanner({ state }: { state: "connecting" | "live" | "reconnecting" | "closed_auth" }) {
  if (state === "live" || state === "connecting") return null;
  if (state === "reconnecting") {
    return (
      <div role="status" className="mx-5 mt-3 rounded-md border border-warn/40 bg-warn/5 px-3 py-2 flex items-center gap-2 text-body-sm text-fg-muted">
        <WifiOff className="h-4 w-4 text-warn" aria-hidden />
        <span>Connection lost — reconnecting…</span>
      </div>
    );
  }
  // closed_auth — terminal; user has to refresh or re-auth.
  return (
    <div role="alert" className="mx-5 mt-3 rounded-md border border-danger/40 bg-pnl-down/5 px-3 py-2 flex items-center gap-2 text-body-sm text-fg">
      <WifiOff className="h-4 w-4 text-danger" aria-hidden />
      <span>Stream auth failed. Refresh the page to retry.</span>
    </div>
  );
}

/* ---------- Skeleton while waiting for first event ---------- */
function StreamSkeleton({ wsState }: { wsState: "connecting" | "live" | "reconnecting" | "closed_auth" }) {
  const subline =
    wsState === "live"
      ? "Connected — waiting for the first event."
      : wsState === "reconnecting"
        ? "Reconnecting to the run stream…"
        : wsState === "closed_auth"
          ? "Stream auth failed. Refresh the page to retry."
          : "Connecting to the run stream…";
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-body-sm text-fg-subtle">
        {wsState === "live" ? (
          <Wifi className="h-4 w-4 text-pnl-up" aria-hidden />
        ) : (
          <WifiOff className="h-4 w-4 text-fg-subtle" aria-hidden />
        )}
        {subline}
      </div>
      <Skeleton className="h-16 w-full" />
      <Skeleton className="h-16 w-5/6" />
      <Skeleton className="h-16 w-4/6" />
    </div>
  );
}

function JsonCard({ title, payload, emptyHint }: { title: string; payload?: unknown; emptyHint: string }) {
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent>
        {payload ? (
          <pre className="text-caption font-mono text-fg whitespace-pre-wrap max-h-96 overflow-auto">
            {safeStringify(payload)}
          </pre>
        ) : (
          <p className="text-body-sm text-fg-subtle">{emptyHint}</p>
        )}
      </CardContent>
    </Card>
  );
}

/* =================================================================== */
/* New run dialog                                                       */
/* =================================================================== */
function NewRunDialog({
  catalog, portfolios, pending, onSubmit,
}: {
  catalog: StrategySchema[];
  portfolios: Portfolio[];
  pending: boolean;
  onSubmit: (v: Record<string, unknown>) => void;
}) {
  const [strategy, setStrategy] = React.useState(catalog[0]?.name ?? "directional");
  const [prompt,   setPrompt]   = React.useState("");
  React.useEffect(() => {
    if (catalog.length && !strategy) setStrategy(catalog[0].name);
  }, [catalog, strategy]);

  return (
    <DialogContent className="w-[min(92vw,520px)]">
      <DialogTitle>Start agent run</DialogTitle>
      <DialogDescription>
        Pick a strategy — the desk will plan, pass @RiskGuard, then execute in paper mode.
      </DialogDescription>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!portfolios[0]) return;
          onSubmit({
            strategy_name: strategy,
            portfolio: portfolios[0].id,
            config: prompt ? { prompt } : {},
          });
        }}
        className="mt-4 space-y-4"
      >
        <div>
          <label className="text-body-sm text-fg mb-1.5 inline-block">Strategy</label>
          <div className="grid grid-cols-1 gap-1.5">
            {catalog.map((s) => (
              <label
                key={s.name}
                className={cn(
                  "flex items-center gap-3 rounded-sm border p-3 cursor-pointer",
                  strategy === s.name
                    ? "border-accent/60 bg-accent/5"
                    : "border-border hover:border-border-strong hover:bg-surface-2",
                )}
              >
                <input
                  type="radio"
                  name="strategy"
                  value={s.name}
                  checked={strategy === s.name}
                  onChange={() => setStrategy(s.name)}
                  className="sr-only"
                />
                <CircleDot className={cn("h-4 w-4", strategy === s.name ? "text-accent" : "text-fg-subtle")} aria-hidden />
                <div className="flex-1 min-w-0">
                  <div className="text-body-sm text-fg">
                    {s.name} <span className="text-fg-subtle">· {s.asset_class} · v{s.version}</span>
                  </div>
                  <div className="text-caption text-fg-subtle">
                    Needs: {s.required_retrievers?.join(", ") || "—"}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        <Input
          label="Prompt (optional)"
          hint="Example: Plan a BUY trade for HDFCBANK if the 15m breakout confirms."
          placeholder="Describe the situation or leave blank to use defaults"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <div className="rounded-sm border border-border bg-surface-2/50 p-3 flex gap-2 items-start">
          <ShieldCheck className="h-4 w-4 text-accent mt-0.5" aria-hidden />
          <p className="text-caption text-fg-muted">
            @RiskGuard will deterministically validate every plan before execution. No LLM bypass.
          </p>
        </div>

        <div className="flex items-center justify-end gap-2">
          <Button type="submit" loading={pending} leading={<Send className="h-4 w-4" />}>
            Start run
          </Button>
        </div>
      </form>
    </DialogContent>
  );
}

/* =================================================================== */
/* helpers                                                              */
/* =================================================================== */
function RunStatusPill({ status }: { status: AgentRun["status"] }) {
  const map = {
    queued:    { tone: "neutral" as const, label: "Queued" },
    running:   { tone: "info" as const,    label: "Running" },
    succeeded: { tone: "success" as const, label: "Succeeded" },
    failed:    { tone: "danger" as const,  label: "Failed" },
    cancelled: { tone: "neutral" as const, label: "Cancelled" },
  }[status];
  return <Badge tone={map.tone} dot>{map.label}</Badge>;
}

function RunDot({ status }: { status: AgentRun["status"] }) {
  const color = {
    queued:    "bg-fg-subtle",
    running:   "bg-info animate-pulse motion-reduce:animate-none",
    succeeded: "bg-pnl-up",
    failed:    "bg-pnl-down",
    cancelled: "bg-fg-subtle",
  }[status];
  return <span aria-hidden className={cn("h-2 w-2 rounded-full shrink-0", color)} />;
}

function safeStringify(v: unknown) {
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}

/* =================================================================== */
/* Agent + StepKind classification                                      */
/* =================================================================== */

// Persona labels per workflow node — these are stable across the three
// workflows (equity / straddle / pyramid). Keep additions in lock-step
// with the strategy plugins' graph nodes.
const AGENT_BY_NODE: Record<string, { label: string; tone: "brand" | "info" | "warning" | "success" | "danger" }> = {
  fetch_data:       { label: "@DataAnalyst",      tone: "info"    },
  retrieve_context: { label: "@PortfolioTracker", tone: "brand"   },
  planner:          { label: "@DirectionalTrader",tone: "brand"   },
  generate_action:  { label: "@OptionsStrategist",tone: "brand"   },
  risk:             { label: "@RiskGuard",        tone: "warning" },
  validate_action:  { label: "@RiskGuard",        tone: "warning" },
  execute:          { label: "@Broker",           tone: "success" },
  journal:          { label: "@Journal",          tone: "info"    },
  init:             { label: "@System",           tone: "info"    },
};

export function agentForNode(node: string) {
  return AGENT_BY_NODE[node] ?? { label: node, tone: "info" as const };
}

// StepKind icons follow the redesign-v2 plan: 🧠 LLM, 🛡 risk, 📡 broker,
// ⚙ deterministic service, 📓 persistence. Operators learn the symbols
// fast and can scan a long timeline at a glance.
export type StepKind = "llm" | "risk" | "broker" | "service" | "persist" | "init";

interface KindInfo {
  kind: StepKind;
  Icon: LucideIcon;
  colorCls: string;
  label: string;
}

const STEP_KIND_BY_NODE: Record<string, KindInfo> = {
  fetch_data:       { kind: "service", Icon: Wrench,   colorCls: "text-fg-muted", label: "Deterministic service" },
  retrieve_context: { kind: "service", Icon: Wrench,   colorCls: "text-fg-muted", label: "Deterministic service" },
  planner:          { kind: "llm",     Icon: Brain,    colorCls: "text-brand",    label: "LLM step" },
  generate_action:  { kind: "llm",     Icon: Brain,    colorCls: "text-brand",    label: "LLM step" },
  risk:             { kind: "risk",    Icon: ShieldCheck,colorCls: "text-warn",   label: "Risk engine" },
  validate_action:  { kind: "risk",    Icon: ShieldCheck,colorCls: "text-warn",   label: "Risk engine" },
  execute:          { kind: "broker",  Icon: Radio,    colorCls: "text-pnl-up",   label: "Broker call" },
  journal:          { kind: "persist", Icon: Database, colorCls: "text-fg-muted", label: "Persistence" },
  init:             { kind: "init",    Icon: Sparkles, colorCls: "text-fg-subtle",label: "Init" },
};

export function inferStepKind(node: string): KindInfo {
  return STEP_KIND_BY_NODE[node] ?? {
    kind: "service",
    Icon: Wrench,
    colorCls: "text-fg-subtle",
    label: node,
  };
}

/* =================================================================== */
/* KPI computation                                                      */
/* =================================================================== */
export interface KpiSummary {
  eventCount: number;
  distinctNodes: number;
  llmCalls: number;
  /** ms between first and last event with ts; null if <2 timestamped events. */
  elapsedMs: number | null;
}

export function computeKpis(events: AgentEvent[]): KpiSummary {
  const nodes = new Set<string>();
  let llmCalls = 0;
  let firstTs: number | null = null;
  let lastTs: number | null = null;
  let timestamped = 0;

  for (const ev of events) {
    nodes.add(ev.node);
    // Count distinct LLM-kind steps that produced a result (not every token).
    // A 50-token streaming planner shouldn't read as "50 LLM calls".
    if (inferStepKind(ev.node).kind === "llm" && ev.type === "result") {
      llmCalls += 1;
    }
    if (ev.ts) {
      const t = new Date(ev.ts).getTime();
      if (!Number.isNaN(t)) {
        timestamped += 1;
        if (firstTs == null || t < firstTs) firstTs = t;
        if (lastTs == null || t > lastTs) lastTs = t;
      }
    }
  }

  // "Elapsed" needs a span. With <2 timestamped events the KPI strip renders
  // "—" rather than misleading "0ms".
  const elapsedMs =
    timestamped >= 2 && firstTs != null && lastTs != null
      ? lastTs - firstTs
      : null;

  return {
    eventCount: events.length,
    distinctNodes: nodes.size,
    llmCalls,
    elapsedMs,
  };
}

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

/* =================================================================== */
/* Strategy → CLI mapping (for the inline OpButton)                     */
/* =================================================================== */
// Best-effort match against StrategyCatalog.name (lowercased). The
// strategy plugins live in backend/plugins/strategy_*; their names map
// onto the equivalent management commands. If we can't match, we hide
// the CLI button rather than show a useless one.
export function cliForStrategy(
  name: string | undefined,
): { command: string; args: string; description: string } | null {
  if (!name) return null;
  const n = name.toLowerCase();
  if (n.includes("directional") || n.includes("equity") || n.includes("intraday")) {
    return {
      command: "run_trading_agent",
      args: "--show-journal",
      description: "Inspect the equity directional journal or fire a new plan.",
    };
  }
  if (n.includes("straddle")) {
    return {
      command: "manage_straddle",
      args: "--list",
      description: "List, register, or close short-straddle positions.",
    };
  }
  if (n.includes("pyramid")) {
    return {
      command: "run_pyramid",
      args: "--strike 24200 --type CE --dry-run",
      description: "Pyramid backtest on intraday option candles (dry-run by default).",
    };
  }
  if (n.includes("screener")) {
    return {
      command: "run_screener",
      args: "",
      description: "Run the live intraday screener.",
    };
  }
  if (n.includes("swing") || n.includes("ok")) {
    return {
      command: "run_ok_scanner",
      args: "--actionable-only",
      description: "Oliver Kell daily/weekly cycle scan.",
    };
  }
  return null;
}
