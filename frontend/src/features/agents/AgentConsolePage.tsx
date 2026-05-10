import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle, Bot, ChevronRight, CircleDot, Clock, Play, Send, ShieldCheck, Sparkles,
} from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { connect } from "@/lib/ws";
import type { AgentEvent, AgentRun, Portfolio, StrategySchema } from "@/types";
import { cn, fmtRel } from "@/lib/utils";
import { useLegacyAudit } from "@/lib/legacy";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import {
  Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger,
} from "@/components/ui/Dialog";

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
  const { data: legacyAudit = [] } = useLegacyAudit(50);

  /* ---------- selected run + stream ---------- */
  const [events, setEvents] = React.useState<AgentEvent[]>([]);
  const feedRef = React.useRef<HTMLDivElement>(null);
  const wsRef = React.useRef<ReturnType<typeof connect>>();

  const selected = runs.find((r) => r.id === runId) ?? runs[0];

  React.useEffect(() => {
    setEvents([]);
    wsRef.current?.close();
    if (!selected) return;
    wsRef.current = connect(`/ws/agents/${selected.id}/`, (msg) => {
      setEvents((prev) => [...prev, msg as AgentEvent]);
    });
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
          <RunDetail run={selected} events={events} feedRef={feedRef} />
        )}
      </section>
    </div>
  );
}

/* =================================================================== */
/* Run detail                                                           */
/* =================================================================== */
function RunDetail({
  run, events, feedRef,
}: {
  run: AgentRun;
  events: AgentEvent[];
  feedRef: React.RefObject<HTMLDivElement>;
}) {
  const planEvt = events.find((e) => e.node === "planner" && e.type === "result");
  const riskEvt = events.find((e) => e.node === "risk"    && e.type === "result");
  const execEvt = events.find((e) => e.node === "execute" && e.type === "result");

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
        <Badge tone="neutral">
          <Clock className="h-3 w-3 mr-1" aria-hidden /> {fmtRel(run.created_at)}
        </Badge>
      </header>

      {/* Risk breach banner */}
      {riskEvt && (riskEvt.payload as any)?.approved === false && (
        <div role="alert" className="mx-5 mt-5 rounded-md border border-danger/40 bg-pnl-down/5 p-4 flex gap-3 items-start">
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
              <div className="text-body-sm text-fg-subtle">
                Waiting for events — the desk will stream reasoning, tool calls, and decisions here.
              </div>
            ) : (
              events.map((e) => <EventBubble key={e.seq} ev={e} />)
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

function EventBubble({ ev }: { ev: AgentEvent }) {
  const agentByNode: Record<string, { label: string; tone: "brand" | "info" | "warning" | "success" | "danger" }> = {
    fetch_data:       { label: "@DataAnalyst",    tone: "info"    },
    retrieve_context: { label: "@PortfolioTracker",tone: "brand"  },
    planner:          { label: "@DirectionalTrader",tone: "brand" },
    generate_action:  { label: "@OptionsStrategist",tone: "brand" },
    risk:             { label: "@RiskGuard",      tone: "warning" },
    validate_action:  { label: "@RiskGuard",      tone: "warning" },
    execute:          { label: "@Broker",         tone: "success" },
    journal:          { label: "@Journal",        tone: "info"    },
  };
  const a = agentByNode[ev.node] ?? { label: ev.node, tone: "info" as const };

  if (ev.type === "token") {
    return (
      <div className="flex gap-2">
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
        <Badge tone={isError ? "danger" : a.tone}>{a.label}</Badge>
        <span className="text-caption text-fg-subtle font-mono uppercase tracking-wider">{ev.type}</span>
        <span className="text-caption text-fg-subtle ml-auto">#{ev.seq}</span>
      </div>
      <pre className="text-caption font-mono text-fg-muted whitespace-pre-wrap break-all max-h-64 overflow-auto">
        {safeStringify(ev.payload)}
      </pre>
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
