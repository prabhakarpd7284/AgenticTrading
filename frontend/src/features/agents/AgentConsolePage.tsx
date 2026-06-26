import * as React from "react";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle, Bot, CircleDot, Clock,
  ExternalLink, Play, Send, ShieldCheck, Sparkles, Terminal,
  Wifi, WifiOff,
} from "lucide-react";
import { toast } from "sonner";

import { api, fetchPage } from "@/lib/api";
import { connect } from "@/lib/ws";
import type { AgentEvent, AgentRun, AgentRunStatus, AgentRunSummary, Portfolio, StrategySchema } from "@/types";
import { cn, fmtInr, fmtRel, formatElapsed, safeStringify } from "@/lib/utils";
import { useAuditFeed, useEvent } from "@/lib/v2";
import { agentForNode, inferStepKind, computeKpis, cliForStrategy } from "./agentConsole.utils";
import type { KpiSummary } from "./agentConsole.utils";

/** Connection status drives the inline banner. "connecting" covers both the
 *  initial handshake and exponential-backoff reconnects (lib/ws auto-retries
 *  up to 30s). "closed_auth" is terminal — server rejected the JWT and the
 *  lib stops retrying. "live" means the socket is open right now. */
type WsState = "connecting" | "live" | "reconnecting" | "closed_auth";

/** Cap on retained events in the stream. Token-streaming workflows can fire
 *  thousands of events over a long run; rendering all of them re-runs every
 *  EventBubble + every memoised find/reduce on each frame. Older events
 *  remain in the database (AgentStep) — this only trims the in-memory feed. */
const MAX_RETAINED_EVENTS = 2000;

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
import {
  Sheet, SheetBody, SheetContent, SheetDescription, SheetFooter, SheetHeader, SheetTitle,
} from "@/components/ui/Sheet";
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
  /* ---------- runs list: filtered + cursor-paginated ---------- */
  const [fStrategy, setFStrategy] = React.useState("");
  const [fStatus, setFStatus] = React.useState<AgentRunStatus | "">("");

  const runsQuery = useInfiniteQuery({
    queryKey: ["agent-runs", fStrategy, fStatus],
    queryFn: ({ pageParam }) =>
      fetchPage<AgentRunSummary>("/agents/runs/", {
        strategy: fStrategy || undefined,
        status: fStatus || undefined,
        cursor: pageParam || undefined,
      }),
    initialPageParam: "" as string,
    getNextPageParam: (last) => cursorOf(last.next) ?? undefined,
    refetchInterval: 8_000,
  });
  const runs = React.useMemo(
    () => runsQuery.data?.pages.flatMap((p) => p.results) ?? [],
    [runsQuery.data],
  );

  // Legacy audit feed — shown below v2 runs so the console has real content on
  // day one (the legacy DB has AuditLog rows from the Streamlit-era pipelines).
  const { data: legacyAudit = [] } = useAuditFeed(50);
  const [auditEventId, setAuditEventId] = React.useState<number | undefined>();

  /* ---------- selected run: light row + full detail ---------- */
  const selectedId = runId ?? runs[0]?.id;
  const selectedRow = runs.find((r) => r.id === selectedId);
  // Sim runs (scalp) carry their data in config/result and never emit agent
  // events — skip the agent WS + the event-stream chrome for them.
  const selectedIsSim = isSimRun(selectedRow);

  const detailQuery = useQuery({
    queryKey: ["agent-run", selectedId],
    queryFn: () => api.get<AgentRun>(`/agents/runs/${selectedId}/`).then((r) => r.data),
    enabled: !!selectedId,
  });
  const detail = detailQuery.data;

  /* ---------- event stream (agent runs only) ---------- */
  const [events, setEvents] = React.useState<AgentEvent[]>([]);
  const [wsState, setWsState] = React.useState<WsState>("connecting");
  const feedRef = React.useRef<HTMLDivElement>(null);
  const wsRef = React.useRef<ReturnType<typeof connect>>();

  React.useEffect(() => {
    setEvents([]);
    wsRef.current?.close();
    if (!selectedId || selectedIsSim) return;   // sim runs don't stream agent events
    setWsState("connecting");

    // 1) Hydrate the timeline from REST. Without this, opening a run
    // page AFTER the run completed shows an empty stream — the WS
    // broadcast is fire-and-forget, so any events that fired before
    // the client subscribed are lost. The /steps/ endpoint replays
    // them so the operator sees the full timeline regardless of when
    // they navigated in. WS events are then merged on top, de-duped
    // by `seq` to avoid double-rendering anything still in-flight.
    let cancelled = false;
    api
      .get<{ events: AgentEvent[] }>(`/agents/runs/${selectedId}/steps/`)
      .then((r) => {
        if (cancelled) return;
        const past = r.data?.events ?? [];
        if (past.length) {
          setEvents((prev) => {
            const seen = new Set(prev.map((e) => e.seq));
            const merged = [...prev, ...past.filter((e) => !seen.has(e.seq))];
            merged.sort((a, b) => a.seq - b.seq);
            return merged.slice(-MAX_RETAINED_EVENTS);
          });
        }
      })
      .catch(() => { /* non-fatal — stream will still populate via WS */ });

    // 2) Open the live stream for events that haven't happened yet.
    wsRef.current = connect(
      `/ws/agents/${selectedId}/`,
      (msg) => setEvents((prev) => {
        const incoming = msg as unknown as AgentEvent;
        // Skip if we already hydrated this seq from REST.
        if (prev.some((e) => e.seq === incoming.seq)) return prev;
        const next = prev.length >= MAX_RETAINED_EVENTS
          ? prev.slice(-(MAX_RETAINED_EVENTS - 1))
          : prev;
        return [...next, incoming];
      }),
      {
        onOpen: () => setWsState("live"),
        onClose: (ev) => {
          // 4401/4403 = auth failure, lib stops retrying.
          if (ev.code === 4401 || ev.code === 4403) setWsState("closed_auth");
          else setWsState("reconnecting");
        },
      },
    );
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [selectedId, selectedIsSim]);

  // Autoscroll to newest event. Plain `auto` (not `smooth`) — a smooth-scroll
  // animation queues per-event and visibly stutters when tokens stream at
  // 30 Hz; the instant snap keeps the latest in view without animation thrash.
  React.useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: "auto" });
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

        <RunFilters
          catalog={catalog}
          strategy={fStrategy} onStrategy={setFStrategy}
          status={fStatus} onStatus={setFStatus}
        />

        <div className="flex-1 overflow-auto">
          {runsQuery.isLoading ? (
            <div className="p-3 space-y-2">
              <Skeleton className="h-14 w-full" />
              <Skeleton className="h-14 w-full" />
              <Skeleton className="h-14 w-5/6" />
            </div>
          ) : runs.length === 0 && legacyAudit.length === 0 ? (
            <EmptyState
              className="m-3"
              icon={<Sparkles />}
              title={fStrategy || fStatus ? "No matching runs" : "No runs yet"}
              description={fStrategy || fStatus
                ? "No runs match the current filter."
                : "Start your first run to see the desk think."}
            />
          ) : (
            <>
              {runs.length > 0 && (
                <ul>
                  {groupRuns(runs).map((it) =>
                    it.type === "header" ? (
                      <li
                        key={`h-${it.label}`}
                        className="px-4 py-1.5 text-caption uppercase tracking-wider text-fg-subtle bg-surface-2/50 border-y border-border"
                      >
                        {it.label}
                      </li>
                    ) : (
                      <li key={it.run.id} className="border-b border-border">
                        <RunRow
                          run={it.run}
                          selected={selectedId === it.run.id}
                          onClick={() => nav(`/agents/${it.run.id}`)}
                        />
                      </li>
                    ),
                  )}
                </ul>
              )}
              {runsQuery.hasNextPage && (
                <div className="p-3">
                  <Button
                    variant="secondary"
                    size="sm"
                    className="w-full"
                    loading={runsQuery.isFetchingNextPage}
                    onClick={() => runsQuery.fetchNextPage()}
                  >
                    Load more
                  </Button>
                </div>
              )}
              {legacyAudit.length > 0 && (
                <div className="border-t border-border">
                  <div className="px-4 py-2 text-caption uppercase tracking-wider text-fg-subtle bg-surface-2/40">
                    Legacy audit log · {legacyAudit.length}
                  </div>
                  <ul className="divide-y divide-border">
                    {legacyAudit.map((e, i) => {
                      // Older bridge payloads may omit `id`. Rows without an
                      // id can't link to a detail page so we render them as
                      // plain <li> with a "no detail" hint instead of a
                      // dead button.
                      const inner = (
                        <>
                          <div className="flex-1 min-w-0">
                            <div className="text-body-sm text-fg truncate">{e.detail}</div>
                            <div className="text-caption text-fg-subtle font-mono">
                              {e.time}{e.symbol && ` · ${e.symbol}`}
                            </div>
                          </div>
                          <Badge tone="neutral" className="shrink-0">{e.type.split("_")[0]}</Badge>
                        </>
                      );
                      return (
                        <li key={e.id ?? i}>
                          {e.id != null ? (
                            <button
                              type="button"
                              onClick={() => setAuditEventId(e.id)}
                              className="w-full text-left px-4 py-2.5 flex items-start gap-2 hover:bg-surface-2 focus-visible:bg-surface-2 focus-visible:outline-none"
                              aria-label={`Open detail for ${e.type} at ${e.time}`}
                            >
                              {inner}
                            </button>
                          ) : (
                            <div
                              className="px-4 py-2.5 flex items-start gap-2 opacity-80"
                              title="No detail available for this entry"
                            >
                              {inner}
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      </aside>

      {/* ----- detail pane ----- */}
      <section className="flex flex-col min-w-0">
        {!selectedId ? (
          <div className="flex-1 flex items-center justify-center p-6">
            <EmptyState
              icon={<Bot />}
              title="Select or start a run"
              description="The left rail lists recent runs; press New to start one."
              action={<Button onClick={() => setNewOpen(true)} leading={<Play className="h-4 w-4" />}>New run</Button>}
            />
          </div>
        ) : !detail ? (
          <div className="p-5 space-y-3">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        ) : selectedIsSim ? (
          <SimRunDetail key={detail.id} run={detail} />
        ) : (
          <RunDetail key={detail.id} run={detail} events={events} feedRef={feedRef} wsState={wsState} />
        )}
      </section>

      {/* Detail panel for a clicked legacy audit row. Right-edge slide-in
          Sheet (non-modal, so the rail stays clickable) — useEvent only
          fires when an id is set, so this stays cheap when nothing is
          selected. Clicking another rail row while the panel is open just
          updates the panel content; Esc or the X button dismiss. */}
      <EventDetailPanel
        id={auditEventId}
        onClose={() => setAuditEventId(undefined)}
        onJumpToRun={(runUuid) => {
          setAuditEventId(undefined);
          nav(`/agents/${runUuid}`);
        }}
      />
    </div>
  );
}

/* =================================================================== */
/* Run detail                                                           */
/* =================================================================== */
/** A sim/data run (scalp etc.) keeps its data in config/result, not agent
 *  events. Detect so the console renders the right view for the type. */
function isSimRun(run?: { strategy_name: string; result?: Record<string, unknown> | null }): boolean {
  if (!run) return false;
  if (run.strategy_name === "scalp") return true;
  return !!run.result && ("kpis" in run.result || "log" in run.result);
}

function RunDetail({
  run, events, feedRef, wsState,
}: {
  run: AgentRun;
  events: AgentEvent[];
  feedRef: React.RefObject<HTMLDivElement>;
  wsState: WsState;
}) {

  // Memoise the per-render scans of `events`. Without these, every WS token
  // append re-runs three O(n) finds, a fresh KPI reduce, and the CLI lookup
  // — the bigger cost is the cascade re-rendering 200+ EventBubbles below.
  const { planEvt, riskEvt, execEvt } = React.useMemo(() => {
    let p: AgentEvent | undefined, r: AgentEvent | undefined, e: AgentEvent | undefined;
    for (const ev of events) {
      if (!p && ev.type === "result" && ev.node === "planner") p = ev;
      if (!r && ev.type === "result" && ev.node === "risk")    r = ev;
      if (!e && ev.type === "result" && ev.node === "execute") e = ev;
      if (p && r && e) break;
    }
    return { planEvt: p, riskEvt: r, execEvt: e };
  }, [events]);

  const lastTs = events.length ? events[events.length - 1].ts : undefined;
  const kpis = React.useMemo(() => computeKpis(events), [events]);
  const cli = React.useMemo(() => cliForStrategy(run.strategy_name), [run.strategy_name]);

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
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="plan">Plan</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="execution">Execution</TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="flex-1 min-h-0 overflow-auto px-5 pb-5 pt-3">
          <SummaryView run={run} />
        </TabsContent>

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

// Memoised so a 200-event stream doesn't re-render every bubble when one new
// event arrives. Both `ev` and `prev` are stable refs from a useMemo-friendly
// parent — append-only `events` means an existing bubble's props never change.
const EventBubble = React.memo(function EventBubble({
  ev, prev,
}: { ev: AgentEvent; prev?: AgentEvent }) {
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
});

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
function WsStatusBanner({ state }: { state: WsState }) {
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
const STREAM_SKELETON_SUBLINE: Record<WsState, string> = {
  live:         "Connected — waiting for the first event.",
  connecting:   "Connecting to the run stream…",
  reconnecting: "Reconnecting to the run stream…",
  closed_auth:  "Stream auth failed. Refresh the page to retry.",
};

function StreamSkeleton({ wsState }: { wsState: WsState }) {
  const subline = STREAM_SKELETON_SUBLINE[wsState];
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

/* =================================================================== */
/* Event detail panel (right-edge Sheet — legacy audit row drilldown)   */
/* =================================================================== */
function EventDetailPanel({
  id, onClose, onJumpToRun,
}: {
  id: number | undefined;
  onClose: () => void;
  onJumpToRun: (runUuid: string) => void;
}) {
  const { data: ev, isLoading, isError } = useEvent(id);
  const open = id != null;

  return (
    <Sheet open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent aria-describedby={undefined}>
        <SheetHeader>
          <SheetTitle>Event detail</SheetTitle>
          <SheetDescription>
            Full row from the unified Event log. Type, severity, payload, and any
            workflow / trade / order linkage.
          </SheetDescription>
        </SheetHeader>

        <SheetBody>
          {isLoading && (
            <div className="space-y-2" aria-label="Loading event">
              <Skeleton className="h-5 w-1/2" />
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-5 w-1/3" />
            </div>
          )}

          {isError && (
            <div role="alert" className="rounded-md border border-danger/40 bg-pnl-down/5 p-3 text-body-sm text-fg">
              Failed to load event. The row may have been pruned or you lack permission.
            </div>
          )}

          {ev && (
            <div className="space-y-4">
              {/* ── Identity strip ── */}
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={severityTone(ev.severity)}>{ev.severity}</Badge>
                <span className="text-body-sm font-mono text-fg">{ev.type}</span>
                <span className="text-caption text-fg-subtle font-mono">
                  {new Date(ev.ts).toLocaleString("en-IN", { hour12: false })}
                </span>
                <span className="text-caption text-fg-subtle ml-auto">#{ev.id}</span>
              </div>

              {/* ── Human-readable line ── */}
              {ev.text && (
                <p className="text-body-sm text-fg whitespace-pre-wrap border-l-2 border-border pl-3">
                  {ev.text}
                </p>
              )}

              {/* ── Cross-links ── */}
              <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-caption font-mono">
                {(
                  [
                    {
                      label: "actor",
                      value: `${ev.actor_kind}${ev.actor_user != null ? ` · user ${ev.actor_user}` : ""}`,
                    },
                    { label: "step",    value: ev.step_name },
                    { label: "request", value: ev.request_id, breakAll: true },
                    { label: "trade",   value: ev.trade_id,   breakAll: true },
                    { label: "order",   value: ev.order,      breakAll: true },
                    { label: "signal",  value: ev.signal_id != null ? String(ev.signal_id) : "" },
                  ] as const
                )
                  .filter((row) => row.value)
                  .map((row) => (
                    <React.Fragment key={row.label}>
                      <dt className="text-fg-subtle">{row.label}</dt>
                      <dd className={cn("text-fg", "breakAll" in row && row.breakAll && "break-all")}>
                        {row.value}
                      </dd>
                    </React.Fragment>
                  ))}
              </dl>

              {/* ── Payload JSON ── */}
              <div>
                <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">payload</div>
                <pre className="text-caption font-mono text-fg-muted whitespace-pre-wrap break-all max-h-72 overflow-auto rounded-sm border border-border bg-surface-2 p-3">
                  {ev.payload ? safeStringify(ev.payload) : "(empty)"}
                </pre>
              </div>
            </div>
          )}
        </SheetBody>

        <SheetFooter>
          {ev?.workflow_run && (
            <Button
              size="sm"
              onClick={() => onJumpToRun(ev.workflow_run!)}
              leading={<ExternalLink className="h-3.5 w-3.5" />}
            >
              Open run
            </Button>
          )}
          <Button variant="secondary" size="sm" onClick={onClose}>Close</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function severityTone(s: "info" | "warn" | "error"): "info" | "warning" | "danger" {
  return s === "error" ? "danger" : s === "warn" ? "warning" : "info";
}

/* ---------- Sim / data run detail (scalp etc. — no agent stream) ---------- */
function SimRunDetail({ run }: { run: AgentRun }) {
  return (
    <>
      <header className="h-12 px-5 border-b border-border flex items-center gap-3 sticky top-0 bg-bg/80 backdrop-blur z-sticky">
        <RunStatusPill status={run.status} />
        <div className="flex-1 min-w-0">
          <div className="text-body-sm text-fg truncate">
            {run.strategy_name} <span className="text-fg-subtle">v{run.strategy_version}</span>
          </div>
          <div className="text-caption text-fg-subtle font-mono truncate">run {run.id}</div>
        </div>
        <Badge tone="neutral">
          <Clock className="h-3 w-3 mr-1" aria-hidden /> {fmtRel(run.created_at)} ago
        </Badge>
      </header>
      <div className="flex-1 overflow-auto p-5">
        <SummaryView run={run} />
      </div>
    </>
  );
}

/* ---------- Universal run summary (config + result, any run type) ---------- */
function fmtVal(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "object") return Array.isArray(v) ? `[${v.length}]` : "{…}";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toLocaleString();
  return String(v);
}

function SummaryView({ run }: { run: AgentRun }) {
  const cfg = run.config ?? {};
  const res = run.result;
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader><CardTitle>Config</CardTitle></CardHeader>
        <CardContent>
          {Object.keys(cfg).length ? (
            <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-5 gap-y-1 text-body-sm">
              {Object.entries(cfg).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between gap-3 border-b border-border/40 py-1">
                  <dt className="text-fg-subtle">{k}</dt>
                  <dd className="text-fg font-mono truncate max-w-[60%]" title={String(v)}>{fmtVal(v)}</dd>
                </div>
              ))}
            </dl>
          ) : <p className="text-body-sm text-fg-subtle">No config recorded.</p>}
        </CardContent>
      </Card>

      {run.error && (
        <div role="alert" className="rounded-md border border-danger/40 bg-pnl-down/5 p-3 text-body-sm text-fg">
          {run.error}
        </div>
      )}

      {res ? <ResultView result={res} /> : (
        <Card><CardContent className="py-6">
          <p className="text-body-sm text-fg-subtle">
            {run.status === "running" || run.status === "queued"
              ? "Run in progress — results appear when it completes. Live runs that stream (e.g. scalp) update on their own page."
              : "No result recorded for this run."}
          </p>
        </CardContent></Card>
      )}
    </div>
  );
}

function ResultView({ result }: { result: Record<string, unknown> }) {
  const kpis = result.kpis as Record<string, unknown> | undefined;
  const log = result.log as string[] | undefined;
  const hasKpis = kpis && typeof kpis === "object";
  const hasLog = Array.isArray(log) && log.length > 0;
  return (
    <Card>
      <CardHeader><CardTitle>Result</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        {hasKpis && (
          <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
            {Object.entries(kpis).map(([k, v]) => (
              <div key={k} className="rounded-sm border border-border/60 px-3 py-1.5">
                <dt className="text-caption text-fg-subtle uppercase tracking-wider truncate" title={k}>{k}</dt>
                <dd className="text-body-sm text-fg font-mono mt-0.5">{fmtVal(v)}</dd>
              </div>
            ))}
          </dl>
        )}
        {hasLog && (
          <div>
            <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">log · {log!.length}</div>
            <div className="max-h-72 overflow-auto rounded-sm border border-border bg-surface-2 p-3 font-mono text-caption space-y-0.5">
              {log!.map((l, i) => <div key={i} className="text-fg-muted whitespace-pre-wrap break-words">{l}</div>)}
            </div>
          </div>
        )}
        {!hasKpis && !hasLog && (
          <pre className="text-caption font-mono text-fg-muted whitespace-pre-wrap break-all max-h-96 overflow-auto">
            {safeStringify(result)}
          </pre>
        )}
      </CardContent>
    </Card>
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
  // Sim strategies (scalp) have their own pages and don't run as agent graphs.
  const choices = React.useMemo(() => catalog.filter((s) => s.name !== "scalp"), [catalog]);
  const [strategy, setStrategy] = React.useState(choices[0]?.name ?? "directional");
  const [prompt,   setPrompt]   = React.useState("");
  React.useEffect(() => {
    if (choices.length && !choices.some((s) => s.name === strategy)) setStrategy(choices[0].name);
  }, [choices, strategy]);

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
            {choices.map((s) => (
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

function RunDot({ status }: { status: AgentRunStatus }) {
  const color = {
    queued:    "bg-fg-subtle",
    running:   "bg-info animate-pulse motion-reduce:animate-none",
    succeeded: "bg-pnl-up",
    failed:    "bg-pnl-down",
    cancelled: "bg-fg-subtle",
  }[status];
  return <span aria-hidden className={cn("h-2 w-2 rounded-full shrink-0", color)} />;
}

/* ---------- runs list: cursor, grouping, row, filters ---------- */
/** Pull the opaque `cursor` token out of a DRF next/previous link so we can
 *  re-request the relative endpoint (keeps everything behind the dev proxy). */
function cursorOf(url: string | null): string | null {
  if (!url) return null;
  try { return new URL(url, window.location.origin).searchParams.get("cursor"); }
  catch { return null; }
}

function dateGroup(iso: string): string {
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(new Date()) - day(new Date(iso))) / 86_400_000);
  if (diff <= 0) return "Today";
  if (diff === 1) return "Yesterday";
  if (diff < 7) return "Earlier this week";
  if (diff < 30) return "Earlier this month";
  return "Older";
}

type RunListItem = { type: "header"; label: string } | { type: "run"; run: AgentRunSummary };
function groupRuns(runs: AgentRunSummary[]): RunListItem[] {
  const out: RunListItem[] = [];
  let last = "";
  for (const run of runs) {
    const g = dateGroup(run.created_at);
    if (g !== last) { out.push({ type: "header", label: g }); last = g; }
    out.push({ type: "run", run });
  }
  return out;
}

function RunRow({ run, selected, onClick }: { run: AgentRunSummary; selected: boolean; onClick: () => void }) {
  const pnl = run.summary?.realized_pnl_inr ?? run.summary?.total_pnl_inr;
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full text-left px-4 py-2.5 flex items-center gap-2.5 hover:bg-surface-2",
        selected && "bg-surface-2 border-l-2 border-l-accent",
      )}
    >
      <RunDot status={run.status} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-body-sm text-fg truncate">{run.strategy_name}</span>
          <span className="text-caption text-fg-subtle">v{run.strategy_version}</span>
          {pnl != null && (
            <span className={cn("ml-auto text-caption font-mono shrink-0", pnl >= 0 ? "text-pnl-up" : "text-pnl-down")}>
              {fmtInr(pnl)}
            </span>
          )}
        </div>
        <div className="text-caption text-fg-subtle mt-0.5 font-mono flex items-center gap-1.5 min-w-0">
          <span className="truncate">{run.id.slice(0, 8)}</span>
          <span>·</span>
          <span className="shrink-0">{fmtRel(run.created_at)} ago</span>
          {run.summary?.trades != null && (<><span>·</span><span className="shrink-0">{run.summary.trades} trades</span></>)}
        </div>
      </div>
    </button>
  );
}

const STATUS_FILTERS: (AgentRunStatus | "")[] = ["", "running", "succeeded", "failed", "cancelled"];

function RunFilters({
  catalog, strategy, onStrategy, status, onStatus,
}: {
  catalog: StrategySchema[];
  strategy: string;
  onStrategy: (v: string) => void;
  status: AgentRunStatus | "";
  onStatus: (v: AgentRunStatus | "") => void;
}) {
  return (
    <div className="px-3 py-2 border-b border-border space-y-2 bg-surface/40">
      <select
        value={strategy}
        onChange={(e) => onStrategy(e.target.value)}
        className="w-full h-8 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg outline-none focus:border-border-strong"
      >
        <option value="">All strategies</option>
        {catalog.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
      </select>
      <div className="flex flex-wrap gap-1">
        {STATUS_FILTERS.map((s) => (
          <button
            key={s || "all"}
            type="button"
            onClick={() => onStatus(s)}
            className={cn(
              "px-2 py-0.5 rounded-full text-caption border capitalize transition-colors",
              status === s ? "bg-accent/15 text-accent border-accent/40" : "border-border text-fg-subtle hover:text-fg",
            )}
          >
            {s || "all"}
          </button>
        ))}
      </div>
    </div>
  );
}
