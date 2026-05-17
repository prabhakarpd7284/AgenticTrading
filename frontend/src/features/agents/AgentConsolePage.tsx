import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle, Bot, ChevronRight, CircleDot, Clock, Layers, Play, Send, ShieldCheck, Sparkles, Triangle,
} from "lucide-react";
import { toast } from "sonner";

import { api, legacyApi } from "@/lib/api";
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
  run: listRun, events, feedRef,
}: {
  run: AgentRun;
  events: AgentEvent[];
  feedRef: React.RefObject<HTMLDivElement>;
}) {
  // Refetch the single-run detail so we get the populated `result` blob
  // (the list endpoint omits it for completed runs in some serializers).
  const isLive = listRun.status === "queued" || listRun.status === "running";
  const { data: run = listRun } = useQuery({
    queryKey: ["agent-run", listRun.id],
    queryFn: () => api.get<AgentRun>(`/agents/runs/${listRun.id}/`).then((r) => r.data),
    refetchInterval: isLive ? 2_000 : false,
    initialData: listRun,
  });

  const result = (run.result ?? {}) as RunResult;
  const isStraddle = run.strategy_name === "short_straddle";
  const isPyramid = run.strategy_name === "pyramid";
  const isVerticalSpread = run.strategy_name === "vertical_spread";

  // Merge persisted steps from the detail endpoint with live WS events,
  // de-duplicated by seq. This makes the Stream tab populated for
  // completed runs even though the WS doesn't replay history.
  const mergedEvents = React.useMemo(() => {
    const replay: AgentEvent[] = (run.steps ?? []).map((s) => ({
      seq: s.seq, node: s.node, type: s.event_type, payload: s.payload,
    }));
    const seen = new Map<number, AgentEvent>();
    for (const e of replay) seen.set(e.seq, e);
    for (const e of events) seen.set(e.seq, e);
    return Array.from(seen.values()).sort((a, b) => a.seq - b.seq);
  }, [run.steps, events]);

  // Risk verdict — directional has .risk.approved, straddle has .validated.approved
  const riskApproved = isStraddle
    ? result.validated?.approved
    : result.risk?.approved;
  const riskReason = isStraddle
    ? result.validated?.reason
    : result.risk?.reason;

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

      {/* Risk breach banner — only when explicitly rejected */}
      {riskApproved === false && (
        <div role="alert" className="mx-5 mt-5 rounded-md border border-danger/40 bg-pnl-down/5 p-4 flex gap-3 items-start">
          <AlertTriangle className="h-5 w-5 text-danger shrink-0 mt-0.5" aria-hidden />
          <div className="flex-1">
            <div className="text-body-sm font-semibold text-fg">@RiskGuard blocked this plan</div>
            <p className="text-body-sm text-fg-muted mt-0.5">
              {riskReason ?? "Deterministic risk rule violated. No order was placed."}
            </p>
          </div>
          <Button variant="secondary" size="sm" disabled>Execute (blocked)</Button>
        </div>
      )}

      {/* Plugin-level error banner — run "succeeded" but the plugin caught
          something fatal (e.g. missing position_id, broker login failed). */}
      {result.error && (
        <div role="alert" className="mx-5 mt-5 rounded-md border border-warning/40 bg-warning/5 p-4 flex gap-3 items-start">
          <AlertTriangle className="h-5 w-5 text-warning shrink-0 mt-0.5" aria-hidden />
          <div className="flex-1">
            <div className="text-body-sm font-semibold text-fg">Plugin error</div>
            <p className="text-body-sm text-fg-muted mt-0.5 font-mono">{result.error}</p>
          </div>
        </div>
      )}

      <Tabs defaultValue="overview" className="flex-1 min-h-0 flex flex-col">
        <TabsList className="px-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          {!isStraddle && <TabsTrigger value="chart">Chart</TabsTrigger>}
          <TabsTrigger value="stream">Stream</TabsTrigger>
          <TabsTrigger value="raw">Raw</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="flex-1 overflow-auto px-5 pb-5">
          {!run.result ? (
            <EmptyState
              icon={<Sparkles />}
              title={isLive ? "Run is in flight" : "No result yet"}
              description={isLive
                ? "Cards will populate as the graph progresses."
                : "Open the Stream tab for the event timeline."}
            />
          ) : isStraddle ? (
            <StraddleOverview result={result} />
          ) : isPyramid ? (
            <PyramidOverview result={result} />
          ) : isVerticalSpread ? (
            <VerticalSpreadOverview result={result} />
          ) : (
            <DirectionalOverview result={result} />
          )}
        </TabsContent>

        {!isStraddle && (
          <TabsContent value="chart" className="flex-1 min-h-0 px-5 pb-5">
            <DirectionalChart result={result} />
          </TabsContent>
        )}

        <TabsContent value="stream" className="flex-1 min-h-0 px-5 pb-5">
          <div className="h-full flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <div className="text-caption text-fg-subtle font-mono">
                {mergedEvents.length} event{mergedEvents.length === 1 ? "" : "s"}
                {(run.steps?.length ?? 0) > 0 && events.length === 0 && (
                  <> · {run.steps!.length} replayed from history</>
                )}
              </div>
              {mergedEvents.length > 0 && (
                <CopyButton value={JSON.stringify(mergedEvents, null, 2)} label="Copy events JSON" />
              )}
            </div>
            <div
              ref={feedRef}
              className="flex-1 overflow-auto rounded-md border border-border bg-surface p-4 space-y-3"
              aria-live="polite"
              aria-label="Agent event stream"
            >
              {mergedEvents.length === 0 ? (
                <div className="text-body-sm text-fg-subtle">
                  {isLive
                    ? "Waiting for events — the desk will stream reasoning, tool calls, and decisions here."
                    : "No events were persisted for this run."}
                </div>
              ) : (
                mergedEvents.map((e) => <EventBubble key={e.seq} ev={e} />)
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="raw" className="px-5 pb-5">
          <JsonCard title="run.result" payload={run.result} emptyHint="No result yet." />
        </TabsContent>
      </Tabs>
    </>
  );
}

/* ---------- typed shape of run.result ---------- */
type DirectionalPlan = {
  symbol?: string;
  side?: "BUY" | "SELL";
  entry_price?: number;
  stop_loss?: number;
  target?: number;
  quantity?: number;
  confidence?: number;
  reasoning?: string;
  error?: string;
};

type DirectionalRisk = {
  approved?: boolean;
  reason?: string;
  details?: {
    risk_amount?: number;
    max_risk_allowed?: number;
    risk_pct_of_capital?: number;
    position_value?: number;
    max_position_value?: number;
    daily_loss_so_far?: number;
    max_daily_loss?: number;
    risk_reward_ratio?: number;
    rr_ratio?: number; // legacy field name in some payloads
    regime?: Record<string, unknown>;
  };
};

type ExecutionResult = {
  success?: boolean;
  skipped?: boolean;
  order_id?: string;
  fill_price?: number;
  fill_quantity?: number;
  mode?: "paper" | "live";
  message?: string;
  reason?: string;
  intent?: string;
  dry_run?: boolean;
  note?: string;
};

type MarketData = {
  symbol?: string;
  candle_count?: number;
  last_close?: number;
  day_high?: number;
  day_low?: number;
  range_pct?: number;
  summary?: string;
  candles?: Array<{ date: string; open: number; high: number; low: number; close: number; volume?: number }>;
};

type StraddleAction = {
  action?: string;
  urgency?: string;
  confidence?: number;
  ce_action?: string;
  pe_action?: string;
  reasoning?: string;
  key_risk?: string;
  source?: string;
  hedge_side?: string | null;
  hedge_lots?: number;
  roll_to_strike?: number | null;
};

type StraddleScenario = {
  label: string;
  nifty_level: number;
  ce_expiry_value: number;
  pe_expiry_value: number;
  net_pnl_inr: number;
};

type StraddleAnalysis = {
  net_pnl_inr?: number;
  net_pnl_pts?: number;
  premium_decayed_pct?: number;
  vix_phase?: "CALM" | "ELEVATED" | "SPIKE";
  vix_current?: number;
  vix_prev_close?: number;
  vix_change_pct?: number;
  market_phase?: string;
  net_delta?: number;
  ce_delta?: number;
  pe_delta?: number;
  delta_bias?: string;
  is_underwater?: boolean;
  stop_triggered?: boolean;
  is_expiry_day?: boolean;
  expiry_tomorrow?: boolean;
  days_to_expiry?: number;
  nifty_spot?: number;
  nifty_prev_close?: number;
  nifty_gap_pts?: number;
  nifty_gap_pct?: number;
  combined_sold?: number;
  combined_current?: number;
  ce_ltp?: number;
  pe_ltp?: number;
  ce_sell_price?: number;
  pe_sell_price?: number;
  ce_itm_by?: number;
  pe_itm_by?: number;
  nearest_itm_leg?: "CE" | "PE" | "BOTH_OTM";
  scenarios?: StraddleScenario[];
  summary_text?: string;
};

type StraddleSnapshotLeg = {
  ltp?: number;
  open?: number;
  high?: number;
  low?: number;
  prev_close?: number;
};

type CandleRow = [string, number, number, number, number, number]; // [ts, o, h, l, c, v]
type StraddleSnapshot = {
  nifty?: StraddleSnapshotLeg;
  vix?: StraddleSnapshotLeg;
  ce?: StraddleSnapshotLeg;
  pe?: StraddleSnapshotLeg;
  candles?: CandleRow[];     // NIFTY spot 5-min
  ce_candles?: CandleRow[];
  pe_candles?: CandleRow[];
};

type StraddlePosition = {
  id?: number;
  underlying?: string;
  strike?: number;
  expiry?: string;
  lots?: number;
  lot_size?: number;
  ce_symbol?: string;
  pe_symbol?: string;
  ce_sell?: number;
  pe_sell?: number;
};

type PyramidEntry = {
  bar_index: number;
  timestamp: string;
  price: number;
  lots: number;
  sl_at_entry: number;
  reason: string;
};

type PyramidPlan = {
  symbol?: string;
  entries?: PyramidEntry[];
  exit_price?: number;
  exit_time?: string;
  exit_reason?: string;
  total_lots?: number;
  peak_lots?: number;
  total_cost?: number;
  realized_pnl?: number;
  peak_unrealized?: number;
  lot_size?: number;
  avg_entry?: number;
  pnl_per_lot?: number;
  total_pnl_points?: number;
  total_pnl_rupees?: number;
  log_tail?: string[];
  error?: string;
};

type VerticalSpreadPlan = {
  underlying?: string;
  side?: "BULL" | "BEAR";
  option_type?: "CE" | "PE";
  expiry?: string;
  long_strike?: number;
  short_strike?: number;
  long_ltp?: number;
  short_ltp?: number;
  net_debit?: number;
  net_credit?: number;
  lots?: number;
  lot_size?: number;
  qty_per_leg?: number;
  max_profit_inr?: number;
  max_loss_inr?: number;
  breakeven?: number;
  capital_used?: number;
  rr_ratio?: number;
  spot?: number;
  error?: string;
};

type RunResult = {
  plan?: DirectionalPlan | PyramidPlan | VerticalSpreadPlan;
  risk?: DirectionalRisk;
  execution?: ExecutionResult;
  market_data?: MarketData;
  symbol?: string;
  rag_context?: string;
  action?: StraddleAction;
  analysis?: StraddleAnalysis;
  position?: StraddlePosition;
  snapshot?: StraddleSnapshot;
  validated?: { approved?: boolean; override?: string | null; reason?: string };
  error?: string;
  // Vertical spread + pyramid extras stashed at top level:
  underlying?: string;
  spot?: number;
  long_ltp?: number;
  short_ltp?: number;
  candles_raw?: unknown[];
};

/* =================================================================== */
/* Directional overview                                                 */
/* =================================================================== */
function DirectionalOverview({ result }: { result: RunResult }) {
  const plan = (result.plan ?? {}) as DirectionalPlan;
  const risk = result.risk ?? {};
  const exe  = result.execution ?? {};
  const md   = result.market_data ?? {};

  const sideTone = plan.side === "BUY" ? "success" : plan.side === "SELL" ? "danger" : "neutral";
  const rr = plan.entry_price && plan.stop_loss && plan.target
    ? Math.abs((plan.target - plan.entry_price) / (plan.entry_price - plan.stop_loss))
    : null;
  const risked = plan.entry_price && plan.stop_loss && plan.quantity
    ? Math.abs(plan.entry_price - plan.stop_loss) * plan.quantity
    : null;
  const reward = plan.entry_price && plan.target && plan.quantity
    ? Math.abs(plan.target - plan.entry_price) * plan.quantity
    : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
      {/* Plan card */}
      <Card className="lg:col-span-2">
        <CardHeader className="flex flex-row items-center gap-3">
          <CardTitle className="flex items-center gap-2">
            <Badge tone={sideTone as any}>{plan.side ?? "—"}</Badge>
            <span className="text-h3">{plan.symbol ?? "—"}</span>
            <span className="text-fg-subtle">×{plan.quantity ?? 0}</span>
          </CardTitle>
          {plan.confidence != null && (
            <Badge tone={plan.confidence >= 0.7 ? "success" : plan.confidence >= 0.5 ? "warning" : "danger"}>
              conf {plan.confidence.toFixed(2)}
            </Badge>
          )}
        </CardHeader>
        <CardContent>
          {plan.error ? (
            <p className="text-body-sm text-danger">{plan.error}</p>
          ) : (
            <>
              <div className="grid grid-cols-4 gap-2 mb-4">
                <KvBlock label="Entry"   value={fmtNum(plan.entry_price)} />
                <KvBlock label="Stop"    value={fmtNum(plan.stop_loss)}    tone="danger" />
                <KvBlock label="Target"  value={fmtNum(plan.target)}       tone="success" />
                <KvBlock label="R:R"     value={rr ? `${rr.toFixed(2)}×` : "—"} />
              </div>
              <div className="grid grid-cols-3 gap-2 mb-4">
                <KvBlock label="Risked (₹)"  value={fmtINR(risked)} tone="danger" />
                <KvBlock label="Reward (₹)"  value={fmtINR(reward)} tone="success" />
                <KvBlock label="Position (₹)" value={fmtINR(plan.entry_price && plan.quantity ? plan.entry_price * plan.quantity : null)} />
              </div>
              {plan.reasoning && (
                <div>
                  <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Reasoning</div>
                  <p className="text-body-sm text-fg-muted whitespace-pre-wrap">{plan.reasoning}</p>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Risk card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-accent" aria-hidden />
            @RiskGuard
            {risk.approved != null && (
              <Badge tone={risk.approved ? "success" : "danger"}>
                {risk.approved ? "Approved" : "Rejected"}
              </Badge>
            )}
          </CardTitle>
          <CardDescription className="font-mono text-caption">{risk.reason ?? "—"}</CardDescription>
        </CardHeader>
        <CardContent>
          {risk.details ? (
            <>
              <div className="grid grid-cols-2 gap-x-4">
                <KvRow label="Risk amount"      value={fmtINR(risk.details.risk_amount)} />
                <KvRow label="Max risk allowed" value={fmtINR(risk.details.max_risk_allowed)} />
                <KvRow
                  label="Risk % of capital"
                  value={risk.details.risk_pct_of_capital != null ? `${risk.details.risk_pct_of_capital.toFixed(2)}%` : "—"}
                />
                <KvRow
                  label="R:R"
                  value={(() => {
                    const r = risk.details.risk_reward_ratio ?? risk.details.rr_ratio;
                    return r != null ? `${r.toFixed(2)}×` : "—";
                  })()}
                />
                <KvRow label="Position value"   value={fmtINR(risk.details.position_value)} />
                <KvRow label="Max position"     value={fmtINR(risk.details.max_position_value)} />
                <KvRow label="Daily loss"       value={fmtINR(risk.details.daily_loss_so_far)} />
                <KvRow label="Max daily loss"   value={fmtINR(risk.details.max_daily_loss)} />
              </div>
              {risk.details.regime && (
                <div className="mt-3 rounded-sm border border-border bg-surface-2/40 p-2.5">
                  <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Regime gate</div>
                  <RegimeSummary regime={risk.details.regime} />
                </div>
              )}
            </>
          ) : <p className="text-body-sm text-fg-subtle">No detail.</p>}
        </CardContent>
      </Card>

      {/* Execution card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Execution
            {exe.success && <Badge tone="success">Filled</Badge>}
            {exe.skipped && <Badge tone="neutral">Skipped</Badge>}
            {!exe.success && !exe.skipped && exe.message && <Badge tone="danger">Error</Badge>}
            {exe.mode && <Badge tone={exe.mode === "paper" ? "info" : "warning"}>{exe.mode}</Badge>}
          </CardTitle>
          <CardDescription>{exe.reason || exe.message || exe.note || ""}</CardDescription>
        </CardHeader>
        <CardContent>
          {exe.order_id && <KvRow label="Order ID"  value={<code>{exe.order_id}</code>} />}
          {exe.fill_price != null && <KvRow label="Fill price" value={fmtNum(exe.fill_price)} />}
          {exe.fill_quantity != null && <KvRow label="Filled qty" value={String(exe.fill_quantity)} />}
          {!exe.order_id && !exe.fill_price && !exe.fill_quantity && (
            <p className="text-body-sm text-fg-subtle">
              {exe.reason === "risk_rejected"
                ? "No order placed — @RiskGuard blocked the plan."
                : exe.reason === "dry_run"
                ? "Dry run — no order placed."
                : "Nothing executed."}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Market data card */}
      <Card className="lg:col-span-2">
        <CardHeader><CardTitle>Market context</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <KvBlock label="Last close" value={fmtNum(md.last_close)} />
            <KvBlock label="Day high"   value={fmtNum(md.day_high)} />
            <KvBlock label="Day low"    value={fmtNum(md.day_low)} />
            <KvBlock label="Range"      value={md.range_pct != null ? `${md.range_pct.toFixed(2)}%` : "—"} />
          </div>
          {md.summary && (
            <pre className="text-caption font-mono text-fg-muted whitespace-pre-wrap max-h-48 overflow-auto">
              {md.summary}
            </pre>
          )}
        </CardContent>
      </Card>

      {/* RAG context — recent trades + strategy rules fed to the planner */}
      {result.rag_context && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>RAG context</CardTitle>
            <CardDescription>Recent trades + strategy rules injected into the planner prompt.</CardDescription>
          </CardHeader>
          <CardContent>
            <RagContextPanel text={result.rag_context} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function RegimeSummary({ regime }: { regime: Record<string, unknown> }) {
  // Two shapes possible: cache-miss { cache: "miss"/"bad" } or full
  // { vol, trend, global_tone, tradeable, summary }
  if (regime.cache) {
    return (
      <p className="text-body-sm text-fg-muted font-mono">
        Pulse cache <Badge tone="warning">{String(regime.cache)}</Badge> · soft-skipped in dev
      </p>
    );
  }
  const vol = regime.vol as string | undefined;
  const trend = regime.trend as string | undefined;
  const tone = regime.global_tone as string | undefined;
  const tradeable = regime.tradeable as boolean | undefined;
  const summary = regime.summary as string | undefined;
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-1.5">
        {vol && <Badge tone={vol === "extreme" ? "danger" : vol === "elevated" ? "warning" : "neutral"}>vol: {vol}</Badge>}
        {trend && <Badge tone="neutral">trend: {trend}</Badge>}
        {tone && <Badge tone="neutral">global: {tone}</Badge>}
        {tradeable != null && <Badge tone={tradeable ? "success" : "danger"}>{tradeable ? "tradeable" : "stand down"}</Badge>}
      </div>
      {summary && <p className="text-body-sm text-fg-muted">{summary}</p>}
    </div>
  );
}

function RagContextPanel({ text }: { text: string }) {
  // The legacy retrieve_context() emits sections separated by "---" and
  // headed by ALL-CAPS labels like "RECENT TRADES FOR ITC", "OTHER RECENT
  // TRADES", "ACTIVE STRATEGY RULES". Split + label the sections so they
  // render as collapsible blocks instead of one blob.
  const sections = text.split(/\n\n---\n\n/).map((s) => s.trim()).filter(Boolean);
  if (sections.length === 0) return null;

  return (
    <div className="space-y-3">
      {sections.map((section, i) => {
        const firstLine = section.split("\n", 1)[0] ?? "";
        const titleMatch = firstLine.match(/^([A-Z][A-Z0-9_ ()/]+):/);
        const title = titleMatch ? titleMatch[1] : `Section ${i + 1}`;
        const body = section.slice(firstLine.length + 1).trim();
        return (
          <details key={i} className="rounded-sm border border-border bg-surface-2/40" open={i === 0}>
            <summary className="cursor-pointer px-3 py-2 text-body-sm text-fg flex items-center justify-between">
              <span className="font-medium">{title}</span>
              <span className="text-caption text-fg-subtle font-mono">{body.length} chars</span>
            </summary>
            <pre className="px-3 pb-3 text-caption font-mono text-fg-muted whitespace-pre-wrap max-h-72 overflow-auto">
              {body}
            </pre>
          </details>
        );
      })}
    </div>
  );
}

/* =================================================================== */
/* Directional candle chart (lightweight-charts)                        */
/* =================================================================== */
function DirectionalChart({ result }: { result: RunResult }) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const md = result.market_data ?? {};
  const plan = result.plan ?? {};
  const candles = md.candles ?? [];

  React.useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;

    let chart: any;
    let cleanup = () => {};

    (async () => {
      const mod = await import("lightweight-charts");
      chart = mod.createChart(containerRef.current!, {
        layout: { background: { color: "transparent" }, textColor: "#8b949e" },
        grid: { vertLines: { color: "#161b22" }, horzLines: { color: "#161b22" } },
        timeScale: { timeVisible: true, secondsVisible: false },
        rightPriceScale: { borderColor: "#30363d" },
        crosshair: { mode: 0 },
        autoSize: true,
      });
      const series = chart.addCandlestickSeries({
        upColor: "#3fb950", downColor: "#f85149",
        wickUpColor: "#3fb950", wickDownColor: "#f85149",
        borderVisible: false,
      });
      // lightweight-charts expects time as a unix-second number OR "YYYY-MM-DD HH:MM".
      // The legacy fetch_historical returns ISO-ish date strings; coerce to seconds.
      const data = candles
        .map((c) => ({
          time: Math.floor(new Date(c.date).getTime() / 1000) as any,
          open: c.open, high: c.high, low: c.low, close: c.close,
        }))
        .filter((c) => Number.isFinite(c.time) && c.time > 0)
        .sort((a, b) => a.time - b.time);
      series.setData(data);

      // Overlay entry / SL / target
      if (plan.entry_price) series.createPriceLine({
        price: plan.entry_price, color: "#58a6ff",
        lineWidth: 1, lineStyle: 0, axisLabelVisible: true, title: "ENTRY",
      });
      if (plan.stop_loss) series.createPriceLine({
        price: plan.stop_loss, color: "#f85149",
        lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "SL",
      });
      if (plan.target) series.createPriceLine({
        price: plan.target, color: "#3fb950",
        lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "TGT",
      });

      chart.timeScale().fitContent();
      cleanup = () => chart.remove();
    })();

    return () => cleanup();
  }, [candles, plan.entry_price, plan.stop_loss, plan.target]);

  if (candles.length === 0) {
    return (
      <EmptyState
        icon={<Sparkles />}
        title="No candles"
        description={`The fetch_data node returned no candles for ${md.symbol ?? "this symbol"}. Try a different symbol or check the Angel One session.`}
      />
    );
  }

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{md.symbol ?? plan.symbol ?? "Chart"} · 5-min · {candles.length} candles</CardTitle>
        <CardDescription>
          Plan overlay — entry (blue) · stop (red, dashed) · target (green, dashed).
        </CardDescription>
      </CardHeader>
      <CardContent className="h-[480px]">
        <div ref={containerRef} className="h-full w-full" />
      </CardContent>
    </Card>
  );
}

/* =================================================================== */
/* Straddle overview                                                    */
/* =================================================================== */
/* =================================================================== */
/* Pyramid overview                                                     */
/* =================================================================== */
function PyramidOverview({ result }: { result: RunResult }) {
  const plan = (result.plan ?? {}) as PyramidPlan;
  const entries = plan.entries ?? [];
  const candles = (result.candles_raw ?? []) as Array<[string, number, number, number, number, number]>;
  const pnl = plan.total_pnl_rupees ?? 0;
  const pnlTone = pnl >= 0 ? "success" : "danger";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
      {/* Intraday chart with entry/exit markers (full-width) */}
      {candles.length > 0 && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Intraday — entries, SL, exit</CardTitle>
            <CardDescription>
              5-min candlesticks for {plan.symbol ?? result.symbol}. Entries shown as colored arrows,
              SL-at-entry as dashed lines, exit price as a horizontal reference.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[420px]">
            <PyramidIntradayChart candles={candles} entries={entries} exitPrice={plan.exit_price} />
          </CardContent>
        </Card>
      )}

      {/* Headline */}
      <Card className="lg:col-span-2">
        <CardHeader className="flex flex-row items-center gap-3 flex-wrap">
          <CardTitle className="flex items-center gap-2">
            <Triangle className="h-4 w-4 text-accent" aria-hidden />
            <span className="font-mono text-h3">{plan.symbol ?? result.symbol ?? "—"}</span>
            <Badge tone="info">PYRAMID</Badge>
            {plan.exit_reason && <Badge tone="neutral">exit: {plan.exit_reason}</Badge>}
          </CardTitle>
          <Badge tone={pnlTone as any}>
            P&amp;L {fmtINR(pnl)}{plan.total_pnl_points != null ? ` · ${plan.total_pnl_points.toFixed(1)} pts` : ""}
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <KvBlock label="Underlying" value={result.underlying ?? "—"} />
            <KvBlock label="Spot at fetch" value={fmtNum(result.spot)} />
            <KvBlock label="Candles" value={String((result.candles_raw as any[] | undefined)?.length ?? 0)} />
            <KvBlock label="Lot size" value={String(plan.lot_size ?? 0)} />
            <KvBlock label="Entries" value={String(entries.length)} />
            <KvBlock label="Peak lots" value={String(plan.peak_lots ?? 0)} />
            <KvBlock label="Total lots" value={String(plan.total_lots ?? 0)} />
            <KvBlock label="Avg entry" value={fmtNum(plan.avg_entry)} />
            <KvBlock label="Exit price" value={fmtNum(plan.exit_price)} />
            <KvBlock label="Exit time" value={plan.exit_time ?? "—"} />
            <KvBlock label="Peak unrealized" value={plan.peak_unrealized != null ? `${plan.peak_unrealized.toFixed(1)} pts` : "—"} />
            <KvBlock label="P&L / lot" value={plan.pnl_per_lot != null ? plan.pnl_per_lot.toFixed(2) : "—"} tone={pnlTone} />
          </div>
          {plan.error && (
            <div className="mt-3 rounded-sm border border-warning/40 bg-warning/5 p-2.5 text-body-sm text-fg-muted">
              <strong className="text-fg">No simulation:</strong> {plan.error}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Entries table */}
      {entries.length > 0 && (
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle>Entries ({entries.length})</CardTitle>
            <CopyButton value={safeStringify(entries)} label="Copy entries" />
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-body-sm">
                <thead>
                  <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
                    <th className="text-right py-2 px-3">#</th>
                    <th className="text-left py-2 px-3">Time</th>
                    <th className="text-right py-2 px-3">Price</th>
                    <th className="text-right py-2 px-3">Lots</th>
                    <th className="text-right py-2 px-3">SL at entry</th>
                    <th className="text-left py-2 px-3">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map((e, i) => (
                    <tr key={i} className="border-b border-border last:border-b-0 hover:bg-surface-2">
                      <td className="text-right py-2 px-3 font-mono text-fg-subtle">{e.bar_index}</td>
                      <td className="py-2 px-3 font-mono text-caption">{e.timestamp}</td>
                      <td className="text-right py-2 px-3 font-mono">{fmtNum(e.price)}</td>
                      <td className="text-right py-2 px-3 font-mono">{e.lots}</td>
                      <td className="text-right py-2 px-3 font-mono text-pnl-down">{fmtNum(e.sl_at_entry)}</td>
                      <td className="py-2 px-3 text-fg-muted">{e.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Simulator log */}
      {plan.log_tail && plan.log_tail.length > 0 && (
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <div>
              <CardTitle>Simulator log (tail)</CardTitle>
              <CardDescription>Last {plan.log_tail.length} lines from the pure-Python pyramid run.</CardDescription>
            </div>
            <CopyButton value={(plan.log_tail ?? []).join("\n")} label="Copy log" />
          </CardHeader>
          <CardContent>
            <pre className="text-caption font-mono text-fg-muted whitespace-pre-wrap max-h-80 overflow-auto bg-surface-2/40 border border-border rounded-sm p-3">
              {plan.log_tail.join("\n")}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/* =================================================================== */
/* Vertical spread overview                                             */
/* =================================================================== */
function VerticalSpreadOverview({ result }: { result: RunResult }) {
  const plan = (result.plan ?? {}) as VerticalSpreadPlan;
  const debit = plan.net_debit ?? 0;
  const credit = plan.net_credit ?? 0;
  const rr = plan.rr_ratio;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
      <Card className="lg:col-span-2">
        <CardHeader className="flex flex-row items-center gap-3 flex-wrap">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-accent" aria-hidden />
            <Badge tone={plan.side === "BULL" ? "success" : "danger"}>{plan.side ?? "—"}</Badge>
            <span className="font-mono text-h3">{plan.underlying ?? result.underlying ?? "—"}</span>
            <span className="text-fg-subtle">
              {plan.long_strike ?? "—"}/{plan.short_strike ?? "—"} {plan.option_type}
            </span>
            <Badge tone="neutral">{plan.expiry}</Badge>
          </CardTitle>
          {rr != null && <Badge tone={rr >= 1.5 ? "success" : rr >= 1 ? "warning" : "danger"}>R:R {rr.toFixed(2)}×</Badge>}
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <KvBlock label="Spot" value={fmtNum(plan.spot ?? result.spot)} />
            <KvBlock label="Long LTP" value={fmtNum(plan.long_ltp)} />
            <KvBlock label="Short LTP" value={fmtNum(plan.short_ltp)} />
            <KvBlock label={debit > 0 ? "Net debit" : "Net credit"} value={fmtNum(debit > 0 ? debit : credit)} />
            <KvBlock label="Breakeven" value={fmtNum(plan.breakeven)} />
            <KvBlock label="Lots" value={String(plan.lots ?? "—")} />
            <KvBlock label="Capital used" value={fmtINR(plan.capital_used)} />
            <KvBlock label="Qty per leg" value={String(plan.qty_per_leg ?? "—")} />
            <KvBlock label="Max profit" value={fmtINR(plan.max_profit_inr)} tone="success" />
            <KvBlock label="Max loss" value={fmtINR(plan.max_loss_inr)} tone="danger" />
          </div>
          {plan.error && (
            <div className="rounded-sm border border-warning/40 bg-warning/5 p-2.5 text-body-sm text-fg-muted">
              <strong className="text-fg">Couldn't size:</strong> {plan.error}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Payoff diagram — P&L at expiry vs underlying price */}
      {plan.long_strike != null && plan.short_strike != null && plan.lots && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Expiry payoff diagram</CardTitle>
            <CardDescription>
              Net P&amp;L if held to expiry across a ±10% range around current spot.
              Vertical lines: current spot · breakeven · long/short strikes.
              Green region = profit · red region = loss.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[320px]">
            <VerticalSpreadPayoffChart plan={plan} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function VerticalSpreadPayoffChart({ plan }: { plan: VerticalSpreadPlan }) {
  const [R, setR] = React.useState<any>(null);
  React.useEffect(() => { import("recharts").then(setR); }, []);

  const data = React.useMemo(() => {
    const spot = plan.spot ?? plan.long_strike ?? 0;
    if (!spot) return [];
    const longK = plan.long_strike ?? 0;
    const shortK = plan.short_strike ?? 0;
    const isCall = plan.option_type === "CE";
    const lotQty = (plan.lots ?? 0) * (plan.lot_size ?? 1);
    const netDebit = plan.net_debit ?? 0;
    const netCredit = plan.net_credit ?? 0;
    // Bull call spread payoff at expiry per unit:
    //   long  CE @ K_long  pays max(0, S - K_long) - long_premium
    //   short CE @ K_short pays short_premium - max(0, S - K_short)
    // Symmetric for bear put: long PE @ K_long, short PE @ K_short (K_long > K_short).
    const lo = Math.max(1, spot * 0.9);
    const hi = spot * 1.1;
    const step = (hi - lo) / 80;
    const out: Array<{ s: number; pnl: number }> = [];
    for (let s = lo; s <= hi; s += step) {
      let payoff: number;
      if (isCall) {
        const longPay = Math.max(0, s - longK) - (plan.long_ltp ?? 0);
        const shortPay = (plan.short_ltp ?? 0) - Math.max(0, s - shortK);
        payoff = (longPay + shortPay) * lotQty;
      } else {
        const longPay = Math.max(0, longK - s) - (plan.long_ltp ?? 0);
        const shortPay = (plan.short_ltp ?? 0) - Math.max(0, shortK - s);
        payoff = (longPay + shortPay) * lotQty;
      }
      out.push({ s: +s.toFixed(2), pnl: +payoff.toFixed(0) });
    }
    return out;
  }, [plan]);

  if (data.length === 0) {
    return <EmptyState icon={<Sparkles />} title="No payoff to plot" description="Plan is missing spot or strike data." />;
  }
  if (!R) return <div className="h-full flex items-center justify-center text-fg-subtle text-body-sm">Loading…</div>;

  const { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } = R;
  const spot = plan.spot ?? plan.long_strike ?? 0;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 12, right: 32, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#161b22" />
        <XAxis dataKey="s" type="number" domain={["dataMin", "dataMax"]}
               tick={{ fill: "#8b949e", fontSize: 11 }}
               tickFormatter={(v: number) => v.toLocaleString("en-IN", { maximumFractionDigits: 0 })} />
        <YAxis tick={{ fill: "#8b949e", fontSize: 11 }}
               tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`} width={50} />
        <Tooltip
          contentStyle={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6, fontSize: 12 }}
          formatter={(v: number) => [`₹${Math.round(v).toLocaleString("en-IN")}`, "P&L at expiry"]}
          labelFormatter={(v: number) => `Underlying ${v.toLocaleString("en-IN")}`} />
        <ReferenceLine y={0} stroke="#484f58" />
        {spot > 0 && <ReferenceLine x={spot} stroke="#58a6ff" strokeDasharray="3 3" label={{ value: `spot ${spot.toFixed(0)}`, fill: "#58a6ff", fontSize: 11, position: "top" }} />}
        {plan.breakeven && <ReferenceLine x={plan.breakeven} stroke="#d29922" strokeDasharray="3 3" label={{ value: `BE ${plan.breakeven.toFixed(0)}`, fill: "#d29922", fontSize: 11, position: "top" }} />}
        {plan.long_strike  && <ReferenceLine x={plan.long_strike}  stroke="#3fb950" strokeOpacity={0.5} label={{ value: `L ${plan.long_strike}`, fill: "#3fb950", fontSize: 11, position: "insideBottomLeft" }} />}
        {plan.short_strike && <ReferenceLine x={plan.short_strike} stroke="#f85149" strokeOpacity={0.5} label={{ value: `S ${plan.short_strike}`, fill: "#f85149", fontSize: 11, position: "insideBottomRight" }} />}
        <Line type="monotone" dataKey="pnl" stroke="#58a6ff" strokeWidth={2} dot={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function StraddleOverview({ result }: { result: RunResult }) {
  const pos = result.position ?? {};
  const an  = result.analysis ?? {};
  const act = result.action ?? {};
  const v   = result.validated ?? {};
  const snap = result.snapshot ?? {};

  // Pull every run that has touched this position so we can plot the
  // lifecycle on the intraday chart ("when + which trade taken").
  const { data: allRuns = [] } = useQuery({
    queryKey: ["agent-runs", "for-position", pos.id],
    queryFn: () => api.get<AgentRun[]>("/agents/runs/?limit=500").then((r) => r.data),
    enabled: pos.id != null,
  });
  const positionRuns = React.useMemo(
    () => allRuns
      .filter((r) => r.strategy_name === "short_straddle" && (r.config as any)?.position_id === pos.id)
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()),
    [allRuns, pos.id],
  );

  const actionTone = act.action === "HOLD" ? "success"
    : act.action === "CLOSE_BOTH" ? "danger"
    : act.action === "MONITOR" ? "info"
    : "warning";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-5">
      {/* Action card — the LLM's verdict */}
      <Card className="lg:col-span-2">
        <CardHeader className="flex flex-row items-center gap-3 flex-wrap">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-4 w-4 text-accent" aria-hidden />
            <Badge tone={actionTone as any}>{act.action ?? "—"}</Badge>
            <span className="text-h3">
              {pos.underlying} {pos.strike}
            </span>
            <span className="text-fg-subtle text-body-sm">{pos.expiry}</span>
          </CardTitle>
          {act.confidence != null && (
            <Badge tone={act.confidence >= 0.7 ? "success" : act.confidence >= 0.5 ? "warning" : "danger"}>
              conf {act.confidence.toFixed(2)}
            </Badge>
          )}
          {act.urgency && <Badge tone={act.urgency === "IMMEDIATE" ? "danger" : "neutral"}>{act.urgency}</Badge>}
          {act.source && <Badge tone="info">{act.source}</Badge>}
          {v.override && <Badge tone="warning">override → {v.override}</Badge>}
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            <KvBlock label="CE action" value={act.ce_action ?? "—"} />
            <KvBlock label="PE action" value={act.pe_action ?? "—"} />
            {act.hedge_side && act.hedge_side !== "NONE" && (
              <KvBlock label="Hedge" value={`${act.hedge_side} ${act.hedge_lots ?? 0}`} />
            )}
            {act.roll_to_strike && (
              <KvBlock label="Roll → strike" value={String(act.roll_to_strike)} />
            )}
          </div>
          {act.reasoning && (
            <div className="mb-3">
              <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Reasoning</div>
              <p className="text-body-sm text-fg-muted whitespace-pre-wrap">{act.reasoning}</p>
            </div>
          )}
          {act.key_risk && (
            <div className="mb-3">
              <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Key risk</div>
              <p className="text-body-sm text-fg-muted whitespace-pre-wrap">{act.key_risk}</p>
            </div>
          )}
          {v.override && v.reason && (
            <div className="rounded-sm border border-warning/40 bg-warning/5 p-2.5">
              <div className="text-caption uppercase tracking-wider text-warning mb-0.5">@RiskGuard override</div>
              <p className="text-body-sm text-fg-muted">{v.reason}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Expiry P&L scenarios — full-width chart */}
      {an.scenarios && an.scenarios.length > 0 && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Expiry P&amp;L scenarios</CardTitle>
            <CardDescription>
              Net P&amp;L at expiry across NIFTY levels · current spot marked.
              Crossings of zero are the breakeven points.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[280px]">
            <ScenariosChart scenarios={an.scenarios} currentSpot={an.nifty_spot} />
          </CardContent>
        </Card>
      )}

      {/* Intraday close chart — CE + PE + combined since position open */}
      {((snap.ce_candles?.length ?? 0) > 0 || (snap.pe_candles?.length ?? 0) > 0) && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Lifecycle — intraday close + agent actions</CardTitle>
            <CardDescription>
              CE / PE / combined 5-min closes since position open · NIFTY on right axis ·
              every agent run plotted at its timestamp + action.
              {an.days_to_expiry != null && (
                <span className="ml-2">DTE {an.days_to_expiry}{pos.expiry ? ` · expiry ${pos.expiry}` : ""}</span>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[420px]">
            <IntradayClosesChart
              ceCandles={snap.ce_candles ?? []}
              peCandles={snap.pe_candles ?? []}
              niftyCandles={snap.candles ?? []}
              ceSold={pos.ce_sell}
              peSold={pos.pe_sell}
              runs={positionRuns}
            />
          </CardContent>
        </Card>
      )}

      {/* Position card */}
      <Card>
        <CardHeader>
          <CardTitle>Position</CardTitle>
          <CardDescription>{pos.underlying} {pos.strike} · {pos.expiry}</CardDescription>
        </CardHeader>
        <CardContent>
          <KvRow label="Lots"     value={`${pos.lots} × ${pos.lot_size ?? 0}`} />
          <KvRow label="CE sold @" value={fmtNum(pos.ce_sell)} />
          <KvRow label="PE sold @" value={fmtNum(pos.pe_sell)} />
          <KvRow label="Combined sold" value={fmtNum(an.combined_sold)} />
          <KvRow label="CE symbol" value={<code className="text-caption">{pos.ce_symbol}</code>} />
          <KvRow label="PE symbol" value={<code className="text-caption">{pos.pe_symbol}</code>} />
        </CardContent>
      </Card>

      {/* P&L + Phases card */}
      <Card>
        <CardHeader>
          <CardTitle>P&amp;L + phase</CardTitle>
          <CardDescription>Live mark-to-market · regime context</CardDescription>
        </CardHeader>
        <CardContent>
          <KvRow
            label="Net P&L"
            value={fmtINR(an.net_pnl_inr)}
            tone={an.net_pnl_inr != null && an.net_pnl_inr >= 0 ? "success" : "danger"}
          />
          <KvRow label="P&L (pts)" value={an.net_pnl_pts != null ? an.net_pnl_pts.toFixed(2) : "—"} />
          <KvRow label="Premium decayed" value={an.premium_decayed_pct != null ? `${an.premium_decayed_pct.toFixed(1)}%` : "—"} />
          <KvRow label="Combined current" value={fmtNum(an.combined_current)} />
          <KvRow label="Market phase" value={an.market_phase ?? "—"} />
          <KvRow label="VIX phase"  value={an.vix_phase ?? "—"} />
          <KvRow label="DTE"        value={an.days_to_expiry != null ? String(an.days_to_expiry) : "—"} />
          <KvRow
            label="Underwater?"
            value={an.is_underwater ? "Yes" : "No"}
            tone={an.is_underwater ? "danger" : "success"}
          />
          <KvRow
            label="Stop triggered?"
            value={an.stop_triggered ? "Yes" : "No"}
            tone={an.stop_triggered ? "danger" : "success"}
          />
        </CardContent>
      </Card>

      {/* Greeks + ITM exposure */}
      <Card>
        <CardHeader>
          <CardTitle>Greeks &amp; moneyness</CardTitle>
          <CardDescription>Delta exposure · which leg is in-the-money</CardDescription>
        </CardHeader>
        <CardContent>
          <KvRow label="CE delta"   value={an.ce_delta != null ? an.ce_delta.toFixed(3) : "—"} />
          <KvRow label="PE delta"   value={an.pe_delta != null ? an.pe_delta.toFixed(3) : "—"} />
          <KvRow
            label="Net delta"
            value={an.net_delta != null ? an.net_delta.toFixed(3) : "—"}
            tone={
              an.net_delta == null ? "neutral"
              : Math.abs(an.net_delta) > 0.5 ? "danger"
              : Math.abs(an.net_delta) > 0.25 ? "warning" : "success"
            }
          />
          <KvRow label="Delta bias" value={an.delta_bias ?? "—"} />
          <KvRow
            label="CE ITM by"
            value={an.ce_itm_by != null && an.ce_itm_by > 0 ? `+${an.ce_itm_by.toFixed(1)} pts` : "OTM"}
            tone={an.ce_itm_by != null && an.ce_itm_by > 0 ? "danger" : "success"}
          />
          <KvRow
            label="PE ITM by"
            value={an.pe_itm_by != null && an.pe_itm_by > 0 ? `+${an.pe_itm_by.toFixed(1)} pts` : "OTM"}
            tone={an.pe_itm_by != null && an.pe_itm_by > 0 ? "danger" : "success"}
          />
          <KvRow label="Nearest ITM leg" value={an.nearest_itm_leg ?? "—"} />
        </CardContent>
      </Card>

      {/* Live snapshot */}
      <Card>
        <CardHeader>
          <CardTitle>Snapshot</CardTitle>
          <CardDescription>NIFTY · VIX · CE · PE — live broker pull</CardDescription>
        </CardHeader>
        <CardContent>
          <SnapshotRow label="NIFTY" leg={snap.nifty} fallbackLTP={an.nifty_spot} prev={an.nifty_prev_close} changePct={an.nifty_gap_pct} />
          <SnapshotRow label="VIX"   leg={snap.vix}   fallbackLTP={an.vix_current} prev={an.vix_prev_close} changePct={an.vix_change_pct} />
          <SnapshotRow label="CE"    leg={snap.ce}    fallbackLTP={an.ce_ltp} prev={an.ce_sell_price} changeLabel="vs sold" />
          <SnapshotRow label="PE"    leg={snap.pe}    fallbackLTP={an.pe_ltp} prev={an.pe_sell_price} changeLabel="vs sold" />
        </CardContent>
      </Card>

      {/* The exact text the LLM saw */}
      {an.summary_text && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>LLM analysis input</CardTitle>
            <CardDescription>The summary text passed verbatim to Claude before the action was generated.</CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="text-caption font-mono text-fg-muted whitespace-pre-wrap max-h-80 overflow-auto bg-surface-2/40 border border-border rounded-sm p-3">
              {an.summary_text}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function SnapshotRow({
  label, leg, fallbackLTP, prev, changePct, changeLabel,
}: {
  label: string;
  leg?: StraddleSnapshotLeg;
  fallbackLTP?: number;
  prev?: number;
  changePct?: number;
  changeLabel?: string;
}) {
  const ltp = leg?.ltp ?? fallbackLTP;
  const prevVal = leg?.prev_close ?? prev;
  const pct = changePct != null ? changePct : (ltp != null && prevVal && prevVal !== 0) ? ((ltp - prevVal) / prevVal) * 100 : null;
  const pctTone = pct == null ? "neutral" : pct >= 0 ? "success" : "danger";
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border last:border-b-0">
      <div className="text-caption text-fg-subtle uppercase tracking-wider w-12 shrink-0">{label}</div>
      <div className="flex-1 grid grid-cols-3 gap-3 ml-3 text-body-sm font-mono">
        <div><span className="text-fg-subtle text-caption">LTP </span>{fmtNum(ltp)}</div>
        <div><span className="text-fg-subtle text-caption">{leg?.high || leg?.low ? "H/L " : "Prev "}</span>{leg?.high != null && leg?.low != null ? `${fmtNum(leg.high)} / ${fmtNum(leg.low)}` : fmtNum(prevVal)}</div>
        <div className={cn(
          pctTone === "success" ? "text-pnl-up" : pctTone === "danger" ? "text-pnl-down" : "text-fg-subtle",
        )}>
          {pct != null ? `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%` : "—"}
          {changeLabel && <span className="text-fg-subtle text-caption ml-1">{changeLabel}</span>}
        </div>
      </div>
    </div>
  );
}

/* =================================================================== */
/* Pyramid intraday chart with entry/SL/exit markers                    */
/* =================================================================== */
function PyramidIntradayChart({
  candles, entries, exitPrice,
}: {
  candles: Array<[string, number, number, number, number, number]>;
  entries: PyramidEntry[];
  exitPrice?: number;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;
    let cleanup = () => {};

    (async () => {
      const mod = await import("lightweight-charts");
      const chart = mod.createChart(containerRef.current!, {
        layout: { background: { color: "transparent" }, textColor: "#8b949e" },
        grid:   { vertLines: { color: "#161b22" }, horzLines: { color: "#161b22" } },
        timeScale: { timeVisible: true, secondsVisible: false, borderColor: "#30363d" },
        rightPriceScale: { borderColor: "#30363d" },
        crosshair: { mode: 0 },
        autoSize: true,
      });
      const series = chart.addCandlestickSeries({
        upColor: "#3fb950", downColor: "#f85149",
        wickUpColor: "#3fb950", wickDownColor: "#f85149",
        borderVisible: false,
      });
      const data = candles
        .map(([ts, o, h, l, c]) => ({
          time: Math.floor(new Date(ts).getTime() / 1000) as any,
          open: o, high: h, low: l, close: c,
        }))
        .filter((d) => Number.isFinite(d.time) && d.time > 0)
        .sort((a, b) => a.time - b.time);
      series.setData(data);

      // Mark entries on the candle series with up arrows (initial) / down
      // arrows (no — they're always BUYs in pyramid), each annotated with lots.
      if (entries.length > 0) {
        series.setMarkers(entries.map((e, i) => ({
          time: Math.floor(new Date(e.timestamp).getTime() / 1000) as any,
          position: "belowBar" as any,
          color: i === 0 ? "#79c0ff" : "#a371f7",
          shape: i === 0 ? "arrowUp" : "circle",
          text: i === 0 ? `IN ${e.lots}L @ ${e.price.toFixed(1)}` : `+${e.lots}L`,
          size: 1,
        })));
      }

      // SL-at-entry as a horizontal price line per entry (initial only — the
      // pyramid trails its SL between bars; legacy log captures every raise).
      if (entries.length > 0 && entries[0].sl_at_entry) {
        series.createPriceLine({
          price: entries[0].sl_at_entry, color: "#f85149",
          lineWidth: 1, lineStyle: 2, axisLabelVisible: true,
          title: `SL ${entries[0].sl_at_entry.toFixed(1)}`,
        });
      }

      // Exit price reference.
      if (exitPrice) {
        series.createPriceLine({
          price: exitPrice, color: "#d29922",
          lineWidth: 1, lineStyle: 0, axisLabelVisible: true,
          title: `EXIT ${exitPrice.toFixed(1)}`,
        });
      }

      chart.timeScale().fitContent();
      cleanup = () => chart.remove();
    })();

    return () => cleanup();
  }, [candles, entries, exitPrice]);

  return <div ref={containerRef} className="h-full w-full" />;
}


function IntradayClosesChart({
  ceCandles, peCandles, niftyCandles, ceSold, peSold, runs = [],
}: {
  ceCandles: CandleRow[];
  peCandles: CandleRow[];
  niftyCandles?: CandleRow[];
  ceSold?: number;
  peSold?: number;
  runs?: AgentRun[];
}) {
  const [R, setR] = React.useState<any>(null);
  React.useEffect(() => { import("recharts").then(setR); }, []);

  // Merge candles on timestamp so the series has {t, ce, pe, combined, nifty}.
  const data = React.useMemo(() => {
    const map = new Map<string, { t: number; ce?: number; pe?: number; combined?: number; nifty?: number }>();
    const push = (rows: CandleRow[] | undefined, key: "ce" | "pe" | "nifty") => {
      if (!rows) return;
      for (const row of rows) {
        const [ts, , , , close] = row;
        const t = new Date(ts).getTime();
        if (!Number.isFinite(t)) continue;
        const existing = map.get(ts) ?? { t };
        existing[key] = close;
        map.set(ts, existing);
      }
    };
    push(ceCandles, "ce");
    push(peCandles, "pe");
    push(niftyCandles, "nifty");
    const out = Array.from(map.values()).sort((a, b) => a.t - b.t);
    for (const row of out) {
      if (row.ce != null && row.pe != null) row.combined = +(row.ce + row.pe).toFixed(2);
    }
    return out;
  }, [ceCandles, peCandles, niftyCandles]);

  // Map every agent run to a data point on the combined series so it
  // renders as a colored dot at the closest 5-min bar to its timestamp.
  const lifecyclePoints = React.useMemo(() => {
    if (data.length === 0) return [];
    return runs
      .map((r) => {
        const at = new Date(r.completed_at ?? r.started_at ?? r.created_at).getTime();
        if (!Number.isFinite(at)) return null;
        // nearest bar (binary search would be fine; linear is fine for ~300)
        let nearest = data[0];
        let nearestDiff = Math.abs(data[0].t - at);
        for (let i = 1; i < data.length; i++) {
          const d = Math.abs(data[i].t - at);
          if (d < nearestDiff) { nearest = data[i]; nearestDiff = d; }
        }
        const action = ((r.result ?? {}) as any).action?.action as string | undefined;
        return {
          t: nearest.t,
          combined: nearest.combined,
          ce: nearest.ce,
          pe: nearest.pe,
          action: action ?? "?",
          confidence: ((r.result ?? {}) as any).action?.confidence as number | undefined,
          runId: r.id,
        };
      })
      .filter(Boolean) as Array<{ t: number; combined?: number; ce?: number; pe?: number; action: string; confidence?: number; runId: string }>;
  }, [runs, data]);

  if (data.length === 0) {
    return <EmptyState icon={<Sparkles />} title="No intraday candles" description="The broker returned no 5-min candles for this position." />;
  }
  if (!R) return <div className="h-full flex items-center justify-center text-fg-subtle text-body-sm">Loading…</div>;

  const sold = (ceSold ?? 0) + (peSold ?? 0);
  const hasNifty = data.some((d) => d.nifty != null);

  const {
    ResponsiveContainer, ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip,
    CartesianGrid, ReferenceLine, Legend,
  } = R;

  // Action → dot color
  const actionColor = (a: string): string => {
    if (a === "HOLD") return "#3fb950";
    if (a === "CLOSE_BOTH" || a === "CLOSE_CE" || a === "CLOSE_PE") return "#f85149";
    if (a === "MONITOR") return "#79c0ff";
    if (a === "SHIFT_TO_ATM" || a === "ROLL_PE" || a === "ROLL_CE") return "#a371f7";
    if (a === "HEDGE_FUTURES") return "#d29922";
    return "#8b949e";
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 12, right: hasNifty ? 56 : 24, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#161b22" />
        <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={["dataMin", "dataMax"]}
          tick={{ fill: "#8b949e", fontSize: 11 }}
          tickFormatter={(v: number) => {
            const d = new Date(v);
            const dd = String(d.getDate()).padStart(2, "0");
            const mm = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][d.getMonth()];
            const hh = String(d.getHours()).padStart(2, "0");
            const mi = String(d.getMinutes()).padStart(2, "0");
            return `${dd}${mm} ${hh}:${mi}`;
          }}
          minTickGap={70}
        />
        <YAxis yAxisId="opt" tick={{ fill: "#8b949e", fontSize: 11 }} width={48} label={{ value: "Option ₹", angle: -90, position: "insideLeft", fill: "#6e7681", fontSize: 11 }} />
        {hasNifty && (
          <YAxis yAxisId="nifty" orientation="right" tick={{ fill: "#8b949e", fontSize: 11 }} width={56} tickFormatter={(v: number) => v.toFixed(0)} label={{ value: "NIFTY", angle: 90, position: "insideRight", fill: "#6e7681", fontSize: 11 }} />
        )}
        <Tooltip
          contentStyle={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6, fontSize: 12 }}
          labelFormatter={(v: number) => new Date(v).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
          formatter={(v: number, name: string) => v != null ? [name === "NIFTY" ? v.toFixed(2) : v.toFixed(2), name] : ["—", name]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {sold > 0 && (
          <ReferenceLine yAxisId="opt" y={sold} stroke="#d29922" strokeDasharray="4 2" label={{ value: `sold ${sold.toFixed(0)}`, fill: "#d29922", fontSize: 11, position: "insideTopRight" }} />
        )}
        <Line yAxisId="opt" type="monotone" dataKey="ce"       name="CE close"   stroke="#79c0ff" strokeWidth={1.5} dot={false} connectNulls />
        <Line yAxisId="opt" type="monotone" dataKey="pe"       name="PE close"   stroke="#a371f7" strokeWidth={1.5} dot={false} connectNulls />
        <Line yAxisId="opt" type="monotone" dataKey="combined" name="Combined"   stroke="#3fb950" strokeWidth={2}   dot={false} connectNulls />
        {hasNifty && (
          <Line yAxisId="nifty" type="monotone" dataKey="nifty" name="NIFTY" stroke="#8b949e" strokeWidth={1} dot={false} connectNulls strokeDasharray="2 4" />
        )}
        {lifecyclePoints.length > 0 && (
          <Scatter
            yAxisId="opt"
            name="Agent action"
            data={lifecyclePoints}
            shape={(props: any) => {
              const { cx, cy, payload } = props;
              if (cx == null || cy == null) return null;
              const c = actionColor(payload.action);
              return (
                <g>
                  <circle cx={cx} cy={cy} r={6} fill={c} stroke="#0d1117" strokeWidth={1.5} />
                  <text x={cx} y={cy - 10} fill={c} fontSize={10} textAnchor="middle" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    {payload.action}
                  </text>
                </g>
              );
            }}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function ScenariosChart({ scenarios, currentSpot }: { scenarios: StraddleScenario[]; currentSpot?: number }) {
  // Lazy import recharts so we don't bloat the auth bundle.
  const [R, setR] = React.useState<any>(null);
  React.useEffect(() => { import("recharts").then(setR); }, []);
  if (!R) return <div className="h-full flex items-center justify-center text-fg-subtle text-body-sm">Loading chart…</div>;

  const { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ReferenceDot, CartesianGrid } = R;

  const sorted = [...scenarios].sort((a, b) => a.nifty_level - b.nifty_level);
  // Identify breakeven crossings (sign change between adjacent points)
  const breakevens: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const a = sorted[i - 1], b = sorted[i];
    if ((a.net_pnl_inr <= 0) !== (b.net_pnl_inr <= 0) && a.net_pnl_inr !== b.net_pnl_inr) {
      const t = -a.net_pnl_inr / (b.net_pnl_inr - a.net_pnl_inr);
      breakevens.push(a.nifty_level + t * (b.nifty_level - a.nifty_level));
    }
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={sorted} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#161b22" />
        <XAxis dataKey="nifty_level" tick={{ fill: "#8b949e", fontSize: 11 }} tickFormatter={(v: number) => v.toLocaleString("en-IN")} />
        <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`} width={50} />
        <Tooltip
          contentStyle={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6, fontSize: 12 }}
          labelStyle={{ color: "#e6edf3" }}
          formatter={(v: number, _name: string, props: any) => [`₹${Math.round(v).toLocaleString("en-IN")}`, props.payload.label]}
          labelFormatter={(v: number) => `NIFTY ${v.toLocaleString("en-IN")}`}
        />
        <ReferenceLine y={0} stroke="#484f58" />
        {currentSpot != null && currentSpot > 0 && (
          <ReferenceLine x={currentSpot} stroke="#58a6ff" strokeDasharray="3 3" label={{ value: `spot ${currentSpot.toFixed(0)}`, fill: "#58a6ff", fontSize: 11, position: "top" }} />
        )}
        {breakevens.map((b, i) => (
          <ReferenceDot key={i} x={b} y={0} r={4} fill="#d29922" stroke="none" />
        ))}
        <Line type="monotone" dataKey="net_pnl_inr" stroke="#58a6ff" strokeWidth={2} dot={{ r: 3, fill: "#58a6ff" }} activeDot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* =================================================================== */
/* Small UI helpers                                                     */
/* =================================================================== */
function KvBlock({ label, value, tone = "neutral" }: { label: string; value: React.ReactNode; tone?: "neutral" | "success" | "danger" | "warning" | "info" }) {
  const color =
    tone === "success" ? "text-pnl-up"
    : tone === "danger" ? "text-pnl-down"
    : tone === "warning" ? "text-warning"
    : tone === "info" ? "text-info"
    : "text-fg";
  return (
    <div className="rounded-sm border border-border bg-surface-2/40 px-3 py-2">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn("text-body-sm font-mono mt-0.5", color)}>{value}</div>
    </div>
  );
}

function KvRow({ label, value, tone = "neutral" }: { label: string; value: React.ReactNode; tone?: "neutral" | "success" | "danger" | "warning" | "info" }) {
  const color =
    tone === "success" ? "text-pnl-up"
    : tone === "danger" ? "text-pnl-down"
    : tone === "warning" ? "text-warning"
    : tone === "info" ? "text-info"
    : "text-fg";
  return (
    <div className="flex items-center justify-between py-1 border-b border-border last:border-b-0">
      <div className="text-caption text-fg-subtle">{label}</div>
      <div className={cn("text-body-sm font-mono", color)}>{value ?? "—"}</div>
    </div>
  );
}

function fmtNum(n?: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function fmtINR(n?: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
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
  const json = payload ? safeStringify(payload) : "";
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2">
        <CardTitle>{title}</CardTitle>
        {payload != null && <CopyButton value={json} />}
      </CardHeader>
      <CardContent>
        {payload != null ? (
          <pre className="text-caption font-mono text-fg whitespace-pre-wrap max-h-[600px] overflow-auto">
            {json}
          </pre>
        ) : (
          <p className="text-body-sm text-fg-subtle">{emptyHint}</p>
        )}
      </CardContent>
    </Card>
  );
}

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // Some browsers block clipboard outside https/localhost — fall back to a textarea hack
          const ta = document.createElement("textarea");
          ta.value = value; document.body.appendChild(ta); ta.select();
          try { document.execCommand("copy"); setCopied(true); setTimeout(() => setCopied(false), 1500); }
          finally { document.body.removeChild(ta); }
        }
      }}
      className={cn(
        "px-2 py-1 rounded-sm border text-caption font-mono inline-flex items-center gap-1 transition",
        copied
          ? "border-pnl-up/40 bg-pnl-up/10 text-pnl-up"
          : "border-border bg-surface hover:bg-surface-2 text-fg-muted hover:text-fg",
      )}
      aria-label={label}
    >
      {copied ? "Copied" : label}
    </button>
  );
}

/* =================================================================== */
/* New run dialog — schema-driven                                       */
/* =================================================================== */
type SchemaProp = {
  type?: "string" | "integer" | "number" | "boolean";
  default?: unknown;
  enum?: string[];
  minimum?: number;
  maximum?: number;
  description?: string;
};

function NewRunDialog({
  catalog, portfolios, pending, onSubmit,
}: {
  catalog: StrategySchema[];
  portfolios: Portfolio[];
  pending: boolean;
  onSubmit: (v: Record<string, unknown>) => void;
}) {
  const [strategy, setStrategy] = React.useState(catalog[0]?.name ?? "directional");
  const [portfolioId, setPortfolioId] = React.useState(portfolios[0]?.id ?? "");
  const [config, setConfig] = React.useState<Record<string, unknown>>({});

  React.useEffect(() => {
    if (catalog.length && !catalog.find((c) => c.name === strategy)) {
      setStrategy(catalog[0].name);
    }
  }, [catalog, strategy]);
  React.useEffect(() => {
    if (portfolios.length && !portfolioId) setPortfolioId(portfolios[0].id);
  }, [portfolios, portfolioId]);

  const selected = catalog.find((c) => c.name === strategy);
  const properties: Record<string, SchemaProp> =
    ((selected?.params as any)?.properties as Record<string, SchemaProp>) ?? {};
  const required: string[] = (selected?.params as any)?.required ?? [];

  // Seed defaults whenever the selected strategy changes.
  React.useEffect(() => {
    if (!selected) return;
    const seeded: Record<string, unknown> = {};
    Object.entries(properties).forEach(([k, p]) => {
      if (p.default !== undefined) seeded[k] = p.default;
    });
    setConfig(seeded);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategy]);

  const setField = (k: string, v: unknown) =>
    setConfig((prev) => ({ ...prev, [k]: v }));

  const missingRequired = required.filter((k) => {
    const v = config[k];
    return v === undefined || v === null || (typeof v === "string" && !v.trim());
  });

  return (
    <DialogContent className="w-[min(92vw,560px)] max-h-[90vh] overflow-auto">
      <DialogTitle>Start agent run</DialogTitle>
      <DialogDescription>
        Pick a strategy and fill its parameters — the desk plans, passes @RiskGuard, then executes in paper mode.
      </DialogDescription>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!portfolioId || !selected || missingRequired.length) return;
          onSubmit({
            strategy_name: strategy,
            strategy_version: selected.version,
            portfolio: portfolioId,
            config,
          });
        }}
        className="mt-4 space-y-4"
      >
        {/* Strategy picker */}
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

        {/* Portfolio picker (relevant when there are multiple) */}
        {portfolios.length > 1 && (
          <div>
            <label className="text-body-sm text-fg mb-1.5 inline-block">Portfolio</label>
            <select
              value={portfolioId}
              onChange={(e) => setPortfolioId(e.target.value)}
              className="w-full bg-surface border border-border rounded-sm px-3 py-2 text-body-sm text-fg"
            >
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>{p.name} · ₹{p.capital}</option>
              ))}
            </select>
          </div>
        )}

        {/* Schema-driven params */}
        {selected && (
          <div className="border-t border-border pt-3">
            <div className="text-caption uppercase tracking-wider text-fg-subtle mb-2">Parameters</div>
            <div className="space-y-3">
              {Object.entries(properties).map(([key, prop]) => (
                <SchemaField
                  key={`${strategy}:${key}`}
                  name={key}
                  prop={prop}
                  required={required.includes(key)}
                  value={config[key]}
                  onChange={(v) => setField(key, v)}
                  formValues={config}
                />
              ))}
            </div>
          </div>
        )}

        <div className="rounded-sm border border-border bg-surface-2/50 p-3 flex gap-2 items-start">
          <ShieldCheck className="h-4 w-4 text-accent mt-0.5" aria-hidden />
          <p className="text-caption text-fg-muted">
            @RiskGuard will deterministically validate every plan before execution. No LLM bypass.
          </p>
        </div>

        {missingRequired.length > 0 && (
          <p className="text-caption text-danger">
            Missing required: {missingRequired.join(", ")}
          </p>
        )}

        <div className="flex items-center justify-end gap-2">
          <Button
            type="submit"
            loading={pending}
            disabled={missingRequired.length > 0 || !portfolioId}
            leading={<Send className="h-4 w-4" />}
          >
            Start run
          </Button>
        </div>
      </form>
    </DialogContent>
  );
}

function SchemaField({
  name, prop, required, value, onChange, formValues,
}: {
  name: string;
  prop: SchemaProp;
  required: boolean;
  value: unknown;
  onChange: (v: unknown) => void;
  formValues?: Record<string, unknown>;
}) {
  const label = (
    <span className="text-body-sm text-fg">
      {name}
      {required && <span className="text-danger ml-1">*</span>}
    </span>
  );

  // ── Special case: position_id → live picker of existing straddles ──
  if (name === "position_id") {
    return (
      <StraddlePositionPicker
        value={typeof value === "number" ? value : undefined}
        onChange={onChange}
        required={required}
      />
    );
  }

  // ── Special case: expiry → dropdown of real expiries for the underlying ──
  if (name === "expiry") {
    const underlying = String(formValues?.underlying ?? formValues?.symbol ?? "NIFTY").toUpperCase();
    return (
      <ExpiryPicker
        underlying={underlying}
        value={typeof value === "string" ? value : ""}
        onChange={(v) => onChange(v || undefined)}
        required={required}
      />
    );
  }

  // Boolean → checkbox
  if (prop.type === "boolean") {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 accent-accent"
        />
        {label}
      </label>
    );
  }

  // Enum (string) → select
  if (prop.enum && prop.enum.length) {
    return (
      <label className="block">
        {label}
        <select
          value={value == null ? "" : String(value)}
          onChange={(e) => onChange(e.target.value)}
          className="mt-1 w-full bg-surface border border-border rounded-sm px-3 py-2 text-body-sm text-fg"
        >
          <option value="" disabled>Select…</option>
          {prop.enum.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      </label>
    );
  }

  // Number / integer
  if (prop.type === "integer" || prop.type === "number") {
    return (
      <Input
        label={`${name}${required ? " *" : ""}`}
        type="number"
        step={prop.type === "integer" ? 1 : "any"}
        min={prop.minimum}
        max={prop.maximum}
        value={value == null ? "" : String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return onChange(undefined);
          const n = prop.type === "integer" ? parseInt(raw, 10) : parseFloat(raw);
          onChange(Number.isFinite(n) ? n : undefined);
        }}
        hint={defaultHint(prop)}
      />
    );
  }

  // Default → text input
  return (
    <Input
      label={`${name}${required ? " *" : ""}`}
      value={value == null ? "" : String(value)}
      onChange={(e) => onChange(e.target.value)}
      placeholder={prop.default != null ? String(prop.default) : undefined}
      hint={defaultHint(prop)}
    />
  );
}

function defaultHint(prop: SchemaProp): string | undefined {
  if (prop.default !== undefined) return `default: ${String(prop.default)}`;
  if (prop.minimum != null && prop.maximum != null) return `${prop.minimum} – ${prop.maximum}`;
  return undefined;
}

type LegacyStraddle = {
  id: number;
  underlying: string;
  strike: string | number;
  expiry: string;
  trade_date: string;
  status: "ACTIVE" | "PARTIAL" | "HEDGED" | "CLOSED";
  lots: number;
  ce_symbol?: string;
  pe_symbol?: string;
  premium_sold?: number;
  pnl_inr?: number;
};

function StraddlePositionPicker({
  value, onChange, required,
}: {
  value?: number;
  onChange: (v: number | undefined) => void;
  required: boolean;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["legacy", "straddles"],
    queryFn: () => legacyApi.get<{ results: LegacyStraddle[] }>("/legacy/straddles/").then((r) => r.data.results ?? []),
  });

  // Default to the first ACTIVE position once data lands.
  React.useEffect(() => {
    if (value != null || !data || data.length === 0) return;
    const active = data.find((p) => p.status === "ACTIVE") ?? data[0];
    if (active) onChange(active.id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  if (isLoading) {
    return (
      <div className="block">
        <span className="text-body-sm text-fg">
          position_id{required && <span className="text-danger ml-1">*</span>}
        </span>
        <div className="mt-1 h-9 bg-surface-2 border border-border rounded-sm animate-pulse" />
        <p className="text-caption text-fg-subtle mt-1">Loading positions…</p>
      </div>
    );
  }

  if (error || !data || data.length === 0) {
    return (
      <div className="block">
        <span className="text-body-sm text-fg">
          position_id{required && <span className="text-danger ml-1">*</span>}
        </span>
        <div className="mt-1 rounded-sm border border-warning/40 bg-warning/5 p-3 text-body-sm text-fg-muted">
          No straddle positions found. Register one with{" "}
          <code className="text-caption">python manage.py manage_straddle --register ...</code>{" "}
          (or use the Django shell), then reopen this dialog.
        </div>
      </div>
    );
  }

  // ACTIVE first, then by trade_date desc.
  const sorted = [...data].sort((a, b) => {
    if (a.status !== b.status) return a.status === "ACTIVE" ? -1 : 1;
    return (b.trade_date ?? "").localeCompare(a.trade_date ?? "");
  });

  const selected = sorted.find((p) => p.id === value);

  return (
    <div className="block">
      <span className="text-body-sm text-fg">
        Position{required && <span className="text-danger ml-1">*</span>}
        <span className="text-fg-subtle font-mono ml-2 text-caption">position_id</span>
      </span>
      <select
        value={value == null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
        className="mt-1 w-full bg-surface border border-border rounded-sm px-3 py-2 text-body-sm text-fg font-mono"
      >
        <option value="" disabled>Select a position…</option>
        {sorted.map((p) => {
          const dte = daysBetween(new Date(), new Date(p.expiry));
          return (
            <option key={p.id} value={p.id}>
              #{p.id} · {p.underlying} {p.strike} · {p.expiry} (DTE {dte}) · {p.lots}L · {p.status}
            </option>
          );
        })}
      </select>
      {selected && (
        <div className="mt-2 grid grid-cols-2 gap-2 rounded-sm border border-border bg-surface-2/40 p-2.5">
          <KvRow label="Underlying" value={`${selected.underlying} ${selected.strike}`} />
          <KvRow label="Expiry"     value={`${selected.expiry} (DTE ${daysBetween(new Date(), new Date(selected.expiry))})`} />
          <KvRow label="Premium sold" value={fmtINR(selected.premium_sold)} />
          <KvRow label="Live P&L"     value={fmtINR(selected.pnl_inr)} tone={(selected.pnl_inr ?? 0) >= 0 ? "success" : "danger"} />
          <KvRow label="CE leg" value={<code className="text-caption">{selected.ce_symbol}</code>} />
          <KvRow label="PE leg" value={<code className="text-caption">{selected.pe_symbol}</code>} />
        </div>
      )}
    </div>
  );
}

function daysBetween(a: Date, b: Date): number {
  return Math.max(0, Math.ceil((b.getTime() - a.getTime()) / (1000 * 60 * 60 * 24)));
}

type LegacyExpiry = {
  expiry: string;       // DDMMMYY
  iso: string;          // YYYY-MM-DD
  dte: number;
  kind: "weekly" | "monthly";
  strikes: number;
};

function ExpiryPicker({
  underlying, value, onChange, required,
}: {
  underlying: string;
  value: string;
  onChange: (v: string) => void;
  required: boolean;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["legacy", "expiries", underlying],
    queryFn: () =>
      legacyApi
        .get<{ results: LegacyExpiry[] }>(`/legacy/expiries/?underlying=${underlying}&limit=12`)
        .then((r) => r.data.results ?? []),
    enabled: !!underlying,
  });

  // Default to nearest monthly when data lands, if user hasn't picked.
  React.useEffect(() => {
    if (value || !data || data.length === 0) return;
    const nearest = data.find((e) => e.kind === "monthly") ?? data[0];
    onChange(nearest.expiry);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  if (isLoading) {
    return (
      <div className="block">
        <span className="text-body-sm text-fg">expiry{required && <span className="text-danger ml-1">*</span>}</span>
        <div className="mt-1 h-9 bg-surface-2 border border-border rounded-sm animate-pulse" />
      </div>
    );
  }
  if (error || !data || data.length === 0) {
    return (
      <Input
        label={`expiry${required ? " *" : ""}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="DDMMMYY (e.g. 29MAY26)"
        hint={`No expiries found for ${underlying} in the scrip master. Falling back to free text.`}
      />
    );
  }

  return (
    <label className="block">
      <span className="text-body-sm text-fg">
        Expiry{required && <span className="text-danger ml-1">*</span>}
        <span className="text-fg-subtle font-mono ml-2 text-caption">{underlying}</span>
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full bg-surface border border-border rounded-sm px-3 py-2 text-body-sm text-fg font-mono"
      >
        <option value="" disabled>Select an expiry…</option>
        {data.map((e) => (
          <option key={e.expiry} value={e.expiry}>
            {e.expiry} · {e.iso} · DTE {e.dte} · {e.kind} ({e.strikes} strikes)
          </option>
        ))}
      </select>
    </label>
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
