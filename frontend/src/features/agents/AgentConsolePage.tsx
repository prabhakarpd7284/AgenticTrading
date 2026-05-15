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
          <div
            ref={feedRef}
            className="h-full overflow-auto rounded-md border border-border bg-surface p-4 space-y-3"
            aria-live="polite"
            aria-label="Agent event stream"
          >
            {events.length === 0 ? (
              <div className="text-body-sm text-fg-subtle">
                {isLive
                  ? "Waiting for events — the desk will stream reasoning, tool calls, and decisions here."
                  : "Run completed. Open Overview for the structured result."}
              </div>
            ) : (
              events.map((e) => <EventBubble key={e.seq} ev={e} />)
            )}
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
    rr_ratio?: number;
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

type StraddleAnalysis = {
  net_pnl_inr?: number;
  premium_decayed_pct?: number;
  vix_phase?: "CALM" | "ELEVATED" | "SPIKE";
  market_phase?: string;
  net_delta?: number;
  delta_bias?: string;
  is_underwater?: boolean;
  stop_triggered?: boolean;
  days_to_expiry?: number;
  nifty_spot?: number;
  combined_sold?: number;
  combined_current?: number;
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

type RunResult = {
  plan?: DirectionalPlan;
  risk?: DirectionalRisk;
  execution?: ExecutionResult;
  market_data?: MarketData;
  symbol?: string;
  action?: StraddleAction;
  analysis?: StraddleAnalysis;
  position?: StraddlePosition;
  validated?: { approved?: boolean; override?: string | null; reason?: string };
  error?: string;
};

/* =================================================================== */
/* Directional overview                                                 */
/* =================================================================== */
function DirectionalOverview({ result }: { result: RunResult }) {
  const plan = result.plan ?? {};
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
            <div className="grid grid-cols-2 gap-2">
              <KvRow label="Risk amount"      value={fmtINR(risk.details.risk_amount)} />
              <KvRow label="Max risk allowed" value={fmtINR(risk.details.max_risk_allowed)} />
              <KvRow label="Position value"   value={fmtINR(risk.details.position_value)} />
              <KvRow label="Max position"     value={fmtINR(risk.details.max_position_value)} />
              <KvRow label="Daily loss"       value={fmtINR(risk.details.daily_loss_so_far)} />
              <KvRow label="Max daily loss"   value={fmtINR(risk.details.max_daily_loss)} />
              {risk.details.rr_ratio != null && (
                <KvRow label="R:R" value={`${risk.details.rr_ratio.toFixed(2)}×`} />
              )}
              {(risk.details.regime as any)?.cache && (
                <KvRow label="Regime" value={String((risk.details.regime as any).cache)} />
              )}
            </div>
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
function StraddleOverview({ result }: { result: RunResult }) {
  const pos = result.position ?? {};
  const an  = result.analysis ?? {};
  const act = result.action ?? {};
  const v   = result.validated ?? {};

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
          <div className="grid grid-cols-2 gap-2 mb-3">
            <KvBlock label="CE action" value={act.ce_action ?? "—"} />
            <KvBlock label="PE action" value={act.pe_action ?? "—"} />
          </div>
          {act.reasoning && (
            <div className="mb-3">
              <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Reasoning</div>
              <p className="text-body-sm text-fg-muted whitespace-pre-wrap">{act.reasoning}</p>
            </div>
          )}
          {act.key_risk && (
            <div>
              <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Key risk</div>
              <p className="text-body-sm text-fg-muted whitespace-pre-wrap">{act.key_risk}</p>
            </div>
          )}
        </CardContent>
      </Card>

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
          <KvRow label="CE symbol" value={<code>{pos.ce_symbol}</code>} />
          <KvRow label="PE symbol" value={<code>{pos.pe_symbol}</code>} />
        </CardContent>
      </Card>

      {/* Analysis card */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis</CardTitle>
          <CardDescription>Pure-Python — P&amp;L, delta, market phase</CardDescription>
        </CardHeader>
        <CardContent>
          <KvRow
            label="Net P&L"
            value={fmtINR(an.net_pnl_inr)}
            tone={an.net_pnl_inr != null && an.net_pnl_inr >= 0 ? "success" : "danger"}
          />
          <KvRow label="Premium decayed" value={an.premium_decayed_pct != null ? `${an.premium_decayed_pct.toFixed(1)}%` : "—"} />
          <KvRow label="Net delta"  value={an.net_delta != null ? an.net_delta.toFixed(2) : "—"} />
          <KvRow label="Delta bias" value={an.delta_bias ?? "—"} />
          <KvRow label="Market phase" value={an.market_phase ?? "—"} />
          <KvRow label="VIX phase"  value={an.vix_phase ?? "—"} />
          <KvRow label="DTE"        value={an.days_to_expiry != null ? String(an.days_to_expiry) : "—"} />
          <KvRow
            label="Underwater?"
            value={an.is_underwater ? "Yes" : "No"}
            tone={an.is_underwater ? "danger" : "success"}
          />
        </CardContent>
      </Card>
    </div>
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
  name, prop, required, value, onChange,
}: {
  name: string;
  prop: SchemaProp;
  required: boolean;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = (
    <span className="text-body-sm text-fg">
      {name}
      {required && <span className="text-danger ml-1">*</span>}
    </span>
  );

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
