import * as React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  Activity, AlertTriangle, ArrowRight, Bot, Briefcase, RefreshCcw, ShieldCheck, Sparkles,
} from "lucide-react";
import {
  LineChart, Line, ResponsiveContainer, YAxis, Tooltip as ReTooltip,
  CartesianGrid, ReferenceLine,
} from "recharts";

import { api } from "@/lib/api";
import { connect } from "@/lib/ws";
import type { AgentRun, Portfolio } from "@/types";
import { clsPnl, fmtInr } from "@/lib/utils";
import {
  useAuditFeed, useRiskAlerts, usePortfolioSummary, usePositions,
  useRiskOverview, useSystemStatus,
} from "@/lib/v2";

import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { KPI } from "@/components/ui/KPI";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";

type PnlMsg = { portfolio_id: string; mtm: number; day_pnl: number; unrealized: number };

export function DashboardPage() {
  const nav = useNavigate();
  const qc = useQueryClient();

  /* ------------------------------------------------------------ */
  /* Page queries — each panel below pulls from one of these.      */
  /* ------------------------------------------------------------ */
  const portfolioQ = usePortfolioSummary();
  const positionsQ = usePositions();
  const auditQ     = useAuditFeed(8);
  const alertsQ    = useRiskAlerts();
  const riskQ      = useRiskOverview();
  const systemQ    = useSystemStatus();
  const {
    data: legacyPortfolio,
    isLoading: pfLoading,
    dataUpdatedAt: portfolioUpdatedAt,
  } = portfolioQ;
  const { data: legacyPositions }  = positionsQ;
  const { data: auditFeed = [] }   = auditQ;
  const { data: alerts = [] }      = alertsQ;
  const { data: risk }             = riskQ;
  const { data: system }           = systemQ;

  /* v2 queries — kept so the page still works once the native v2   */
  /* schema is populated. These are currently empty for most users.  */
  const portfoliosQ = useQuery({
    queryKey: ["portfolios"],
    queryFn: () => api.get<Portfolio[]>("/portfolios/").then((r) => r.data),
  });
  const runsQ = useQuery({
    queryKey: ["agent-runs", "recent"],
    queryFn: () => api.get<AgentRun[]>("/agents/runs/?limit=5").then((r) => r.data),
  });
  const { data: v2portfolios } = portfoliosQ;
  const { data: v2runs = [] } = runsQ;
  const primary = v2portfolios?.[0];

  /* ── Manual refresh: refetch every panel on the page in one go.  */
  const isAnyFetching =
    portfolioQ.isFetching || positionsQ.isFetching || auditQ.isFetching ||
    alertsQ.isFetching   || riskQ.isFetching     || systemQ.isFetching ||
    portfoliosQ.isFetching || runsQ.isFetching;
  const refreshAll = React.useCallback(() => {
    // Invalidate by prefix so the page picks up data from *any* in-flight
    // op (e.g. an enrich_signals run that touched the audit feed).
    qc.invalidateQueries({ queryKey: ["portfolio-summary"] });
    qc.invalidateQueries({ queryKey: ["positions"] });
    qc.invalidateQueries({ queryKey: ["audit"] });
    qc.invalidateQueries({ queryKey: ["risk-alerts"] });
    qc.invalidateQueries({ queryKey: ["risk"] });
    qc.invalidateQueries({ queryKey: ["system-status"] });
    qc.invalidateQueries({ queryKey: ["portfolios"] });
    qc.invalidateQueries({ queryKey: ["agent-runs"] });
  }, [qc]);

  /* ------------------------------------------------------------ */
  /* Live MTM curve from WebSocket                                 */
  /* ------------------------------------------------------------ */
  const [live, setLive] = React.useState<PnlMsg | null>(null);
  const [curve, setCurve] = React.useState<{ t: number; v: number }[]>([]);
  // Last PnL push time — drives the freshness pill on the equity-curve card
  // so the operator can tell the difference between "flat market" and
  // "WebSocket has been silent for 3 minutes".
  const [lastPnlAt, setLastPnlAt] = React.useState<number | null>(null);
  React.useEffect(() => {
    const ws = connect("/ws/pnl/", (msg) => {
      const m = msg as PnlMsg;
      setLive(m);
      setCurve((c) => [...c.slice(-199), { t: Date.now(), v: m.mtm }]);
      setLastPnlAt(Date.now());
    });
    return () => ws.close();
  }, []);

  /* ------------------------------------------------------------ */
  /* Derived numbers — prefer legacy, fall back to v2 / WS         */
  /* ------------------------------------------------------------ */
  const capital    = legacyPortfolio?.capital    ?? parseFloat(primary?.capital ?? "0");
  const dayPnl     = legacyPortfolio?.combined_pnl
                    ?? live?.day_pnl
                    ?? parseFloat(primary?.day_pnl ?? "0");
  const unrealized = legacyPortfolio?.combined?.options_pnl
                    ?? live?.unrealized
                    ?? 0;
  const openCount  = (legacyPositions?.equity.length ?? 0) +
                     (legacyPositions?.options.length ?? 0);
  const dayReturnPct = capital ? (dayPnl / capital) * 100 : 0;

  const greeting = getGreeting();
  const mode = system?.trading_mode ?? primary?.mode ?? "paper";
  const marketOpen = system?.is_market_open ?? false;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      {/* Header */}
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">{greeting}</p>
          <h1 className="text-h1 text-fg">
            Your desk is <span className="text-accent">watching the market</span>.
          </h1>
          <p className="text-body-sm text-fg-muted mt-1 flex items-center gap-2 flex-wrap">
            <span className="uppercase">{mode}</span> mode
            <span aria-hidden>·</span>
            <Badge tone={marketOpen ? "success" : "neutral"} dot>
              {marketOpen ? "Market open" : (system?.session?.session_phase ?? "Closed")}
            </Badge>
            <span aria-hidden>·</span>
            NIFTY50
            <span aria-hidden>·</span>
            capital {fmtInr(capital)}
            <span aria-hidden>·</span>
            <FreshnessIndicator
              variant="muted"
              label="Portfolio"
              timestamp={portfolioUpdatedAt}
              freshMs={20_000}
              staleMs={60_000}
            />
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={refreshAll}
            disabled={isAnyFetching}
            aria-label="Refresh desk data"
            title="Refetch every panel on this page"
          >
            <RefreshCcw className={`h-4 w-4 ${isAnyFetching ? "animate-spin" : ""}`} />
          </Button>
          <Button
            variant="secondary"
            leading={<ShieldCheck className="h-4 w-4" />}
            onClick={() => nav("/brokers")}
          >
            Link broker
          </Button>
          <Button
            leading={<Bot className="h-4 w-4" />}
            onClick={() => nav("/agents?new=1")}
          >
            New agent run
          </Button>
        </div>
      </header>

      {/* Alert banner — surface critical alerts from legacy */}
      {alerts.length > 0 && (
        <div role="alert" className="rounded-md border border-warning/40 bg-warning/5 p-3 flex gap-3 items-start">
          <AlertTriangle className="h-5 w-5 text-warning mt-0.5 shrink-0" aria-hidden />
          <div className="flex-1 min-w-0">
            <div className="text-body-sm font-semibold text-fg">
              {alerts.length} active alert{alerts.length > 1 ? "s" : ""}
            </div>
            <ul className="mt-1 space-y-1">
              {alerts.slice(0, 3).map((a, i) => (
                <li key={i} className="text-body-sm text-fg-muted">
                  <Badge
                    tone={a.severity === "critical" ? "danger" : a.severity === "warning" ? "warning" : "neutral"}
                    className="mr-2"
                  >
                    {a.severity}
                  </Badge>
                  {a.message} <span className="text-fg-subtle">— {a.action}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* KPI grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <KPI
          label="Day P&L"
          value={dayPnl}
          valueFormat="inr"
          delta={dayReturnPct}
          deltaFormat="pct"
          live
          loading={pfLoading}
          hint="Equity + options, realized + unrealized"
        />
        <KPI
          label="Options P&L"
          value={unrealized}
          valueFormat="inr"
          live
          loading={pfLoading}
          hint={`${legacyPortfolio?.straddle_count ?? 0} active straddles`}
        />
        <KPI
          label="Capital"
          value={capital}
          valueFormat="inr"
          loading={pfLoading}
          hint={
            legacyPortfolio
              ? `${fmtInr(legacyPortfolio.available_cash, { compact: true })} free`
              : mode === "live" ? "Live trading" : "Paper trading"
          }
        />
        <KPI
          label="Daily loss used"
          value={risk?.daily_loss_pct ?? 0}
          valueFormat="pct"
          loading={!risk}
          hint={
            risk
              ? `${fmtInr(risk.daily_loss)} / ${fmtInr(risk.max_daily_loss)} cap`
              : "—"
          }
        />
      </div>

      {/* Row 2 — equity curve + agent feed */}
      <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6">
        <Card>
          <CardHeader>
            <div className="flex items-start justify-between gap-3">
              <div>
                <CardTitle>Equity curve</CardTitle>
                <CardDescription>
                  Mark-to-market since market open, updated live.
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <FreshnessIndicator
                  label="Last tick"
                  timestamp={lastPnlAt}
                  freshMs={5_000}
                  staleMs={30_000}
                />
                <Badge tone="info" dot>Live</Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="h-64">
            {curve.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <Skeleton className="h-48 w-full" />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curve}>
                  <CartesianGrid stroke="rgb(var(--border))" vertical={false} strokeDasharray="3 3" />
                  <YAxis
                    domain={["auto", "auto"]}
                    stroke="rgb(var(--fg-subtle))"
                    fontSize={11}
                    tickFormatter={(v) => fmtInr(v, { compact: true })}
                    width={60}
                  />
                  <ReferenceLine
                    y={capital}
                    stroke="rgb(var(--border-strong))"
                    strokeDasharray="4 4"
                    label={{ value: "open", fill: "rgb(var(--fg-subtle))", fontSize: 10 }}
                  />
                  <ReTooltip
                    contentStyle={{
                      background: "rgb(var(--surface))",
                      border: "1px solid rgb(var(--border-strong))",
                      borderRadius: 6,
                      color: "rgb(var(--fg))",
                      fontSize: 12,
                    }}
                    labelFormatter={(l) => new Date(l).toLocaleTimeString("en-IN")}
                    formatter={(v: number) => [fmtInr(v), "MTM"]}
                  />
                  <Line
                    dataKey="v"
                    stroke={dayPnl >= 0 ? "rgb(var(--pnl-up))" : "rgb(var(--pnl-down))"}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-start justify-between gap-3">
              <div>
                <CardTitle>AI desk — recent activity</CardTitle>
                <CardDescription>Last 8 decisions from the audit log.</CardDescription>
              </div>
              <Button
                variant="link"
                size="sm"
                onClick={() => nav("/agents")}
                trailing={<ArrowRight className="h-3.5 w-3.5" />}
              >
                View all
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {auditFeed.length === 0 && v2runs.length === 0 ? (
              <EmptyState
                className="m-4"
                icon={<Sparkles />}
                title="No AI activity yet"
                description="Start a directional or straddle run and the desk will begin thinking."
                action={
                  <Button size="sm" onClick={() => nav("/agents?new=1")}>
                    Start a run
                  </Button>
                }
              />
            ) : (
              <ul className="divide-y divide-border">
                {auditFeed.slice(0, 8).map((e, i) => (
                  <li key={i} className="px-5 py-2.5 flex items-start gap-3">
                    <AuditTypeBadge type={e.type} />
                    <div className="flex-1 min-w-0">
                      <div className="text-body-sm text-fg truncate">{e.detail}</div>
                      <div className="text-caption text-fg-subtle font-mono">
                        {e.time}{e.symbol && ` · ${e.symbol}`}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Row 3 — open positions preview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Open positions</CardTitle>
              <CardDescription>
                {openCount} open · equity {legacyPositions?.equity.length ?? 0} ·
                options {legacyPositions?.options.length ?? 0}
              </CardDescription>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => nav("/positions")}
              leading={<Briefcase className="h-4 w-4" />}
            >
              Go to positions
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {openCount === 0 ? (
            <EmptyState
              className="m-4"
              icon={<Activity />}
              title="No open positions"
              description="When the desk executes a trade, it will show up here with a live mark."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="text-left text-caption uppercase tracking-wider text-fg-subtle bg-surface-2">
                    {["Symbol", "Side", "Qty", "Entry", "Target / SL", "P&L"].map((h) => (
                      <th key={h} className="px-5 py-2">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(legacyPositions?.equity ?? []).slice(0, 5).map((p) => (
                    <tr key={`eq-${p.id}`} className="border-t border-border/60 hover:bg-surface-2/40">
                      <td className="px-5 py-2 text-body-sm text-fg font-medium">{p.symbol}</td>
                      <td className="px-5 py-2">
                        <Badge tone={p.side === "BUY" ? "success" : "danger"}>{p.side}</Badge>
                      </td>
                      <td className="px-5 py-2 text-body-sm font-mono tabular text-fg">{p.quantity}</td>
                      <td className="px-5 py-2 text-body-sm font-mono tabular text-fg">
                        {fmtInr(p.entry_price)}
                      </td>
                      <td className="px-5 py-2 text-body-sm font-mono tabular text-fg-muted">
                        {fmtInr(p.target)} / {fmtInr(p.stop_loss)}
                      </td>
                      <td className={`px-5 py-2 text-body-sm font-mono tabular ${clsPnl(p.pnl ?? 0)}`}>
                        {p.pnl == null ? "—" : fmtInr(p.pnl)}
                      </td>
                    </tr>
                  ))}
                  {(legacyPositions?.options ?? []).slice(0, 5).map((p) => (
                    <tr key={`op-${p.id}`} className="border-t border-border/60 hover:bg-surface-2/40">
                      <td className="px-5 py-2 text-body-sm text-fg font-medium">
                        {p.underlying} {p.strike} straddle
                      </td>
                      <td className="px-5 py-2">
                        <Badge tone="info">SHORT</Badge>
                      </td>
                      <td className="px-5 py-2 text-body-sm font-mono tabular text-fg">
                        {p.lots}×{p.lot_size}
                      </td>
                      <td className="px-5 py-2 text-body-sm font-mono tabular text-fg">
                        {fmtInr(p.ce_sell + p.pe_sell)}
                      </td>
                      <td className="px-5 py-2 text-body-sm font-mono tabular text-fg-muted">
                        Δ {p.net_delta.toFixed(2)} · {p.dte}d
                      </td>
                      <td className={`px-5 py-2 text-body-sm font-mono tabular ${clsPnl(p.pnl_inr)}`}>
                        {fmtInr(p.pnl_inr)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */
function getGreeting() {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  return "Good evening";
}

function AuditTypeBadge({ type }: { type: string }) {
  const map: Record<string, { tone: "neutral" | "info" | "success" | "danger" | "warning"; label: string }> = {
    PLANNER_REQ:    { tone: "info",    label: "PLAN" },
    PLANNER_RES:    { tone: "info",    label: "PLAN" },
    PLANNER_ERR:    { tone: "danger",  label: "PLAN" },
    RISK_APPROVE:   { tone: "success", label: "RISK" },
    RISK_REJECT:    { tone: "danger",  label: "RISK" },
    EXECUTION:      { tone: "success", label: "EXEC" },
    RECONCILE:      { tone: "neutral", label: "SYNC" },
  };
  const m = map[type] ?? { tone: "neutral" as const, label: type.slice(0, 4) };
  return <Badge tone={m.tone}>{m.label}</Badge>;
}

