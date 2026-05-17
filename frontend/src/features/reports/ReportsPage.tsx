/**
 * Reports — cross-run analytics derived from /api/v1/agents/runs/.
 *
 * Reads every run the user has access to, then aggregates:
 *   - Per-strategy summary: total · success rate · @RiskGuard approval rate
 *     · avg confidence · avg duration · paper / dry-run mix
 *   - Action distribution (straddle) — pie of HOLD / CLOSE_BOTH / MONITOR ...
 *   - Symbol leaderboard (directional) — top 10 symbols by run count
 *   - Recent runs table — clickable into /agents/{id}
 *
 * No new backend endpoint; everything is derived from run.result + status +
 * duration. When stream-history replay lands we can add per-node latency here.
 */
import * as React from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, Bot, ShieldCheck, Sparkles, Triangle } from "lucide-react";

import { api } from "@/lib/api";
import type { AgentRun } from "@/types";
import { cn, fmtRel } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";

type Filter = "all" | "directional" | "short_straddle";

export function ReportsPage() {
  const { data: runs = [], isLoading } = useQuery({
    queryKey: ["agent-runs", "all"],
    queryFn: () => api.get<AgentRun[]>("/agents/runs/?limit=500").then((r) => r.data),
    refetchInterval: 15_000,
  });

  const [filter, setFilter] = React.useState<Filter>("all");
  const filtered = React.useMemo(
    () => filter === "all" ? runs : runs.filter((r) => r.strategy_name === filter),
    [runs, filter],
  );

  const topline = React.useMemo(() => deriveTopline(filtered), [filtered]);
  const byStrategy = React.useMemo(() => groupBy(runs, (r) => r.strategy_name), [runs]);

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Cross-run reporting</p>
          <h1 className="text-h1 text-fg flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-accent" aria-hidden />
            Reports
          </h1>
          <p className="text-body-sm text-fg-muted mt-1">
            Aggregates derived from every agent run. Click into a run to see the full graph state.
          </p>
        </div>
        <Tabs value={filter} onValueChange={(v) => setFilter(v as Filter)}>
          <TabsList>
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="directional">Directional</TabsTrigger>
            <TabsTrigger value="short_straddle">Short straddle</TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      {/* Topline KPIs */}
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Runs" value={String(topline.total)} hint={`last ${runs.length} known`} />
        <KPI
          label="Completion"
          value={pct(topline.succeeded, topline.total)}
          hint={`${topline.succeeded} succeeded · ${topline.failed} failed`}
          tone={topline.failed ? "warning" : "success"}
        />
        <KPI
          label="@RiskGuard approval"
          value={pct(topline.riskApproved, topline.riskDecided)}
          hint={`${topline.riskApproved}/${topline.riskDecided} approved`}
        />
        <KPI
          label="Avg duration"
          value={topline.avgDurSec != null ? `${topline.avgDurSec.toFixed(1)} s` : "—"}
          hint={topline.avgConf != null ? `avg conf ${topline.avgConf.toFixed(2)}` : ""}
        />
      </section>

      {/* Per-strategy summary table */}
      <Card>
        <CardHeader>
          <CardTitle>Per strategy</CardTitle>
          <CardDescription>One row per strategy across all runs (filter is ignored here).</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto -mx-4 sm:mx-0">
            <table className="w-full text-body-sm">
              <thead>
                <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
                  <th className="text-left py-2 px-3">Strategy</th>
                  <th className="text-right py-2 px-3">Runs</th>
                  <th className="text-right py-2 px-3">Success</th>
                  <th className="text-right py-2 px-3">RiskGuard ✓</th>
                  <th className="text-right py-2 px-3">Avg conf</th>
                  <th className="text-right py-2 px-3">Avg dur</th>
                  <th className="text-right py-2 px-3">Latest</th>
                </tr>
              </thead>
              <tbody>
                {Array.from(byStrategy.entries()).map(([name, group]) => {
                  const s = deriveTopline(group);
                  const latest = group[0];
                  return (
                    <tr key={name} className="border-b border-border last:border-b-0 hover:bg-surface-2">
                      <td className="py-2 px-3">
                        <Badge tone="info">{name}</Badge>
                      </td>
                      <td className="text-right py-2 px-3 font-mono">{s.total}</td>
                      <td className="text-right py-2 px-3 font-mono">{pct(s.succeeded, s.total)}</td>
                      <td className="text-right py-2 px-3 font-mono">{pct(s.riskApproved, s.riskDecided)}</td>
                      <td className="text-right py-2 px-3 font-mono">{s.avgConf != null ? s.avgConf.toFixed(2) : "—"}</td>
                      <td className="text-right py-2 px-3 font-mono">{s.avgDurSec != null ? `${s.avgDurSec.toFixed(1)}s` : "—"}</td>
                      <td className="text-right py-2 px-3 text-caption text-fg-subtle">
                        {latest ? fmtRel(latest.created_at) : "—"}
                      </td>
                    </tr>
                  );
                })}
                {byStrategy.size === 0 && (
                  <tr><td colSpan={7} className="py-6 text-center text-fg-subtle">No runs yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Action distribution (straddle) */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Triangle className="h-4 w-4 text-accent" aria-hidden /> Straddle action mix</CardTitle>
            <CardDescription>What the LLM decided across every short-straddle run.</CardDescription>
          </CardHeader>
          <CardContent className="h-[280px]">
            <ActionDistributionChart runs={runs.filter((r) => r.strategy_name === "short_straddle")} />
          </CardContent>
        </Card>

        {/* Symbol leaderboard (directional) */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-accent" aria-hidden /> Directional symbol leaderboard</CardTitle>
            <CardDescription>Top symbols by run count + RiskGuard approval rate.</CardDescription>
          </CardHeader>
          <CardContent>
            <SymbolLeaderboard runs={runs.filter((r) => r.strategy_name === "directional")} />
          </CardContent>
        </Card>
      </div>

      {/* Recent runs table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent runs</CardTitle>
          <CardDescription>
            {filter === "all" ? "All strategies" : filter} · click a row to open in the agent console.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-8 w-full" />)}
            </div>
          ) : filtered.length === 0 ? (
            <EmptyState icon={<Sparkles />} title="No runs match this filter" description="Try a different strategy filter or start a new run from /agents." />
          ) : (
            <div className="overflow-x-auto -mx-4 sm:mx-0">
              <table className="w-full text-body-sm">
                <thead>
                  <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
                    <th className="text-left py-2 px-3">When</th>
                    <th className="text-left py-2 px-3">Strategy</th>
                    <th className="text-left py-2 px-3">Status</th>
                    <th className="text-left py-2 px-3">Outcome</th>
                    <th className="text-right py-2 px-3">Conf</th>
                    <th className="text-right py-2 px-3">Dur</th>
                    <th className="text-left py-2 px-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, 50).map((r) => {
                    const o = perRunOutcome(r);
                    return (
                      <tr key={r.id} className="border-b border-border last:border-b-0 hover:bg-surface-2">
                        <td className="py-2 px-3 text-caption text-fg-subtle">{fmtRel(r.created_at)}</td>
                        <td className="py-2 px-3"><Badge tone="info">{r.strategy_name}</Badge></td>
                        <td className="py-2 px-3"><StatusBadge status={r.status} /></td>
                        <td className="py-2 px-3 font-mono">
                          {o.label}
                          {o.subtitle && <span className="text-fg-subtle ml-2 text-caption">{o.subtitle}</span>}
                        </td>
                        <td className="text-right py-2 px-3 font-mono">{o.confidence != null ? o.confidence.toFixed(2) : "—"}</td>
                        <td className="text-right py-2 px-3 font-mono text-caption">{o.durSec != null ? `${o.durSec.toFixed(1)}s` : "—"}</td>
                        <td className="py-2 px-3">
                          <Link to={`/agents/${r.id}`} className="text-accent text-caption font-mono hover:underline">open →</Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────── */
/* Aggregations                                                        */
/* ─────────────────────────────────────────────────────────────────── */
type Topline = {
  total: number;
  succeeded: number;
  failed: number;
  riskDecided: number;
  riskApproved: number;
  avgConf: number | null;
  avgDurSec: number | null;
};

function deriveTopline(runs: AgentRun[]): Topline {
  let succeeded = 0, failed = 0, riskDecided = 0, riskApproved = 0;
  const confs: number[] = [];
  const durs: number[] = [];
  for (const r of runs) {
    if (r.status === "succeeded") succeeded++;
    if (r.status === "failed") failed++;
    const result = (r.result ?? {}) as any;
    const risk = result.risk ?? result.validated;
    if (risk && typeof risk.approved === "boolean") {
      riskDecided++;
      if (risk.approved) riskApproved++;
    }
    const conf = result.plan?.confidence ?? result.action?.confidence;
    if (typeof conf === "number" && Number.isFinite(conf)) confs.push(conf);
    if (r.started_at && r.completed_at) {
      const d = (new Date(r.completed_at).getTime() - new Date(r.started_at).getTime()) / 1000;
      if (Number.isFinite(d) && d >= 0) durs.push(d);
    }
  }
  return {
    total: runs.length,
    succeeded, failed,
    riskDecided, riskApproved,
    avgConf: confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null,
    avgDurSec: durs.length ? durs.reduce((a, b) => a + b, 0) / durs.length : null,
  };
}

function perRunOutcome(r: AgentRun): { label: string; subtitle?: string; confidence?: number; durSec?: number } {
  const result = (r.result ?? {}) as any;
  const dur = r.started_at && r.completed_at
    ? (new Date(r.completed_at).getTime() - new Date(r.started_at).getTime()) / 1000
    : undefined;
  if (result.error) {
    return { label: "ERROR", subtitle: String(result.error).slice(0, 60), durSec: dur };
  }
  if (r.strategy_name === "short_straddle") {
    const act = result.action ?? {};
    return {
      label: act.action ?? "—",
      subtitle: result.position ? `${result.position.underlying} ${result.position.strike}` : undefined,
      confidence: act.confidence,
      durSec: dur,
    };
  }
  // directional
  const plan = result.plan ?? {};
  const risk = result.risk ?? {};
  return {
    label: plan.symbol ? `${plan.side ?? "?"} ${plan.symbol} ×${plan.quantity ?? 0}` : "—",
    subtitle: risk.approved === false ? "rejected" : risk.approved === true ? "approved" : undefined,
    confidence: plan.confidence,
    durSec: dur,
  };
}

function groupBy<T, K>(items: T[], key: (item: T) => K): Map<K, T[]> {
  const m = new Map<K, T[]>();
  for (const item of items) {
    const k = key(item);
    if (!m.has(k)) m.set(k, []);
    m.get(k)!.push(item);
  }
  return m;
}

/* ─────────────────────────────────────────────────────────────────── */
/* Charts                                                              */
/* ─────────────────────────────────────────────────────────────────── */
function ActionDistributionChart({ runs }: { runs: AgentRun[] }) {
  const [R, setR] = React.useState<any>(null);
  React.useEffect(() => { import("recharts").then(setR); }, []);

  const counts = React.useMemo(() => {
    const c = new Map<string, number>();
    for (const r of runs) {
      const a = ((r.result ?? {}) as any).action?.action ?? "—";
      c.set(a, (c.get(a) ?? 0) + 1);
    }
    return Array.from(c.entries()).map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [runs]);

  if (runs.length === 0) {
    return <EmptyState icon={<Triangle />} title="No straddle runs yet" description="Fire a short_straddle run to populate this chart." />;
  }
  if (!R) return <div className="h-full flex items-center justify-center text-fg-subtle text-body-sm">Loading…</div>;

  const colorByAction: Record<string, string> = {
    HOLD: "#3fb950",
    CLOSE_BOTH: "#f85149",
    CLOSE_CE: "#d29922",
    CLOSE_PE: "#d29922",
    MONITOR: "#79c0ff",
    SHIFT_TO_ATM: "#58a6ff",
    HEDGE_FUTURES: "#a371f7",
    ROLL_PE: "#a371f7",
    ROLL_CE: "#a371f7",
    "—": "#6e7681",
  };
  const COLORS = counts.map((c) => colorByAction[c.name] ?? "#6e7681");

  const { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } = R;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie data={counts} dataKey="value" nameKey="name" innerRadius={40} outerRadius={90} paddingAngle={2}>
          {counts.map((_: any, i: number) => <Cell key={i} fill={COLORS[i]} />)}
        </Pie>
        <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6, fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

function SymbolLeaderboard({ runs }: { runs: AgentRun[] }) {
  type SymbolStat = { symbol: string; total: number; approved: number; rejected: number; lastSide?: string };

  const rows = React.useMemo(() => {
    const m = new Map<string, SymbolStat>();
    for (const r of runs) {
      const plan = ((r.result ?? {}) as any).plan ?? {};
      const sym = plan.symbol;
      if (!sym) continue;
      if (!m.has(sym)) m.set(sym, { symbol: sym, total: 0, approved: 0, rejected: 0 });
      const s = m.get(sym)!;
      s.total++;
      s.lastSide = plan.side ?? s.lastSide;
      const risk = ((r.result ?? {}) as any).risk;
      if (risk?.approved === true) s.approved++;
      if (risk?.approved === false) s.rejected++;
    }
    return Array.from(m.values()).sort((a, b) => b.total - a.total).slice(0, 10);
  }, [runs]);

  if (rows.length === 0) {
    return <EmptyState icon={<Bot />} title="No directional runs yet" description="Run the directional strategy to see symbols here." />;
  }

  return (
    <table className="w-full text-body-sm">
      <thead>
        <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
          <th className="text-left py-1.5 px-2">Symbol</th>
          <th className="text-right py-1.5 px-2">Runs</th>
          <th className="text-right py-1.5 px-2">Approved</th>
          <th className="text-right py-1.5 px-2">Rejected</th>
          <th className="text-right py-1.5 px-2">Last side</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((s) => (
          <tr key={s.symbol} className="border-b border-border last:border-b-0">
            <td className="py-1.5 px-2 font-mono">{s.symbol}</td>
            <td className="text-right py-1.5 px-2 font-mono">{s.total}</td>
            <td className="text-right py-1.5 px-2 font-mono text-pnl-up">{s.approved}</td>
            <td className="text-right py-1.5 px-2 font-mono text-pnl-down">{s.rejected}</td>
            <td className="text-right py-1.5 px-2 font-mono">
              {s.lastSide ? <Badge tone={s.lastSide === "BUY" ? "success" : "danger"}>{s.lastSide}</Badge> : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ─────────────────────────────────────────────────────────────────── */
/* Tiny UI helpers                                                     */
/* ─────────────────────────────────────────────────────────────────── */
function KPI({ label, value, hint, tone = "neutral" }: { label: string; value: string; hint?: string; tone?: "neutral" | "success" | "warning" | "danger" }) {
  const color =
    tone === "success" ? "text-pnl-up"
    : tone === "warning" ? "text-warning"
    : tone === "danger" ? "text-pnl-down"
    : "text-fg";
  return (
    <div className="rounded-md border border-border bg-surface p-3">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn("text-h3 font-mono mt-0.5", color)}>{value}</div>
      {hint && <div className="text-caption text-fg-subtle mt-0.5">{hint}</div>}
    </div>
  );
}

function StatusBadge({ status }: { status: AgentRun["status"] }) {
  const map = {
    queued:    { tone: "neutral" as const, label: "Queued" },
    running:   { tone: "info" as const,    label: "Running" },
    succeeded: { tone: "success" as const, label: "Succeeded" },
    failed:    { tone: "danger" as const,  label: "Failed" },
    cancelled: { tone: "neutral" as const, label: "Cancelled" },
  }[status];
  return <Badge tone={map.tone} dot>{map.label}</Badge>;
}

function pct(n: number, d: number): string {
  if (!d) return "—";
  return `${Math.round((n / d) * 100)}%`;
}
