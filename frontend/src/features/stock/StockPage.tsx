/**
 * Stock view — desk-level summary for a single symbol.
 *
 * Single backend call: GET /api/v1/legacy/stock-summary/?symbol=X&period=W|M|H|Y
 * powers everything below. Frontend never aggregates — backend does the
 * heavy filtering across TradeJournal · StraddlePosition · AgentRun and
 * returns a typed payload with KPIs, buckets, open positions (with
 * margin + leverage + expected exit), per-period rollups, strategies, and
 * the latest indicator snapshot.
 */
import * as React from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Bot, Briefcase, ChevronRight, Info, LineChart as LineChartIcon, Search,
  ShieldCheck, Sparkles, Triangle, TrendingUp, Wallet,
} from "lucide-react";

import { api, legacyApi } from "@/lib/api";
import { cn, fmtRel } from "@/lib/utils";
import { INDICATORS, getIndicator, indicatorsByStrategy, type IndicatorDef } from "@/lib/indicators";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { PlanStockButton, PlanReport, getStoredPlan } from "./PlanStockButton";

type Period = "weekly" | "monthly" | "half-yearly" | "yearly";

type Bucket = {
  key: string;
  count: number;
  notional: number;
  total_margin: number;
  premium_received: number;
  premium_paid: number;
  leverage: number;
};

type OpenPosition = {
  kind: "equity" | "options";
  strategy: string;
  id: number;
  leg: string;
  side: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  stop_loss?: number;
  expected_exit: number;
  pnl_inr: number;
  status: string;
  opened_at?: string;
  expiry?: string;
  notional: number;
  margin: number;
  leverage: number;
  premium_received?: number;
};

type Rollup = {
  period_label: string;
  from: string;
  to: string;
  trades_planned: number;
  trades_taken: number;
  capital_deployed: number;
  pnl: number;
  positions_opened: number;
  positions_closed: number;
  win_rate: number;
};

type StockSummary = {
  symbol: string;
  kind: "equity" | "index_underlying";
  period: Period;
  kpis: {
    capital_deployed: number;
    money_in_play: number;
    open_count: number;
    period_pnl: number;
    live_leverage: number;
    period_window: [string, string];
  };
  buckets: Bucket[];
  bucket_totals: {
    notional: number; total_margin: number;
    premium_received: number; premium_paid: number; leverage: number;
  };
  open_positions: OpenPosition[];
  rollups: Rollup[];
  strategies: Array<{ name: string; runs: number; approved: number; rejected: number; pnl: number }>;
  indicators: {
    source: string | null;
    run_id?: string;
    values: Record<string, number | string | boolean | null>;
  };
};

export function StockPage() {
  const { symbol: rawSymbol } = useParams<{ symbol?: string }>();
  const symbol = (rawSymbol ?? "").trim().toUpperCase();
  const nav = useNavigate();

  if (!symbol) {
    return (
      <div className="px-6 py-8 max-w-[720px] mx-auto">
        <div className="mb-4">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Per-stock desk</p>
          <h1 className="text-h1 text-fg flex items-center gap-2">
            <Search className="h-6 w-6 text-accent" aria-hidden /> Stock view
          </h1>
          <p className="text-body-sm text-fg-muted mt-1">
            Type a symbol — equity (HDFCBANK, ITC, …) or an index underlying (NIFTY, BANKNIFTY).
          </p>
        </div>
        <SymbolSearch onPick={(s) => nav(`/stock/${s}`)} />
      </div>
    );
  }

  return <StockSymbolView symbol={symbol} onChangeSymbol={(s) => nav(`/stock/${s}`)} />;
}

/* ─────────────────────────────────────────────────────────────────── */

function StockSymbolView({ symbol, onChangeSymbol }: { symbol: string; onChangeSymbol: (s: string) => void }) {
  // ─── ALL HOOKS UP FRONT — Rules of Hooks: order must be stable across renders ───
  const [period, setPeriod] = React.useState<Period>("monthly");
  // Bumped on each orchestrator run so the Plan tab re-reads localStorage.
  const [planTick, setPlanTick] = React.useState(0);
  const storedPlan = React.useMemo(() => getStoredPlan(symbol), [symbol, planTick]);

  const summaryQ = useQuery({
    queryKey: ["stock-summary", symbol, period],
    queryFn: () =>
      legacyApi
        .get<StockSummary>(`/legacy/stock-summary/?symbol=${symbol}&period=${period}`)
        .then((r) => r.data),
    refetchInterval: 30_000,
  });

  const { data: recentRuns = [] } = useQuery({
    queryKey: ["agent-runs", "recent-for-stock", symbol],
    queryFn: () => api.get<any[]>("/agents/runs/?limit=50").then((r) => r.data),
    refetchInterval: 60_000,
  });

  const open_positions = summaryQ.data?.open_positions ?? [];
  const positionRuns = React.useMemo(() => {
    const positionIds = new Set(open_positions.filter((p) => p.kind === "options").map((p) => p.id));
    return recentRuns.filter((r: any) =>
      (r.strategy_name === "directional" && ((r.result?.plan?.symbol ?? r.config?.symbol) ?? "").toUpperCase() === symbol)
      || (r.strategy_name === "short_straddle" && positionIds.has(r.config?.position_id))
    );
  }, [recentRuns, open_positions, symbol]);

  // ─── Now safe to early-return ───
  if (summaryQ.isLoading) {
    return (
      <div className="px-6 py-6 space-y-4 max-w-[1440px] mx-auto">
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }
  if (summaryQ.isError || !summaryQ.data) {
    return (
      <div className="px-6 py-6 max-w-[1440px] mx-auto">
        <EmptyState icon={<Sparkles />} title={`Couldn't load ${symbol}`} description="Backend returned an error. Check logs/v2.log." />
      </div>
    );
  }

  const s = summaryQ.data;
  const { kpis, buckets, rollups, strategies, indicators } = s;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      {/* ── Header ── */}
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Per-stock desk</p>
          <h1 className="text-h1 text-fg flex items-center gap-3 flex-wrap">
            <span className="font-mono">{symbol}</span>
            <Badge tone="info">{s.kind === "index_underlying" ? "index underlying" : "equity"}</Badge>
          </h1>
          <p className="text-body-sm text-fg-muted mt-1">
            {summarisePeriod(period, kpis.period_window)} · {open_positions.length} open positions · live leverage {kpis.live_leverage.toFixed(2)}×
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            <PlanStockButton symbol={symbol} onPlanned={() => setPlanTick((t) => t + 1)} />
            <PeriodPicker value={period} onChange={setPeriod} />
          </div>
          <SymbolSearch onPick={onChangeSymbol} initial={symbol} compact />
        </div>
      </header>

      {/* ── KPI strip ── */}
      <section className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <KPI label="Capital deployed" value={fmtINR(kpis.capital_deployed)} hint="notional across open positions" icon={<Wallet className="h-4 w-4" />} />
        <KPI label="Money in play" value={fmtINR(kpis.money_in_play)} hint="margin actually locked" icon={<ShieldCheck className="h-4 w-4" />} tone={kpis.money_in_play > 0 ? "info" : "neutral"} />
        <KPI label="Open positions" value={String(kpis.open_count)} hint={strategies.map((x) => `${x.name} ${x.runs}`).join(" · ")} icon={<Briefcase className="h-4 w-4" />} />
        <KPI label="Period P&L" value={fmtINR(kpis.period_pnl)} hint={`${period} window`} icon={<TrendingUp className="h-4 w-4" />} tone={kpis.period_pnl >= 0 ? "success" : "danger"} />
        <KPI label="Live leverage" value={`${kpis.live_leverage.toFixed(2)}×`} hint="notional / margin" icon={<LineChartIcon className="h-4 w-4" />} tone={kpis.live_leverage > 5 ? "warning" : "neutral"} />
      </section>

      {/* ── Capital split + open positions ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Capital split</CardTitle>
            <CardDescription>By bucket · notional + margin per slice.</CardDescription>
          </CardHeader>
          <CardContent>
            <BucketSplit buckets={buckets} totals={s.bucket_totals} />
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Open positions</CardTitle>
            <CardDescription>What's running right now · expected exit comes from the strategy's target.</CardDescription>
          </CardHeader>
          <CardContent>
            {open_positions.length === 0 ? (
              <p className="text-body-sm text-fg-subtle">No open positions on {symbol}.</p>
            ) : (
              <OpenPositionsTable positions={open_positions} />
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Period rollups + strategies + indicators ── */}
      <Tabs defaultValue={storedPlan ? "plan" : "rollups"}>
        <TabsList>
          <TabsTrigger value="plan">Plan report{storedPlan ? "" : " (none)"}</TabsTrigger>
          <TabsTrigger value="rollups">History · {period}</TabsTrigger>
          <TabsTrigger value="strategies">Strategies · {strategies.length}</TabsTrigger>
          <TabsTrigger value="indicators">Indicators</TabsTrigger>
          <TabsTrigger value="runs">Recent runs · {positionRuns.length}</TabsTrigger>
        </TabsList>

        <TabsContent value="plan" className="pt-4">
          {storedPlan ? (
            <PlanReport
              plan={storedPlan.plan}
              savedAt={storedPlan.savedAt}
              onReplan={() => setPlanTick((t) => t + 1)}
            />
          ) : (
            <EmptyState
              icon={<Sparkles />}
              title="No saved plan for this stock"
              description={`Click "Plan stock" above to run the orchestrator. The result is saved here and survives page reloads.`}
            />
          )}
        </TabsContent>

        <TabsContent value="rollups" className="pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Activity by {period} period</CardTitle>
              <CardDescription>Trades planned (every agent run) vs taken (actual fills) · capital deployed · P&L · win rate.</CardDescription>
            </CardHeader>
            <CardContent>
              <RollupsTable rollups={rollups} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="strategies" className="pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Strategies used on {symbol}</CardTitle>
              <CardDescription>How each strategy has interacted with this stock.</CardDescription>
            </CardHeader>
            <CardContent>
              {strategies.length === 0 ? (
                <p className="text-body-sm text-fg-subtle">No agent runs for this stock yet.</p>
              ) : (
                <div className="space-y-4">
                  {strategies.map((st) => <StrategyCard key={st.name} stat={st} />)}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="indicators" className="pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Indicators · latest values</CardTitle>
              <CardDescription>
                {indicators.source
                  ? <>From the most recent <code>{indicators.source}</code> run{indicators.run_id ? <> · <Link to={`/agents/${indicators.run_id}`} className="text-accent hover:underline">open run</Link></> : null}.</>
                  : <>No run with structured indicators on this stock yet.</>}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <IndicatorsPanel values={indicators.values} source={indicators.source} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="runs" className="pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent agent runs</CardTitle>
              <CardDescription>Newest first. Click to open in the agent console.</CardDescription>
            </CardHeader>
            <CardContent>
              {positionRuns.length === 0 ? (
                <p className="text-body-sm text-fg-subtle">No recent runs.</p>
              ) : (
                <ul className="divide-y divide-border">
                  {positionRuns.slice(0, 20).map((r: any) => (
                    <li key={r.id} className="py-2 flex items-center gap-3">
                      <Badge tone="info">{r.strategy_name}</Badge>
                      <div className="flex-1 min-w-0">
                        <div className="text-body-sm text-fg truncate">
                          {summariseRun(r)}
                        </div>
                        <div className="text-caption text-fg-subtle font-mono">{fmtRel(r.created_at)}</div>
                      </div>
                      <Link to={`/agents/${r.id}`} className="text-accent text-caption font-mono hover:underline shrink-0">
                        open <ChevronRight className="inline h-3 w-3" />
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────── */
/* Sub-components                                                      */
/* ─────────────────────────────────────────────────────────────────── */

function PeriodPicker({ value, onChange }: { value: Period; onChange: (p: Period) => void }) {
  const opts: { key: Period; label: string }[] = [
    { key: "weekly", label: "Weekly" },
    { key: "monthly", label: "Monthly" },
    { key: "half-yearly", label: "Half-yearly" },
    { key: "yearly", label: "Yearly" },
  ];
  return (
    <div className="inline-flex rounded-sm border border-border bg-surface overflow-hidden">
      {opts.map((o) => (
        <button
          key={o.key}
          type="button"
          onClick={() => onChange(o.key)}
          className={cn(
            "px-3 py-1.5 text-body-sm font-mono",
            value === o.key ? "bg-accent/15 text-accent" : "text-fg-muted hover:bg-surface-2",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function KPI({ label, value, hint, tone = "neutral", icon }: {
  label: string; value: string; hint?: string;
  tone?: "neutral" | "success" | "warning" | "danger" | "info"; icon?: React.ReactNode;
}) {
  const color =
    tone === "success" ? "text-pnl-up"
    : tone === "warning" ? "text-warning"
    : tone === "danger" ? "text-pnl-down"
    : tone === "info" ? "text-info" : "text-fg";
  return (
    <div className="rounded-md border border-border bg-surface p-3">
      <div className="text-caption uppercase tracking-wider text-fg-subtle flex items-center gap-1.5">
        {icon}{label}
      </div>
      <div className={cn("text-h3 font-mono mt-0.5", color)}>{value}</div>
      {hint && <div className="text-caption text-fg-subtle mt-0.5 truncate">{hint}</div>}
    </div>
  );
}

function BucketSplit({ buckets, totals }: { buckets: Bucket[]; totals: StockSummary["bucket_totals"] }) {
  if (buckets.length === 0) {
    return <p className="text-body-sm text-fg-subtle">No deployed capital on this stock.</p>;
  }
  const totalNotional = totals.notional || 1;
  // Sorted by notional desc so the biggest slice is at the top.
  const sorted = [...buckets].sort((a, b) => b.notional - a.notional);

  return (
    <div className="space-y-3">
      {sorted.map((b) => {
        const pct = (b.notional / totalNotional) * 100;
        return (
          <div key={b.key}>
            <div className="flex items-center justify-between text-body-sm">
              <span className="font-mono text-fg">{prettyBucket(b.key)}</span>
              <span className="text-fg-subtle font-mono text-caption">{pct.toFixed(1)}%</span>
            </div>
            <div className="h-1.5 bg-surface-2 rounded-full overflow-hidden mt-1">
              <div className={cn("h-full", bucketColor(b.key))} style={{ width: `${pct}%` }} />
            </div>
            <div className="grid grid-cols-3 gap-2 mt-1 text-caption text-fg-subtle font-mono">
              <div>{b.count} pos</div>
              <div>margin {fmtINR(b.total_margin)}</div>
              <div>{b.leverage.toFixed(2)}× lev</div>
            </div>
          </div>
        );
      })}
      <div className="pt-3 mt-2 border-t border-border grid grid-cols-2 gap-2 text-body-sm">
        <KvRow label="Total notional" value={fmtINR(totals.notional)} />
        <KvRow label="Total margin" value={fmtINR(totals.total_margin)} />
        <KvRow label="Premium received" value={fmtINR(totals.premium_received)} tone={totals.premium_received > 0 ? "success" : "neutral"} />
        <KvRow label="Total leverage" value={`${totals.leverage.toFixed(2)}×`} tone={totals.leverage > 5 ? "warning" : "neutral"} />
      </div>
    </div>
  );
}

function OpenPositionsTable({ positions }: { positions: OpenPosition[] }) {
  return (
    <div className="overflow-x-auto -mx-4 sm:mx-0">
      <table className="w-full text-body-sm">
        <thead>
          <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
            <th className="text-left py-2 px-3">Leg</th>
            <th className="text-right py-2 px-3">Qty</th>
            <th className="text-right py-2 px-3">Entry</th>
            <th className="text-right py-2 px-3">Current</th>
            <th className="text-right py-2 px-3">Expected exit</th>
            <th className="text-right py-2 px-3">P&L</th>
            <th className="text-right py-2 px-3">Margin</th>
            <th className="text-right py-2 px-3">Leverage</th>
            <th className="text-left py-2 px-3">Strategy</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => {
            const targetPct = p.entry_price ? ((p.expected_exit - p.entry_price) / p.entry_price) * 100 : 0;
            const pnlTone = p.pnl_inr >= 0 ? "text-pnl-up" : "text-pnl-down";
            return (
              <tr key={`${p.kind}-${p.id}`} className="border-b border-border last:border-b-0 hover:bg-surface-2">
                <td className="py-2 px-3 font-mono">
                  <div className="text-fg">{p.leg}</div>
                  <div className="text-caption text-fg-subtle">{p.side} · {p.status}{p.expiry ? ` · exp ${p.expiry}` : ""}</div>
                </td>
                <td className="text-right py-2 px-3 font-mono">{p.quantity}</td>
                <td className="text-right py-2 px-3 font-mono">{fmtNum(p.entry_price)}</td>
                <td className="text-right py-2 px-3 font-mono">{fmtNum(p.current_price)}</td>
                <td className="text-right py-2 px-3 font-mono text-pnl-up">
                  {fmtNum(p.expected_exit)}
                  <div className="text-caption text-fg-subtle">{targetPct >= 0 ? "+" : ""}{targetPct.toFixed(1)}%</div>
                </td>
                <td className={cn("text-right py-2 px-3 font-mono", pnlTone)}>{fmtINR(p.pnl_inr)}</td>
                <td className="text-right py-2 px-3 font-mono">{fmtINR(p.margin)}</td>
                <td className={cn("text-right py-2 px-3 font-mono", p.leverage > 5 ? "text-warning" : "text-fg-muted")}>
                  {p.leverage.toFixed(2)}×
                </td>
                <td className="py-2 px-3">
                  <Badge tone={p.strategy === "directional" ? "info" : "brand"}>{p.strategy}</Badge>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RollupsTable({ rollups }: { rollups: Rollup[] }) {
  if (rollups.length === 0) return <p className="text-body-sm text-fg-subtle">No history yet.</p>;
  return (
    <div className="overflow-x-auto -mx-4 sm:mx-0">
      <table className="w-full text-body-sm">
        <thead>
          <tr className="text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
            <th className="text-left py-2 px-3">Period</th>
            <th className="text-right py-2 px-3">Planned</th>
            <th className="text-right py-2 px-3">Taken</th>
            <th className="text-right py-2 px-3">Opened</th>
            <th className="text-right py-2 px-3">Closed</th>
            <th className="text-right py-2 px-3">Capital</th>
            <th className="text-right py-2 px-3">P&L</th>
            <th className="text-right py-2 px-3">Win rate</th>
          </tr>
        </thead>
        <tbody>
          {rollups.map((r) => (
            <tr key={r.period_label} className="border-b border-border last:border-b-0 hover:bg-surface-2">
              <td className="py-2 px-3 font-mono text-fg">{r.period_label}</td>
              <td className="text-right py-2 px-3 font-mono">{r.trades_planned}</td>
              <td className="text-right py-2 px-3 font-mono">{r.trades_taken}</td>
              <td className="text-right py-2 px-3 font-mono">{r.positions_opened}</td>
              <td className="text-right py-2 px-3 font-mono">{r.positions_closed}</td>
              <td className="text-right py-2 px-3 font-mono">{fmtINR(r.capital_deployed)}</td>
              <td className={cn("text-right py-2 px-3 font-mono", r.pnl >= 0 ? "text-pnl-up" : "text-pnl-down")}>{fmtINR(r.pnl)}</td>
              <td className="text-right py-2 px-3 font-mono">{r.win_rate.toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StrategyCard({ stat }: { stat: { name: string; runs: number; approved: number; rejected: number; pnl: number } }) {
  const inds = indicatorsByStrategy(stat.name);
  return (
    <div className="rounded-md border border-border p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {stat.name === "directional" ? <Bot className="h-4 w-4 text-accent" /> : <Triangle className="h-4 w-4 text-accent" />}
          <span className="text-body-sm font-medium font-mono">{stat.name}</span>
          <Badge tone="neutral">{stat.runs} runs</Badge>
        </div>
        <div className={cn("text-body-sm font-mono", stat.pnl >= 0 ? "text-pnl-up" : "text-pnl-down")}>{fmtINR(stat.pnl)}</div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-caption text-fg-subtle font-mono mb-2">
        <div>approved {stat.approved}</div>
        <div>rejected {stat.rejected}</div>
        <div>{stat.runs ? Math.round((stat.approved / stat.runs) * 100) : 0}% approval</div>
      </div>
      <div>
        <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Indicators it uses</div>
        <div className="flex flex-wrap gap-1.5">
          {inds.map((i) => (
            <IndicatorChip key={i.key} ind={i} />
          ))}
          {inds.length === 0 && <span className="text-caption text-fg-subtle">No indicators registered.</span>}
        </div>
      </div>
    </div>
  );
}

function IndicatorChip({ ind, value }: { ind: IndicatorDef; value?: number | string | boolean | null }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        className="px-2 py-0.5 rounded-sm border border-border bg-surface-2 text-caption font-mono hover:border-accent/40 inline-flex items-center gap-1.5"
      >
        {ind.label}
        {value !== undefined && value !== null && <span className="text-fg">{String(value)}</span>}
        <Info className="h-3 w-3 text-fg-subtle" />
      </button>
      {open && (
        <div className="absolute z-popover top-full mt-1 left-0 w-80 rounded-md border border-border bg-surface shadow-lg p-3 text-body-sm">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-fg">{ind.label}</span>
            <Badge tone="neutral">{ind.kind}</Badge>
            {ind.unit && <Badge tone="neutral">{ind.unit}</Badge>}
          </div>
          <p className="text-fg-muted">{ind.oneLiner}</p>
          {ind.formula && (
            <div className="mt-2">
              <div className="text-caption uppercase tracking-wider text-fg-subtle">Formula</div>
              <code className="text-caption font-mono">{ind.formula}</code>
            </div>
          )}
          <div className="mt-2">
            <div className="text-caption uppercase tracking-wider text-fg-subtle">How {ind.strategies.join(", ")} uses it</div>
            <p className="text-fg-muted text-caption">{ind.usage}</p>
          </div>
          {ind.goodRange && (
            <p className="text-caption text-fg-subtle mt-2">Range: {ind.goodRange}</p>
          )}
        </div>
      )}
    </div>
  );
}

function IndicatorsPanel({ values, source }: { values: Record<string, any>; source: string | null }) {
  // Show every indicator we have a value for, plus all the catalog
  // indicators for the source strategy (with placeholder if value missing).
  const sourceInds = source ? indicatorsByStrategy(source) : INDICATORS;
  const seen = new Set<string>();
  const rows: Array<{ def: IndicatorDef; value: any }> = [];
  for (const i of sourceInds) {
    seen.add(i.key);
    rows.push({ def: i, value: values[i.key] });
  }
  // Surface any extra keys we don't have a catalog entry for, as info.
  for (const k of Object.keys(values)) {
    if (seen.has(k)) continue;
    rows.push({
      def: { key: k, label: k, kind: "structure", oneLiner: "Uncatalogued indicator — add to lib/indicators.ts.", usage: "—", strategies: [source ?? ""] },
      value: values[k],
    });
  }

  if (rows.length === 0) return <p className="text-body-sm text-fg-subtle">No indicators emitted.</p>;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
      {rows.map(({ def, value }) => (
        <div key={def.key} className="rounded-sm border border-border bg-surface-2/40 p-2.5">
          <div className="flex items-center justify-between mb-0.5">
            <div className="text-caption uppercase tracking-wider text-fg-subtle">{def.label}</div>
            {def.unit && <Badge tone="neutral">{def.unit}</Badge>}
          </div>
          <div className="text-body-sm font-mono text-fg">
            {value === undefined || value === null ? <span className="text-fg-subtle">—</span> : String(value)}
          </div>
          <p className="text-caption text-fg-subtle mt-1 line-clamp-2">{def.oneLiner}</p>
          <details className="mt-1">
            <summary className="text-caption text-accent cursor-pointer">how it's used</summary>
            <p className="text-caption text-fg-muted mt-1">{def.usage}</p>
            {def.formula && <p className="text-caption text-fg-subtle font-mono mt-1">{def.formula}</p>}
          </details>
        </div>
      ))}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────── */
/* Symbol search                                                       */
/* ─────────────────────────────────────────────────────────────────── */
function SymbolSearch({ onPick, initial = "", compact = false }: { onPick: (s: string) => void; initial?: string; compact?: boolean }) {
  const [value, setValue] = React.useState(initial);
  React.useEffect(() => setValue(initial), [initial]);
  const quickPicks = ["NIFTY", "BANKNIFTY", "SENSEX", "HDFCBANK", "RELIANCE", "ITC", "TCS", "INFY"];

  return (
    <div className={cn("space-y-2", compact && "min-w-[280px]")}>
      <form
        onSubmit={(e) => { e.preventDefault(); const s = value.trim().toUpperCase(); if (s) onPick(s); }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="HDFCBANK, NIFTY, RELIANCE, …"
          className="flex-1 bg-surface border border-border rounded-sm px-3 py-2 text-body-sm text-fg font-mono"
          autoFocus={!compact}
        />
        <button
          type="submit"
          className="px-3 py-2 rounded-sm border border-accent/40 bg-accent/10 text-body-sm text-accent hover:bg-accent/20"
        >
          View
        </button>
      </form>
      {!compact && (
        <div className="flex flex-wrap gap-1.5">
          {quickPicks.map((s) => (
            <button
              key={s}
              onClick={() => onPick(s)}
              className="px-2 py-1 rounded-sm border border-border bg-surface-2 text-caption text-fg-muted hover:text-fg hover:border-accent/40 font-mono"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────── */
/* Helpers                                                             */
/* ─────────────────────────────────────────────────────────────────── */
function KvRow({ label, value, tone = "neutral" }: { label: string; value: React.ReactNode; tone?: "neutral" | "success" | "warning" | "danger" }) {
  const color = tone === "success" ? "text-pnl-up" : tone === "warning" ? "text-warning" : tone === "danger" ? "text-pnl-down" : "text-fg";
  return (
    <div className="flex items-center justify-between py-1 border-b border-border last:border-b-0">
      <div className="text-caption text-fg-subtle">{label}</div>
      <div className={cn("text-body-sm font-mono", color)}>{value ?? "—"}</div>
    </div>
  );
}

function summarisePeriod(p: Period, window: [string, string]) {
  return `${p} window · ${window[0]} → ${window[1]}`;
}

function summariseRun(r: any) {
  if (r.strategy_name === "directional") {
    const plan = r.result?.plan ?? {};
    return plan.symbol ? `${plan.side ?? "?"} ${plan.symbol} ×${plan.quantity ?? 0} @ ${plan.entry_price ?? "?"} (conf ${plan.confidence?.toFixed?.(2) ?? "—"})` : "Directional plan";
  }
  if (r.strategy_name === "short_straddle") {
    const act = r.result?.action ?? {};
    return `${act.action ?? "—"} · conf ${act.confidence?.toFixed?.(2) ?? "—"}`;
  }
  return r.strategy_name;
}

function prettyBucket(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function bucketColor(key: string): string {
  if (key.startsWith("equity")) return "bg-info";
  if (key.includes("straddle")) return "bg-accent";
  if (key.startsWith("options")) return "bg-brand";
  if (key === "futures") return "bg-warning";
  return "bg-fg-subtle";
}

function fmtNum(n?: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function fmtINR(n?: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const v = Math.round(n);
  return `${v < 0 ? "-" : ""}₹${Math.abs(v).toLocaleString("en-IN")}`;
}
