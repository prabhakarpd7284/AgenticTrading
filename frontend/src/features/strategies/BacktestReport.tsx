/**
 * Backtest report: provenance badge first (where the numbers came from and
 * how much to trust them), then headline metrics, in-sample vs out-of-sample,
 * equity curve and the trade list. Every number is from the deterministic
 * engine; nothing here is computed by an LLM.
 */
import * as React from "react";
import { AlertTriangle, Database, Info, Loader2, ShieldAlert, XCircle } from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip as ReTooltip, XAxis, YAxis,
} from "recharts";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn, clsPnl, fmtInr, fmtNum } from "@/lib/utils";
import {
  TERMINAL_BT, useBacktest, useBacktestTrades, useCancelBacktest,
  type Backtest, type BacktestSummary, type BacktestTrade, type Metrics, type Provenance,
} from "./useStrategies";

const r2 = (n: number | null | undefined, d = 2) => (n === null || n === undefined ? "—" : fmtNum(n, d));

/** Compact provenance badge used in lists and at the top of a report. */
export function ProvenanceBadge({ p, className }: { p: Partial<Provenance> | null | undefined; className?: string }) {
  if (!p || !p.timeframe) return <Badge tone="neutral" className={className}>no result</Badge>;
  const tone = p.synthetic ? "danger" : p.sample?.too_few ? "warning" : "success";
  const n = p.sample?.trades ?? 0;
  const txt = p.synthetic
    ? "synthetic data"
    : `real · ${p.timeframe} · ${p.sessions ?? "?"} sess · n=${n}${p.sample?.too_few ? " (too few)" : ""}`;
  const title = [p.data_source, p.label, p.research_oos_overlap ? "overlaps the research OOS window" : ""]
    .filter(Boolean).join(" · ");
  return <Badge tone={tone} className={className} title={title} data-testid="provenance-badge">{txt}</Badge>;
}

export function BacktestReportById({ id, onCancelled }: { id: string; onCancelled?: () => void }) {
  const q = useBacktest(id);
  if (q.isLoading) return <Skeleton className="h-64 w-full" />;
  if (q.isError || !q.data) return <p className="text-body-sm text-pnl-down">Could not load this backtest.</p>;
  return <BacktestReport bt={q.data} onCancelled={onCancelled} />;
}

export function BacktestReport({ bt, onCancelled }: { bt: Backtest; onCancelled?: () => void }) {
  const cancel = useCancelBacktest();
  const done = bt.status === "done";
  const trades = useBacktestTrades(bt.id, done);

  if (!TERMINAL_BT.includes(bt.status)) {
    return (
      <Card data-testid="backtest-progress">
        <CardContent className="py-6 space-y-3">
          <div className="flex items-center gap-2 text-body-sm text-fg">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            Backtest {bt.status} · {bt.progress}% {bt.progress_note && `· ${bt.progress_note}`}
          </div>
          <div className="h-1.5 w-full rounded-full bg-surface-2 overflow-hidden" role="progressbar"
               aria-valuenow={bt.progress} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-accent transition-all" style={{ width: `${Math.max(2, bt.progress)}%` }} />
          </div>
          <Button size="sm" variant="ghost" loading={cancel.isPending} disabled={bt.cancel_requested}
                  onClick={() => cancel.mutate(bt.id, { onSuccess: () => onCancelled?.() })}>
            {bt.cancel_requested ? "Cancelling…" : "Cancel"}
          </Button>
        </CardContent>
      </Card>
    );
  }
  if (bt.status === "failed" || bt.status === "cancelled") {
    return (
      <Card data-testid="backtest-failed">
        <CardContent className="py-5 flex gap-2 text-body-sm">
          <XCircle className="h-4 w-4 text-pnl-down shrink-0 mt-0.5" aria-hidden />
          <div>
            <p className="text-fg font-medium">Backtest {bt.status}</p>
            <p className="text-fg-muted">{bt.error || "No error recorded."}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const m = bt.metrics as Metrics;
  const p = bt.provenance as Provenance;
  return (
    <div className="space-y-4" data-testid="backtest-report">
      <ProvenancePanel p={p} bt={bt} />
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <Stat label="Trades" value={String(m.trades)} hint={m.too_few_trades ? `under ${m.min_trades ?? 30} — anecdotal` : undefined}
              warn={m.too_few_trades} />
        <Stat label="Win rate" value={m.win_rate === null ? "—" : `${r2(m.win_rate, 1)}%`} />
        <Stat label="Avg R (net)" value={r2(m.avg_r, 3)} cls={clsPnl(m.avg_r ?? 0)} hint="expectancy per trade, in R" />
        <Stat label="Profit factor" value={r2(m.profit_factor)} />
        <Stat label="Net after costs" value={fmtInr(m.net_pnl)} cls={clsPnl(m.net_pnl)}
              hint={`gross ${fmtInr(m.gross_pnl ?? 0)} · costs ${fmtInr(m.costs)}`} />
        <Stat label="Max drawdown" value={fmtInr(m.max_drawdown)} hint={m.max_drawdown_pct !== undefined ? `${r2(m.max_drawdown_pct)}% of capital` : undefined} />
      </div>
      <SplitTable bt={bt} />
      <EquityChart bt={bt} />
      <Card>
        <CardHeader>
          <CardTitle>Trades ({bt.trade_count})</CardTitle>
          <CardDescription>Net of slippage and costs. Entries fill at the open of the bar after the signal.</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <TradesTable rows={trades.data ?? []} loading={trades.isLoading} />
        </CardContent>
      </Card>
    </div>
  );
}

function Stat({ label, value, hint, cls, warn }: { label: string; value: string; hint?: string; cls?: string; warn?: boolean }) {
  return (
    <div className={cn("rounded-md border bg-surface px-3 py-2.5", warn ? "border-warn/50" : "border-border")}>
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn("font-mono tabular text-h3 mt-0.5 text-fg", cls)}>{value}</div>
      {hint && <div className={cn("text-caption mt-0.5", warn ? "text-warn" : "text-fg-subtle")}>{hint}</div>}
    </div>
  );
}

function ProvenancePanel({ p, bt }: { p: Provenance; bt: Backtest }) {
  return (
    <Card className={cn(p.synthetic && "border-pnl-down")}>
      <CardContent className="py-3 space-y-2 text-body-sm">
        {p.synthetic && (
          <div className="flex gap-2 rounded-sm bg-pnl-down/10 px-3 py-2 text-pnl-down" role="alert">
            <ShieldAlert className="h-4 w-4 shrink-0 mt-0.5" aria-hidden />
            Synthetic data — these numbers mean nothing.
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <ProvenanceBadge p={p} />
          <Badge tone="info">{p.label}</Badge>
          {p.definition?.version && <Badge tone="neutral">v{p.definition.version}</Badge>}
          {p.research_oos_overlap && <Badge tone="warning" title={p.research_note}>overlaps research OOS</Badge>}
          <span className="text-caption text-fg-subtle font-mono">{p.definition?.hash?.slice(0, 10)}</span>
        </div>
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1 text-caption">
          <Row k="Data" v={<span className="inline-flex items-center gap-1"><Database className="h-3 w-3" aria-hidden />{p.data_source}</span>} />
          <Row k="Period" v={`${p.requested.from} → ${p.requested.to} (data ${p.actual.first ?? "—"} → ${p.actual.last ?? "—"}, ${p.sessions} sessions)`} />
          <Row k="Universe" v={`${p.symbols_loaded.length} loaded${p.symbols_missing.length ? `, missing: ${p.symbols_missing.join(", ")}` : ""} · ${fmtNum(p.bars)} bars · ${p.gap_sessions} gap session(s)`} />
          <Row k="Split" v={`${p.split.method}${p.split.oos ? ` — IS ${p.split.is[0]}→${p.split.is[1]}, OOS ${p.split.oos[0]}→${p.split.oos[1]}` : ""}`} />
          <Row k="Costs" v={`${p.costs.slippage_pct_per_side}% slippage/side · ${p.costs.description} (as of ${p.costs.as_of})`} />
          <Row k="Signals" v={`${p.signals.detected} detected · ${p.signals.taken_to_fill} filled${Object.keys(p.signals.rejected).length ? ` · skipped: ${Object.entries(p.signals.rejected).map(([k, n]) => `${k.replace(/_/g, " ")} ${n}`).join(", ")}` : ""}${Object.keys(p.signals.dropped ?? {}).length ? ` · dropped: ${Object.entries(p.signals.dropped).map(([k, n]) => `${k.replace(/_/g, " ")} ${n}`).join(", ")}` : ""}`} />
          <Row k="Engine" v={p.engine} />
          <Row k="Capital" v={fmtInr(bt.config?.capital ?? 0)} />
        </dl>
        {p.warnings.length > 0 && (
          <ul className="space-y-1">
            {p.warnings.map((w) => (
              <li key={w} className="flex gap-1.5 text-caption text-warn">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-px" aria-hidden />{w}
              </li>
            ))}
          </ul>
        )}
        {p.notes.length > 0 && (
          <ul className="space-y-1">
            {p.notes.map((n) => (
              <li key={n} className="flex gap-1.5 text-caption text-fg-muted">
                <Info className="h-3.5 w-3.5 shrink-0 mt-px" aria-hidden />{n}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex gap-2 min-w-0">
      <dt className="text-fg-subtle w-16 shrink-0">{k}</dt>
      <dd className="text-fg-muted min-w-0 break-words">{v}</dd>
    </div>
  );
}

function SplitTable({ bt }: { bt: Backtest }) {
  const s = bt.split_metrics;
  const rows: [string, Metrics | undefined][] = [["All", bt.metrics as Metrics], ["In-sample", s.is], ["Out-of-sample", s.oos]];
  return (
    <Card>
      <CardHeader>
        <CardTitle>In-sample vs out-of-sample</CardTitle>
        <CardDescription>
          The last {Math.round((s.oos_fraction ?? 0.3) * 100)}% of sessions are held out
          {s.oos_start ? ` (from ${s.oos_start})` : ""}. A result that only holds in-sample is probably fitted noise.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0 overflow-x-auto">
        <table className="w-full text-body-sm" aria-label="In-sample vs out-of-sample">
          <thead className="bg-surface-2 text-caption uppercase tracking-wider text-fg-subtle">
            <tr>
              {["Segment", "Trades", "Win %", "Avg R", "PF", "Net ₹", "Costs ₹", "Max DD ₹"].map((h) => (
                <th key={h} className={cn("px-3 py-2", h === "Segment" ? "text-left" : "text-right")}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, m]) => (
              <tr key={name} className="border-t border-border/60">
                <td className="px-3 py-2 text-fg">{name}{m?.too_few_trades && <Badge tone="warning" className="ml-2">too few</Badge>}</td>
                <td className="px-3 py-2 text-right font-mono">{m?.trades ?? 0}</td>
                <td className="px-3 py-2 text-right font-mono">{m?.win_rate === null || m?.win_rate === undefined ? "—" : r2(m.win_rate, 1)}</td>
                <td className={cn("px-3 py-2 text-right font-mono", clsPnl(m?.avg_r ?? 0))}>{r2(m?.avg_r, 3)}</td>
                <td className="px-3 py-2 text-right font-mono">{r2(m?.profit_factor)}</td>
                <td className={cn("px-3 py-2 text-right font-mono", clsPnl(m?.net_pnl ?? 0))}>{fmtNum(m?.net_pnl ?? 0, 0)}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(m?.costs ?? 0, 0)}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(m?.max_drawdown ?? 0, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function EquityChart({ bt }: { bt: Backtest }) {
  const data = bt.equity_curve.map((p) => ({ ...p, label: p.t.slice(0, 16).replace("T", " ") }));
  if (data.length < 2) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Equity curve (net)</CardTitle>
        <CardDescription>Closed-trade equity from {fmtInr(bt.config?.capital ?? 0)}; the red band is drawdown.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-56" aria-label="Equity curve chart" role="img">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border, #333)" opacity={0.4} />
              <XAxis dataKey="label" hide />
              <YAxis yAxisId="eq" domain={["auto", "auto"]} width={70} tickFormatter={(v) => fmtNum(v, 0)} fontSize={11} />
              <YAxis yAxisId="dd" orientation="right" width={60} tickFormatter={(v) => fmtNum(v, 0)} fontSize={11} />
              <ReTooltip formatter={(v: number) => fmtInr(v)} />
              <Area yAxisId="dd" type="stepAfter" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} />
              <Area yAxisId="eq" type="monotone" dataKey="equity" stroke="#6366f1" fill="#6366f1" fillOpacity={0.1} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function TradesTable({ rows, loading }: { rows: BacktestTrade[]; loading: boolean }) {
  const [seg, setSeg] = React.useState<"all" | "is" | "oos">("all");
  const shown = seg === "all" ? rows : rows.filter((r) => r.segment === seg);
  const cols: Column<BacktestTrade>[] = [
    { key: "seq", header: "#", kind: "num" },
    { key: "symbol", header: "Symbol", render: (t) => <span className="font-medium">{t.symbol}</span> },
    { key: "side", header: "Side", render: (t) => <Badge tone={t.side === "BUY" ? "success" : "danger"}>{t.side}</Badge> },
    { key: "entry_time", header: "Entry", render: (t) => <span className="font-mono text-caption">{t.entry_time.slice(0, 16).replace("T", " ")}</span> },
    { key: "exit_time", header: "Exit", render: (t) => <span className="font-mono text-caption">{t.exit_time.slice(0, 16).replace("T", " ")}</span> },
    { key: "entry_price", header: "In", kind: "num" },
    { key: "exit_price", header: "Out", kind: "num" },
    { key: "qty", header: "Qty", kind: "num" },
    { key: "costs", header: "Costs", kind: "num", render: (t) => <span className="font-mono">{fmtNum(t.costs, 0)}</span> },
    { key: "net_pnl", header: "Net ₹", kind: "num", render: (t) => <span className={cn("font-mono", clsPnl(t.net_pnl))}>{fmtNum(t.net_pnl, 0)}</span> },
    { key: "r_multiple", header: "R", kind: "num", render: (t) => <span className={cn("font-mono", clsPnl(t.r_multiple))}>{fmtNum(t.r_multiple, 2)}</span> },
    { key: "exit_reason", header: "Exit reason" },
    { key: "segment", header: "Seg", render: (t) => <Badge tone={t.segment === "oos" ? "info" : "neutral"}>{t.segment}</Badge> },
  ];
  return (
    <div>
      <div className="flex gap-1 px-4 py-2" role="group" aria-label="Filter trades by segment">
        {(["all", "is", "oos"] as const).map((s) => (
          <Button key={s} size="sm" variant={seg === s ? "secondary" : "ghost"} onClick={() => setSeg(s)}>
            {s === "all" ? "All" : s === "is" ? "In-sample" : "Out-of-sample"}
          </Button>
        ))}
      </div>
      <DataTable<BacktestTrade> columns={cols} rows={shown.slice(0, 500)} loading={loading} dense rowKey={(t) => String(t.seq)}
                                emptyTitle="No trades" emptyDescription="The strategy never entered in this period." />
      {shown.length > 500 && <p className="px-4 py-2 text-caption text-fg-subtle">Showing the first 500 of {shown.length}.</p>}
    </div>
  );
}

/** One-line summary of a backtest, for lists. */
export function BacktestLine({ b }: { b: BacktestSummary }) {
  const m = b.metrics ?? {};
  if (b.status !== "done") {
    return <span className="text-caption text-fg-muted">{b.status}{b.status === "running" ? ` ${b.progress}%` : ""}</span>;
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-2 text-caption">
      <span className="font-mono">n={m.trades ?? 0}</span>
      <span className={cn("font-mono", clsPnl(m.avg_r ?? 0))}>avgR {r2(m.avg_r, 2)}</span>
      <span className={cn("font-mono", clsPnl(m.net_pnl ?? 0))}>{fmtInr(m.net_pnl ?? 0)}</span>
      <ProvenanceBadge p={b.provenance} />
    </span>
  );
}
