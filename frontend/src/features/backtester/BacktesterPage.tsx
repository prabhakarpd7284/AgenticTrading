/**
 * Backtester — unified backtest hub for all strategies.
 *
 * Tabs:
 *   Daily Swing  — OK cycle phases on daily candles, trailing SL, walk-forward
 *   Intraday     — 3m/5m/15m grid search (TF × SL ATR × R:R), RSI momentum
 *
 * Both use the unified BacktestEngine via GET /api/v1/market-data/ok-backtest/
 * Smart universe: scans NIFTY 100 for active OK phases, backtests only those.
 */
import * as React from "react";
import { Link } from "react-router-dom";
import {
  Calendar,
  ChevronDown,
  ChevronUp,
  Play,
  TrendingUp,
} from "lucide-react";
import {
  AreaChart, Area, ResponsiveContainer, Tooltip as ReTooltip, XAxis, YAxis, CartesianGrid,
} from "recharts";

import {
  useOKBacktest,
  phaseTone,
  type OKBacktestPayload,
  type OKBacktestTrade,
  type CyclePhase,
} from "@/lib/market-pulse";
import { cn, fmtNum, fmtInr } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { KPI } from "@/components/ui/KPI";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/Tabs";
import {
  listHistory, recordRun, deleteHistoryEntry, clearHistory,
  type BacktestHistoryEntry,
} from "./history";
import { Clock, History, Trash2, X } from "lucide-react";

export function BacktesterPage() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">
            Backtester
          </p>
          <h1 className="text-h1 text-fg">Test before you trust</h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            Replay OK cycle strategies against historical candles. Same risk gates,
            same exit management, no live capital.
          </p>
        </div>
        <Link to="/swing-scanner"
              className="inline-flex items-center gap-1.5 text-body-sm text-fg-muted hover:text-fg">
          <TrendingUp className="h-4 w-4" /> Live Scanner
        </Link>
      </header>

      <Tabs defaultValue="daily">
        <TabsList>
          <TabsTrigger value="daily">Daily Swing</TabsTrigger>
          <TabsTrigger value="intraday">Intraday Multi-TF</TabsTrigger>
          <TabsTrigger value="basket">Morning Basket</TabsTrigger>
          <TabsTrigger value="history" className="gap-1.5">
            <History className="h-3.5 w-3.5" /> History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="daily">
          <BacktestTab mode="daily" />
        </TabsContent>
        <TabsContent value="intraday">
          <BacktestTab mode="intraday" />
        </TabsContent>
        <TabsContent value="basket">
          <BacktestTab mode="basket" />
        </TabsContent>
        <TabsContent value="history">
          <HistoryTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

/* ================================================================== */
/* Backtest Tab (shared by both modes)                                  */
/* ================================================================== */

function BacktestTab({ mode }: { mode: "daily" | "intraday" | "basket" }) {
  const today = new Date().toISOString().slice(0, 10);
  const thirtyDaysAgo = new Date(Date.now() - 30 * 86400_000).toISOString().slice(0, 10);

  const [from, setFrom] = React.useState(thirtyDaysAgo);
  const [to, setTo] = React.useState(today);
  const [run, setRun] = React.useState(false);

  const { data, isLoading, isError, error } = useOKBacktest({
    mode,
    from_date: from,
    to_date: to,
    enabled: run,
  });

  // Snapshot every completed run into localStorage so the History tab can
  // replay it without re-paying broker cost. recordRun() de-dups identical
  // mode+window pairs within 10s (React Query refetch quirk).
  React.useEffect(() => {
    if (data && !isLoading) recordRun(mode, data);
  }, [data, isLoading, mode]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>
            {mode === "daily" ? "Daily Swing Backtest"
             : mode === "basket" ? "Morning Basket Backtest"
             : "Intraday Multi-TF Backtest"}
          </CardTitle>
          <CardDescription>
            {mode === "daily"
              ? "OK cycle phases on daily candles. Trailing SL + breakeven. Walk-forward with ₹5L capital."
              : mode === "basket"
              ? "Equity + ATM options momentum play. 5 EMA + BB in the 9:45-10:30 window. Scale-in + trailing SL."
              : "Grid search: 3m/5m/15m × SL(1.0/1.5/2.0 ATR) × RR(1.5/2.0/2.5). Finds optimal config."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            onSubmit={(e) => { e.preventDefault(); setRun(true); }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-end"
          >
            <Input label="From" type="date" value={from}
                   onChange={(e) => { setFrom(e.target.value); setRun(false); }}
                   leading={<Calendar className="h-4 w-4" />} />
            <Input label="To" type="date" value={to}
                   onChange={(e) => { setTo(e.target.value); setRun(false); }}
                   leading={<Calendar className="h-4 w-4" />} />
            <Button type="submit" loading={isLoading}
                    leading={<Play className="h-4 w-4" />}>
              Run backtest
            </Button>
          </form>
        </CardContent>
      </Card>

      {isLoading && (
        <Card>
          <CardContent className="py-12 text-center">
            <Skeleton className="h-8 w-48 mx-auto mb-4" />
            <p className="text-body-sm text-fg-muted">
              {mode === "daily"
                ? "Scanning NIFTY 100 → filtering active phases → running backtest..."
                : mode === "basket"
                ? "Fetching 5m candles → scanning 9:45-10:30 window → testing basket entries..."
                : "Scanning universe → fetching 3m/5m/15m candles → testing 27 combos..."}
            </p>
          </CardContent>
        </Card>
      )}

      {isError && (
        <EmptyState
          title="Backtest failed"
          description={(error as Error)?.message ?? "Unknown error"}
          action={<Button onClick={() => setRun(true)}>Retry</Button>}
        />
      )}

      {data && !isLoading && <BacktestResults data={data} />}
    </div>
  );
}

/* ================================================================== */
/* Results display                                                      */
/* ================================================================== */

function BacktestResults({ data }: { data: OKBacktestPayload }) {
  // Default to OPEN — operators couldn't find the trade list before because
  // the chevron was easy to miss. Win-rate / PF means nothing without seeing
  // which stocks traded.
  const [showTrades, setShowTrades] = React.useState(true);
  const [tradeSort, setTradeSort] = React.useState<{ key: string; dir: "asc" | "desc" }>({ key: "pnl", dir: "desc" });

  if (data.errors.length > 0 && data.total_trades === 0) {
    return (
      <Card><CardContent>
        <EmptyState title="No results" description={data.errors.join("; ")} />
      </CardContent></Card>
    );
  }

  // Sort trades
  const sortedTrades = React.useMemo(() => {
    const trades = [...data.trades];
    const { key, dir } = tradeSort;
    trades.sort((a, b) => {
      const av = (a as Record<string, unknown>)[key];
      const bv = (b as Record<string, unknown>)[key];
      const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
      return dir === "asc" ? cmp : -cmp;
    });
    return trades;
  }, [data.trades, tradeSort]);

  const handleSort = (key: string) => {
    setTradeSort((prev) =>
      prev.key === key ? { key, dir: prev.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" }
    );
  };

  const tradeColumns: Column<OKBacktestTrade>[] = [
    { key: "symbol", header: "Symbol", sortable: true,
      render: (t) => <span className="font-mono text-body-sm">{t.symbol}</span> },
    { key: "phase", header: "Type", sortable: true,
      render: (t) => <Badge tone={phaseTone(t.phase as CyclePhase)}>{t.phase}</Badge> },
    { key: "side", header: "Side",
      render: (t) => <Badge tone={t.side === "BUY" ? "success" : "danger"}>{t.side}</Badge> },
    { key: "entry", header: "Entry", kind: "num", align: "right", sortable: true,
      render: (t) => <span className="font-mono tabular">{fmtNum(t.entry, 2)}</span> },
    { key: "exit", header: "Exit", kind: "num", align: "right",
      render: (t) => <span className="font-mono tabular">{fmtNum(t.exit, 2)}</span> },
    { key: "exit_reason", header: "Reason", sortable: true,
      render: (t) => <span className="text-body-sm text-fg-muted">{t.exit_reason}</span> },
    { key: "pnl", header: "P&L", kind: "num", align: "right", sortable: true,
      render: (t) => (
        <span className={cn("font-mono tabular", t.won ? "text-pnl-up" : "text-pnl-down")}>
          {fmtInr(t.pnl)}
        </span>
      )},
    { key: "rr", header: "R:R", kind: "num", align: "right", sortable: true,
      render: (t) => <span className={cn("font-mono tabular", t.rr >= 1 ? "text-pnl-up" : t.rr < 0 ? "text-pnl-down" : "text-fg")}>{t.rr.toFixed(1)}</span> },
    { key: "bars_held", header: "Bars", kind: "num", align: "right", sortable: true,
      render: (t) => <span className="font-mono tabular">{t.bars_held}</span> },
  ];

  // Summary verdict
  const verdict = data.profit_factor >= 1.5 ? "Strong edge" :
                  data.profit_factor >= 1.0 ? "Marginal" : "No edge";
  const verdictTone = data.profit_factor >= 1.5 ? "success" :
                      data.profit_factor >= 1.0 ? "warning" : "danger";

  return (
    <div className="space-y-6">
      {/* Smart universe banner */}
      {data.scanned_universe > 0 && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-3 flex-wrap text-body-sm">
              <Badge tone="brand">Smart Universe</Badge>
              <span className="text-fg-muted">
                Scanned <strong className="text-fg">{data.scanned_universe}</strong> stocks
                {" → "}
                <strong className="text-fg">{data.active_phases}</strong> with active OK phases
                {" → "}
                backtested <strong className="text-fg">{data.symbols_count}</strong>
              </span>
              {data.universe_symbols.length > 0 && data.universe_symbols.length <= 20 && (
                <span className="text-fg-subtle font-mono text-caption">
                  {data.universe_symbols.join(", ")}
                </span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Verdict + KPIs */}
      <div className="flex items-center gap-3 flex-wrap">
        <Badge tone={verdictTone as "success" | "warning" | "danger"} dot className="text-body-sm px-3 py-1">
          {verdict}
        </Badge>
        <span className="text-body-sm text-fg-muted">
          {data.total_trades} trades · {(data.win_rate * 100).toFixed(0)}% win · PF {data.profit_factor.toFixed(2)}
        </span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        <KPI label="Trades" value={data.total_trades}
             hint={`${data.winners}W / ${data.total_trades - data.winners}L`} />
        <KPI label="Win Rate" value={data.win_rate} valueFormat="pct"
             delta={data.win_rate - 0.5} deltaFormat="pct" />
        <KPI label="Profit Factor" value={data.profit_factor} valueFormat="num"
             hint={data.profit_factor >= 1.5 ? "strong" : data.profit_factor >= 1.0 ? "marginal" : "weak"}
             className={data.profit_factor >= 1.5 ? "border-pnl-up/30" : data.profit_factor < 1.0 ? "border-pnl-down/30" : ""} />
        <KPI label="Total P&L" value={data.total_pnl} valueFormat="inr"
             delta={data.total_pnl_pct / 100} deltaFormat="pct" />
        <KPI label="Max DD" value={data.max_drawdown} valueFormat="inr"
             hint={`${data.max_drawdown_pct.toFixed(1)}% of capital`}
             className={data.max_drawdown_pct > 5 ? "border-pnl-down/30" : ""} />
        <KPI label="Avg Win" value={data.avg_win} valueFormat="inr"
             hint={data.avg_win > 0 && data.avg_loss < 0 ? `${(data.avg_win / Math.abs(data.avg_loss)).toFixed(1)}x loss` : ""} />
        <KPI label="Avg Loss" value={data.avg_loss} valueFormat="inr" />
      </div>

      {/* Equity curve */}
      {data.equity_curve.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-body">Equity Curve</CardTitle>
            <CardDescription>
              {data.from_date} → {data.to_date} · {data.symbols_count} symbols ·
              {data.mode === "daily" ? " daily swing" : " best intraday config"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.equity_curve}>
                  <defs>
                    <linearGradient id="bt-eq-fill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="rgb(var(--accent))" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="rgb(var(--accent))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgb(var(--border))" vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="t" stroke="rgb(var(--fg-subtle))" fontSize={11}
                         tickFormatter={(t) => String(t).slice(5, 10)} />
                  <YAxis stroke="rgb(var(--fg-subtle))" fontSize={11}
                         tickFormatter={(v) => fmtInr(v, { compact: true })} width={70} />
                  <ReTooltip
                    contentStyle={{
                      background: "rgb(var(--surface))",
                      border: "1px solid rgb(var(--border-strong))",
                      borderRadius: 6, color: "rgb(var(--fg))", fontSize: 12,
                    }}
                    formatter={(v: number) => [fmtInr(v), "Equity"]}
                  />
                  <Area dataKey="v" stroke="rgb(var(--accent))" strokeWidth={2}
                        fill="url(#bt-eq-fill)" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Intraday TF grid */}
      {data.tf_grid.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-body">TF / Risk Grid</CardTitle>
            {data.best_config.tf && (
              <CardDescription>
                Best: <strong>{data.best_config.tf}</strong> SL:{data.best_config.sl_atr} ATR,
                RR:{data.best_config.rr} → PF {data.best_config.pf?.toFixed(2)},{" "}
                {((data.best_config.win_rate ?? 0) * 100).toFixed(0)}% win
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-body-sm">
              <thead>
                <tr className="border-b border-border">
                  {["TF", "SL", "RR", "Trades", "Win%", "PF", "P&L", "MaxDD"].map((h) => (
                    <th key={h} className="text-right p-2 text-fg-subtle first:text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.tf_grid.map((r, i) => {
                  const pfColor = r.pf >= 1.5 ? "text-pnl-up" : r.pf >= 1.0 ? "text-fg" : "text-pnl-down";
                  const isBest = r.tf === data.best_config.tf &&
                    r.sl_atr === data.best_config.sl_atr && r.rr === data.best_config.rr;
                  return (
                    <tr key={i} className={cn("border-b border-border/40", isBest && "bg-accent/5")}>
                      <td className="p-2 font-mono">{r.tf}{isBest && " ★"}</td>
                      <td className="p-2 text-right font-mono">{r.sl_atr}</td>
                      <td className="p-2 text-right font-mono">{r.rr}</td>
                      <td className="p-2 text-right font-mono">{r.trades}</td>
                      <td className="p-2 text-right font-mono">{(r.win_rate * 100).toFixed(0)}%</td>
                      <td className={cn("p-2 text-right font-mono font-semibold", pfColor)}>{r.pf.toFixed(2)}</td>
                      <td className={cn("p-2 text-right font-mono", r.pnl >= 0 ? "text-pnl-up" : "text-pnl-down")}>
                        {fmtInr(r.pnl)}
                      </td>
                      <td className="p-2 text-right font-mono text-pnl-down">{fmtInr(-r.max_dd)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Phase breakdown */}
      {Object.keys(data.phase_stats).length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-body">Per-Phase Breakdown</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(data.phase_stats).sort(([, a], [, b]) => b.pnl - a.pnl).map(([ph, s]) => (
                <div key={ph} className="rounded-sm border border-border/60 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge tone={phaseTone(ph as CyclePhase)}>{ph}</Badge>
                    <span className="text-caption text-fg-subtle">{s.trades}t</span>
                  </div>
                  <div className="text-body-sm font-mono">
                    {(s.win_rate * 100).toFixed(0)}% win ·{" "}
                    <span className={s.pnl >= 0 ? "text-pnl-up" : "text-pnl-down"}>{fmtInr(s.pnl)}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Trade log — collapsible */}
      {data.trades.length > 0 && (
        <Card>
          <CardHeader
            className="cursor-pointer select-none"
            onClick={() => setShowTrades((o) => !o)}
            role="button"
            aria-expanded={showTrades}
          >
            <CardTitle className="flex items-center gap-2 text-body">
              {showTrades ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              Trade Log ({data.trades.length})
              <span className="text-caption text-fg-subtle font-normal ml-2">
                {data.trades.filter((t) => t.won).length}W / {data.trades.filter((t) => !t.won).length}L
              </span>
            </CardTitle>
          </CardHeader>
          {showTrades && (
            <CardContent className="p-0">
              <DataTable<OKBacktestTrade>
                columns={tradeColumns}
                rows={sortedTrades}
                sort={tradeSort}
                onSort={handleSort}
                rowKey={(t) => `${t.symbol}-${t.entry_date}-${t.phase}`}
                emptyTitle="No trades"
              />
            </CardContent>
          )}
        </Card>
      )}

      {/* Errors */}
      {data.errors.length > 0 && (
        <Card>
          <CardContent>
            <ul className="list-disc pl-5 text-body-sm text-fg-muted">
              {data.errors.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/* ================================================================== */
/* History tab — last 20 runs from localStorage                        */
/* ================================================================== */

function HistoryTab() {
  const [entries, setEntries] = React.useState<BacktestHistoryEntry[]>(() => listHistory());
  const [open, setOpen] = React.useState<BacktestHistoryEntry | null>(null);

  const refresh = () => setEntries(listHistory());

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteHistoryEntry(id);
    refresh();
    if (open?.id === id) setOpen(null);
  };

  const handleClear = () => {
    if (!confirm("Clear all backtest history? This can't be undone.")) return;
    clearHistory();
    refresh();
    setOpen(null);
  };

  if (entries.length === 0) {
    return (
      <Card>
        <CardContent className="py-12">
          <EmptyState
            title="No backtest history yet"
            description="Run a Daily Swing, Intraday Multi-TF, or Morning Basket backtest — every result is logged here for the last 20 runs, replayable without re-fetching."
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <div>
            <CardTitle>Past runs</CardTitle>
            <CardDescription>
              Last {entries.length} backtest{entries.length === 1 ? "" : "s"} on this device. Click any row to re-display its full result.
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={handleClear} leading={<Trash2 className="h-3.5 w-3.5" />}>
            Clear all
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-body-sm">
            <thead>
              <tr className="border-b border-border text-fg-subtle">
                <th className="text-left p-2 pl-4">Ran</th>
                <th className="text-left p-2">Mode</th>
                <th className="text-left p-2">Window</th>
                <th className="text-right p-2">Symbols</th>
                <th className="text-right p-2">Trades</th>
                <th className="text-right p-2">Win%</th>
                <th className="text-right p-2">PF</th>
                <th className="text-right p-2">P&L</th>
                <th className="text-right p-2 pr-4"></th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => {
                const isOpen = open?.id === e.id;
                const pfColor = e.profit_factor >= 1.5 ? "text-pnl-up"
                  : e.profit_factor >= 1.0 ? "text-fg" : "text-pnl-down";
                return (
                  <tr key={e.id}
                      onClick={() => setOpen(isOpen ? null : e)}
                      className={cn(
                        "border-b border-border/40 cursor-pointer hover:bg-surface-2/40",
                        isOpen && "bg-accent/5",
                      )}>
                    <td className="p-2 pl-4 text-fg-muted">
                      <span className="inline-flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {new Date(e.ran_at).toLocaleString()}
                      </span>
                    </td>
                    <td className="p-2"><Badge tone="brand">{e.mode}</Badge></td>
                    <td className="p-2 font-mono text-caption">{e.from_date} → {e.to_date}</td>
                    <td className="p-2 text-right font-mono">{e.symbols_count}</td>
                    <td className="p-2 text-right font-mono">{e.total_trades}</td>
                    <td className="p-2 text-right font-mono">{(e.win_rate * 100).toFixed(0)}%</td>
                    <td className={cn("p-2 text-right font-mono", pfColor)}>{e.profit_factor.toFixed(2)}</td>
                    <td className={cn("p-2 text-right font-mono", e.total_pnl >= 0 ? "text-pnl-up" : "text-pnl-down")}>
                      {fmtInr(e.total_pnl)}
                    </td>
                    <td className="p-2 pr-4 text-right">
                      <button
                        onClick={(ev) => handleDelete(e.id, ev)}
                        className="text-fg-subtle hover:text-danger p-1"
                        aria-label="Delete run"
                        title="Delete"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {open && (
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-2 px-1">
            <h3 className="text-body font-semibold">
              Replay: {open.mode} · {open.from_date} → {open.to_date}
            </h3>
            <Button variant="ghost" size="sm" onClick={() => setOpen(null)} leading={<X className="h-3.5 w-3.5" />}>
              Close replay
            </Button>
          </div>
          <BacktestResults data={open.payload} />
        </div>
      )}
    </div>
  );
}
