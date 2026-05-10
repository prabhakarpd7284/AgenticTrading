/**
 * Swing Scanner — Oliver Kell Cycle of Price Action (Live Scanner).
 *
 * Scans NIFTY 100 on daily/weekly charts to detect cycle phases.
 * Backtesting is consolidated in the Backtester page.
 */
import * as React from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  BarChart3,
  ChevronDown,
  ChevronUp,
  RefreshCcw,
  TrendingUp,
} from "lucide-react";

import {
  useMarketPulse,
  useSwingScanner,
  phaseTone,
  trendStateTone,
  PHASE_INFO,
  type SwingStock,
  type CyclePhase,
} from "@/lib/market-pulse";
import { cn, fmtNum } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { KPI } from "@/components/ui/KPI";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

/* ------------------------------------------------------------------ */
/* Main page — live scanner only (backtest is in BacktesterPage)        */
/* ------------------------------------------------------------------ */

export function SwingScannerPage() {
  const { data: pulse } = useMarketPulse();
  const { data, isLoading, isError, error, refetch, isFetching } =
    useSwingScanner({ isOpen: pulse?.is_market_open ?? false });

  const [showRules, setShowRules] = React.useState(false);
  const [filterAligned, setFilterAligned] = React.useState(false);

  if (isLoading) return <ScannerLoading />;
  if (isError) return <ScannerError error={error as Error} onRetry={() => refetch()} />;
  if (!data) return null;

  const stocks = filterAligned ? data.stocks.filter((s) => s.aligned) : data.stocks;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">
            Swing Scanner · Oliver Kell Cycles
          </p>
          <h1 className="text-h1 text-fg">Swing Scanner</h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            Live OK cycle phase detection across NIFTY 100 with RSI momentum confirmation.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone="neutral">{data.total} scanned</Badge>
          <Badge tone="success">{data.active} active</Badge>
          <span className="text-caption text-fg-subtle">{data.scan_date}</span>
          <Button variant="ghost" size="icon" onClick={() => refetch()}
                  aria-label="Refresh" disabled={isFetching}>
            <RefreshCcw className={cn("h-4 w-4", isFetching && "animate-spin")} />
          </Button>
          <Link to="/backtester">
            <Button variant="ghost" size="sm" leading={<BarChart3 className="h-4 w-4" />}>
              Backtest
            </Button>
          </Link>
        </div>
      </header>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KPI label="Scanned" value={data.total} />
        <KPI label="Active Phases" value={data.active} />
        <KPI label="BUY (aligned)" value={data.buy_aligned} />
        <KPI label="SHORT (aligned)" value={data.short_aligned} />
        <KPI label="WATCH" value={data.watch} />
      </div>

      {/* Phase distribution */}
      {Object.keys(data.phase_distribution).length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-body">Phase Distribution</CardTitle></CardHeader>
          <CardContent>
            <div className="flex gap-2 flex-wrap">
              {Object.entries(data.phase_distribution).sort(([, a], [, b]) => b - a).map(([phase, count]) => {
                const info = PHASE_INFO[phase];
                return (
                  <div key={phase} className={cn("inline-flex items-center gap-1.5 rounded-sm border px-3 py-1.5",
                    info?.color ?? "bg-surface-2 border-border")}>
                    <span className="font-mono text-body-sm font-semibold">{phase}</span>
                    <span className="text-body-sm">{count}</span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Filter */}
      <div className="flex items-center gap-3">
        <Button variant={filterAligned ? "secondary" : "ghost"} size="sm"
                onClick={() => setFilterAligned(!filterAligned)}>
          {filterAligned ? "Showing aligned only" : "Show all phases"}
        </Button>
        <span className="text-caption text-fg-subtle">
          {stocks.length} stock{stocks.length !== 1 && "s"}
        </span>
      </div>

      {/* Stock cards */}
      {stocks.length === 0 ? (
        <EmptyState
          title="No active phases"
          description={filterAligned
            ? "No stocks with aligned trends. Try showing all."
            : "No cycle phases detected today."}
        />
      ) : (
        <div className="space-y-3">
          {stocks.map((s) => <StockCard key={s.symbol} stock={s} />)}
        </div>
      )}

      {/* Errors */}
      {data.errors.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-body text-warn">Warnings</CardTitle></CardHeader>
          <CardContent>
            <ul className="list-disc pl-5 text-body-sm text-fg-muted space-y-1">
              {data.errors.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Trading rules */}
      <Card>
        <CardHeader className="cursor-pointer select-none" onClick={() => setShowRules((o) => !o)}
                    role="button" aria-expanded={showRules}>
          <CardTitle className="flex items-center gap-2 text-body">
            {showRules ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            Oliver Kell Trading Rules
          </CardTitle>
          <CardDescription>Phase definitions, entry rules, and stop-loss guidance.</CardDescription>
        </CardHeader>
        {showRules && (
          <CardContent>
            <div className="space-y-3">
              {Object.entries(PHASE_INFO).map(([code, info]) => (
                <div key={code} className="flex items-start gap-3 py-2 border-b border-border/40 last:border-0">
                  <Badge tone={phaseTone(code as CyclePhase)}>{code}</Badge>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-body-sm font-semibold text-fg">{info.label}</span>
                      <span className={cn("text-caption font-mono",
                        info.action === "BUY" ? "text-pnl-up" :
                        info.action === "SHORT" || info.action === "AVOID" ? "text-pnl-down" : "text-fg-muted"
                      )}>{info.action}</span>
                    </div>
                    <p className="text-caption text-fg-muted mt-0.5">{info.description}</p>
                  </div>
                </div>
              ))}
              <div className="pt-2 text-caption text-fg-subtle">
                <strong>Stop Loss:</strong> Below EMA10/20 for longs, above for shorts.{" "}
                <strong>EMAs:</strong> 10/20/50.{" "}
                <strong>Trend:</strong> EMA10 &gt; EMA20 &gt; EMA50 = Bullish.
              </div>
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  );
}

/* ================================================================== */
/* Stock card                                                           */
/* ================================================================== */

function StockCard({ stock: s }: { stock: SwingStock }) {
  const phaseInfo = PHASE_INFO[s.phase] ?? { label: s.phase_label, action: s.action, color: "bg-surface-2 border-border" };

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <CardTitle className="font-mono">
              <Link to={`/setup/${encodeURIComponent(s.symbol)}`}
                    className="hover:text-accent focus-visible:outline-none focus-visible:underline">
                {s.symbol}
              </Link>
            </CardTitle>
            <Badge tone={phaseTone(s.phase)}>{s.phase} — {phaseInfo.action}</Badge>
            {s.aligned && <Badge tone="brand">Aligned</Badge>}
          </div>
          <CardDescription className="mt-1">{phaseInfo.label} · Last {fmtNum(s.close, 2)}</CardDescription>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <Badge tone="neutral">{(s.confidence * 100).toFixed(0)}% conf</Badge>
          <Link to={`/setup/${encodeURIComponent(s.symbol)}`}
                className="inline-flex items-center gap-1 text-caption text-fg-subtle hover:text-fg">
            Setup <ArrowRight className="h-3 w-3" aria-hidden />
          </Link>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          <Metric label="EMA 10" value={fmtNum(s.ema10, 2)} />
          <Metric label="EMA 20" value={fmtNum(s.ema20, 2)} />
          <Metric label="EMA 50" value={fmtNum(s.ema50, 2)} />
          <Metric label="Upper Ext" value={fmtNum(s.upper_ext, 2)} />
          <Metric label="Lower Ext" value={fmtNum(s.lower_ext, 2)} />
          <Metric label="Volume" value={`${s.volume_ratio.toFixed(1)}x`} />
          <Metric label="Close" value={fmtNum(s.close, 2)} />
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5">
            <TrendingUp className="h-3.5 w-3.5 text-fg-subtle" />
            <span className="text-caption text-fg-subtle">Daily</span>
            <Badge tone={trendStateTone(s.trend_daily)}>{s.trend_daily}</Badge>
          </div>
          <div className="flex items-center gap-1.5">
            <TrendingUp className="h-3.5 w-3.5 text-fg-subtle" />
            <span className="text-caption text-fg-subtle">Weekly</span>
            <Badge tone={trendStateTone(s.trend_weekly)}>{s.trend_weekly}</Badge>
          </div>
          {s.aligned && <span className="text-caption text-success font-medium">Trends aligned</span>}
        </div>
      </CardContent>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border/60 p-2">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className="text-body font-mono tabular text-fg">{value}</div>
    </div>
  );
}

function ScannerLoading() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <Skeleton className="h-12 w-80" />
      <div className="grid grid-cols-5 gap-3">
        {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-16 w-full" />)}
      </div>
      <div className="space-y-3">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-40 w-full" />)}
      </div>
    </div>
  );
}

function ScannerError({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="px-6 py-6 max-w-[800px] mx-auto">
      <EmptyState
        title="Couldn't load swing scanner"
        description={error?.message ?? "Unknown error"}
        action={<Button onClick={onRetry}>Try again</Button>}
      />
    </div>
  );
}
