import * as React from "react";
import {
  Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ComposedChart, Line, Bar, ReferenceLine, ReferenceDot,
  CartesianGrid,
} from "recharts";
import {
  TrendingUp, Play, ChevronDown, ChevronUp, Layers,
  Target, Shield, ArrowUpRight, ArrowDownRight, Zap,
  AlertTriangle, Clock, Send, CheckCircle2,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { KPI } from "@/components/ui/KPI";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";
import { cn, fmtInr, fmtNum, clsPnl } from "@/lib/utils";
import { api } from "@/lib/api";
import { INDICES, INDEX_LIST, getLotSize, isMonthlyOnly, getExpiryWeekday } from "@/lib/market-config";
import { useMarketPulse, type Quote } from "@/lib/market-pulse";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface PyramidEntry {
  t: string;
  price: number;
  lots: number;
  reason: string;
  sl: number;
  cumulative_lots: number;
}

interface PyramidExit {
  t: string;
  price: number;
  reason: string;
  lots: number;
}

interface PyramidKPIs {
  total_pnl_pts: number;
  total_pnl_inr: number;
  peak_unrealized_pts: number;
  peak_unrealized_inr: number;
  total_lots: number;
  peak_lots: number;
  pyramid_count: number;
  avg_entry: number;
  exit_price: number;
  exit_reason: string;
  lot_size: number;
  won: boolean;
  capital_deployed: number;
  initial_risk_inr: number;
  roi_pct: number;
}

interface CandleRow {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  ema5?: number;
  bb_upper?: number;
  bb_mid?: number;
  bb_lower?: number;
  rsi?: number;
  rsi_ema3?: number;
  rsi_wma21?: number;
}

interface TrailSLPoint { t: string; sl: number; }
interface PositionPoint { t: string; lots: number; avg_entry: number; unrealized: number; }

interface PyramidData {
  symbol: string;
  kpis: PyramidKPIs;
  candles: CandleRow[];
  trail_sl: TrailSLPoint[];
  position: PositionPoint[];
  entries: PyramidEntry[];
  exit: PyramidExit | null;
  log: string[];
  config: Record<string, any>;
  telegram_sent?: boolean;
}

/* ------------------------------------------------------------------ */
/* Hook                                                                */
/* ------------------------------------------------------------------ */

function usePyramid(params: Record<string, string>, enabled: boolean) {
  return useQuery<PyramidData>({
    queryKey: ["pyramid", params],
    queryFn: async () => {
      // Synchronous backtest — the v2 native endpoint at
      // /strategies/pyramid/backtest/ runs the engine in-process and
      // returns the chart payload. For long iterations use the Ops
      // Console (run_pyramid via /ws/ops/).
      const r = await api.get<PyramidData>("/strategies/pyramid/backtest/", {
        params,
        timeout: 120_000,
      });
      return r.data;
    },
    enabled,
    staleTime: 300_000,
    retry: false,
  });
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function fmtTime(ts: string) {
  if (!ts) return "";
  const m = ts.match(/T(\d{2}:\d{2})/);
  return m ? m[1] : ts.slice(11, 16);
}

/* ------------------------------------------------------------------ */
/* Underlying config                                                   */
/* ------------------------------------------------------------------ */

/* Constants sourced from @/lib/market-config — single source of truth. */

const MONTH_ABBR = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];

/**
 * Compute next expiry in DDMMMYY format (e.g. "13MAY26").
 * Uses market-config for expiry weekday and monthly-only flag.
 */
function computeNextExpiry(underlying: string): string {
  const today = new Date();
  const weekday = getExpiryWeekday(underlying);

  if (isMonthlyOnly(underlying)) {
    // Last occurrence of expiry weekday in current month; if past, next month
    const lastExp = lastWeekdayOfMonth(today.getFullYear(), today.getMonth(), weekday);
    let exp: Date;
    if (lastExp > today || lastExp.toDateString() === today.toDateString()) {
      exp = lastExp;
    } else {
      const nextMonth = today.getMonth() === 11 ? 0 : today.getMonth() + 1;
      const nextYear = today.getMonth() === 11 ? today.getFullYear() + 1 : today.getFullYear();
      exp = lastWeekdayOfMonth(nextYear, nextMonth, weekday);
    }
    return fmtExpiry(exp);
  }

  // Weekly expiry
  const dow = today.getDay(); // Sun=0 ... Sat=6
  const todayMon = (dow + 6) % 7; // Mon=0
  let daysAhead = (weekday - todayMon + 7) % 7;
  if (daysAhead === 0) {
    const ist = new Date(today.getTime() + 5.5 * 3600_000);
    if (ist.getHours() >= 16) daysAhead = 7;
  }
  const exp = new Date(today);
  exp.setDate(exp.getDate() + daysAhead);
  return fmtExpiry(exp);
}

/** Last occurrence of a weekday (Mon=0..Sun=6) in a given month. */
function lastWeekdayOfMonth(year: number, month: number, targetDow: number): Date {
  // Start from last day of month and walk backwards
  const last = new Date(year, month + 1, 0); // last day
  const dow = (last.getDay() + 6) % 7; // Mon=0
  let diff = (dow - targetDow + 7) % 7;
  return new Date(year, month, last.getDate() - diff);
}

function fmtExpiry(d: Date): string {
  const dd = String(d.getDate()).padStart(2, "0");
  const mmm = MONTH_ABBR[d.getMonth()];
  const yy = String(d.getFullYear()).slice(2);
  return `${dd}${mmm}${yy}`;
}

/* ------------------------------------------------------------------ */
/* Index ticker strip                                                  */
/* ------------------------------------------------------------------ */

const INDEX_KEYS = ["NIFTY", "BANKNIFTY", "SENSEX"];

function IndexTicker({ quotes }: { quotes: Quote[] }) {
  const indexed = React.useMemo(() => {
    const map: Record<string, Quote> = {};
    for (const q of quotes) {
      // pulse uses symbol keys like "NIFTY", "BANKNIFTY", etc.
      if (INDEX_KEYS.includes(q.symbol)) map[q.symbol] = q;
    }
    return map;
  }, [quotes]);

  return (
    <div className="flex items-center gap-3">
      {INDEX_KEYS.map((key) => {
        const q = indexed[key];
        if (!q || q.last == null) return null;
        const meta = INDICES[key];
        const chg = q.change_pct ?? 0;
        return (
          <div
            key={key}
            className="flex items-center gap-1.5 rounded-sm border border-border/60 bg-surface-2/50 px-2.5 py-1.5"
          >
            <span className="text-caption text-fg-subtle font-medium">
              {meta?.label ?? key}
            </span>
            <span className="font-mono tabular text-body-sm text-fg font-semibold">
              {fmtNum(q.last, 0)}
            </span>
            <span className={cn(
              "font-mono tabular text-caption font-medium",
              clsPnl(chg),
            )}>
              {chg > 0 ? "+" : ""}{chg.toFixed(2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function PyramidPage() {
  // Live index prices from market pulse
  const { data: pulse } = useMarketPulse();

  // Config state
  const [strike, setStrike] = React.useState("24200");
  const [optType, setOptType] = React.useState("CE");
  const [underlying, setUnderlying] = React.useState("NIFTY");
  const [expiry, setExpiry] = React.useState(() => computeNextExpiry("NIFTY"));
  const [date, setDate] = React.useState("");
  const [capital, setCapital] = React.useState("100000");
  const [riskPct, setRiskPct] = React.useState("2.0");
  const [profitRisk, setProfitRisk] = React.useState("0.80");
  const [maxPyramids, setMaxPyramids] = React.useState("5");
  const [lotSize, setLotSize] = React.useState("65");

  // Auto-set lot size and expiry when underlying changes
  const handleUnderlying = React.useCallback((u: string) => {
    setUnderlying(u);
    setLotSize(String(getLotSize(u)));
    setExpiry(computeNextExpiry(u));
  }, []);
  const [dryRun, setDryRun] = React.useState(false);
  const [telegram, setTelegram] = React.useState(false);
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [showLog, setShowLog] = React.useState(false);
  const [run, setRun] = React.useState(false);

  const params = React.useMemo(() => {
    const p: Record<string, string> = {
      strike,
      type: optType,
      underlying,
      capital,
      risk_pct: riskPct,
      profit_risk: profitRisk,
      max_pyramids: maxPyramids,
      lot_size: lotSize,
      dry_run: dryRun ? "true" : "false",
      telegram: telegram ? "true" : "false",
    };
    if (expiry) p.expiry = expiry;
    if (date) p.date = date;
    return p;
  }, [strike, optType, underlying, expiry, date, capital, riskPct, profitRisk, maxPyramids, lotSize, dryRun, telegram]);

  const { data, isLoading, isError, error, dataUpdatedAt } = usePyramid(params, run);

  // Reset run flag after data loads or errors
  React.useEffect(() => {
    if (data || isError) setRun(false);
  }, [data, isError]);

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-accent/15">
            <Layers className="h-5 w-5 text-accent" />
          </div>
          <div>
            <h1 className="text-h2 font-semibold text-fg">Pyramid Strategy</h1>
            <p className="text-body-sm text-fg-muted">
              Aggressive momentum pyramiding on options
            </p>
          </div>
        </div>

        {/* Live index prices */}
        <div className="flex items-center gap-4">
          {pulse?.quotes?.indices_in && <IndexTicker quotes={pulse.quotes.indices_in} />}
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-body-sm text-fg-muted cursor-pointer select-none"
                 title="Use synthetic sample candles instead of real broker data. Results will differ from live.">
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
              className="rounded border-border accent-accent"
            />
            Sample Data
          </label>
          <label className="flex items-center gap-2 text-body-sm text-fg-muted cursor-pointer select-none">
            <input
              type="checkbox"
              checked={telegram}
              onChange={(e) => setTelegram(e.target.checked)}
              className="rounded border-border accent-accent"
            />
            <Send className="h-3.5 w-3.5" />
            Telegram
          </label>
          <Button
            onClick={() => setRun(true)}
            loading={isLoading}
            leading={<Play className="h-4 w-4" />}
          >
            Run
          </Button>
        </div>
      </div>

      {/* ── Config ── */}
      <Card>
        <CardContent className="py-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <Input label="Strike" type="number" value={strike} onChange={(e) => setStrike(e.target.value)} />
            <div>
              <label className="mb-1.5 inline-flex text-body-sm text-fg">Type</label>
              <div className="flex gap-1">
                {(["CE", "PE"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setOptType(t)}
                    className={cn(
                      "flex-1 h-9 rounded-sm text-body-sm font-medium transition-colors duration-120",
                      optType === t
                        ? t === "CE" ? "bg-pnl-up/20 text-pnl-up border border-pnl-up/40"
                                     : "bg-pnl-down/20 text-pnl-down border border-pnl-down/40"
                        : "bg-surface-2 text-fg-muted border border-border hover:border-border-strong",
                    )}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="mb-1.5 inline-flex text-body-sm text-fg">Underlying</label>
              <div className="flex gap-1">
                {INDEX_LIST.map((u) => (
                  <button
                    key={u}
                    onClick={() => handleUnderlying(u)}
                    className={cn(
                      "flex-1 h-9 rounded-sm text-body-sm font-medium transition-colors duration-120",
                      underlying === u
                        ? "bg-accent/15 text-accent border border-accent/40"
                        : "bg-surface-2 text-fg-muted border border-border hover:border-border-strong",
                    )}
                  >
                    {u === "BANKNIFTY" ? "BNFTY" : u}
                  </button>
                ))}
              </div>
            </div>
            <Input label="Expiry" value={expiry} onChange={(e) => setExpiry(e.target.value)} hint={`Next: ${computeNextExpiry(underlying)}`} />
            <Input label="Date" type="date" value={date} onChange={(e) => setDate(e.target.value)} hint="Default: last trading day" />
            <Input label="Capital" type="number" value={capital} onChange={(e) => setCapital(e.target.value)} leading="₹" />
          </div>

          {/* Advanced toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1 mt-3 text-caption text-fg-muted hover:text-fg transition-colors"
          >
            {showAdvanced ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            Advanced Parameters
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3 pt-3 border-t border-border/60">
              <Input label="Init Risk %" type="number" step="0.1" value={riskPct} onChange={(e) => setRiskPct(e.target.value)} hint="% of capital on first entry" />
              <Input label="Profit Risk" type="number" step="0.05" value={profitRisk} onChange={(e) => setProfitRisk(e.target.value)} hint="Fraction risked on pyramids" />
              <Input label="Max Pyramids" type="number" value={maxPyramids} onChange={(e) => setMaxPyramids(e.target.value)} hint="Maximum add-ons" />
              <Input label="Lot Size" type="number" value={lotSize} onChange={(e) => setLotSize(e.target.value)} hint={`Auto: ${getLotSize(underlying)}`} />
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Error state ── */}
      {isError && (
        <Card>
          <CardContent className="flex items-center gap-3 text-pnl-down">
            <AlertTriangle className="h-5 w-5 flex-shrink-0" />
            <div>
              <p className="font-medium">Simulation failed</p>
              <p className="text-body-sm text-fg-muted">
                {(error as any)?.response?.data?.detail || (error as any)?.message || "Unknown error"}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Results ── */}
      {data && (
        <PyramidResults
          data={data}
          showLog={showLog}
          setShowLog={setShowLog}
          dataUpdatedAt={dataUpdatedAt}
        />
      )}

      {/* ── Empty state ── */}
      {!data && !isLoading && !isError && (
        <Card>
          <CardContent className="py-16 text-center">
            <Layers className="h-12 w-12 mx-auto text-fg-subtle mb-4" />
            <h3 className="text-h3 font-semibold text-fg mb-2">Ready to simulate</h3>
            <p className="text-body-sm text-fg-muted max-w-md mx-auto">
              Configure your option strike & parameters, then hit{" "}
              <span className="font-medium text-fg">Run Simulation</span>.
              Enable <span className="font-medium text-fg">Dry Run</span> to test
              with sample momentum data without broker login.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Results Panel                                                       */
/* ------------------------------------------------------------------ */

function PyramidResults({
  data,
  showLog,
  setShowLog,
  dataUpdatedAt,
}: {
  data: PyramidData;
  showLog: boolean;
  setShowLog: (v: boolean) => void;
  dataUpdatedAt: number;
}) {
  const { kpis, entries, exit, candles, trail_sl, position } = data;
  const won = kpis.won;

  return (
    <div className="space-y-4">
      {/* ── Symbol + Result badge ── */}
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-h3 font-semibold text-fg">{data.symbol}</h2>
        <Badge tone={won ? "success" : "danger"} dot>
          {won ? "WINNER" : "LOSS"}
        </Badge>
        {data.config?.dry_run && <Badge tone="warning">Sample Data</Badge>}
        {data.telegram_sent && (
          <Badge tone="success">
            <CheckCircle2 className="h-3 w-3" /> Sent to Telegram
          </Badge>
        )}
        {/* Pyramid is a one-shot backtest, not a stream — generous thresholds
            (1min fresh, 10min stale) flag re-run if the operator left it up
            for a while. */}
        <FreshnessIndicator
          label="Simulated"
          timestamp={dataUpdatedAt}
          freshMs={60_000}
          staleMs={10 * 60_000}
        />
      </div>

      {/* ── KPIs ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-9 gap-3">
        <KPI label="Total P&L" value={kpis.total_pnl_inr} valueFormat="inr" />
        <KPI label="P&L (points)" value={kpis.total_pnl_pts} hint={`Exit: ${kpis.exit_reason}`} />
        <KPI label="Capital Deployed" value={kpis.capital_deployed} valueFormat="inr" hint="Premium paid" />
        <KPI label="ROI" value={kpis.roi_pct} valueFormat="pct" hint="P&L / Capital" />
        <KPI label="Initial Risk" value={kpis.initial_risk_inr} valueFormat="inr" hint="Max loss on entry" />
        <KPI label="Peak Unrealized" value={kpis.peak_unrealized_inr} valueFormat="inr" />
        <KPI label="Final Lots" value={kpis.total_lots} hint={`Peak: ${kpis.peak_lots}`} />
        <KPI label="Pyramids" value={kpis.pyramid_count} hint={`${entries.length} total entries`} />
        <KPI label="Avg Entry" value={kpis.avg_entry} hint={`Exit @ ${fmtNum(kpis.exit_price, 2)}`} />
      </div>

      {/* ── Pyramid Flow ── */}
      {entries.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-accent" />
              Pyramid Flow
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex items-stretch gap-0 overflow-x-auto pb-2">
              {entries.map((e, i) => (
                <React.Fragment key={i}>
                  {i > 0 && (
                    <div className="flex items-center px-1">
                      <div className="h-px w-6 bg-border" />
                      <ArrowUpRight className="h-3.5 w-3.5 text-fg-subtle -ml-1" />
                    </div>
                  )}
                  <div className={cn(
                    "flex-shrink-0 rounded-md border px-3 py-2.5 min-w-[120px]",
                    i === 0 ? "border-accent/40 bg-accent/10"
                            : "border-pnl-up/30 bg-pnl-up/5",
                  )}>
                    <div className="text-caption text-fg-subtle">{fmtTime(e.t)}</div>
                    <div className="text-body-sm font-semibold text-fg mt-0.5">
                      {i === 0 ? "Entry" : `Add #${i}`}
                    </div>
                    <div className="font-mono text-num text-fg mt-1">
                      {fmtNum(e.price, 2)}
                    </div>
                    <div className="flex items-center justify-between mt-1.5 gap-2">
                      <span className="text-caption text-fg-muted">
                        +{e.lots} lots
                      </span>
                      <span className="text-caption text-fg-subtle">
                        &Sigma;{e.cumulative_lots}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 mt-1">
                      <Shield className="h-3 w-3 text-warn" />
                      <span className="text-caption text-warn font-mono">
                        SL {fmtNum(e.sl, 2)}
                      </span>
                    </div>
                  </div>
                </React.Fragment>
              ))}

              {/* Exit card */}
              {exit && (
                <>
                  <div className="flex items-center px-1">
                    <div className="h-px w-6 bg-border" />
                    <ArrowDownRight className="h-3.5 w-3.5 text-fg-subtle -ml-1" />
                  </div>
                  <div className={cn(
                    "flex-shrink-0 rounded-md border px-3 py-2.5 min-w-[120px]",
                    won ? "border-pnl-up/40 bg-pnl-up/10" : "border-pnl-down/40 bg-pnl-down/10",
                  )}>
                    <div className="text-caption text-fg-subtle">{fmtTime(exit.t)}</div>
                    <div className="text-body-sm font-semibold text-fg mt-0.5">
                      Exit
                    </div>
                    <div className="font-mono text-num text-fg mt-1">
                      {fmtNum(exit.price, 2)}
                    </div>
                    <div className="text-caption text-fg-muted mt-1.5">
                      {exit.lots} lots
                    </div>
                    <div className="flex items-center gap-1 mt-1">
                      <Clock className="h-3 w-3 text-fg-subtle" />
                      <span className="text-caption text-fg-subtle">
                        {exit.reason}
                      </span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Price Chart with Indicators ── */}
      <PriceChart candles={candles} trail_sl={trail_sl} entries={entries} exit={exit} />

      {/* ── RSI Chart ── */}
      <RSIChart candles={candles} />

      {/* ── Position Size Timeline ── */}
      <PositionChart position={position} />

      {/* ── Risk Analysis ── */}
      <RiskBreakdown entries={entries} exit={exit} kpis={kpis} />

      {/* ── Trade Log ── */}
      <Card>
        <CardHeader
          className="cursor-pointer pb-2"
          onClick={() => setShowLog(!showLog)}
        >
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              Trade Log
              <Badge tone="neutral">{data.log.length} events</Badge>
            </CardTitle>
            {showLog
              ? <ChevronUp className="h-4 w-4 text-fg-muted" />
              : <ChevronDown className="h-4 w-4 text-fg-muted" />}
          </div>
        </CardHeader>
        {showLog && (
          <CardContent className="pt-0">
            <div className="max-h-80 overflow-y-auto rounded-sm bg-surface-2 border border-border p-3 space-y-0.5">
              {data.log.map((line, i) => {
                const isEntry = line.includes("ENTRY") || line.includes("PYRAMID");
                const isExit = line.includes("SL HIT") || line.includes("EOD EXIT");
                const isSL = line.includes("SL raised");
                return (
                  <div
                    key={i}
                    className={cn(
                      "font-mono text-num-sm leading-relaxed",
                      isEntry ? "text-pnl-up" :
                      isExit ? "text-pnl-down" :
                      isSL ? "text-warn" :
                      "text-fg-muted",
                    )}
                  >
                    {line}
                  </div>
                );
              })}
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Price Chart                                                         */
/* ------------------------------------------------------------------ */

function PriceChart({
  candles, trail_sl, entries, exit,
}: {
  candles: CandleRow[];
  trail_sl: TrailSLPoint[];
  entries: PyramidEntry[];
  exit: PyramidExit | null;
}) {
  // Merge candle + trail SL + indicators into one series for recharts
  const slMap = new Map(trail_sl.map((s) => [s.t, s.sl]));
  const chartData = candles.map((c) => ({
    t: fmtTime(c.t),
    price: c.c,
    high: c.h,
    low: c.l,
    ema5: c.ema5 ?? null,
    bb_upper: c.bb_upper ?? null,
    bb_mid: c.bb_mid ?? null,
    bb_lower: c.bb_lower ?? null,
    trail_sl: slMap.get(c.t) ?? null,
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-accent" />
          Price &amp; Indicators
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" opacity={0.4} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: "rgb(var(--fg-subtle))" }} stroke="rgb(var(--fg-subtle))" interval="preserveStartEnd" />
              <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "rgb(var(--fg-subtle))" }} stroke="rgb(var(--fg-subtle))" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgb(var(--surface-2))",
                  border: "1px solid rgb(var(--border))",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />

              {/* BB bands */}
              <Area dataKey="bb_upper" stroke="none" fill="rgb(var(--accent) / 0.06)" stackId="bb" name="BB Upper" />
              <Area dataKey="bb_lower" stroke="none" fill="transparent" stackId="bb" name="BB Lower" />
              <Line dataKey="bb_upper" stroke="rgb(var(--fg-subtle))" strokeWidth={1} dot={false} strokeDasharray="2 2" name="BB Upper" />
              <Line dataKey="bb_lower" stroke="rgb(var(--fg-subtle))" strokeWidth={1} dot={false} strokeDasharray="2 2" name="BB Lower" />
              <Line dataKey="bb_mid" stroke="rgb(var(--fg-muted))" strokeWidth={1} dot={false} strokeDasharray="4 2" name="BB Mid" />

              {/* EMA5 */}
              <Line dataKey="ema5" stroke="rgb(var(--accent))" strokeWidth={1.5} dot={false} name="EMA 5" />

              {/* Trail SL */}
              <Line dataKey="trail_sl" stroke="rgb(var(--warn))" strokeWidth={2} dot={false} strokeDasharray="4 2" name="Trail SL" connectNulls={false} />

              {/* Price */}
              <Line dataKey="price" stroke="rgb(var(--fg))" strokeWidth={2} dot={false} name="Close" />

              {/* Entry/exit markers */}
              {entries.map((e, i) => (
                <ReferenceDot
                  key={`entry-${i}`}
                  x={fmtTime(e.t)}
                  y={e.price}
                  r={i === 0 ? 6 : 4}
                  fill={i === 0 ? "rgb(var(--accent))" : "rgb(var(--pnl-up))"}
                  stroke="rgb(var(--bg))"
                  strokeWidth={2}
                />
              ))}
              {exit && (
                <ReferenceDot
                  x={fmtTime(exit.t)}
                  y={exit.price}
                  r={6}
                  fill="rgb(var(--pnl-down))"
                  stroke="rgb(var(--bg))"
                  strokeWidth={2}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center gap-4 mt-2 text-caption text-fg-subtle">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-accent" /> EMA 5
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-6 border-t border-dashed border-fg-subtle" /> BB
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-6 border-t-2 border-dashed border-warn" /> Trail SL
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-accent" /> Entry
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-pnl-up" /> Pyramid
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-pnl-down" /> Exit
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* RSI Chart                                                           */
/* ------------------------------------------------------------------ */

function RSIChart({ candles }: { candles: CandleRow[] }) {
  const chartData = candles
    .filter((c) => c.rsi != null)
    .map((c) => ({
      t: fmtTime(c.t),
      rsi: c.rsi,
      rsi_ema3: c.rsi_ema3,
      rsi_wma21: c.rsi_wma21,
    }));

  if (!chartData.length) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          RSI (14) + EMA(3) + WMA(21)
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="h-36">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" opacity={0.3} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: "rgb(var(--fg-subtle))" }} stroke="rgb(var(--fg-subtle))" interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "rgb(var(--fg-subtle))" }} stroke="rgb(var(--fg-subtle))" ticks={[30, 50, 70]} />

              <ReferenceLine y={70} stroke="rgb(var(--pnl-down))" strokeDasharray="3 3" strokeOpacity={0.5} />
              <ReferenceLine y={30} stroke="rgb(var(--pnl-up))" strokeDasharray="3 3" strokeOpacity={0.5} />
              <ReferenceLine y={50} stroke="rgb(var(--fg-subtle))" strokeDasharray="2 2" strokeOpacity={0.3} />

              <Area dataKey="rsi" stroke="rgb(var(--accent))" fill="rgb(var(--accent) / 0.08)" strokeWidth={1.5} dot={false} name="RSI" />
              <Line dataKey="rsi_ema3" stroke="rgb(var(--pnl-up))" strokeWidth={1} dot={false} name="EMA(3)" />
              <Line dataKey="rsi_wma21" stroke="rgb(var(--warn))" strokeWidth={1} dot={false} strokeDasharray="3 2" name="WMA(21)" />

              <Tooltip
                contentStyle={{
                  backgroundColor: "rgb(var(--surface-2))",
                  border: "1px solid rgb(var(--border))",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Position Size Chart                                                 */
/* ------------------------------------------------------------------ */

function PositionChart({ position }: { position: PositionPoint[] }) {
  if (!position.length) return null;

  const chartData = position.map((p) => ({
    t: fmtTime(p.t),
    lots: p.lots,
    unrealized: p.unrealized,
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Target className="h-4 w-4 text-accent" />
            Position Build-up
          </CardTitle>
          <span className="text-caption text-fg-subtle">
            Lots held over time + unrealized P&L
          </span>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" opacity={0.3} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: "rgb(var(--fg-subtle))" }} stroke="rgb(var(--fg-subtle))" interval="preserveStartEnd" />
              <YAxis yAxisId="lots" tick={{ fontSize: 10, fill: "rgb(var(--fg-subtle))" }} stroke="rgb(var(--fg-subtle))" />
              <YAxis yAxisId="pnl" orientation="right" tick={{ fontSize: 10, fill: "rgb(var(--fg-subtle))" }} stroke="rgb(var(--fg-subtle))" />

              <Bar
                yAxisId="lots"
                dataKey="lots"
                fill="rgb(var(--accent) / 0.25)"
                stroke="rgb(var(--accent) / 0.5)"
                name="Lots"
              />
              <Line
                yAxisId="pnl"
                dataKey="unrealized"
                stroke="rgb(var(--pnl-up))"
                strokeWidth={1.5}
                dot={false}
                name="Unrealized P&L"
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: "rgb(var(--surface-2))",
                  border: "1px solid rgb(var(--border))",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Risk Breakdown                                                      */
/* ------------------------------------------------------------------ */

function RiskBreakdown({
  entries,
  exit,
  kpis,
}: {
  entries: PyramidEntry[];
  exit: PyramidExit | null;
  kpis: PyramidKPIs;
}) {
  if (!entries.length) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-4 w-4 text-warn" />
          Risk Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="overflow-x-auto">
          <table className="w-full text-body-sm">
            <thead>
              <tr className="border-b border-border text-caption uppercase text-fg-subtle">
                <th className="py-2 text-left font-medium">#</th>
                <th className="py-2 text-left font-medium">Time</th>
                <th className="py-2 text-right font-medium">Entry</th>
                <th className="py-2 text-right font-medium">Lots</th>
                <th className="py-2 text-right font-medium">&Sigma; Lots</th>
                <th className="py-2 text-right font-medium">SL</th>
                <th className="py-2 text-right font-medium">Risk/Lot</th>
                <th className="py-2 text-right font-medium">Risk (INR)</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => {
                const riskPerLot = e.price - e.sl;
                const riskInr = riskPerLot * e.lots * kpis.lot_size;
                return (
                  <tr key={i} className="border-b border-border/50 hover:bg-surface-2/50 transition-colors">
                    <td className="py-2 font-mono">
                      <Badge tone={i === 0 ? "brand" : "success"}>
                        {i === 0 ? "Entry" : `P${i}`}
                      </Badge>
                    </td>
                    <td className="py-2 font-mono text-fg-muted">{fmtTime(e.t)}</td>
                    <td className="py-2 text-right font-mono tabular">{fmtNum(e.price, 2)}</td>
                    <td className="py-2 text-right font-mono tabular">+{e.lots}</td>
                    <td className="py-2 text-right font-mono tabular font-semibold">{e.cumulative_lots}</td>
                    <td className="py-2 text-right font-mono tabular text-warn">{fmtNum(e.sl, 2)}</td>
                    <td className="py-2 text-right font-mono tabular">{fmtNum(riskPerLot, 2)}</td>
                    <td className="py-2 text-right font-mono tabular text-pnl-down">{fmtInr(riskInr)}</td>
                  </tr>
                );
              })}
            </tbody>
            {exit && (
              <tfoot>
                <tr className="border-t-2 border-border">
                  <td colSpan={2} className="py-2 font-semibold text-fg">
                    Exit ({exit.reason})
                  </td>
                  <td className="py-2 text-right font-mono tabular font-semibold">{fmtNum(exit.price, 2)}</td>
                  <td className="py-2 text-right font-mono tabular">{exit.lots}</td>
                  <td colSpan={2} />
                  <td className="py-2 text-right text-caption text-fg-subtle">Total P&L</td>
                  <td className={cn("py-2 text-right font-mono tabular font-semibold", clsPnl(kpis.total_pnl_inr))}>
                    {fmtInr(kpis.total_pnl_inr)}
                  </td>
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
