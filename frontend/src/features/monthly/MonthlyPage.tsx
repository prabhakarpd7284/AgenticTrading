/**
 * Monthly — post-trade feedback loop.
 *
 * Where the Cascade stages (1-6) answer "what should I trade today?", this
 * page answers "what did I actually earn / risk / deploy this month?".  It's
 * the dashboard a trader opens at end-of-day or end-of-week to close the
 * learning loop — see whether the plan + execution actually compounded.
 *
 * Layout
 * ────────────────────────────────────────────────────────────────
 *  [ Header — month picker · paper badge · refresh                 ]
 *  [ YTD strip — headline stats + 12-month bar chart (click to jump)]
 *  [ Current-month card — totals + asset-class tabs                 ]
 *  [   Cash | F&O | Commodity                                        ]
 *  [     Underlying roll-ups (expandable → leg detail)               ]
 *
 * Design choices
 * ──────────────
 *  • Three-level drill (Month → Asset → Underlying → Legs) keeps the
 *    common case — "what did each stock earn?" — one glance away.
 *  • YTD strip is a *navigator*, not a summary — clicking a bar reseats
 *    the month card below without a page reload.
 *  • Everything flows from a single `useMonthlyView()` hook; the PAPER
 *    badge reads off the payload so once a live endpoint lands and a
 *    user is on live-mode the label flips automatically.
 */
import * as React from "react";
import {
  AlertTriangle, BarChart3, Calendar, ChevronDown, ChevronUp,
  FlaskConical, Lightbulb, LineChart, RefreshCcw, Shield, Target,
  TrendingDown, TrendingUp,
} from "lucide-react";

import {
  useMonthlyView,
  getMonthlySource, setMonthlySource,
  ASSET_CLASS_LABEL,
  type Analytics, type AssetClass, type BenchmarkComparison,
  type EquityCurve, type MonthGroup, type MonthlyPayload, type PositionLeg,
  type RejectionReview, type SignalAudit, type StockCapture,
  type UnderlyingRoll, type YtdMonthBar, type YtdSummary,
} from "@/lib/monthly";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { clsPnl, cn, fmtInr, fmtNum, fmtPct } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { OpButton } from "@/features/ops/OpButton";
import { EmptyState } from "@/components/ui/EmptyState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";
import { TradeChartModal } from "./TradeChartModal";

/* ================================================================== */
/* Page                                                                 */
/* ================================================================== */

/**
 * Refresh-control context — every card on the monthly page reads the
 * same `dataUpdatedAt` + `onRefresh` from here and renders a small
 * freshness + refresh control in its top-right corner. Single source of
 * truth means clicking refresh on ANY card re-fetches the whole report.
 */
interface MonthlyRefreshCtx {
  dataUpdatedAt: number;
  isFetching: boolean;
  onRefresh: () => void;
  // Sticky context used by every empty state to render a "populate me"
  // OpButton that triggers the right upstream CLI for the viewed month.
  monthStart: string;
  monthEnd: string;
  onOpFinished: () => void;
}
const MonthlyRefreshContext = React.createContext<MonthlyRefreshCtx | null>(null);
function useMonthlyCtx(): MonthlyRefreshCtx {
  const ctx = React.useContext(MonthlyRefreshContext);
  if (!ctx) throw new Error("useMonthlyCtx must be inside MonthlyRefreshContext.Provider");
  return ctx;
}

/**
 * Action-oriented empty state. When a section has no data, we tell the
 * operator WHICH upstream CLI populates it and let them trigger it inline
 * (no terminal, no recipe-hunting). Both buttons auto-invalidate the
 * monthly view on success so the section fills in.
 */
function PopulateEmpty({
  title, description,
  primaryLabel, primaryCommand, primaryArgs,
  secondaryLabel, secondaryCommand, secondaryArgs,
}: {
  title: string;
  description: string;
  primaryLabel: string;
  primaryCommand: string;
  primaryArgs?: string;
  secondaryLabel?: string;
  secondaryCommand?: string;
  secondaryArgs?: string;
}) {
  const ctx = useMonthlyCtx();
  const defArgs = primaryArgs
    ?? `--backtest --from ${ctx.monthStart} --to ${ctx.monthEnd} --persist-signals`;
  return (
    <EmptyState
      title={title}
      description={description}
      action={
        <div className="flex items-center gap-2 flex-wrap justify-center">
          <OpButton
            command={primaryCommand}
            defaultArgs={defArgs}
            label={primaryLabel}
            description={`Triggers ${primaryCommand} ${defArgs} — populates this section.`}
            onSuccess={ctx.onOpFinished}
          />
          {secondaryCommand && (
            <OpButton
              command={secondaryCommand}
              defaultArgs={secondaryArgs ?? ""}
              label={secondaryLabel ?? secondaryCommand}
              description={`Triggers ${secondaryCommand} ${secondaryArgs ?? ""}.`}
              onSuccess={ctx.onOpFinished}
            />
          )}
        </div>
      }
    />
  );
}

function SectionTools({ className }: { className?: string }) {
  const ctx = React.useContext(MonthlyRefreshContext);
  if (!ctx) return null;
  return (
    <div className={cn("flex items-center gap-1.5 shrink-0", className)}>
      <FreshnessIndicator
        timestamp={ctx.dataUpdatedAt}
        freshMs={5 * 60_000}
        staleMs={60 * 60_000}
        variant="muted"
        label=""
      />
      <Button
        variant="ghost" size="icon"
        onClick={ctx.onRefresh}
        disabled={ctx.isFetching}
        aria-label="Refresh this section"
        title="Refresh — re-runs the whole monthly view"
      >
        <RefreshCcw className={cn("h-3.5 w-3.5", ctx.isFetching && "animate-spin")} />
      </Button>
    </div>
  );
}

export function MonthlyPage() {
  const queryClient = useQueryClient();

  const [source, setSource] = React.useState(getMonthlySource);

  const [selectedMonth, setSelectedMonth] = React.useState<string | null>(
    () => new URLSearchParams(window.location.search).get("month"),
  );

  // The focused month drives which month the backend computes the feedback
  // sections for — the YTD strip + month list always come back full, so the
  // bar chart stays a complete navigator regardless of what's focused.
  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } =
    useMonthlyView(selectedMonth);

  // Sync selected month to URL
  const handleSelectMonth = React.useCallback((m: string) => {
    setSelectedMonth(m);
    const url = new URL(window.location.href);
    url.searchParams.set("month", m);
    window.history.replaceState(null, "", url.toString());
  }, []);

  const handleToggleSource = React.useCallback(() => {
    const next = source === "mock" ? "live" : "mock";
    setMonthlySource(next);
    setSource(next);
    queryClient.invalidateQueries({ queryKey: ["monthly-view"] });
  }, [source, queryClient]);

  // Force-refresh: server caches the monthly report for 120s, so plain
  // refetch() inside that window returns the same payload. Hit the endpoint
  // with ?force=1 to bust the server cache, then invalidate React Query.
  const handleForceRefresh = React.useCallback(async () => {
    if (source !== "live") {
      await refetch();
      return;
    }
    try {
      // Bust the server cache for the *focused* month — the cache key is
      // per-month, so a force without ?month= would only refresh the default.
      const q = selectedMonth ? `&month=${encodeURIComponent(selectedMonth)}` : "";
      await api.get(`portfolios/monthly/?force=1${q}`);
    } catch {
      // Server might 5xx; we still invalidate so the user sees an error state.
    }
    queryClient.invalidateQueries({ queryKey: ["monthly-view"] });
  }, [source, refetch, queryClient, selectedMonth]);

  // Stable callback for op-buttons — has to live above the conditional
  // returns below, otherwise the hook count changes between renders
  // (loading → ready) and React throws "Rendered more hooks than ...".
  const ctxOpFinished = React.useCallback(
    () => queryClient.invalidateQueries({ queryKey: ["monthly-view"] }),
    [queryClient],
  );

  // Derive everything needed for the context BEFORE the conditional return,
  // otherwise the hook count changes between loading and ready states.
  const monthKey = selectedMonth ?? data?.current_month ?? "";
  const month = React.useMemo(
    () => (data ? data.months.find((m) => m.month === monthKey) ?? data.months[0] : null),
    [data, monthKey],
  );

  // Derive --from / --to for the currently-viewed month — used by both
  // the Header op-buttons and the empty-state op-buttons.
  const { ctxMonthStart, ctxMonthEnd } = React.useMemo(() => {
    if (!monthKey) return { ctxMonthStart: "", ctxMonthEnd: "" };
    const [year, mon] = monthKey.split("-").map(Number);
    const lastDay = new Date(Date.UTC(year, mon, 0)).getUTCDate();
    const mm = String(mon).padStart(2, "0");
    return {
      ctxMonthStart: `${year}-${mm}-01`,
      ctxMonthEnd: `${year}-${mm}-${String(lastDay).padStart(2, "0")}`,
    };
  }, [monthKey]);

  const ctxValue = React.useMemo<MonthlyRefreshCtx>(
    () => ({
      dataUpdatedAt, isFetching, onRefresh: handleForceRefresh,
      monthStart: ctxMonthStart, monthEnd: ctxMonthEnd, onOpFinished: ctxOpFinished,
    }),
    [dataUpdatedAt, isFetching, handleForceRefresh, ctxMonthStart, ctxMonthEnd, ctxOpFinished],
  );

  if (isLoading) return <MonthlyLoading />;
  if (isError) return <MonthlyError error={error as Error} onRetry={() => refetch()} />;
  if (!data || !month) return null;

  return (
    <MonthlyRefreshContext.Provider value={ctxValue}>
      <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
        <Header
          data={data}
          onRefresh={handleForceRefresh}
          isFetching={isFetching}
          dataUpdatedAt={dataUpdatedAt}
          source={source}
          onToggleSource={handleToggleSource}
          monthKey={monthKey}
          onOpFinished={() => queryClient.invalidateQueries({ queryKey: ["monthly-view"] })}
        />

        <PipelineStatusStrip data={data} month={month} />

        <TradeFreshnessBanner data={data} />

        <YtdStrip
          ytd={data.ytd}
          selected={monthKey}
          onSelect={handleSelectMonth}
        />

        <MonthCard
          month={month}
          isCurrent={month.month === data.current_month}
        />

        {/* Row: Equity curve + Benchmark side-by-side */}
        <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6">
          <EquityCurveCard curve={data.equity_curve} />
          <BenchmarkCard benchmark={data.benchmark} />
        </div>

        {/* Analytics: hour heatmap + day-of-week + sector */}
        <AnalyticsCard analytics={data.analytics} />

        <CaptureMatrixCard matrix={data.capture_matrix} />

        <SignalAuditCard audit={data.signal_audit} />

        {data.rejections.length > 0 && (
          <RejectionsCard rejections={data.rejections} />
        )}

        <LessonsCard lessons={data.lessons} />
      </div>
    </MonthlyRefreshContext.Provider>
  );
}

/* ================================================================== */
/* Header                                                               */
/* ================================================================== */

function Header({
  data, onRefresh, isFetching, dataUpdatedAt, source, onToggleSource,
  monthKey, onOpFinished,
}: {
  data: MonthlyPayload;
  onRefresh: () => void;
  isFetching: boolean;
  dataUpdatedAt: number;
  source: "mock" | "live";
  onToggleSource: () => void;
  monthKey: string;
  onOpFinished: () => void;
}) {
  // Derive --from / --to for the currently-viewed month so the backtest
  // ops drawer opens with the right window pre-filled.
  const [year, mon] = monthKey.split("-").map(Number);
  const monthStart = `${year}-${String(mon).padStart(2, "0")}-01`;
  // Last day of month: roll to next month then back one day.
  const lastDay = new Date(Date.UTC(year, mon, 0)).getUTCDate();
  const monthEnd = `${year}-${String(mon).padStart(2, "0")}-${String(lastDay).padStart(2, "0")}`;
  return (
    <header className="flex items-end justify-between gap-4 flex-wrap">
      <div className="min-w-0">
        <p className="text-caption uppercase tracking-wider text-fg-subtle">
          Post-trade · Monthly feedback report
        </p>
        <h1 className="text-h1 text-fg">Monthly</h1>
        <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
          Every position grouped by month and asset class.  See exposure,
          capital deployed, risk, and running P&amp;L at a glance — drill into
          any underlying to audit individual legs.
        </p>
      </div>
      <div className="flex items-center gap-2">
        {/* Data source toggle */}
        <button
          type="button"
          onClick={onToggleSource}
          className={cn(
            "flex items-center gap-1.5 rounded-sm border px-2 py-1 text-caption transition-colors",
            source === "live"
              ? "border-pnl-up/40 bg-pnl-up/10 text-pnl-up"
              : "border-border/60 bg-surface-2 text-fg-subtle hover:text-fg",
          )}
          title={source === "mock"
            ? "Using mock data — click to switch to live backend"
            : "Using live backend — click to switch to mock data"
          }
        >
          <span className={cn(
            "h-1.5 w-1.5 rounded-full",
            source === "live" ? "bg-pnl-up animate-pulse" : "bg-fg-subtle",
          )} />
          {source === "live" ? "Live" : "Mock"}
        </button>

        {data.paper_mode && (
          <Badge tone="warning" dot title="All figures are from paper trades">
            <FlaskConical className="h-3 w-3" aria-hidden />
            Paper
          </Badge>
        )}
        {/* Reports are aggregated EOD — softer thresholds (5min fresh, 1h
            stale) match how often the data could realistically change. */}
        <FreshnessIndicator
          label="Report as of"
          timestamp={dataUpdatedAt}
          freshMs={5 * 60_000}
          staleMs={60 * 60_000}
        />

        {/* Replay the live screener over the viewed month — populates
            apps.strategies.Signal rows for every strategy fired. The
            capture matrix + signal audit + rejections sections all depend
            on this. Run first, then "Refresh signals" to compute outcomes. */}
        <OpButton
          command="run_screener"
          defaultArgs={`--backtest --from ${monthStart} --to ${monthEnd} --persist-signals`}
          label="Run screener"
          description={`Replay 8 intraday strategies over ${monthStart} → ${monthEnd} and persist every signal that fired. Populates the capture matrix / signal audit / rejections sections.`}
          onSuccess={onOpFinished}
        />

        {/* Re-run the signal-outcome enrichment (capture rates, win/loss labels)
            then re-fetch the monthly view so the new numbers appear inline. */}
        <OpButton
          command="enrich_signals"
          defaultArgs="--all"
          label="Refresh signals"
          description="Backfill EOD outcomes (max favorable / adverse move, traded/expired/rejected outcome) for every apps.strategies.Signal row — drives the capture matrix + signal audit on this page."
          onSuccess={onOpFinished}
        />

        {/* Derive paper trades for the viewed month by replaying the intraday
            agent over real candles. Populates the trade-driven sections —
            month P&L, equity curve, analytics, benchmark. */}
        <OpButton
          command="derive_trades"
          defaultArgs={`--from ${monthStart} --to ${monthEnd}`}
          label="Derive intraday"
          description={`Replay the intraday agent over ${monthStart} → ${monthEnd} — creates paper trades (FILLED → CLOSED with P&L) from real intraday structure. Feeds the month P&L / equity curve / analytics. Skips days that already have trades.`}
          onSuccess={onOpFinished}
        />

        {/* Derive swing (Oliver Kell) trades for the month — the other trade
            source feeding the cash bucket. Reuses the /backtester engine. */}
        <OpButton
          command="derive_swing_trades"
          defaultArgs={`--from ${monthStart} --to ${monthEnd}`}
          label="Derive swing"
          description={`Run the Oliver Kell swing backtest over ${monthStart} → ${monthEnd} and book each closed swing trade (cash, multi-day) into the Monthly report. Idempotent — replaces prior swing rows for the window.`}
          onSuccess={onOpFinished}
        />

        {/* Re-backtest the swing strategy for the currently-viewed month. */}
        <OpButton
          command="run_ok_backtest"
          defaultArgs={`--from ${monthStart} --to ${monthEnd}`}
          label="Backtest swing"
          description={`Run the Oliver-Kell cycle backtest over ${monthStart} → ${monthEnd}.`}
          onSuccess={onOpFinished}
        />

        <Button
          variant="ghost"
          size="icon"
          onClick={onRefresh}
          aria-label="Refresh monthly view"
          disabled={isFetching}
        >
          <RefreshCcw className={cn("h-4 w-4", isFetching && "animate-spin")} />
        </Button>
      </div>
    </header>
  );
}

/* ================================================================== */
/* Trade-freshness banner                                                */
/* ------------------------------------------------------------------- */
/* Signals flow automatically from the scan pipeline, but trades only    */
/* appear once the intraday agent (live or replay) runs. When signals    */
/* are newer than the last derived trade, the P&L-driven sections look   */
/* stale — so we say so explicitly and offer a one-click backfill that   */
/* replays the agent over the viewed month (same OpButton pattern as the */
/* "Run screener" signal backfill).                                      */
/* ================================================================== */

function TradeFreshnessBanner({ data }: { data: MonthlyPayload }) {
  const ctx = useMonthlyCtx();
  const fresh = data.data_freshness;
  if (!fresh || !fresh.trades_stale) return null;

  return (
    <div className="flex items-start gap-3 rounded-sm border border-warning/40 bg-warning/10 px-4 py-3">
      <AlertTriangle className="h-4 w-4 text-warning shrink-0 mt-0.5" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-body-sm text-fg">
          Signals are current through{" "}
          <span className="font-mono">{fresh.latest_signal_date ?? "—"}</span>, but trades have only
          been derived through{" "}
          <span className="font-mono">{fresh.latest_trade_date ?? "never"}</span>. The P&amp;L, equity
          curve, and analytics below reflect derived trades only.
        </p>
        <p className="text-caption text-fg-muted mt-0.5">
          Replays the intraday agent (and run “Derive swing” in the header for Oliver Kell trades)
          over the viewed month to create paper trades from real structure.
        </p>
      </div>
      <OpButton
        command="derive_trades"
        defaultArgs={`--from ${ctx.monthStart} --to ${ctx.monthEnd}`}
        label="Derive intraday"
        description={`Replay the intraday agent over ${ctx.monthStart} → ${ctx.monthEnd} — creates paper trades (FILLED → CLOSED with P&L) from real intraday structure. Feeds the P&L / equity curve / analytics / benchmark sections.`}
        onSuccess={ctx.onOpFinished}
      />
    </div>
  );
}

/* ================================================================== */
/* Pipeline status strip                                                 */
/* ------------------------------------------------------------------- */
/* Shows the health of each upstream data source feeding this page —    */
/* trades, signals, broker snapshots. Any tile that says "0 / stale"    */
/* tells the operator which CLI they need to run to populate that part  */
/* of the report.                                                       */
/* ================================================================== */

function PipelineStatusStrip({
  data, month,
}: {
  data: MonthlyPayload;
  month: MonthGroup;
}) {
  const sigTotal = data.signal_audit.total_signals;
  const tradeCount = month.trade_count;
  const captureSymbols = data.capture_matrix.length;
  const rejectionsCount = data.rejections.length;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <PipelineTile
        label="Trades in scope"
        value={tradeCount.toString()}
        status={tradeCount > 0 ? "ok" : "empty"}
        emptyHint="No trades for this month yet"
        source="apps.trading.Trade"
      />
      <PipelineTile
        label="Signals recorded"
        value={sigTotal.toString()}
        status={sigTotal > 0 ? "ok" : "empty"}
        emptyHint="Run the screener to populate"
        source="apps.strategies.Signal"
      />
      <PipelineTile
        label="Stocks tracked"
        value={captureSymbols.toString()}
        status={captureSymbols > 0 ? "ok" : "empty"}
        emptyHint="Needs enriched signals"
        source="Signal × Trade join"
      />
      <PipelineTile
        label="Risk rejections"
        value={rejectionsCount.toString()}
        status={rejectionsCount > 0 ? "ok" : "muted"}
        emptyHint="Nothing blocked this month"
        source="apps.events.Event"
      />
    </div>
  );
}

function PipelineTile({
  label, value, status, emptyHint, source,
}: {
  label: string;
  value: string;
  status: "ok" | "empty" | "muted";
  emptyHint: string;
  source: string;
}) {
  const tone = status === "ok" ? "text-fg" : status === "empty" ? "text-warning" : "text-fg-muted";
  const dot = status === "ok" ? "bg-success" : status === "empty" ? "bg-warning" : "bg-fg-subtle";
  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-1.5 text-caption uppercase tracking-wider text-fg-subtle">
          <span className={cn("h-1.5 w-1.5 rounded-full", dot)} />
          {label}
        </div>
        <div className={cn("text-h2 mt-1 tabular-nums", tone)}>{value}</div>
        <div className="text-caption text-fg-subtle mt-1">
          {status === "ok" ? source : emptyHint}
        </div>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* YTD strip — headline stats + 12-month bar chart                       */
/* ================================================================== */

function YtdStrip({
  ytd, selected, onSelect,
}: {
  ytd: YtdSummary;
  selected: string;
  onSelect: (month: string) => void;
}) {
  const pctOnCapital = ytd.capital_base > 0
    ? (ytd.total_pnl / ytd.capital_base) * 100
    : 0;

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-fg-muted" aria-hidden />
            Year-to-date
          </CardTitle>
          <CardDescription>
            This financial year (Apr onward) · click a bar to inspect any month below
          </CardDescription>
        </div>
        <div className="flex items-center gap-2 text-caption uppercase tracking-wider text-fg-subtle">
          Capital base {fmtInr(ytd.capital_base, { compact: true })}
          <SectionTools />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <YtdStat label="YTD P&L"      value={fmtInr(ytd.total_pnl)}      tone={ytd.total_pnl} suffix={fmtPct(pctOnCapital, 2)} />
          <YtdStat label="Trades"       value={fmtNum(ytd.trade_count)} />
          <YtdStat label="Win rate"     value={`${(ytd.win_rate * 100).toFixed(0)}%`} />
          <YtdStat label="Best · Worst" value={`${ytd.best_month.slice(5)} · ${ytd.worst_month.slice(5)}`} mono />
        </div>

        <BarChart months={ytd.months} selected={selected} onSelect={onSelect} />
      </CardContent>
    </Card>
  );
}

function YtdStat({
  label, value, tone, suffix, mono,
}: {
  label: string;
  value: string;
  tone?: number;
  suffix?: string;
  mono?: boolean;
}) {
  const toneCls = tone == null ? "text-fg" : clsPnl(tone);
  return (
    <div className="rounded-sm border border-border/60 p-3">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn(
        "text-h3 mt-0.5",
        toneCls,
        mono && "font-mono tabular",
      )}>
        {value}
      </div>
      {suffix && (
        <div className={cn("text-caption mt-0.5", toneCls)}>{suffix}</div>
      )}
    </div>
  );
}

function BarChart({
  months, selected, onSelect,
}: {
  months: YtdMonthBar[];
  selected: string;
  onSelect: (m: string) => void;
}) {
  const max = Math.max(...months.map((m) => Math.abs(m.pnl)), 1);
  return (
    <div
      role="group"
      aria-label="Financial-year P&L by month"
      className="flex items-end gap-2 h-28 border-t border-dashed border-border/60 pt-2"
    >
      {months.map((m) => {
        const h = (Math.abs(m.pnl) / max) * 100;
        const isPos = m.pnl >= 0;
        const isSelected = m.month === selected;
        return (
          <button
            type="button"
            key={m.month}
            onClick={() => onSelect(m.month)}
            aria-pressed={isSelected}
            aria-label={`${m.month_label} — ${fmtInr(m.pnl)}`}
            className={cn(
              "group flex-1 flex flex-col items-center gap-1 h-full justify-end",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 rounded-xs",
            )}
          >
            <span
              className={cn(
                "w-full rounded-t-xs transition-[height,background-color] duration-120",
                isPos ? "bg-pnl-up/60 group-hover:bg-pnl-up" : "bg-pnl-down/60 group-hover:bg-pnl-down",
                isSelected && (isPos ? "bg-pnl-up ring-2 ring-pnl-up" : "bg-pnl-down ring-2 ring-pnl-down"),
              )}
              style={{ height: `${Math.max(6, h)}%` }}
            />
            <span className={cn(
              "text-caption",
              isSelected ? "text-fg font-semibold" : "text-fg-subtle",
            )}>
              {m.month_label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ================================================================== */
/* Month card — totals + asset-class tabs                                */
/* ================================================================== */

function MonthCard({ month, isCurrent }: { month: MonthGroup; isCurrent: boolean }) {
  const totalPnlCls = clsPnl(month.total_pnl);
  const hasCash = month.by_asset_class.cash.length > 0;
  const hasFno = month.by_asset_class.fno.length > 0;
  const hasCommodity = month.by_asset_class.commodity.length > 0;

  const defaultTab: AssetClass =
    hasCash ? "cash" : hasFno ? "fno" : "commodity";

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div className="min-w-0">
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-fg-muted" aria-hidden />
            {month.month_label}
            {isCurrent && <Badge tone="brand">Current</Badge>}
          </CardTitle>
          <CardDescription>
            {month.trade_count} trade{month.trade_count === 1 ? "" : "s"} ·
            {" "}win rate {(month.win_rate * 100).toFixed(0)}% ·
            {" "}capital deployed {fmtInr(month.capital_deployed, { compact: true })}
          </CardDescription>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <div className="flex items-center gap-2">
            <div className={cn("text-h2 font-mono tabular", totalPnlCls)}>
              {formatSignedInr(month.total_pnl)}
            </div>
            <SectionTools />
          </div>
          <div className="flex items-center gap-2 text-caption text-fg-subtle">
            <span>Realised {formatSignedInr(month.realized_pnl)}</span>
            <span aria-hidden>·</span>
            <span>Open {formatSignedInr(month.unrealized_pnl)}</span>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue={defaultTab}>
          <TabsList>
            <TabsTrigger value="cash" disabled={!hasCash}>
              <span className="flex items-center gap-2">
                Cash <CountPill n={month.by_asset_class.cash.length} />
              </span>
            </TabsTrigger>
            <TabsTrigger value="fno" disabled={!hasFno}>
              <span className="flex items-center gap-2">
                F&amp;O <CountPill n={month.by_asset_class.fno.length} />
              </span>
            </TabsTrigger>
            <TabsTrigger value="commodity" disabled={!hasCommodity}>
              <span className="flex items-center gap-2">
                Commodity <CountPill n={month.by_asset_class.commodity.length} />
              </span>
            </TabsTrigger>
          </TabsList>

          {(["cash", "fno", "commodity"] as AssetClass[]).map((ac) => {
            const rolls = month.by_asset_class[ac];
            return (
              <TabsContent key={ac} value={ac}>
                {rolls.length === 0 ? (
                  <EmptyState
                    title={`No ${ASSET_CLASS_LABEL[ac]} positions this month`}
                    description="Trades logged in this bucket will appear here."
                  />
                ) : (
                  <UnderlyingTable rolls={rolls} />
                )}
              </TabsContent>
            );
          })}
        </Tabs>
      </CardContent>
    </Card>
  );
}

function CountPill({ n }: { n: number }) {
  return (
    <span className="inline-flex items-center justify-center min-w-4 h-4 rounded-xs text-caption text-fg-subtle bg-surface-2 px-1">
      {n}
    </span>
  );
}

/* ================================================================== */
/* Underlying table — expandable rows                                    */
/* ================================================================== */

function UnderlyingTable({ rolls }: { rolls: UnderlyingRoll[] }) {
  const sorted = React.useMemo(
    () => [...rolls].sort((a, b) => b.running_pnl - a.running_pnl),
    [rolls],
  );
  return (
    <div className="divide-y divide-border/60">
      <Header4Col />
      {sorted.map((r) => <UnderlyingRow key={r.underlying + r.asset_class} r={r} />)}
      <Totals rolls={sorted} />
    </div>
  );
}

function Header4Col() {
  return (
    <div className="hidden md:grid grid-cols-[minmax(160px,1.4fr)_repeat(5,1fr)_32px] gap-3 py-2 text-caption uppercase tracking-wider text-fg-subtle">
      <span>Underlying</span>
      <span className="text-right">Capital</span>
      <span className="text-right">Exposure</span>
      <span className="text-right">Target ₹</span>
      <span className="text-right">Risk ₹</span>
      <span className="text-right">Running P&amp;L</span>
      <span />
    </div>
  );
}

function UnderlyingRow({ r }: { r: UnderlyingRoll }) {
  const [open, setOpen] = React.useState(false);
  const pnlCls = clsPnl(r.running_pnl);
  const rrLabel = r.avg_rr > 0 ? `${r.avg_rr.toFixed(2)}:1` : "—";
  const hasLegs = r.legs.length > 0;

  return (
    <div>
      <button
        type="button"
        onClick={() => hasLegs && setOpen((o) => !o)}
        disabled={!hasLegs}
        aria-expanded={open}
        className={cn(
          "w-full text-left grid grid-cols-[minmax(160px,1.4fr)_repeat(5,1fr)_32px] gap-3 py-3",
          "items-center",
          "focus-visible:outline-none focus-visible:bg-surface-2",
          hasLegs && "hover:bg-surface-2 cursor-pointer",
        )}
      >
        <div className="min-w-0 flex flex-col">
          <span className="font-mono text-body text-fg">{r.underlying}</span>
          <span className="text-caption text-fg-subtle">
            {r.trade_count} trade{r.trade_count === 1 ? "" : "s"} ·
            {" "}win {r.winning_trades}/{r.winning_trades + r.losing_trades || 0} ·
            {" "}R:R {rrLabel} ·
            {" "}{r.days_in_position}d ({Math.round(r.pct_of_month * 100)}% of month)
          </span>
        </div>
        <Num value={fmtInr(r.capital_deployed, { compact: true })} />
        <Num value={fmtInr(r.exposure, { compact: true })} />
        <Num value={fmtInr(r.target_total, { compact: true })} tone="success" />
        <Num value={fmtInr(r.risk_total, { compact: true })} tone="danger" />
        <Num
          value={formatSignedInr(r.running_pnl)}
          className={cn("font-semibold", pnlCls)}
          showTrend={r.running_pnl}
        />
        <span className="text-fg-subtle flex justify-end">
          {hasLegs ? (open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />) : null}
        </span>
      </button>

      {open && hasLegs && (
        <LegList legs={r.legs} />
      )}
    </div>
  );
}

function Num({
  value, className, tone, showTrend,
}: {
  value: string;
  className?: string;
  tone?: "success" | "danger" | "neutral";
  showTrend?: number;
}) {
  const toneCls =
    tone === "success" ? "text-pnl-up" :
    tone === "danger"  ? "text-pnl-down" :
    undefined;
  return (
    <span className={cn(
      "text-right font-mono tabular text-body-sm text-fg flex items-center justify-end gap-1",
      toneCls,
      className,
    )}>
      {showTrend != null && showTrend !== 0 && (
        showTrend > 0
          ? <TrendingUp className="h-3 w-3" aria-hidden />
          : <TrendingDown className="h-3 w-3" aria-hidden />
      )}
      {value}
    </span>
  );
}

function Totals({ rolls }: { rolls: UnderlyingRoll[] }) {
  const { cap, expo, tgt, risk, pnl } = React.useMemo(() => {
    let cap = 0, expo = 0, tgt = 0, risk = 0, pnl = 0;
    for (const r of rolls) {
      cap += r.capital_deployed;
      expo += r.exposure;
      tgt += r.target_total;
      risk += r.risk_total;
      pnl += r.running_pnl;
    }
    return { cap, expo, tgt, risk, pnl };
  }, [rolls]);
  return (
    <div className="grid grid-cols-[minmax(160px,1.4fr)_repeat(5,1fr)_32px] gap-3 py-3 border-t border-border bg-surface-2/50 text-body-sm">
      <span className="uppercase tracking-wider text-caption text-fg-subtle self-center">Totals</span>
      <Num value={fmtInr(cap, { compact: true })} />
      <Num value={fmtInr(expo, { compact: true })} />
      <Num value={fmtInr(tgt, { compact: true })} tone="success" />
      <Num value={fmtInr(risk, { compact: true })} tone="danger" />
      <Num
        value={formatSignedInr(pnl)}
        className={cn("font-semibold", clsPnl(pnl))}
      />
      <span />
    </div>
  );
}

/* ================================================================== */
/* Leg detail — drawer inside an underlying row                          */
/* ================================================================== */

/* Outcome → badge tone. */
const OUTCOME_TONE: Record<string, "success" | "danger" | "neutral"> = {
  TARGET_HIT: "success", SL_HIT: "danger", EOD: "neutral", MANUAL: "neutral", TRAIL: "neutral",
};

/* Shared 6-column template. Literal strings (not interpolated) so Tailwind's
   JIT can see them. */
const LEG_GRID = "grid-cols-[1.5fr_0.9fr_0.9fr_1.2fr_0.9fr_auto]";
const LEG_GRID_MD = "md:grid-cols-[1.5fr_0.9fr_0.9fr_1.2fr_0.9fr_auto]";

function LegList({ legs }: { legs: PositionLeg[] }) {
  const sorted = React.useMemo(
    () => [...legs].sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl)),
    [legs],
  );
  // The chart modal lives here (not per-row) so ←/→ can page through the
  // whole table by index.
  const [openIdx, setOpenIdx] = React.useState<number | null>(null);

  return (
    <div className="px-3 pb-4 pt-1 bg-surface-2/30 rounded-b-sm">
      <div className={cn(
        "hidden md:grid gap-3 px-2 py-2 text-caption uppercase tracking-wider text-fg-subtle border-b border-border/60",
        LEG_GRID,
      )}>
        <span>Trade</span>
        <span className="text-right">Entry · Qty</span>
        <span className="text-right">Exit</span>
        <span>Outcome</span>
        <span className="text-right">P&amp;L</span>
        <span className="text-right">Chart</span>
      </div>
      <div className="divide-y divide-border/60">
        {sorted.map((l, i) => (
          <LegRow key={l.id} l={l} onView={() => setOpenIdx(i)} />
        ))}
      </div>
      <TradeChartModal legs={sorted} index={openIdx} onIndexChange={setOpenIdx} />
    </div>
  );
}

function LegRow({ l, onView }: { l: PositionLeg; onView: () => void }) {
  const closed = l.status === "CLOSED";
  const outcomeTone = OUTCOME_TONE[l.close_reason ?? ""] ?? "neutral";

  return (
    <>
      <div className={cn(
        "grid grid-cols-2 md:gap-3 gap-y-2 gap-x-3 px-2 py-2.5 items-center text-body-sm",
        LEG_GRID_MD,
      )}>
        {/* Trade — symbol, side, source, rationale */}
        <div className="col-span-2 md:col-span-1 min-w-0 flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <span className="font-mono text-fg">{l.symbol}</span>
            <Badge tone={l.side === "BUY" ? "success" : "danger"}>{l.side}</Badge>
            {l.source && (
              <span className="text-caption text-fg-subtle capitalize">{l.source}</span>
            )}
          </div>
          {l.notes && (
            <span className="text-caption text-fg-muted truncate" title={l.notes}>{l.notes}</span>
          )}
        </div>

        {/* Entry · Qty */}
        <div className="text-right font-mono tabular">
          <div className="text-fg">{fmtNum(l.entry_price, 2)}</div>
          <div className="text-caption text-fg-subtle">{l.quantity} qty</div>
        </div>

        {/* Exit · date */}
        <div className="text-right font-mono tabular">
          <div className="text-fg">{l.exit_price != null ? fmtNum(l.exit_price, 2) : "—"}</div>
          <div className="text-caption text-fg-subtle">{(l.exit_date ?? l.entry_date).slice(5)}</div>
        </div>

        {/* Outcome */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <Badge tone={closed ? "neutral" : "info"}>{l.status}</Badge>
          {l.close_reason && (
            <Badge tone={outcomeTone}>{l.close_reason.replace("_", " ")}</Badge>
          )}
        </div>

        {/* P&L */}
        <div className={cn("text-right font-mono tabular font-semibold", clsPnl(l.pnl))}>
          {formatSignedInr(l.pnl)}
        </div>

        {/* Actions */}
        <div className="flex justify-end">
          <Button
            variant="ghost" size="sm"
            onClick={onView}
            title="View this trade on a chart"
          >
            <LineChart className="h-3.5 w-3.5 md:mr-1.5" aria-hidden />
            <span className="hidden md:inline">View chart</span>
          </Button>
        </div>
      </div>
    </>
  );
}

/* ================================================================== */
/* Equity Curve + Drawdown                                               */
/* ================================================================== */

function EquityCurveCard({ curve }: { curve: EquityCurve }) {
  const pts = curve.points;
  if (pts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Equity Curve</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState title="No trade data" description="Equity curve requires closed trades." />
        </CardContent>
      </Card>
    );
  }

  const maxAbs = Math.max(...pts.map((p) => Math.abs(p.cumulative)), 1);
  const minDD = Math.min(...pts.map((p) => p.drawdown), 0);

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-fg-muted" aria-hidden />
            Equity Curve
          </CardTitle>
          <CardDescription>
            Day-by-day cumulative P&amp;L with drawdown
          </CardDescription>
        </div>
        <div className="flex gap-4 text-caption text-right items-center">
          <div>
            <div className="text-fg-subtle">Peak</div>
            <div className="text-pnl-up font-mono">{fmtInr(curve.peak_equity)}</div>
          </div>
          <div>
            <div className="text-fg-subtle">Max DD</div>
            <div className="text-pnl-down font-mono">{fmtInr(curve.max_drawdown)}</div>
          </div>
          <div>
            <div className="text-fg-subtle">Final</div>
            <div className={cn("font-mono font-semibold", clsPnl(curve.final_equity))}>
              {fmtInr(curve.final_equity)}
            </div>
          </div>
          <SectionTools />
        </div>
      </CardHeader>
      <CardContent>
        {/* Cumulative P&L bars */}
        <div className="flex items-end gap-1 h-32 border-b border-border/60 mb-1">
          {pts.map((p) => {
            const h = (Math.abs(p.cumulative) / maxAbs) * 100;
            const isPos = p.cumulative >= 0;
            return (
              <div
                key={p.date}
                className="flex-1 flex flex-col justify-end h-full group relative"
                title={`${p.date.slice(5)}: ${p.cumulative >= 0 ? "+" : ""}${fmtInr(p.cumulative)} (${p.trades}t)`}
              >
                <div
                  className={cn(
                    "w-full rounded-t-xs transition-[height]",
                    isPos ? "bg-pnl-up/70 group-hover:bg-pnl-up" : "bg-pnl-down/70 group-hover:bg-pnl-down",
                  )}
                  style={{ height: `${Math.max(4, h)}%` }}
                />
              </div>
            );
          })}
        </div>
        {/* Drawdown area */}
        {minDD < 0 && (
          <div className="flex items-start gap-1 h-10 mb-1">
            {pts.map((p) => {
              const h = minDD !== 0 ? (Math.abs(p.drawdown) / Math.abs(minDD)) * 100 : 0;
              return (
                <div key={p.date} className="flex-1 flex flex-col h-full">
                  <div
                    className="w-full bg-pnl-down/30 rounded-b-xs"
                    style={{ height: `${h}%` }}
                  />
                </div>
              );
            })}
          </div>
        )}
        {/* Date labels */}
        <div className="flex gap-1">
          {pts.map((p, i) => (
            <span
              key={p.date}
              className={cn(
                "flex-1 text-center text-caption text-fg-subtle",
                i % 2 !== 0 && pts.length > 8 && "invisible",
              )}
            >
              {p.date.slice(8)}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Benchmark Comparison                                                  */
/* ================================================================== */

function BenchmarkCard({ benchmark: b }: { benchmark: BenchmarkComparison }) {
  const hasNifty = b.nifty_start > 0;
  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-fg-muted" aria-hidden />
            vs NIFTY50
          </CardTitle>
          <CardDescription>
            {b.trading_days} trading days
          </CardDescription>
        </div>
        <SectionTools />
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Big alpha number */}
        <div className="text-center py-2">
          <div className="text-caption uppercase tracking-wider text-fg-subtle">Alpha</div>
          <div className={cn("text-h1 font-mono tabular", clsPnl(b.alpha_pct))}>
            {b.alpha_pct > 0 ? "+" : ""}{b.alpha_pct.toFixed(2)}%
          </div>
        </div>

        {/* Comparison bars */}
        <div className="space-y-3">
          <ReturnBar label="Portfolio" pct={b.portfolio_return_pct} />
          <ReturnBar label="NIFTY50" pct={b.nifty_return_pct} muted={!hasNifty} />
        </div>

        {hasNifty && (
          <div className="text-caption text-fg-subtle text-center pt-1">
            NIFTY {fmtNum(b.nifty_start, 0)} → {fmtNum(b.nifty_end, 0)}
          </div>
        )}
        {!hasNifty && (
          <div className="text-caption text-fg-muted text-center pt-1">
            NIFTY data unavailable — broker not connected
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ReturnBar({ label, pct, muted }: { label: string; pct: number; muted?: boolean }) {
  const maxBar = 80; // max width %
  const width = Math.min(maxBar, Math.abs(pct) * 15); // scale: 1% = 15px-equiv
  return (
    <div className="flex items-center gap-3">
      <span className={cn("text-caption w-16 shrink-0", muted ? "text-fg-muted" : "text-fg-subtle")}>
        {label}
      </span>
      <div className="flex-1 h-5 bg-surface-2 rounded-xs overflow-hidden relative">
        <div
          className={cn(
            "h-full rounded-xs transition-[width]",
            muted ? "bg-surface-3" : pct >= 0 ? "bg-pnl-up/70" : "bg-pnl-down/70",
          )}
          style={{ width: `${Math.max(2, width)}%` }}
        />
      </div>
      <span className={cn(
        "font-mono tabular text-body-sm w-14 text-right shrink-0",
        muted ? "text-fg-muted" : clsPnl(pct),
      )}>
        {pct > 0 ? "+" : ""}{pct.toFixed(2)}%
      </span>
    </div>
  );
}

/* ================================================================== */
/* Analytics — hour, day-of-week, sector                                 */
/* ================================================================== */

function AnalyticsCard({ analytics: a }: { analytics: Analytics }) {
  const hasData = a.by_hour.some((h) => h.trades > 0);
  if (!hasData) {
    return (
      <Card>
        <CardHeader><CardTitle>Analytics</CardTitle></CardHeader>
        <CardContent>
          <EmptyState title="No analytics data" description="Need trade data to compute breakdowns." />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
          <CardTitle>Analytics</CardTitle>
          <CardDescription>Performance by time of day, day of week, and sector</CardDescription>
        </div>
        <SectionTools />
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Hour heatmap */}
          <div>
            <div className="text-caption uppercase tracking-wider text-fg-subtle mb-3">
              Time of day (IST)
            </div>
            <div className="space-y-1">
              {a.by_hour.map((h) => (
                <div key={h.hour} className="flex items-center gap-2">
                  <span className="text-caption text-fg-subtle w-10 shrink-0 font-mono">{h.label}</span>
                  <div className="flex-1 h-5 bg-surface-2 rounded-xs overflow-hidden relative">
                    {h.trades > 0 && (
                      <div
                        className={cn(
                          "h-full rounded-xs",
                          h.pnl >= 0 ? "bg-pnl-up/60" : "bg-pnl-down/60",
                        )}
                        style={{ width: `${Math.min(100, h.win_rate * 100)}%` }}
                      />
                    )}
                  </div>
                  <span className={cn(
                    "font-mono tabular text-caption w-16 text-right shrink-0",
                    h.trades === 0 ? "text-fg-muted" : clsPnl(h.pnl),
                  )}>
                    {h.trades > 0 ? `${h.trades}t ${(h.win_rate * 100).toFixed(0)}%` : "—"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Day of week */}
          <div>
            <div className="text-caption uppercase tracking-wider text-fg-subtle mb-3">
              Day of week
            </div>
            <div className="space-y-1">
              {a.by_day_of_week.map((d) => (
                <div key={d.day} className="flex items-center gap-2">
                  <span className="text-caption text-fg-subtle w-10 shrink-0">{d.label}</span>
                  <div className="flex-1 h-5 bg-surface-2 rounded-xs overflow-hidden">
                    {d.trades > 0 && (
                      <div
                        className={cn(
                          "h-full rounded-xs",
                          d.pnl >= 0 ? "bg-pnl-up/60" : "bg-pnl-down/60",
                        )}
                        style={{ width: `${Math.min(100, d.win_rate * 100)}%` }}
                      />
                    )}
                  </div>
                  <span className={cn(
                    "font-mono tabular text-caption w-16 text-right shrink-0",
                    d.trades === 0 ? "text-fg-muted" : clsPnl(d.pnl),
                  )}>
                    {d.trades > 0 ? `${d.trades}t ${(d.win_rate * 100).toFixed(0)}%` : "—"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Sector attribution */}
          <div>
            <div className="text-caption uppercase tracking-wider text-fg-subtle mb-3">
              Sector P&amp;L
            </div>
            <div className="space-y-1">
              {(() => {
                let maxPnl = 1;
                for (const x of a.by_sector) {
                  const v = Math.abs(x.pnl);
                  if (v > maxPnl) maxPnl = v;
                }
                return a.by_sector.map((s) => {
                const w = (Math.abs(s.pnl) / maxPnl) * 100;
                return (
                  <div key={s.sector} className="flex items-center gap-2">
                    <span className="text-caption text-fg-subtle w-14 shrink-0 truncate" title={s.sector}>
                      {s.sector}
                    </span>
                    <div className="flex-1 h-5 bg-surface-2 rounded-xs overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded-xs",
                          s.pnl >= 0 ? "bg-pnl-up/60" : "bg-pnl-down/60",
                        )}
                        style={{ width: `${Math.max(4, w)}%` }}
                      />
                    </div>
                    <span className={cn(
                      "font-mono tabular text-caption w-16 text-right shrink-0",
                      clsPnl(s.pnl),
                    )}>
                      {fmtInr(s.pnl, { compact: true })}
                    </span>
                  </div>
                );
              });
              })()}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Capture Matrix — per-stock: move vs captured                          */
/* ================================================================== */

function CaptureMatrixCard({ matrix }: { matrix: StockCapture[] }) {
  const [sortBy, setSortBy] = React.useState<"potential" | "captured" | "rate">("potential");
  const sorted = React.useMemo(
    () => [...matrix].sort((a, b) => {
      let d = 0;
      if (sortBy === "captured") d = b.captured_pnl - a.captured_pnl;
      else if (sortBy === "rate") d = b.capture_rate_pct - a.capture_rate_pct;
      else d = b.potential_pnl - a.potential_pnl;
      return d !== 0 ? d : a.symbol.localeCompare(b.symbol);
    }),
    [matrix, sortBy],
  );

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-4 w-4 text-fg-muted" aria-hidden />
            Capture Rate Matrix
          </CardTitle>
          <CardDescription>
            Per-stock: how much did the market offer vs how much did AlphaDesk capture?
          </CardDescription>
        </div>
        <div className="flex gap-1 items-center">
          {(["potential", "captured", "rate"] as const).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setSortBy(key)}
              className={cn(
                "text-caption px-2 py-1 rounded-xs",
                sortBy === key
                  ? "bg-accent/20 text-accent font-semibold"
                  : "text-fg-subtle hover:bg-surface-2",
              )}
            >
              {key === "potential" ? "Potential" : key === "captured" ? "Captured" : "Rate"}
            </button>
          ))}
          <SectionTools className="ml-1.5" />
        </div>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <PopulateEmpty
            title="No capture data yet"
            description="The capture matrix needs (1) signals fired during the month and (2) those signals enriched with EOD prices. Click below to backfill."
            primaryLabel="Run screener for this month"
            primaryCommand="run_screener"
            secondaryLabel="Refresh signal outcomes"
            secondaryCommand="enrich_signals"
            secondaryArgs="--all"
          />
        ) : (
          <>
            <div className="hidden md:grid grid-cols-[minmax(100px,1.2fr)_repeat(6,1fr)] gap-3 py-2 text-caption uppercase tracking-wider text-fg-subtle">
              <span>Symbol</span>
              <span className="text-right">Move %</span>
              <span className="text-right">Signals</span>
              <span className="text-right">Trades</span>
              <span className="text-right">Captured</span>
              <span className="text-right">Potential</span>
              <span className="text-right">Rate</span>
            </div>
            <div className="divide-y divide-border/60">
              {sorted.map((s) => (
                <CaptureRow key={s.symbol} s={s} />
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function CaptureRow({ s }: { s: StockCapture }) {
  const rateCls = s.capture_rate_pct >= 50 ? "text-pnl-up" : s.capture_rate_pct >= 25 ? "text-fg" : "text-pnl-down";
  const moveCls = s.month_move_pct >= 0 ? "text-pnl-up" : "text-pnl-down";

  return (
    <div className="grid grid-cols-[minmax(100px,1.2fr)_repeat(6,1fr)] gap-3 py-3 items-center">
      <div className="min-w-0">
        <span className="font-mono text-body text-fg">{s.symbol}</span>
        {s.worst_miss && (
          <span className="block text-caption text-fg-muted">
            Missed: {s.worst_miss.strategy} ({fmtInr(s.worst_miss.potential_pnl)})
          </span>
        )}
      </div>
      <span className={cn("text-right font-mono tabular text-body-sm", moveCls)}>
        {s.month_move_pct > 0 ? "+" : ""}{s.month_move_pct.toFixed(1)}%
      </span>
      <span className="text-right font-mono tabular text-body-sm text-fg">
        {s.signals_fired}
      </span>
      <span className="text-right font-mono tabular text-body-sm text-fg">
        {s.trades_taken}
        {s.trades_skipped > 0 && (
          <span className="text-fg-muted"> / {s.trades_skipped}sk</span>
        )}
      </span>
      <span className={cn("text-right font-mono tabular text-body-sm font-semibold", clsPnl(s.captured_pnl))}>
        {formatSignedInr(s.captured_pnl)}
      </span>
      <span className="text-right font-mono tabular text-body-sm text-fg-muted">
        {fmtInr(s.potential_pnl)}
      </span>
      <div className="flex items-center justify-end gap-2">
        <div className="w-16 h-2 bg-surface-2 rounded-full overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-[width]",
              s.capture_rate_pct >= 50 ? "bg-pnl-up" : s.capture_rate_pct >= 25 ? "bg-accent" : "bg-pnl-down",
            )}
            style={{ width: `${Math.min(100, s.capture_rate_pct)}%` }}
          />
        </div>
        <span className={cn("font-mono tabular text-caption font-semibold min-w-8 text-right", rateCls)}>
          {s.capture_rate_pct.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

/* ================================================================== */
/* Signal Audit — outcome breakdown                                      */
/* ================================================================== */

function SignalAuditCard({ audit }: { audit: SignalAudit }) {
  const total = audit.total_signals;
  const outcomes = React.useMemo(
    () => Object.entries(audit.by_outcome).sort(([, a], [, b]) => b - a),
    [audit.by_outcome],
  );
  const sources = React.useMemo(
    () => Object.entries(audit.by_source).sort(([, a], [, b]) => b - a),
    [audit.by_source],
  );
  const strategies = React.useMemo(
    () => Object.entries(audit.by_strategy).sort(([, a], [, b]) => b.count - a.count),
    [audit.by_strategy],
  );

  const OUTCOME_COLORS: Record<string, string> = {
    TRADED: "bg-pnl-up",
    REJECTED: "bg-pnl-down",
    SKIPPED: "bg-amber-500",
    EXPIRED: "bg-surface-3",
    PENDING: "bg-accent",
  };

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-fg-muted" aria-hidden />
          Signal Audit
        </CardTitle>
        <CardDescription>
          {total} signals detected — {audit.profitable_if_taken} profitable if taken,{" "}
          {audit.loss_avoided} losses avoided
        </CardDescription>
        </div>
        <SectionTools />
      </CardHeader>
      <CardContent className="space-y-5">
        {total === 0 ? (
          <PopulateEmpty
            title="No signals recorded"
            description="Replay the live screener over this month to populate apps.strategies.Signal rows for every strategy that fired."
            primaryLabel="Run screener for this month"
            primaryCommand="run_screener"
            secondaryLabel="Run the swing scanner"
            secondaryCommand="run_ok_scanner"
            secondaryArgs="--actionable-only"
          />
        ) : (
        <>
        {/* Outcome bar */}
        <div>
          <div className="text-caption uppercase tracking-wider text-fg-subtle mb-2">By outcome</div>
          <div className="flex h-4 rounded-xs overflow-hidden gap-px">
            {outcomes.map(([outcome, count]) => (
              <div
                key={outcome}
                className={cn("transition-[width]", OUTCOME_COLORS[outcome] ?? "bg-surface-3")}
                style={{ width: `${(count / total) * 100}%` }}
                title={`${outcome}: ${count}`}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-3 mt-2">
            {outcomes.map(([outcome, count]) => (
              <span key={outcome} className="flex items-center gap-1.5 text-caption text-fg-subtle">
                <span className={cn("w-2 h-2 rounded-full", OUTCOME_COLORS[outcome] ?? "bg-surface-3")} />
                {outcome} {count}
              </span>
            ))}
          </div>
        </div>

        {/* Source breakdown */}
        <div>
          <div className="text-caption uppercase tracking-wider text-fg-subtle mb-2">By source</div>
          <div className="flex gap-3">
            {sources.map(([source, count]) => (
              <div key={source} className="rounded-sm border border-border/60 p-2 flex-1">
                <div className="text-caption text-fg-subtle">{source}</div>
                <div className="text-h3 text-fg">{count}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Strategy table */}
        {strategies.length > 0 && (
          <div>
            <div className="text-caption uppercase tracking-wider text-fg-subtle mb-2">By strategy</div>
            <div className="divide-y divide-border/60">
              <div className="grid grid-cols-4 gap-3 py-2 text-caption uppercase tracking-wider text-fg-subtle">
                <span>Strategy</span>
                <span className="text-right">Signals</span>
                <span className="text-right">Win rate</span>
                <span className="text-right">Avg R:R</span>
              </div>
              {strategies.map(([name, stats]) => (
                <div key={name} className="grid grid-cols-4 gap-3 py-2 text-body-sm">
                  <span className="font-mono text-fg">{name}</span>
                  <span className="text-right font-mono tabular text-fg">{stats.count}</span>
                  <span className="text-right font-mono tabular text-fg">
                    {(stats.win_rate * 100).toFixed(0)}%
                  </span>
                  <span className="text-right font-mono tabular text-fg">
                    {stats.avg_rr.toFixed(1)}x
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        </>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Rejections — risk gate hindsight review                               */
/* ================================================================== */

function RejectionsCard({ rejections }: { rejections: RejectionReview[] }) {
  const profitable = rejections.filter((r) => r.would_have_profited).length;
  const total = rejections.length;

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-4 w-4 text-fg-muted" aria-hidden />
          Risk Rejections
        </CardTitle>
        <CardDescription>
          {total} blocked by @RiskGuard — {profitable} would have been profitable in hindsight
        </CardDescription>
        </div>
        <SectionTools />
      </CardHeader>
      <CardContent>
        <div className="divide-y divide-border/60">
          {rejections.map((r, i) => (
            <div key={`${r.symbol}-${r.date}-${i}`} className="flex items-center gap-3 py-3">
              <div className={cn(
                "shrink-0 w-2 h-2 rounded-full",
                r.would_have_profited ? "bg-pnl-up" : "bg-pnl-down",
              )} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-body text-fg">{r.symbol}</span>
                  <span className="text-caption text-fg-subtle">{r.date}</span>
                  <Badge tone={r.would_have_profited ? "success" : "neutral"}>
                    {r.would_have_profited ? "Would have profited" : "Loss avoided"}
                  </Badge>
                </div>
                <p className="text-caption text-fg-muted mt-0.5">{r.reason}</p>
              </div>
              <div className={cn(
                "font-mono tabular text-body-sm font-semibold shrink-0",
                r.would_have_profited ? "text-pnl-up" : "text-fg-subtle",
              )}>
                {r.would_have_profited ? `+${fmtInr(r.hypothetical_pnl)}` : "—"}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Lessons — AI-generated feedback                                       */
/* ================================================================== */

function LessonsCard({ lessons }: { lessons: string[] }) {
  const filtered = lessons.filter(Boolean);
  if (filtered.length === 0) return null;

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div>
        <CardTitle className="flex items-center gap-2">
          <Lightbulb className="h-4 w-4 text-fg-muted" aria-hidden />
          Lessons
        </CardTitle>
        <CardDescription>
          Insights from this month's trading activity
        </CardDescription>
        </div>
        <SectionTools />
      </CardHeader>
      <CardContent>
        <ul className="space-y-3">
          {filtered.map((lesson, i) => (
            <li key={i} className="flex gap-3 items-start">
              <AlertTriangle className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden />
              <p className="text-body-sm text-fg">{lesson}</p>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Formatters                                                            */
/* ================================================================== */

function formatSignedInr(v: number): string {
  if (!isFinite(v)) return "—";
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}${fmtInr(Math.abs(v))}`;
}

/* ================================================================== */
/* Loading / error states                                                */
/* ================================================================== */

function MonthlyLoading() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <Skeleton className="h-14 w-80" />
      <Skeleton className="h-48 w-full" />
      <Skeleton className="h-[360px] w-full" />
    </div>
  );
}

function MonthlyError({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="px-6 py-6 max-w-[800px] mx-auto">
      <EmptyState
        title="Couldn't load monthly view"
        description={error?.message ?? "Unknown error"}
        action={<Button onClick={onRetry}>Try again</Button>}
      />
    </div>
  );
}
