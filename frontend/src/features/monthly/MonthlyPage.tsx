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
  FlaskConical, Lightbulb, RefreshCcw, Shield, Target,
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
import { clsPnl, cn, fmtInr, fmtNum, fmtPct, fmtRel } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { OpButton } from "@/features/ops/OpButton";
import { EmptyState } from "@/components/ui/EmptyState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";

/* ================================================================== */
/* Page                                                                 */
/* ================================================================== */

export function MonthlyPage() {
  const monthFromUrl = new URLSearchParams(window.location.search).get("month");
  const queryClient = useQueryClient();

  const [source, setSource] = React.useState(getMonthlySource);

  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } =
    useMonthlyView();

  const [selectedMonth, setSelectedMonth] = React.useState<string | null>(monthFromUrl);

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

  if (isLoading) return <MonthlyLoading />;
  if (isError) return <MonthlyError error={error as Error} onRetry={() => refetch()} />;
  if (!data) return null;

  const monthKey = selectedMonth ?? data.current_month;
  const month = data.months.find((m) => m.month === monthKey) ?? data.months[0];

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <Header
        data={data}
        onRefresh={() => refetch()}
        isFetching={isFetching}
        dataUpdatedAt={dataUpdatedAt}
        source={source}
        onToggleSource={handleToggleSource}
        monthKey={monthKey}
        onOpFinished={() => queryClient.invalidateQueries({ queryKey: ["monthly-view"] })}
      />

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
        <span className="text-caption text-fg-subtle">
          {fmtRel(new Date(dataUpdatedAt).toISOString())}
        </span>

        {/* Re-run the signal-outcome enrichment (capture rates, win/loss labels)
            then re-fetch the monthly view so the new numbers appear inline. */}
        <OpButton
          command="enrich_signals"
          defaultArgs="--all"
          label="Refresh signals"
          description="Backfill EOD outcomes for every SignalLog row — drives the capture matrix + signal audit on this page."
          onSuccess={onOpFinished}
        />

        {/* Re-backtest the swing strategy for the currently-viewed month. */}
        <OpButton
          command="run_ok_backtest"
          defaultArgs={`--from ${monthStart} --to ${monthEnd}`}
          label="Backtest this month"
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
            Last 12 months · click a bar to inspect any month below
          </CardDescription>
        </div>
        <div className="flex items-center gap-2 text-caption uppercase tracking-wider text-fg-subtle">
          Capital base {fmtInr(ytd.capital_base, { compact: true })}
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
      aria-label="12 month P&L"
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
          <div className={cn("text-h2 font-mono tabular", totalPnlCls)}>
            {formatSignedInr(month.total_pnl)}
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
  const sorted = [...rolls].sort((a, b) => b.running_pnl - a.running_pnl);
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
  const cap = rolls.reduce((s, r) => s + r.capital_deployed, 0);
  const expo = rolls.reduce((s, r) => s + r.exposure, 0);
  const tgt = rolls.reduce((s, r) => s + r.target_total, 0);
  const risk = rolls.reduce((s, r) => s + r.risk_total, 0);
  const pnl = rolls.reduce((s, r) => s + r.running_pnl, 0);
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

function LegList({ legs }: { legs: PositionLeg[] }) {
  return (
    <div className="px-3 pb-4 pt-1 bg-surface-2/30 rounded-b-sm">
      <ul className="divide-y divide-border/60">
        {legs.map((l) => <LegRow key={l.id} l={l} />)}
      </ul>
    </div>
  );
}

function LegRow({ l }: { l: PositionLeg }) {
  const pnlCls = clsPnl(l.pnl);
  const closed = l.status === "CLOSED";
  return (
    <li className="grid grid-cols-12 gap-3 py-3 items-start text-body-sm">
      <div className="col-span-12 md:col-span-4 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-fg">{l.symbol}</span>
          <Badge tone={l.side === "BUY" ? "success" : "danger"}>{l.side}</Badge>
          <Badge tone={closed ? "neutral" : "info"}>{l.status}</Badge>
        </div>
        {l.notes && (
          <p className="text-caption text-fg-muted mt-1">{l.notes}</p>
        )}
      </div>
      <div className="col-span-6 md:col-span-2 font-mono tabular text-right">
        <div className="text-caption text-fg-subtle">Qty · Entry</div>
        <div className="text-fg">{l.quantity} @ {fmtNum(l.entry_price, 2)}</div>
      </div>
      <div className="col-span-6 md:col-span-2 font-mono tabular text-right">
        <div className="text-caption text-fg-subtle">Target · Stop</div>
        <div className="text-fg">
          <span className="text-pnl-up">{l.target_price != null ? fmtNum(l.target_price, 2) : "—"}</span>
          {" / "}
          <span className="text-pnl-down">{l.stop_price != null ? fmtNum(l.stop_price, 2) : "—"}</span>
        </div>
      </div>
      <div className="col-span-6 md:col-span-2 font-mono tabular text-right">
        <div className="text-caption text-fg-subtle">{closed ? "Exit" : "Opened"}</div>
        <div className="text-fg">
          {closed
            ? `${l.exit_price != null ? fmtNum(l.exit_price, 2) : "—"} · ${l.exit_date?.slice(5) ?? ""}`
            : l.entry_date.slice(5)}
        </div>
      </div>
      <div className={cn("col-span-6 md:col-span-2 font-mono tabular text-right font-semibold", pnlCls)}>
        <div className="text-caption text-fg-subtle font-normal">
          {closed ? "Realised" : "Running"}
        </div>
        {formatSignedInr(l.pnl)}
      </div>
    </li>
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
        <div className="flex gap-4 text-caption text-right">
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
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-fg-muted" aria-hidden />
          vs NIFTY50
        </CardTitle>
        <CardDescription>
          {b.trading_days} trading days
        </CardDescription>
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
      <CardHeader>
        <CardTitle>Analytics</CardTitle>
        <CardDescription>Performance by time of day, day of week, and sector</CardDescription>
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
              {a.by_sector.map((s) => {
                const maxPnl = Math.max(...a.by_sector.map((x) => Math.abs(x.pnl)), 1);
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
              })}
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
  const sorted = [...matrix].sort((a, b) => {
    let d = 0;
    if (sortBy === "captured") d = b.captured_pnl - a.captured_pnl;
    else if (sortBy === "rate") d = b.capture_rate_pct - a.capture_rate_pct;
    else d = b.potential_pnl - a.potential_pnl;
    return d !== 0 ? d : a.symbol.localeCompare(b.symbol);
  });

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
        <div className="flex gap-1">
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
        </div>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <EmptyState
            title="No capture data yet"
            description="Signals need to be enriched with EOD price data to compute capture rates."
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
  const outcomes = Object.entries(audit.by_outcome).sort(([, a], [, b]) => b - a);
  const sources = Object.entries(audit.by_source).sort(([, a], [, b]) => b - a);
  const strategies = Object.entries(audit.by_strategy)
    .sort(([, a], [, b]) => b.count - a.count);

  const OUTCOME_COLORS: Record<string, string> = {
    TRADED: "bg-pnl-up",
    REJECTED: "bg-pnl-down",
    SKIPPED: "bg-amber-500",
    EXPIRED: "bg-surface-3",
    PENDING: "bg-accent",
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-fg-muted" aria-hidden />
          Signal Audit
        </CardTitle>
        <CardDescription>
          {total} signals detected — {audit.profitable_if_taken} profitable if taken,{" "}
          {audit.loss_avoided} losses avoided
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {total === 0 ? (
          <EmptyState
            title="No signals recorded"
            description="Run the screener or scanner to generate signal data for this month."
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
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-4 w-4 text-fg-muted" aria-hidden />
          Risk Rejections
        </CardTitle>
        <CardDescription>
          {total} blocked by @RiskGuard — {profitable} would have been profitable in hindsight
        </CardDescription>
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
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lightbulb className="h-4 w-4 text-fg-muted" aria-hidden />
          Lessons
        </CardTitle>
        <CardDescription>
          Insights from this month's trading activity
        </CardDescription>
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
