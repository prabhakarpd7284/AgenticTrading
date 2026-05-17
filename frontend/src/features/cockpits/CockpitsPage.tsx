/**
 * Cockpits — 10 trader-facing aggregations on one page.
 *
 * Each tab fetches a single `/api/v1/portfolios/<slug>/` endpoint and
 * renders the response as KPIs + a table or chart. Everything is read-only.
 */
import * as React from "react";
import {
  Activity, AlertTriangle, Award, BarChart3, Bell, BookOpen, Briefcase,
  Calculator, Calendar, Check, Clock, Compass, Crosshair, Droplets, Eraser,
  Flag, FlaskConical, Flame, Gauge, GitCompareArrows, Grid3X3, Hourglass,
  Layers, LineChart, Microscope, Pause, Pencil, PieChart, Radar, Receipt,
  RotateCw, Scale, Shield, Sigma, Sparkles, Sunrise, Sunrise as DaybreakIcon,
  Target, TrendingDown, Waves, X, Zap, ZapOff,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart as RLineChart,
  ResponsiveContainer, Tooltip as ReTooltip, XAxis, YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { cn, fmtInr, fmtNum, fmtPct, fmtRel } from "@/lib/utils";

type KPITone = "neutral" | "success" | "warning" | "danger";

function KPI({ label, value, hint, tone = "neutral" }: {
  label: string; value: string; hint?: string; tone?: KPITone;
}) {
  const color =
    tone === "success" ? "text-pnl-up"
    : tone === "warning" ? "text-warning"
    : tone === "danger" ? "text-pnl-down"
    : "text-fg";
  return (
    <div className="rounded-md border border-border bg-surface p-3">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn("text-h3 font-mono mt-0.5", color)}>{value}</div>
      {hint ? <div className="text-caption text-fg-subtle mt-0.5">{hint}</div> : null}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────── *
 * HelpBlock — explainer card shown at the top of every cockpit tab.   *
 * Three slots: what the metric is, why it matters, when to act.       *
 * ─────────────────────────────────────────────────────────────────── */
function HelpBlock({ what, why, act }: { what: string; why: string; act: string }) {
  return (
    <div className="rounded-md border border-border bg-surface-2/50 p-3 text-body-sm">
      <div className="grid md:grid-cols-3 gap-3">
        <Help label="What this is" text={what} />
        <Help label="Why it matters" text={why} />
        <Help label="When to act" text={act} />
      </div>
    </div>
  );
}

function Help({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">{label}</div>
      <p className="text-fg-muted leading-snug">{text}</p>
    </div>
  );
}
import { useQueryClient } from "@tanstack/react-query";
import {
  checkSlippageEdge, flattenAll, pauseSymbol, resetTradingData,
  runSimulation, saveBacktestRun, setCapital,
  simulateSizer, unpauseSymbol, usePastBacktestRuns,
  type BTSimResponse,
  useBaseQuality, useBreakoutClassifier, useBrokerRecon, useCapitalCockpit,
  useCorrelationMatrix, useDepthImbalance, useEarningsOverlay, useEdgeDecay,
  useEdgeLedger, useExpiryCockpit, useFIIDIIFlow, useFirst5Min, useForcedFlat,
  useGapFill, useGapRisk, useGreeksHeatmap, useIntradayRotation,
  useIntradaySectorHeatmap, useLiquidityMap, useMTFStage, useNewsShock,
  useORB, useORBFailure, usePartialFill, usePlanVsActual, usePostMortem,
  useRegimeHeatmap, useRiskBudget, useSecond5Min, useSectorDispersion,
  useSectorRRG, useSignalFunnel, useStockRRG, useStructuralStops,
  useTapeSpeed, useThetaForecast, useVolRegime, useVWAPBands,
  type SizerResponse, type SlippageEdgeResponse,
} from "@/lib/cockpits";

type CategoryId = "capital" | "options" | "setups" | "tape" | "sectors" | "exec" | "tools";

const CATEGORIES: { id: CategoryId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "capital", label: "Capital & Risk",      icon: Briefcase },
  { id: "options", label: "Options & Expiry",    icon: Sigma },
  { id: "setups",  label: "Setups & Signals",    icon: Target },
  { id: "tape",    label: "Tape & Microstructure", icon: Waves },
  { id: "sectors", label: "Sectors & Flow",      icon: Radar },
  { id: "exec",    label: "Execution & Trades",  icon: Receipt },
  { id: "tools",   label: "Tools",               icon: Calculator },
];

const TABS = [
  // Capital & Risk
  { cat: "capital", id: "capital",       label: "Capital",      icon: Briefcase },
  { cat: "capital", id: "risk-budget",   label: "Risk Budget",  icon: Gauge },
  { cat: "capital", id: "gap-risk",      label: "Gap Risk",     icon: Sunrise },
  { cat: "capital", id: "correlation",   label: "Correlation",  icon: Grid3X3 },
  { cat: "capital", id: "stops",         label: "Structural Stops", icon: Shield },

  // Options & Expiry
  { cat: "options", id: "greeks",        label: "Greeks",       icon: Sigma },
  { cat: "options", id: "theta",         label: "Theta",        icon: LineChart },
  { cat: "options", id: "expiry",        label: "Expiry",       icon: Clock },

  // Setups & Signals
  { cat: "setups", id: "signal-funnel", label: "Signal Funnel",icon: Target },
  { cat: "setups", id: "edge-decay",    label: "Edge Decay",   icon: TrendingDown },
  { cat: "setups", id: "regime",        label: "Regime",       icon: Activity },
  { cat: "setups", id: "mtf",           label: "MTF Stages",   icon: Layers },
  { cat: "setups", id: "breakout",      label: "Fresh vs Extended", icon: Activity },
  { cat: "setups", id: "base",          label: "Base Quality", icon: Award },
  { cat: "setups", id: "orb",           label: "Opening Range", icon: Zap },
  { cat: "setups", id: "orb-fail",      label: "OR Failure",   icon: ZapOff },
  { cat: "setups", id: "first5",        label: "First 5-min",  icon: DaybreakIcon },
  { cat: "setups", id: "second5",       label: "2nd 5-min",    icon: BookOpen },
  { cat: "setups", id: "gap-fill",      label: "Gap-Fill Prob", icon: Hourglass },

  // Tape & Microstructure
  { cat: "tape", id: "vwap",          label: "VWAP Bands",   icon: Compass },
  { cat: "tape", id: "vol-regime",    label: "Vol Regime",   icon: Waves },
  { cat: "tape", id: "tape",          label: "Tape Speed",   icon: Activity },
  { cat: "tape", id: "liquidity",     label: "Liquidity",    icon: Droplets },
  { cat: "tape", id: "depth",         label: "Depth Proxy",  icon: BarChart3 },

  // Sectors & Flow
  { cat: "sectors", id: "sector-rrg",    label: "Sector RRG",   icon: Radar },
  { cat: "sectors", id: "stock-rrg",     label: "Stock RRG",    icon: Crosshair },
  { cat: "sectors", id: "dispersion",    label: "Sector Disp.", icon: FlaskConical },
  { cat: "sectors", id: "isector",       label: "Sector Heatmap (5m)", icon: Flame },
  { cat: "sectors", id: "fii-dii",       label: "FII/DII Flow", icon: LineChart },
  { cat: "sectors", id: "news-shock",    label: "News Shocks",  icon: Bell },
  { cat: "sectors", id: "earnings",      label: "Earnings",     icon: Calendar },

  // Execution & Trades
  { cat: "exec", id: "plan-vs-actual",label: "Plan vs Actual", icon: GitCompareArrows },
  { cat: "exec", id: "ledger",        label: "Edge Ledger",  icon: Receipt },
  { cat: "exec", id: "partial-fill",  label: "Partial Fill", icon: PieChart },
  { cat: "exec", id: "post-mortem",   label: "Post-Mortem",  icon: Microscope },
  { cat: "exec", id: "broker-recon",  label: "Broker Recon", icon: GitCompareArrows },
  { cat: "exec", id: "forced-flat",   label: "Forced Flat",  icon: Flag },
  { cat: "exec", id: "rotation",      label: "Capital Rotation", icon: RotateCw },

  // Tools
  { cat: "tools", id: "sizer",         label: "What-If Sizer", icon: Calculator },
  { cat: "tools", id: "slippage-edge", label: "Slippage vs Edge", icon: Scale },
  { cat: "tools", id: "simulator",     label: "Backtester Sim", icon: FlaskConical },
  { cat: "tools", id: "reset",         label: "Reset / Seed", icon: Eraser },
] as const;

export function CockpitsPage() {
  const [category, setCategory] = React.useState<CategoryId>("capital");
  const [search, setSearch] = React.useState("");
  const [activeTab, setActiveTab] = React.useState<string>("capital");

  // Search hits every tab regardless of category.
  const searchHits = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return null;
    return TABS.filter((t) =>
      t.label.toLowerCase().includes(q) || t.id.toLowerCase().includes(q),
    );
  }, [search]);

  const visibleTabs = searchHits ?? TABS.filter((t) => t.cat === category);

  // Switching category jumps to its first tab; search jumps to first hit.
  React.useEffect(() => {
    if (visibleTabs.length > 0 && !visibleTabs.some((t) => t.id === activeTab)) {
      setActiveTab(visibleTabs[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [category, searchHits]);

  const perCategoryCounts = React.useMemo(() => {
    const m: Record<string, number> = {};
    for (const t of TABS) m[t.cat] = (m[t.cat] ?? 0) + 1;
    return m;
  }, []);

  return (
    <div className="px-6 py-6 space-y-4 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Trader cockpits</p>
          <h1 className="text-h1 text-fg flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-accent" aria-hidden />
            Cockpits
            <span className="text-caption text-fg-subtle font-normal ml-1">
              {TABS.length} panels
            </span>
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search panels…"
            className="h-9 px-3 w-56 bg-surface border border-border rounded-sm text-body-sm"
          />
          {search ? (
            <button
              onClick={() => setSearch("")}
              className="h-9 px-2 text-caption text-fg-subtle hover:text-fg"
              aria-label="Clear search"
            >
              <X className="h-4 w-4" />
            </button>
          ) : null}
        </div>
      </header>

      <div className="grid grid-cols-[200px_1fr] gap-4">
        {/* Category sidebar */}
        <nav className="space-y-1" aria-label="Cockpit categories">
          {CATEGORIES.map(({ id, label, icon: Icon }) => {
            const active = !search && id === category;
            return (
              <button
                key={id}
                onClick={() => { setSearch(""); setCategory(id); }}
                className={cn(
                  "w-full flex items-center justify-between gap-2 px-3 h-9 text-body-sm rounded-sm border",
                  "transition-colors",
                  active
                    ? "border-accent/40 bg-accent/10 text-fg"
                    : "border-border bg-surface text-fg-muted hover:bg-surface-2 hover:text-fg",
                )}
              >
                <span className="flex items-center gap-2">
                  <Icon className="h-3.5 w-3.5" />
                  {label}
                </span>
                <span className="text-caption text-fg-subtle">{perCategoryCounts[id]}</span>
              </button>
            );
          })}
          {search ? (
            <div className="px-3 pt-2 text-caption text-fg-subtle">
              {searchHits!.length} match{searchHits!.length === 1 ? "" : "es"} across all categories.
            </div>
          ) : null}
        </nav>

        {/* Right side: per-category tab strip + tab content */}
        <div className="min-w-0">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="flex-wrap h-auto">
              {visibleTabs.length === 0 ? (
                <span className="text-body-sm text-fg-subtle px-2 py-1">No panels match "{search}".</span>
              ) : visibleTabs.map(({ id, label, icon: Icon }) => (
                <TabsTrigger key={id} value={id} className="gap-1.5">
                  <Icon className="h-3.5 w-3.5" aria-hidden /> {label}
                </TabsTrigger>
              ))}
            </TabsList>

        <TabsContent value="capital"><CapitalPanel /></TabsContent>
        <TabsContent value="plan-vs-actual"><PlanVsActualPanel /></TabsContent>
        <TabsContent value="greeks"><GreeksPanel /></TabsContent>
        <TabsContent value="signal-funnel"><SignalFunnelPanel /></TabsContent>
        <TabsContent value="risk-budget"><RiskBudgetPanel /></TabsContent>
        <TabsContent value="expiry"><ExpiryPanel /></TabsContent>
        <TabsContent value="broker-recon"><BrokerReconPanel /></TabsContent>
        <TabsContent value="edge-decay"><EdgeDecayPanel /></TabsContent>
        <TabsContent value="theta"><ThetaPanel /></TabsContent>
        <TabsContent value="regime"><RegimePanel /></TabsContent>
        <TabsContent value="correlation"><CorrelationPanel /></TabsContent>
        <TabsContent value="post-mortem"><PostMortemPanel /></TabsContent>
        <TabsContent value="gap-risk"><GapRiskPanel /></TabsContent>
        <TabsContent value="liquidity"><LiquidityPanel /></TabsContent>
        <TabsContent value="sizer"><SizerPanel /></TabsContent>
        <TabsContent value="stops"><StopsPanel /></TabsContent>
        <TabsContent value="forced-flat"><ForcedFlatPanel /></TabsContent>
        <TabsContent value="slippage-edge"><SlippageEdgePanel /></TabsContent>
        <TabsContent value="orb"><ORBPanel /></TabsContent>
        <TabsContent value="vwap"><VWAPPanel /></TabsContent>
        <TabsContent value="first5"><First5MinPanel /></TabsContent>
        <TabsContent value="base"><BaseQualityPanel /></TabsContent>
        <TabsContent value="mtf"><MTFStagePanel /></TabsContent>
        <TabsContent value="breakout"><BreakoutClassifierPanel /></TabsContent>
        <TabsContent value="orb-fail"><ORBFailurePanel /></TabsContent>
        <TabsContent value="ledger"><EdgeLedgerPanel /></TabsContent>
        <TabsContent value="rotation"><IntradayRotationPanel /></TabsContent>
        <TabsContent value="gap-fill"><GapFillPanel /></TabsContent>
        <TabsContent value="second5"><Second5MinPanel /></TabsContent>
        <TabsContent value="vol-regime"><VolRegimePanel /></TabsContent>
        <TabsContent value="sector-rrg"><SectorRRGPanel /></TabsContent>
        <TabsContent value="dispersion"><SectorDispersionPanel /></TabsContent>
        <TabsContent value="tape"><TapeSpeedPanel /></TabsContent>
        <TabsContent value="news-shock"><NewsShockPanel /></TabsContent>
        <TabsContent value="fii-dii"><FIIDIIFlowPanel /></TabsContent>
        <TabsContent value="depth"><DepthImbalancePanel /></TabsContent>
        <TabsContent value="earnings"><EarningsOverlayPanel /></TabsContent>
        <TabsContent value="stock-rrg"><StockRRGPanel /></TabsContent>
        <TabsContent value="partial-fill"><PartialFillPanel /></TabsContent>
        <TabsContent value="isector"><IntradaySectorHeatmapPanel /></TabsContent>
        <TabsContent value="simulator"><BacktesterSimPanel /></TabsContent>
        <TabsContent value="reset"><ResetPanel /></TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ helpers ----------------------------------- */
function LoadingPanel() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
      {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
    </div>
  );
}

function PanelWrap({ children }: { children: React.ReactNode }) {
  return <div className="mt-4 space-y-6">{children}</div>;
}

/* ------------------------------ 1. Capital -------------------------------- */
function CapitalPanel() {
  const { data, isLoading } = useCapitalCockpit();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Where every rupee of your capital is sitting right now — deployed in positions, locked as margin, or free to deploy."
        why="Notional says how big your bets look on paper; delta-adjusted shows the real directional bet after accounting for option deltas. Leverage above 3× on intraday or 1× on overnight is where stops get triggered by noise."
        act="If leverage > 3× and the regime is choppy, scale a position down. If free margin < 10% of capital, stop opening new positions until something closes."
      />
      <CapitalEditor current={data.total_capital} />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Total capital" value={fmtInr(data.total_capital)} />
        <KPI label="Deployed" value={fmtInr(data.deployed_capital)} />
        <KPI label="Free margin" value={fmtInr(data.free_margin)} tone="success" />
        <KPI label="Leverage" value={`${fmtNum(data.leverage_ratio, 2)}×`}
             tone={data.leverage_ratio > 3 ? "warning" : "neutral"} />
        <KPI label="Margin used" value={fmtInr(data.margin_used)} />
        <KPI label="Notional" value={fmtInr(data.notional_exposure, { compact: true })} />
        <KPI label="Delta-adjusted" value={fmtInr(data.delta_adjusted_exposure, { compact: true })} />
        <KPI label="Premium received" value={fmtInr(data.premium_received)} />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>Buckets</CardTitle>
          <CardDescription>Margin and notional split by instrument bucket.</CardDescription>
        </CardHeader>
        <CardContent>
          {data.buckets.length === 0 ? (
            <EmptyState title="No open positions" description="Deploy a strategy to populate." />
          ) : (
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr><th className="text-left py-1.5">Bucket</th><th className="text-right">Margin</th><th className="text-right">Notional</th></tr>
              </thead>
              <tbody>
                {data.buckets.map((b) => (
                  <tr key={b.name} className="border-b border-border/40">
                    <td className="py-1.5">{b.name}</td>
                    <td className="text-right tabular-nums">{fmtInr(b.margin)}</td>
                    <td className="text-right tabular-nums">{fmtInr(b.notional)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

function CapitalEditor({ current }: { current: number }) {
  const qc = useQueryClient();
  const [editing, setEditing] = React.useState(false);
  const [value, setValue] = React.useState<string>("");
  const [pending, setPending] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const start = () => {
    setValue(String(Math.round(current)));
    setError(null);
    setEditing(true);
  };
  const cancel = () => { setEditing(false); setError(null); };

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) { setError("Enter a positive number"); return; }
    setPending(true);
    setError(null);
    try {
      await setCapital(n);
      // Anything that reads PortfolioSnapshot directly or derives from it
      // needs to refresh: capital, risk budget, sizer, leverage on most cards.
      await qc.invalidateQueries({ queryKey: ["cockpits"] });
      setEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "save failed");
    } finally {
      setPending(false);
    }
  };

  if (!editing) {
    return (
      <div className="flex items-center justify-between gap-2 px-3 py-2 rounded-md border border-border bg-surface">
        <div className="text-body-sm">
          <span className="text-fg-subtle">Capital base:</span>{" "}
          <span className="font-mono text-fg">{fmtInr(current)}</span>
          <span className="text-fg-subtle"> · drives risk-budget % and position sizing</span>
        </div>
        <button
          type="button"
          onClick={start}
          className="inline-flex items-center gap-1 h-7 px-2 text-caption bg-surface-2 hover:bg-surface-3 border border-border rounded-sm"
        >
          <Pencil className="h-3 w-3" /> Edit
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={save} className="flex flex-wrap items-center gap-2 px-3 py-2 rounded-md border border-accent/40 bg-accent/5">
      <label className="text-body-sm text-fg-muted">New capital (INR)</label>
      <input
        autoFocus
        type="number"
        min={1}
        step="any"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="w-40 h-8 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums"
      />
      <button
        type="submit"
        disabled={pending}
        className="inline-flex items-center gap-1 h-8 px-3 bg-accent text-accent-fg rounded-sm text-body-sm disabled:opacity-50"
      >
        <Check className="h-3.5 w-3.5" /> {pending ? "Saving…" : "Save"}
      </button>
      <button
        type="button"
        onClick={cancel}
        className="inline-flex items-center gap-1 h-8 px-3 bg-surface-2 hover:bg-surface-3 border border-border rounded-sm text-body-sm"
      >
        <X className="h-3.5 w-3.5" /> Cancel
      </button>
      {error ? <span className="text-body-sm text-pnl-down">{error}</span> : null}
      <span className="text-caption text-fg-subtle">
        Writes today's PortfolioSnapshot. Used by @RiskGuard for sizing + the 3% daily-loss cap.
      </span>
    </form>
  );
}

/* ------------------------------ 2. Plan vs Actual ------------------------- */
function PlanVsActualPanel() {
  const { data, isLoading } = usePlanVsActual();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="For every trade, what the AI planned vs what the broker actually filled — slippage in basis points (1 bp = 0.01%)."
        why="Slippage is silent edge erosion. A strategy that backtests at +30 bps/trade and slips 35 bps/trade is net-negative without you noticing."
        act="If avg |slippage| > 30 bps, switch to limit orders or trade smaller. Symbols showing consistent positive slippage are illiquid — kick them off the watchlist."
      />
      <section className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <KPI label="Trades analysed" value={String(data.count)} />
        <KPI label="Avg |slippage|" value={`${fmtNum(data.avg_abs_slippage_bps, 1)} bps`}
             tone={data.avg_abs_slippage_bps > 30 ? "warning" : "neutral"} />
        <KPI label="SL hits" value={String(data.rows.filter((r) => r.sl_hit).length)} />
      </section>
      <Card>
        <CardHeader><CardTitle>Slippage breakdown</CardTitle></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? (
            <EmptyState title="No journal rows yet" />
          ) : (
            <div className="overflow-auto max-h-[480px]">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border sticky top-0 bg-surface">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">Side</th>
                    <th className="text-right">Planned</th>
                    <th className="text-right">Actual</th>
                    <th className="text-right">Slippage (bps)</th>
                    <th className="text-left">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.slice(0, 200).map((r) => (
                    <tr key={String(r.trade_id)} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td>{r.side}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.planned_entry, 2)}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.actual_entry, 2)}</td>
                      <td className={`text-right tabular-nums ${Math.abs(r.slippage_bps) > 50 ? "text-warning" : ""}`}>
                        {fmtNum(r.slippage_bps, 1)}
                      </td>
                      <td>{r.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 3. Greeks --------------------------------- */
function GreeksPanel() {
  const { data, isLoading } = useGreeksHeatmap();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Aggregated option Greeks per underlying × expiry. Δ = directional exposure, Γ = how fast Δ moves, Θ = daily decay you collect/pay, V = sensitivity to a 1-vol-point change in IV."
        why="A 'flat' position with Δ near 0 still bleeds if Θ is paying and rips if Γ explodes near expiry. You need to see all four together to know what's actually risky."
        act="On NIFTY Tue / SENSEX Thu, if |Δ| > 50 lots and DTE < 1, hedge with futures. If |Γ| spikes near expiry, close ATM and re-strike further out."
      />
      {data.note ? (
        <Card><CardContent className="py-3 text-body-sm text-fg-muted">{data.note}</CardContent></Card>
      ) : null}
      <Card>
        <CardHeader>
          <CardTitle>Greeks heatmap</CardTitle>
          <CardDescription>Δ / Γ / Θ / V aggregated per underlying × expiry.</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? (
            <EmptyState title="No active option positions" />
          ) : (
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr>
                  <th className="text-left py-1.5">Underlying</th>
                  <th className="text-left">Expiry</th>
                  <th className="text-right">Δ</th>
                  <th className="text-right">Γ</th>
                  <th className="text-right">Θ</th>
                  <th className="text-right">V</th>
                  <th className="text-right">#Pos</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((r, i) => (
                  <tr key={`${r.underlying}-${r.expiry}-${i}`} className="border-b border-border/40">
                    <td className="py-1.5">{r.underlying}</td>
                    <td>{r.expiry}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.delta, 2)}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.gamma, 4)}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.theta, 2)}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.vega, 2)}</td>
                    <td className="text-right tabular-nums">{r.positions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 4. Signal Funnel -------------------------- */
function SignalFunnelPanel() {
  const { data, isLoading } = useSignalFunnel();
  if (isLoading || !data) return <LoadingPanel />;
  const t = data.totals;
  return (
    <PanelWrap>
      <HelpBlock
        what="The four-stage funnel from signal to outcome: how many fired → cleared @RiskGuard → got executed → ended profitable."
        why="A leaky funnel is your real edge problem. If 100 fire and only 5 hit, the bottleneck might be risk rules, sizing, or the strategy itself — knowing where matters more than the total count."
        act="If risk_passed/fired < 30%, your @RiskGuard limits are too tight or your strategies are over-aggressive — fix one. If profitable/executed < 40%, the strategies need re-tuning, not the funnel."
      />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Fired" value={String(t.fired)} />
        <KPI label="Risk passed" value={`${t.risk_passed} (${fmtPct(safeDiv(t.risk_passed, t.fired), 0)})`} />
        <KPI label="Executed" value={`${t.executed} (${fmtPct(safeDiv(t.executed, t.risk_passed), 0)})`} />
        <KPI label="Profitable" value={`${t.profitable} (${fmtPct(safeDiv(t.profitable, t.executed), 0)})`} tone="success" />
      </section>
      <div className="grid md:grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle>By strategy</CardTitle></CardHeader>
          <CardContent>
            {data.by_strategy.length === 0 ? <EmptyState title="No signals logged" /> : (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data.by_strategy}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="strategy" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                  <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                  <ReTooltip />
                  <Bar dataKey="fired" fill="#64748b" />
                  <Bar dataKey="executed" fill="#22c55e" />
                  <Bar dataKey="profitable" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Rejection reasons</CardTitle></CardHeader>
          <CardContent>
            {data.rejection_reasons.length === 0 ? <EmptyState title="No rejections recorded" /> : (
              <ul className="space-y-1.5 text-body-sm">
                {data.rejection_reasons.map((r) => (
                  <li key={r.reason} className="flex items-center justify-between">
                    <span className="text-fg-muted">{r.reason}</span>
                    <Badge tone="danger">{r.count}</Badge>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </PanelWrap>
  );
}

/* ------------------------------ 5. Risk Budget ---------------------------- */
function RiskBudgetPanel() {
  const { data, isLoading } = useRiskBudget();
  if (isLoading || !data) return <LoadingPanel />;
  const used = data.used_risk_pct;
  return (
    <PanelWrap>
      <HelpBlock
        what={`Your daily risk budget — 3% of capital is the hard daily-loss cap (₹${(data.capital * 0.03).toLocaleString("en-IN")}). 'Used risk' is the sum of every open position's loss-if-stop-hit, expressed as % of capital.`}
        why="Going over the 3% cap triggers @RiskGuard to reject every new trade until next session. The drawdown waterfall shows how many days you've actually breached vs survived — emotional state moves with this number."
        act="If used_risk > 50%, stop opening new positions until something closes. If drawdown trends down for 5+ days, halve sizing and review the losing strategies in Edge Decay."
      />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Capital" value={fmtInr(data.capital)} />
        <KPI label="Daily P&amp;L" value={fmtInr(data.daily_pnl)}
             tone={data.daily_pnl >= 0 ? "success" : "danger"} />
        <KPI label="Used risk" value={fmtPct(used / 100, 2)}
             tone={used > 80 ? "danger" : used > 50 ? "warning" : "success"} />
        <KPI label="Open risk @ SL" value={fmtInr(data.open_risk_at_stop)} />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>30-day drawdown</CardTitle>
          <CardDescription>Daily P&amp;L stacked into cumulative drawdown.</CardDescription>
        </CardHeader>
        <CardContent>
          {data.drawdown_waterfall.length === 0 ? <EmptyState title="No realised P&amp;L yet" /> : (
            <ResponsiveContainer width="100%" height={260}>
              <RLineChart data={data.drawdown_waterfall}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <ReTooltip />
                <Line type="monotone" dataKey="cum_pnl" stroke="#22c55e" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="drawdown" stroke="#ef4444" dot={false} strokeWidth={2} />
              </RLineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 6. Expiry --------------------------------- */
function ExpiryPanel() {
  const [underlying, setUnderlying] = React.useState("NIFTY");
  const { data, isLoading } = useExpiryCockpit(underlying);
  return (
    <PanelWrap>
      <HelpBlock
        what="Live countdown to expiry-day close (15:15 IST), the strike where most open interest sits ('pin'), and the close-list of every position you must square off."
        why="ATM gamma explodes after 15:15 — a normal 0.5% move becomes a 5× P&L swing. Holding past 15:15 isn't trading, it's gambling. The pin strike tells you where the market expects the spot to land."
        act="If is_expiry_day = YES and countdown < 1h, close everything in the close-list now. If pin is within 0.25% of spot and you're short ATM, exit immediately — gamma squeeze risk."
      />
      <div className="flex items-center gap-2">
        {["NIFTY", "BANKNIFTY", "SENSEX"].map((u) => (
          <button
            key={u}
            onClick={() => setUnderlying(u)}
            className={`px-3 h-8 text-body-sm rounded-sm border ${underlying === u ? "border-accent text-fg bg-accent/10" : "border-border text-fg-muted hover:bg-surface-2"}`}
          >
            {u}
          </button>
        ))}
      </div>
      {isLoading || !data ? <LoadingPanel /> : (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="Expiry day" value={data.is_expiry_day ? "YES" : "No"}
                 tone={data.is_expiry_day ? "warning" : "neutral"} />
            <KPI label="Countdown" value={fmtCountdown(data.countdown_seconds)} />
            <KPI label="Pin strike" value={data.pin_strike ? String(data.pin_strike) : "—"} />
            <KPI label="Active positions" value={String(data.active_count)} />
          </section>
          <Card>
            <CardHeader>
              <CardTitle>Close list ({data.close_list.length})</CardTitle>
              <CardDescription>
                {data.is_expiry_day
                  ? "Must close before 15:15 IST — gamma risk explodes after."
                  : "No expiry-day positions on this underlying."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.close_list.length === 0 ? <EmptyState title="Nothing to close" /> : (
                <table className="w-full text-body-sm">
                  <thead className="text-fg-subtle border-b border-border">
                    <tr>
                      <th className="text-left py-1.5">Symbol</th>
                      <th className="text-left">CE</th>
                      <th className="text-left">PE</th>
                      <th className="text-right">P&amp;L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.close_list.map((row, i) => (
                      <tr key={i} className="border-b border-border/40">
                        <td className="py-1.5">{row.symbol}</td>
                        <td className="text-fg-muted">{row.ce_symbol ?? "—"}</td>
                        <td className="text-fg-muted">{row.pe_symbol ?? "—"}</td>
                        <td className="text-right tabular-nums">{fmtInr(row.pnl ?? 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 7. Broker Recon --------------------------- */
function BrokerReconPanel() {
  const { data, isLoading } = useBrokerRecon();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Diff between what the broker (Angel One) reports for today and what your TradeJournal believes happened. Lists rows that exist only in one side."
        why="Drift here means the v2 stack lost an order, double-booked a trade, or a paper trade leaked into live numbers. Realised P&L computed on stale data gets every downstream metric wrong."
        act="If mismatched_count > 0, open the offending row in the journal and reconcile by hand before trusting anything else on this dashboard. If the broker side is empty, the broker login failed — re-auth in /brokers."
      />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Broker rows" value={String(data.broker_count)} />
        <KPI label="Journal rows" value={String(data.journal_count)} />
        <KPI label="P&amp;L delta" value={fmtInr(data.pnl_delta)}
             tone={Math.abs(data.pnl_delta) < 1 ? "success" : "warning"} />
        <KPI label="Mismatched" value={String(data.mismatched_count)}
             tone={data.mismatched_count > 0 ? "warning" : "neutral"} />
      </section>
      {data.broker_note ? (
        <Card><CardContent className="py-3 text-body-sm text-fg-muted flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-warning" /> {data.broker_note}
        </CardContent></Card>
      ) : null}
      <div className="grid md:grid-cols-2 gap-4">
        <ReconList title="Only in broker" rows={data.only_in_broker} />
        <ReconList title="Only in journal" rows={data.only_in_journal} />
      </div>
    </PanelWrap>
  );
}

function ReconList({ title, rows }: { title: string; rows: { symbol: string; side?: string; qty?: number; pnl?: number }[] }) {
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent>
        {rows.length === 0 ? <EmptyState title="No discrepancies" /> : (
          <ul className="space-y-1 text-body-sm">
            {rows.map((r, i) => (
              <li key={i} className="flex items-center justify-between">
                <span>{r.symbol} {r.side ? `· ${r.side}` : ""} {r.qty != null ? `· ${r.qty}` : ""}</span>
                {r.pnl != null ? <span className="tabular-nums">{fmtInr(r.pnl)}</span> : null}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/* ------------------------------ 8. Edge Decay ----------------------------- */
function EdgeDecayPanel() {
  const { data, isLoading } = useEdgeDecay();
  if (isLoading || !data) return <LoadingPanel />;
  // Backend returns series as {strategy: [points]}; tolerate the older array shape too.
  const seriesList: { strategy: string; points: { expectancy: number; win_rate: number; as_of?: string; n?: number; avg_r?: number }[] }[] =
    Array.isArray(data.series)
      ? (data.series as { strategy: string; points: { expectancy: number; win_rate: number; as_of?: string; n?: number; avg_r?: number }[] }[])
      : Object.entries(data.series as Record<string, { expectancy: number; win_rate: number; as_of?: string; n?: number; avg_r?: number }[]>)
          .map(([strategy, points]) => ({ strategy, points }));

  if (seriesList.length === 0)
    return (
      <PanelWrap>
        <HelpBlock
          what={`Rolling ${data.window}-trade expectancy and win-rate per strategy, in trade order.`}
          why="Strategies don't stay profitable forever. If expectancy is trending down for 3+ buckets, the setup is decaying — competition caught up, regime shifted, or you're trading a stale signal."
          act="Pause or down-size any strategy where expectancy turns negative or win-rate drops 10+ points over the last 3 windows. Compare against the Regime tab to see if it's the regime, not the setup."
        />
        <EmptyState title="Not enough closed trades yet" description={`Need at least ${data.window} closed trades per strategy.`} />
      </PanelWrap>
    );

  const palette = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];
  return (
    <PanelWrap>
      <HelpBlock
        what={`Rolling ${data.window}-trade expectancy and win-rate per strategy, plotted in trade order.`}
        why="If the green line is sloping down on a strategy, your edge is fading — the market caught up or the regime shifted. Catching this early avoids paying a tuition fee."
        act="Pause or down-size any strategy where expectancy turns negative for 3 windows in a row. Cross-check against the Regime tab — sometimes the strategy is fine, the regime just changed."
      />
      {seriesList.map((s, idx) => {
        const points = (s.points || []).map((p, i) => ({ ...p, idx: i }));
        const latest = points[points.length - 1];
        return (
          <Card key={s.strategy}>
            <CardHeader>
              <CardTitle>{s.strategy}</CardTitle>
              <CardDescription>
                {points.length} window{points.length === 1 ? "" : "s"} of {data.window} trades
                {latest ? ` · latest expectancy ${fmtInr(latest.expectancy)} · win-rate ${fmtNum(latest.win_rate, 0)}%` : ""}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {points.length === 0 ? <EmptyState title="No data" /> : (
                <ResponsiveContainer width="100%" height={200}>
                  <RLineChart data={points}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="idx" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="expectancy" stroke={palette[idx % palette.length]} dot={false} strokeWidth={2} />
                  </RLineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        );
      })}
    </PanelWrap>
  );
}

/* ------------------------------ 9. Theta Forecast ------------------------- */
function ThetaPanel() {
  const { data, isLoading } = useThetaForecast();
  if (isLoading || !data) return <LoadingPanel />;
  if (data.count === 0)
    return (
      <PanelWrap>
        <HelpBlock
          what="Per-minute premium-decay projection for every open short-premium position from now until expiry."
          why="Theta is income from being short premium. The curve tells you the *shape* of that income — most decay happens in the last 24h, so a position that looks 'fine' today might bleed badly tomorrow."
          act="If a position's premium-remaining stops decaying (curve flattens with spot moving), gamma is winning over theta — close before it inverts."
        />
        <EmptyState title="No active short-premium positions" description="Activates when there's at least one short straddle or sold leg open." />
      </PanelWrap>
    );
  return (
    <PanelWrap>
      <HelpBlock
        what="Per-minute premium-decay projection for every open short-premium position from now until expiry."
        why="Theta is income from being short premium. The curve tells you the *shape* of that income — most decay happens in the last 24h, so a position that looks 'fine' today might bleed badly tomorrow."
        act="If a position's premium-remaining stops decaying (curve flattens with spot moving), gamma is winning over theta — close before it inverts."
      />
      {data.positions.map((p) => (
        <Card key={p.position_id}>
          <CardHeader>
            <CardTitle>{p.underlying} · {p.expiry}</CardTitle>
            <CardDescription>
              {fmtNum(p.minutes_to_expiry, 0)} min to expiry · Θ/min ≈ {fmtNum(p.theta_per_minute, 2)}
              · Premium remaining {fmtInr(p.premium_remaining)}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <RLineChart data={p.projection}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="minute" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <ReTooltip />
                <Line type="monotone" dataKey="premium" stroke="#3b82f6" dot={false} strokeWidth={2} />
              </RLineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      ))}
    </PanelWrap>
  );
}

/* ------------------------------ 10. Regime -------------------------------- */
function RegimePanel() {
  const { data, isLoading } = useRegimeHeatmap();
  if (isLoading || !data) return <LoadingPanel />;
  const cellMap = new Map<string, { expectancy: number; trades: number }>();
  for (const c of data.cells) cellMap.set(`${c.strategy}|${c.regime}`, c);
  return (
    <PanelWrap>
      <HelpBlock
        what="Every strategy's average P&L per trade, sliced by which market regime it was traded in. Green cells = strategy makes money in that regime."
        why="Most strategies have ONE regime where they print and another where they bleed. Trading the wrong strategy in the wrong regime is the most common edge-killer — this map shows you exactly which combinations to avoid."
        act="Turn off any strategy whose expectancy in the *current* regime is negative. If a strategy is universally green, scale it up; if universally red, retire it."
      />
      <div className="flex items-center gap-2">
        <span className="text-body-sm text-fg-muted">Current regime:</span>
        <Badge tone="info">{data.current_regime || "unknown"}</Badge>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Strategy × regime expectancy</CardTitle>
          <CardDescription>Mean P&amp;L per trade per regime bucket. Green = positive edge.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr>
                  <th className="text-left py-1.5">Strategy</th>
                  {data.regimes.map((r) => <th key={r} className="text-right px-2">{r}</th>)}
                </tr>
              </thead>
              <tbody>
                {data.strategies.map((s) => (
                  <tr key={s} className="border-b border-border/40">
                    <td className="py-1.5 font-medium">{s}</td>
                    {data.regimes.map((r) => {
                      const cell = cellMap.get(`${s}|${r}`);
                      const e = cell?.expectancy ?? 0;
                      const n = cell?.trades ?? 0;
                      const bg = e > 0 ? `rgba(34,197,94,${Math.min(0.6, Math.abs(e) / 1000)})`
                              : e < 0 ? `rgba(239,68,68,${Math.min(0.6, Math.abs(e) / 1000)})`
                              : "transparent";
                      return (
                        <td key={r} className="text-right px-2 tabular-nums" style={{ background: bg }}>
                          {n > 0 ? <>{fmtInr(e)}<div className="text-caption text-fg-subtle">n={n}</div></> : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 11. Correlation --------------------------- */
function CorrelationPanel() {
  const { data, isLoading } = useCorrelationMatrix();
  if (isLoading || !data) return <LoadingPanel />;
  if (data.symbols.length === 0)
    return (
      <PanelWrap>
        <HelpBlock
          what="Pairwise return correlation across every open underlying + 'independent bets' count. Sector & factor concentration shown below."
          why="Five 'different' trades that all correlate 0.9 are actually ONE bet sized 5×. When the market sneezes, you lose 5× the budget you thought you had."
          act="If independent_bets < 3 and you have > 5 positions, you're concentrated — close one underlying or hedge with a futures short."
        />
        <EmptyState title="No open positions to correlate" />
      </PanelWrap>
    );

  const cell = (rho: number) => {
    const a = Math.min(1, Math.abs(rho));
    const bg = rho >= 0
      ? `rgba(34,197,94,${(a * 0.6).toFixed(2)})`
      : `rgba(239,68,68,${(a * 0.6).toFixed(2)})`;
    return bg;
  };
  return (
    <PanelWrap>
      <HelpBlock
        what="Pairwise return correlation across every open underlying + 'independent bets' count. Sector & factor concentration shown below."
        why="Five 'different' trades that all correlate 0.9 are actually ONE bet sized 5×. When the market sneezes, you lose 5× the budget you thought you had."
        act="If independent_bets < 3 and you have > 5 positions, you're concentrated — close one underlying or hedge with a futures short."
      />
      <section className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <KPI label="Open underlyings" value={String(data.symbols.length)} />
        <KPI label="Independent bets" value={String(data.independent_bets)}
             tone={data.independent_bets < 2 ? "warning" : "neutral"} />
        <KPI label="Top sector" value={topKey(data.sector_weights) ?? "—"}
             hint={topKey(data.sector_weights)
                ? fmtPct(data.sector_weights[topKey(data.sector_weights)!], 0)
                : undefined} />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>Pairwise correlation</CardTitle>
          <CardDescription>60-day daily-return Pearson. Green = positive co-movement.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="text-body-sm border-collapse">
              <thead>
                <tr>
                  <th></th>
                  {data.symbols.map((s) => (
                    <th key={s} className="px-2 py-1 text-fg-subtle font-normal">{s}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.symbols.map((row, i) => (
                  <tr key={row}>
                    <th className="px-2 py-1 text-right text-fg-subtle font-normal">{row}</th>
                    {data.symbols.map((col, j) => (
                      <td key={col} className="px-2 py-1 text-center tabular-nums"
                          style={{ background: cell(data.matrix[i][j]) }}>
                        {fmtNum(data.matrix[i][j], 2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
      <div className="grid md:grid-cols-2 gap-4">
        <ConcentrationCard title="Sector" weights={data.sector_weights} />
        <ConcentrationCard title="Factor" weights={data.factor_weights} />
      </div>
    </PanelWrap>
  );
}

function ConcentrationCard({ title, weights }: { title: string; weights: Record<string, number> }) {
  const rows = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  return (
    <Card>
      <CardHeader><CardTitle>{title} weights</CardTitle></CardHeader>
      <CardContent>
        {rows.length === 0 ? <EmptyState title="No data" /> : (
          <ul className="space-y-2 text-body-sm">
            {rows.map(([k, v]) => (
              <li key={k}>
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-fg-muted">{k}</span>
                  <span className="tabular-nums">{fmtPct(v, 1)}</span>
                </div>
                <div className="h-1.5 bg-surface-2 rounded overflow-hidden">
                  <div className="h-full bg-accent" style={{ width: `${Math.min(100, v * 100)}%` }} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/* ------------------------------ 12. Post-Mortem --------------------------- */
function PostMortemPanel() {
  const [month, setMonth] = React.useState(() => {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
  });
  const { data, isLoading } = usePostMortem(month);
  return (
    <PanelWrap>
      <HelpBlock
        what="Every closed trade in the chosen month, auto-classified into one of seven failure modes (SL too tight, exit too early, sizing too small, slippage, news shock, regime mismatch, thesis wrong)."
        why="Without attribution, every loss feels random and unfixable. With it, you can see 'I lose ₹40k/month to stops triggered by noise' and actually do something about it (widen ATR-based stops, swap to a slower timeframe)."
        act="Tally up the top cause and fix one thing per month. SL too tight → widen by 0.5× ATR. Exit too early → trail with a slower MA. Sizing too small → trust the planner more on high-conviction signals."
      />
      <div className="flex items-center gap-2">
        <label className="text-body-sm text-fg-muted">Month</label>
        <input
          type="month"
          value={month}
          onChange={(e) => setMonth(e.target.value)}
          className="h-8 px-2 bg-surface border border-border rounded-sm text-body-sm"
        />
      </div>
      {isLoading || !data ? <LoadingPanel /> : (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="Closed trades" value={String(data.count)} />
            <KPI label="SL too tight" value={String(data.by_cause.sl_too_tight ?? 0)} tone="warning" />
            <KPI label="Exit too early" value={String(data.by_cause.exit_too_early ?? 0)} tone="warning" />
            <KPI label="Thesis wrong" value={String(data.by_cause.thesis_wrong ?? 0)} tone="danger" />
          </section>
          <Card>
            <CardHeader>
              <CardTitle>Per-trade attribution</CardTitle>
              <CardDescription>Auto-classified by heuristics over plan-vs-actual fields.</CardDescription>
            </CardHeader>
            <CardContent>
              {data.rows.length === 0 ? <EmptyState title="No closed trades in this month" /> : (
                <div className="overflow-auto max-h-[480px]">
                  <table className="w-full text-body-sm">
                    <thead className="text-fg-subtle border-b border-border sticky top-0 bg-surface">
                      <tr>
                        <th className="text-left py-1.5">Symbol</th>
                        <th className="text-left">Side</th>
                        <th className="text-right">Entry</th>
                        <th className="text-right">Exit</th>
                        <th className="text-right">P&amp;L</th>
                        <th className="text-left">Cause</th>
                        <th className="text-left">Evidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.rows.map((r) => (
                        <tr key={r.trade_id} className="border-b border-border/40">
                          <td className="py-1.5">{r.symbol}</td>
                          <td>{r.side}</td>
                          <td className="text-right tabular-nums">{fmtNum(r.entry, 2)}</td>
                          <td className="text-right tabular-nums">{fmtNum(r.exit, 2)}</td>
                          <td className={`text-right tabular-nums ${r.pnl >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>
                            {fmtInr(r.pnl)}
                          </td>
                          <td><Badge tone={r.pnl < 0 ? "danger" : "neutral"}>{r.cause}</Badge></td>
                          <td className="text-fg-muted">{r.evidence}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 13. Gap Risk ------------------------------ */
function GapRiskPanel() {
  const { data, isLoading } = useGapRisk();
  if (isLoading || !data) return <LoadingPanel />;
  if (data.positions.length === 0)
    return (
      <PanelWrap>
        <HelpBlock
          what="For positions held overnight, projected P&L at ±0.5/1/2% gaps + a pre-market hedge checklist."
          why="A gap-down opens at the worst possible price — no stop fires, you wear the loss. Knowing the dollar number BEFORE the bell means you can hedge tonight, not panic at 9:15."
          act="If worst-case at ±2% > daily-loss cap, buy protective wings tonight. Check SGX NIFTY at 8:30 IST — if implied gap > 1%, queue your forced-exit before market open."
        />
        <EmptyState title="No overnight positions" description="Gap risk dashboard activates when at least one position carries to next session." />
      </PanelWrap>
    );

  const steps = ["-2", "-1", "-0.5", "0.5", "1", "2"];
  return (
    <PanelWrap>
      <HelpBlock
        what="For positions held overnight, projected P&L at ±0.5/1/2% gaps + a pre-market hedge checklist."
        why="A gap-down opens at the worst possible price — no stop fires, you wear the loss. Knowing the dollar number BEFORE the bell means you can hedge tonight, not panic at 9:15."
        act="If worst-case at ±2% > daily-loss cap, buy protective wings tonight. Check SGX NIFTY at 8:30 IST — if implied gap > 1%, queue your forced-exit before market open."
      />
      <section className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <KPI label="Positions" value={String(data.positions.length)} />
        <KPI label="Implied gap" value={`${fmtNum(data.implied_gap_pct, 2)}%`}
             hint={`source: ${data.implied_gap_source}`} />
        <KPI label="Worst at ±2%" value={fmtInr(Math.min(
          ...data.positions.map((p) => Math.min(p.pnl_at_gap["-2"], p.pnl_at_gap["2"])),
        ))} tone="warning" />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>P&amp;L at gap (per position)</CardTitle>
          <CardDescription>Projection at -2% / -1% / -0.5% / +0.5% / +1% / +2% gap.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr>
                  <th className="text-left py-1.5">Symbol</th>
                  <th className="text-left">Kind</th>
                  <th className="text-right">Qty</th>
                  {steps.map((s) => <th key={s} className="text-right px-2">{s}%</th>)}
                </tr>
              </thead>
              <tbody>
                {data.positions.map((p) => (
                  <tr key={p.symbol} className="border-b border-border/40">
                    <td className="py-1.5">{p.symbol}</td>
                    <td className="text-fg-muted">{p.kind}</td>
                    <td className="text-right tabular-nums">{p.qty}</td>
                    {steps.map((s) => {
                      const v = p.pnl_at_gap[s];
                      return (
                        <td key={s} className={`text-right tabular-nums px-2 ${v < 0 ? "text-pnl-down" : "text-pnl-up"}`}>
                          {fmtInr(v)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Hedge checklist</CardTitle></CardHeader>
        <CardContent>
          <ul className="space-y-1.5 text-body-sm">
            {data.hedge_checklist.map((item, i) => (
              <li key={i} className="flex items-start gap-2">
                <AlertTriangle className="h-3.5 w-3.5 text-warning mt-0.5 shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 14. Liquidity ----------------------------- */
function LiquidityPanel() {
  const { data, isLoading } = useLiquidityMap();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Bid/ask spread (in basis points), depth proxy, and rolling 50-fill average slippage for every symbol in your watchlist + open positions."
        why="A 25-bp spread is fine on entry but kills you on a quick reverse — you give up 50bps in spread alone. Scaling into illiquid names = paying tuition twice."
        act="Avoid symbols where spread_bps > 20 unless you're using limit orders. If avg_slippage > 30bps for a symbol, switch to slice-by-time execution or drop the symbol."
      />
      {data.note ? (
        <Card><CardContent className="py-3 text-body-sm text-fg-muted">{data.note}</CardContent></Card>
      ) : null}
      <Card>
        <CardHeader>
          <CardTitle>Liquidity & slippage ({data.count} symbols)</CardTitle>
          <CardDescription>Live bid/ask spread + rolling historical fill slippage.</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No watchlist + open symbols" /> : (
            <div className="overflow-auto max-h-[480px]">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border sticky top-0 bg-surface">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-right">Bid</th>
                    <th className="text-right">Ask</th>
                    <th className="text-right">Spread (bps)</th>
                    <th className="text-right">Avg slip (bps)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.symbol} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.bid, 2)}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.ask, 2)}</td>
                      <td className={`text-right tabular-nums ${r.spread_bps > 20 ? "text-warning" : ""}`}>
                        {fmtNum(r.spread_bps, 1)}
                      </td>
                      <td className="text-right tabular-nums">{fmtNum(r.avg_historical_slippage_bps, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 15. What-If Sizer ------------------------- */
function SizerPanel() {
  const [form, setForm] = React.useState({
    symbol: "NIFTY", qty: 65, side: "BUY" as "BUY" | "SELL",
    entry: "", stop: "",
  });
  const [result, setResult] = React.useState<SizerResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [pending, setPending] = React.useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setPending(true);
    try {
      const payload = {
        symbol: form.symbol.toUpperCase(),
        qty: Number(form.qty),
        side: form.side,
        entry: form.entry ? Number(form.entry) : undefined,
        stop: form.stop ? Number(form.stop) : undefined,
      };
      const r = await simulateSizer(payload);
      if (r.error) setError(r.error);
      setResult(r);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setPending(false);
    }
  };

  return (
    <PanelWrap>
      <HelpBlock
        what="Type a hypothetical trade and see exactly what it would do to your portfolio — margin used, free cash, leverage, and worst-case loss if your stop hits."
        why="The difference between '1 lot feels safe' and '1 lot uses 18% of free margin' is the difference between a controlled day and a margin call. Run the math BEFORE you click."
        act="If worst-case loss > daily-loss cap room, halve the qty and try again. If leverage post-trade > 3×, close something else first. Entry blank → pulls live LTP automatically."
      />
      <Card>
        <CardHeader>
          <CardTitle>What-If Position Sizer</CardTitle>
          <CardDescription>Project post-trade margin, leverage, and loss-at-stop before you click buy.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="grid grid-cols-2 sm:grid-cols-5 gap-3 items-end">
            <Field label="Symbol">
              <input value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm" />
            </Field>
            <Field label="Qty">
              <input type="number" value={form.qty} onChange={(e) => setForm({ ...form, qty: Number(e.target.value) })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <Field label="Side">
              <select value={form.side} onChange={(e) => setForm({ ...form, side: e.target.value as "BUY" | "SELL" })}
                      className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm">
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
              </select>
            </Field>
            <Field label="Entry (optional)">
              <input value={form.entry} onChange={(e) => setForm({ ...form, entry: e.target.value })}
                     placeholder="auto" inputMode="decimal"
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <Field label="Stop">
              <input value={form.stop} onChange={(e) => setForm({ ...form, stop: e.target.value })}
                     inputMode="decimal"
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <button type="submit" disabled={pending}
                    className="col-span-2 sm:col-span-5 h-9 px-4 bg-accent text-accent-fg rounded-sm text-body-sm font-medium disabled:opacity-50">
              {pending ? "Simulating…" : "Simulate"}
            </button>
          </form>
          {error ? (
            <div className="mt-3 text-body-sm text-pnl-down flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" /> {error}
            </div>
          ) : null}
        </CardContent>
      </Card>
      {result && !result.error ? (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="Margin used" value={fmtInr(result.margin_used)}
                 hint={`+ ${fmtInr(result.post_trade_delta.margin_added)} this leg`} />
            <KPI label="Free cash" value={fmtInr(result.free_cash)}
                 tone={result.free_cash < 0 ? "danger" : "success"} />
            <KPI label="Leverage" value={`${fmtNum(result.leverage_ratio, 2)}×`}
                 tone={result.leverage_ratio > 3 ? "warning" : "neutral"} />
            <KPI label="Worst-case loss" value={fmtInr(result.worst_case_loss_inr)}
                 tone={result.worst_case_loss_inr > result.daily_loss_cap_inr ? "danger" : "neutral"}
                 hint={`cap ${fmtInr(result.daily_loss_cap_inr)} (3%)`} />
          </section>
          <Card>
            <CardHeader><CardTitle>Daily-loss room</CardTitle></CardHeader>
            <CardContent>
              <p className="text-body-sm">
                {`Distance to 3% daily-loss cap: `}
                <span className="font-mono">{fmtNum(result.distance_to_daily_loss_cap_pct, 2)}%</span>
                {`. Today's realised P&L: `}
                <span className="font-mono">{fmtInr(result.realised_pnl_today)}</span>.
              </p>
            </CardContent>
          </Card>
        </>
      ) : null}
    </PanelWrap>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-caption uppercase tracking-wider text-fg-subtle block mb-1">{label}</span>
      {children}
    </label>
  );
}

function topKey(rec: Record<string, number>): string | null {
  let best: string | null = null;
  let bestV = -Infinity;
  for (const [k, v] of Object.entries(rec)) {
    if (v > bestV) { best = k; bestV = v; }
  }
  return best;
}

/* ------------------------------ 16. Structural Stops ---------------------- */
function StopsPanel() {
  const { data, isLoading } = useStructuralStops();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Three candidate stops per open position — last meaningful swing low, the 10-week MA, and a 2.5× ATR trail off the recent high. 'Recommended' is the tightest one that still sits below entry."
        why="Stops set off feel kill swing trades. Anchoring to structure (a real low, a known mean, a vol-aware trail) means you only get stopped when the thesis is genuinely broken, not by noise."
        act="If pct_loss > 5% on a single position, halve the size or pick a tighter candidate. If recommended is null, the stock has gapped above all three — wait for a pullback before adding."
      />
      <Card>
        <CardHeader>
          <CardTitle>Structural stops · {data.count} open position{data.count === 1 ? "" : "s"}</CardTitle>
          <CardDescription>{data.note}</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No open positions" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-right">Entry</th>
                    <th className="text-right">Swing low</th>
                    <th className="text-right">10W MA</th>
                    <th className="text-right">ATR trail</th>
                    <th className="text-right">Recommended</th>
                    <th className="text-right">% loss</th>
                    <th className="text-right">₹ at stop</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.position_id} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.entry, 2)}</td>
                      <td className="text-right tabular-nums">{r.swing_low ? fmtNum(r.swing_low, 2) : "—"}</td>
                      <td className="text-right tabular-nums">{r.ten_wma ? fmtNum(r.ten_wma, 2) : "—"}</td>
                      <td className="text-right tabular-nums">{r.atr_trail ? fmtNum(r.atr_trail, 2) : "—"}</td>
                      <td className="text-right">
                        {r.recommended ? (
                          <Badge tone="info">{r.recommended} · {fmtNum(r.recommended_value ?? 0, 2)}</Badge>
                        ) : <span className="text-fg-subtle">n/a</span>}
                      </td>
                      <td className={`text-right tabular-nums ${r.pct_loss > 5 ? "text-warning" : ""}`}>
                        {r.pct_loss ? `${fmtNum(r.pct_loss, 2)}%` : "—"}
                      </td>
                      <td className="text-right tabular-nums">{r.loss_at_stop_inr ? fmtInr(r.loss_at_stop_inr) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 17. Forced Flat --------------------------- */
function ForcedFlatPanel() {
  const qc = useQueryClient();
  const { data, isLoading } = useForcedFlat();
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<string | null>(null);

  const doFlatten = async () => {
    if (!confirm("Flatten ALL open intraday positions at current LTP? Paper mode only — no live broker call.")) return;
    setPending(true);
    setResult(null);
    try {
      const r = await flattenAll();
      setResult(`Flattened ${r.flattened} trade${r.flattened === 1 ? "" : "s"}.`);
      await qc.invalidateQueries({ queryKey: ["cockpits"] });
    } catch (e) {
      setResult(e instanceof Error ? e.message : "flatten failed");
    } finally {
      setPending(false);
    }
  };

  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what={`Live countdown to ${data.deadline} (square-off deadline) and every open intraday position with live LTP, P&L, and estimated closing-auction slippage.`}
        why="After 15:15 the closing auction absorbs MIS holders at whatever clears — slippage explodes. A one-click flatten-all keeps you disciplined when the move goes against you in the last hour."
        act="When countdown drops under 30 min, exit losers first. Under 5 min: hit Flatten All if you haven't closed manually. Pair with the Gap Risk tab to decide whether anything deserves NRML conversion."
      />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Now (IST)" value={data.now_ist.split(" ")[1] ?? "—"} hint={data.now_ist.split(" ")[0]} />
        <KPI label="Countdown" value={fmtCountdown(data.countdown_seconds)}
             tone={data.countdown_seconds > 0 && data.countdown_seconds < 1800 ? "warning" : "neutral"} />
        <KPI label="Open MIS" value={String(data.count)} />
        <KPI label="Total P&amp;L" value={fmtInr(data.total_pnl)}
             tone={data.total_pnl >= 0 ? "success" : "danger"} />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>Close list</CardTitle>
          <CardDescription>{data.note}</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="Nothing to flatten" /> : (
            <>
              <div className="overflow-auto">
                <table className="w-full text-body-sm">
                  <thead className="text-fg-subtle border-b border-border">
                    <tr>
                      <th className="text-left py-1.5">Symbol</th>
                      <th className="text-left">Side</th>
                      <th className="text-right">Qty</th>
                      <th className="text-right">Entry</th>
                      <th className="text-right">LTP</th>
                      <th className="text-right">P&amp;L</th>
                      <th className="text-right">Est slip (bps)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={r.trade_id} className="border-b border-border/40">
                        <td className="py-1.5">{r.symbol}</td>
                        <td>{r.side}</td>
                        <td className="text-right tabular-nums">{r.qty}</td>
                        <td className="text-right tabular-nums">{fmtNum(r.entry, 2)}</td>
                        <td className="text-right tabular-nums">{fmtNum(r.ltp, 2)}</td>
                        <td className={`text-right tabular-nums ${r.pnl >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtInr(r.pnl)}</td>
                        <td className="text-right tabular-nums">{fmtNum(r.est_slippage_bps, 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-3 flex items-center gap-3">
                <button onClick={doFlatten} disabled={pending}
                        className="h-9 px-4 bg-pnl-down text-white rounded-sm text-body-sm font-medium disabled:opacity-50">
                  {pending ? "Flattening…" : `Flatten all ${data.count} (paper)`}
                </button>
                {result ? <span className="text-body-sm text-fg-muted">{result}</span> : null}
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 18. Slippage vs Edge ---------------------- */
function SlippageEdgePanel() {
  const [form, setForm] = React.useState({ symbol: "NIFTY", qty: 65, setup_avg_r_inr: 5 });
  const [result, setResult] = React.useState<SlippageEdgeResponse | null>(null);
  const [pending, setPending] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); setPending(true);
    try {
      const r = await checkSlippageEdge({
        symbol: form.symbol.toUpperCase(),
        qty: Number(form.qty),
        setup_avg_r_inr: Number(form.setup_avg_r_inr),
      });
      if (r.error) setError(r.error);
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setPending(false);
    }
  };

  const verdictTone = (v: string): "success" | "warning" | "danger" =>
    v === "green" ? "success" : v === "amber" ? "warning" : "danger";

  return (
    <PanelWrap>
      <HelpBlock
        what="Live half-spread + estimated impact + brokerage for the trade you're about to take, vs the historical expected R for the setup. Returns a green / amber / red verdict."
        why="Backtested edges die in execution. A setup that prints +5 R on paper but pays 2 R in spread + impact each time is net-negative — you wouldn't trade it if you knew."
        act="Green = take it. Amber = halve size or use limit orders. Red = skip; the trade can't pay for itself."
      />
      <Card>
        <CardHeader>
          <CardTitle>Pre-trade check</CardTitle>
          <CardDescription>Net edge = expected R × qty − (half-spread + impact + brokerage)</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="grid grid-cols-2 sm:grid-cols-4 gap-3 items-end">
            <Field label="Symbol">
              <input value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm" />
            </Field>
            <Field label="Qty">
              <input type="number" min={1} step="any" value={form.qty}
                     onChange={(e) => setForm({ ...form, qty: Number(e.target.value) })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <Field label="Expected R / trade (INR)">
              <input type="number" min={0} step="any" value={form.setup_avg_r_inr}
                     onChange={(e) => setForm({ ...form, setup_avg_r_inr: Number(e.target.value) })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <button type="submit" disabled={pending}
                    className="h-9 px-4 bg-accent text-accent-fg rounded-sm text-body-sm font-medium disabled:opacity-50">
              {pending ? "Checking…" : "Check"}
            </button>
          </form>
          {error ? <div className="mt-3 text-body-sm text-pnl-down flex items-center gap-2"><AlertTriangle className="h-4 w-4" /> {error}</div> : null}
        </CardContent>
      </Card>
      {result && !result.error ? (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="Verdict" value={result.verdict.toUpperCase()} tone={verdictTone(result.verdict)} />
            <KPI label="Expected edge" value={fmtInr(result.expected_edge_inr)} />
            <KPI label="Total cost" value={fmtInr(result.total_cost_inr)} />
            <KPI label="Net edge" value={fmtInr(result.net_edge_inr)}
                 tone={result.net_edge_inr > 0 ? "success" : "danger"}
                 hint={`edge:cost ${fmtNum(result.edge_to_cost_ratio, 2)}×`} />
          </section>
          <Card>
            <CardHeader><CardTitle>Cost breakdown</CardTitle></CardHeader>
            <CardContent>
              <ul className="text-body-sm space-y-1.5">
                <li>Half-spread × qty: <span className="font-mono">{fmtInr(result.half_spread_inr)}</span></li>
                <li>Impact (5 bps proxy): <span className="font-mono">{fmtInr(result.impact_inr)}</span></li>
                <li>Brokerage (2 × ₹20): <span className="font-mono">{fmtInr(result.brokerage_inr)}</span></li>
              </ul>
              {result.note ? <p className="text-caption text-fg-subtle mt-3">{result.note}</p> : null}
            </CardContent>
          </Card>
        </>
      ) : null}
    </PanelWrap>
  );
}

/* ------------------------------ 19. Opening Range ------------------------- */
function ORBPanel() {
  const { data, isLoading } = useORB();
  if (isLoading || !data) return <LoadingPanel />;

  const stateTone = (s: string) =>
    s === "breakout_up" ? "success"
    : s === "breakout_down" ? "danger"
    : s === "failed_breakout" ? "warning"
    : "neutral";

  return (
    <PanelWrap>
      <HelpBlock
        what="For every watchlist symbol, today's 9:15-9:30 IST opening range (high/low), the breakout state, the first breakout minute, and how many bars have re-entered the range since."
        why="The first 15-min range filters the day. Wide opens trend; tight opens chop. Retests > 0 after a breakout means it's failing — fading is the play, not chasing."
        act="Trade breakouts only when OR / ATR > 0.6 (wide-trend day). If retests ≥ 2, the setup is broken — exit or flip. Symbols still 'inside' after 11:00 are likely chop all day."
      />
      <Card>
        <CardHeader>
          <CardTitle>ORB tracker · {data.count} symbol{data.count === 1 ? "" : "s"}</CardTitle>
          <CardDescription>{data.note}</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No symbols watchlisted yet" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-right">OR high</th>
                    <th className="text-right">OR low</th>
                    <th className="text-right">Width / ATR</th>
                    <th className="text-left">State</th>
                    <th className="text-left">Breakout @</th>
                    <th className="text-right">Retests</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.symbol} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-right tabular-nums">{r.or_high ? fmtNum(r.or_high, 2) : "—"}</td>
                      <td className="text-right tabular-nums">{r.or_low ? fmtNum(r.or_low, 2) : "—"}</td>
                      <td className={`text-right tabular-nums ${r.or_width_atr > 0.6 ? "text-pnl-up" : r.or_width_atr < 0.3 && r.or_width_atr > 0 ? "text-warning" : ""}`}>
                        {r.or_width_atr ? fmtNum(r.or_width_atr, 2) : "—"}
                      </td>
                      <td><Badge tone={stateTone(r.state)}>{r.state}</Badge></td>
                      <td className="text-fg-muted">{r.breakout_time ?? "—"}</td>
                      <td className={`text-right tabular-nums ${r.retests >= 2 ? "text-warning" : ""}`}>{r.retests}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 20. VWAP bands ---------------------------- */
function VWAPPanel() {
  const [symbol, setSymbol] = React.useState("HDFCBANK");
  const [draft, setDraft] = React.useState("HDFCBANK");
  const { data, isLoading } = useVWAPBands(symbol);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setSymbol(draft.trim().toUpperCase());
  };

  const stateTone = (s: string) =>
    s === "stretched_up" ? "warning"
    : s === "stretched_down" ? "warning"
    : s === "no_data" ? "neutral"
    : "success";

  return (
    <PanelWrap>
      <HelpBlock
        what="Intraday anchored VWAP for one symbol with rolling ±1σ / ±2σ bands. Each minute bar's close is plotted against the bands; the σ comes from the last 30 minutes."
        why="VWAP is where institutions get measured. Trades pinned to VWAP get done at fair value; bars stretched past ±2σ are mean-reversion candidates. The σ width tells you whether the symbol is rangey or trending."
        act="Stretched-up + falling tape = short to VWAP. Stretched-down + rising tape = long to VWAP. Inside ±1σ for 20+ bars = chop, skip."
      />
      <form onSubmit={submit} className="flex items-center gap-2">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               placeholder="Symbol (e.g. RELIANCE)"
               className="h-9 px-2 w-48 bg-surface border border-border rounded-sm text-body-sm" />
        <button type="submit" className="h-9 px-3 bg-accent text-accent-fg rounded-sm text-body-sm">Load</button>
      </form>
      {isLoading || !data ? <LoadingPanel /> : (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="VWAP" value={data.vwap ? fmtNum(data.vwap, 2) : "—"} />
            <KPI label="Last close" value={data.last_close ? fmtNum(data.last_close, 2) : "—"} />
            <KPI label="Distance (σ)" value={`${fmtNum(data.dist_sigma, 2)}σ`}
                 tone={Math.abs(data.dist_sigma) >= 2 ? "warning" : "neutral"} />
            <KPI label="State" value={data.state.replace("_"," ")} tone={stateTone(data.state)} />
          </section>
          <Card>
            <CardHeader>
              <CardTitle>{symbol} — VWAP + ±σ bands</CardTitle>
              <CardDescription>
                ±1σ green band · ±2σ amber band · {data.bar_count ?? 0} bars
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.series.length === 0 ? <EmptyState title={data.note || "No bars yet"} /> : (
                <ResponsiveContainer width="100%" height={260}>
                  <RLineChart data={data.series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="t" hide />
                    <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} domain={["auto","auto"]} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="sigma2_up" stroke="#f59e0b" dot={false} strokeWidth={1} strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="sigma1_up" stroke="#22c55e" dot={false} strokeWidth={1} strokeDasharray="2 2" />
                    <Line type="monotone" dataKey="vwap"      stroke="#3b82f6" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="sigma1_dn" stroke="#22c55e" dot={false} strokeWidth={1} strokeDasharray="2 2" />
                    <Line type="monotone" dataKey="sigma2_dn" stroke="#f59e0b" dot={false} strokeWidth={1} strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="c"         stroke="#e6edf3" dot={false} strokeWidth={1.5} />
                  </RLineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 21. First 5-min --------------------------- */
function First5MinPanel() {
  const { data, isLoading } = useFirst5Min();
  if (isLoading || !data) return <LoadingPanel />;

  const tagTone = (t: string): "success" | "warning" | "danger" | "neutral" =>
    t === "TREND_DAY" ? "success"
    : t === "FADE_DAY" ? "warning"
    : t === "COIL_DAY" ? "neutral"
    : t === "RANGE_DAY" ? "neutral"
    : "neutral";

  return (
    <PanelWrap>
      <HelpBlock
        what="At 09:20 IST, every watchlist symbol's 09:15-09:20 5-min bar gets classified into a day-type — TREND_DAY, RANGE_DAY, FADE_DAY, COIL_DAY — based on body %, gap %, and range vs ATR."
        why="The first 5 minutes set the tone. A wide-range trend-day candle predicts continuation; a doji predicts chop; a gap-and-fade predicts a reversal you can short against. Knowing this at 9:20 saves an hour of bad trades."
        act="TREND_DAY → run your breakout playbook. FADE_DAY → fade the gap. COIL_DAY → wait for expansion. RANGE_DAY → cut size by half, target inner swings."
      />
      <Card>
        <CardHeader>
          <CardTitle>First-5-min profile · {data.count} symbol{data.count === 1 ? "" : "s"}</CardTitle>
          <CardDescription>{data.note}</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No watchlist symbols" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">Classification</th>
                    <th className="text-left">Day type</th>
                    <th className="text-right">Gap %</th>
                    <th className="text-right">Body %</th>
                    <th className="text-right">Range/ATR</th>
                    <th className="text-right">Vol</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.symbol} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td>{r.classification.replace("_"," ")}</td>
                      <td><Badge tone={tagTone(r.day_type_tag)}>{r.day_type_tag}</Badge></td>
                      <td className={`text-right tabular-nums ${Math.abs(r.gap_pct) >= 0.5 ? "text-warning" : ""}`}>
                        {r.gap_pct ? `${fmtNum(r.gap_pct, 2)}%` : "—"}
                      </td>
                      <td className="text-right tabular-nums">{r.body_pct ? fmtNum(r.body_pct, 0) + "%" : "—"}</td>
                      <td className="text-right tabular-nums">{r.range_atr ? fmtNum(r.range_atr, 2) : "—"}</td>
                      <td className="text-right tabular-nums">{r.vol || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 22. Base Quality -------------------------- */
function BaseQualityPanel() {
  const [draft, setDraft] = React.useState("");
  const [symbols, setSymbols] = React.useState<string | undefined>(undefined);
  const { data, isLoading } = useBaseQuality(symbols);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setSymbols(draft.trim() || undefined);
  };

  const tagTone = (t: string): "success" | "warning" | "danger" | "neutral" =>
    t === "VCP_TIGHT" ? "success"
    : t === "FLAT_BASE" ? "success"
    : t === "DEEP_BASE" ? "warning"
    : "neutral";

  return (
    <PanelWrap>
      <HelpBlock
        what="For every watchlist symbol (or a custom list), score the most recent base on depth, length, tightness of last-3 weekly closes, and volume dry-up. 0-100 + pattern tag."
        why="Setups bought from tight, dry, well-defined bases work. Setups bought from raw ranges fail. The score lets you ignore 'kinda looks like a base' opinions and trade only the bases that statistically pay."
        act="80+ = ready to break out — set a buy-stop at the pivot. 60-79 = watchlist; needs another week. <60 = pass; the base hasn't formed yet."
      />
      <form onSubmit={submit} className="flex items-center gap-2">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               placeholder="Comma-separated symbols (blank = watchlist)"
               className="h-9 px-2 flex-1 max-w-md bg-surface border border-border rounded-sm text-body-sm" />
        <button type="submit" className="h-9 px-3 bg-accent text-accent-fg rounded-sm text-body-sm">Score</button>
      </form>
      {isLoading || !data ? <LoadingPanel /> : (
        <Card>
          <CardHeader>
            <CardTitle>Base quality · {data.count} symbol{data.count === 1 ? "" : "s"}</CardTitle>
            <CardDescription>{data.note}</CardDescription>
          </CardHeader>
          <CardContent>
            {data.rows.length === 0 ? <EmptyState title="No data" /> : (
              <div className="overflow-auto">
                <table className="w-full text-body-sm">
                  <thead className="text-fg-subtle border-b border-border">
                    <tr>
                      <th className="text-left py-1.5">Symbol</th>
                      <th className="text-right">Score</th>
                      <th className="text-left">Pattern</th>
                      <th className="text-right">Pivot</th>
                      <th className="text-right">Depth %</th>
                      <th className="text-right">Length (w)</th>
                      <th className="text-right">Tightness</th>
                      <th className="text-right">Vol dry-up</th>
                      <th className="text-right">From pivot</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={r.symbol} className="border-b border-border/40">
                        <td className="py-1.5">{r.symbol}</td>
                        <td className={`text-right tabular-nums font-mono font-semibold ${r.score >= 80 ? "text-pnl-up" : r.score >= 60 ? "text-fg" : "text-fg-subtle"}`}>{r.score}</td>
                        <td><Badge tone={tagTone(r.pattern_tag)}>{r.pattern_tag}</Badge></td>
                        <td className="text-right tabular-nums">{fmtNum(r.pivot, 2)}</td>
                        <td className="text-right tabular-nums">{fmtNum(r.depth_pct, 1)}%</td>
                        <td className="text-right tabular-nums">{fmtNum(r.length_weeks, 1)}</td>
                        <td className="text-right tabular-nums">{fmtNum(r.tightness_pct, 2)}%</td>
                        <td className="text-right tabular-nums">{r.volume_dryup ? fmtNum(r.volume_dryup, 2) + "×" : "—"}</td>
                        <td className={`text-right tabular-nums ${r.pct_from_pivot > -3 ? "text-warning" : ""}`}>{fmtNum(r.pct_from_pivot, 2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 23. MTF Stages ---------------------------- */
function MTFStagePanel() {
  const [draft, setDraft] = React.useState("");
  const [symbols, setSymbols] = React.useState<string | undefined>(undefined);
  const { data, isLoading } = useMTFStage(symbols);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setSymbols(draft.trim() || undefined);
  };

  const alignmentTone = (a: string): "success" | "warning" | "danger" | "neutral" =>
    a === "long_aligned" ? "success"
    : a === "short_aligned" ? "danger"
    : a === "conflict" ? "warning"
    : "neutral";

  const stageBadge = (stage?: string) => {
    if (!stage) return <span className="text-fg-subtle">—</span>;
    const tone =
      stage === "STAGE_2" ? "success"
      : stage === "STAGE_4" ? "danger"
      : stage === "STAGE_3" ? "warning"
      : stage === "STAGE_1" ? "neutral"
      : "neutral";
    return <Badge tone={tone}>{stage.replace("STAGE_", "S")}</Badge>;
  };

  return (
    <PanelWrap>
      <HelpBlock
        what="Daily / Weekly / Monthly each classified into a Stan-Weinstein / Oliver-Kell phase (Stage 1 base · 2 uptrend · 3 top · 4 downtrend) using a 30-period MA + slope."
        why="Trading a daily-S2 long when the weekly is S4 is fighting the bigger trend — you'll be right tactically and wrong strategically. Long-aligned means all three timeframes agree, the strongest setup in the book."
        act="Take longs only on long_aligned (or daily-S2 + weekly-S1/S2). Take shorts only on short_aligned. Conflict / mixed = sit out or trade much smaller."
      />
      <form onSubmit={submit} className="flex items-center gap-2">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               placeholder="Comma-separated symbols (blank = watchlist)"
               className="h-9 px-2 flex-1 max-w-md bg-surface border border-border rounded-sm text-body-sm" />
        <button type="submit" className="h-9 px-3 bg-accent text-accent-fg rounded-sm text-body-sm">Scan</button>
      </form>
      {isLoading || !data ? <LoadingPanel /> : (
        <Card>
          <CardHeader>
            <CardTitle>MTF stage scanner · {data.count} symbol{data.count === 1 ? "" : "s"}</CardTitle>
            <CardDescription>{data.note}</CardDescription>
          </CardHeader>
          <CardContent>
            {data.rows.length === 0 ? <EmptyState title="No data" /> : (
              <div className="overflow-auto">
                <table className="w-full text-body-sm">
                  <thead className="text-fg-subtle border-b border-border">
                    <tr>
                      <th className="text-left py-1.5">Symbol</th>
                      <th className="text-left">Daily</th>
                      <th className="text-left">Weekly</th>
                      <th className="text-left">Monthly</th>
                      <th className="text-left">Alignment</th>
                      <th className="text-left">Weinstein S2</th>
                      <th className="text-right">D close</th>
                      <th className="text-right">D-MA slope</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={r.symbol} className="border-b border-border/40">
                        <td className="py-1.5">{r.symbol}</td>
                        <td>{stageBadge(r.daily?.stage)}</td>
                        <td>{stageBadge(r.weekly?.stage)}</td>
                        <td>{stageBadge(r.monthly?.stage)}</td>
                        <td><Badge tone={alignmentTone(r.alignment)}>{r.alignment.replace("_"," ")}</Badge></td>
                        <td>{r.stage2_aligned
                          ? <Badge tone="success">YES</Badge>
                          : <span className="text-fg-subtle">—</span>}</td>
                        <td className="text-right tabular-nums">{r.daily?.close ? fmtNum(r.daily.close, 2) : "—"}</td>
                        <td className={`text-right tabular-nums ${
                          r.daily && r.daily.slope_pct > 0.5 ? "text-pnl-up"
                          : r.daily && r.daily.slope_pct < -0.5 ? "text-pnl-down" : ""
                        }`}>{r.daily ? `${fmtNum(r.daily.slope_pct, 2)}%` : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 24. Fresh vs Extended --------------------- */
function BreakoutClassifierPanel() {
  const [draft, setDraft] = React.useState("");
  const [symbols, setSymbols] = React.useState<string | undefined>(undefined);
  const { data, isLoading } = useBreakoutClassifier(symbols);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setSymbols(draft.trim() || undefined);
  };

  const stateTone = (s: string): "success" | "warning" | "danger" | "neutral" =>
    s === "fresh" ? "success"
    : s === "extended" ? "danger"
    : s === "base_too_shallow" ? "warning"
    : "neutral";

  return (
    <PanelWrap>
      <HelpBlock
        what="For every watchlist symbol, the distance from the recent 60-day pivot + distance from the 20-day MA + recent base depth, classified as fresh / extended / consolidating / base_too_shallow / neutral."
        why="Buying 'fresh' breakouts within 3% of the pivot pays. Buying 'extended' moves 10%+ above the 20DMA pays for someone else's exit. This panel separates the two at a glance."
        act="Trade only `fresh` longs. Skip `extended` — wait for a pullback to the 20DMA. `base_too_shallow` means the setup isn't ready; revisit in a week."
      />
      <form onSubmit={submit} className="flex items-center gap-2">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               placeholder="Comma-separated symbols (blank = watchlist)"
               className="h-9 px-2 flex-1 max-w-md bg-surface border border-border rounded-sm text-body-sm" />
        <button type="submit" className="h-9 px-3 bg-accent text-accent-fg rounded-sm text-body-sm">Classify</button>
      </form>
      {isLoading || !data ? <LoadingPanel /> : (
        <Card>
          <CardHeader>
            <CardTitle>Breakout classifier · {data.count} symbol{data.count === 1 ? "" : "s"}</CardTitle>
            <CardDescription>{data.note}</CardDescription>
          </CardHeader>
          <CardContent>
            {data.rows.length === 0 ? <EmptyState title="No data" /> : (
              <div className="overflow-auto">
                <table className="w-full text-body-sm">
                  <thead className="text-fg-subtle border-b border-border">
                    <tr>
                      <th className="text-left py-1.5">Symbol</th>
                      <th className="text-left">State</th>
                      <th className="text-right">Close</th>
                      <th className="text-right">Pivot</th>
                      <th className="text-right">From pivot</th>
                      <th className="text-right">From 20DMA</th>
                      <th className="text-right">Base depth</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={r.symbol} className="border-b border-border/40">
                        <td className="py-1.5">{r.symbol}</td>
                        <td><Badge tone={stateTone(r.state)}>{r.state.replace("_"," ")}</Badge></td>
                        <td className="text-right tabular-nums">{r.close ? fmtNum(r.close, 2) : "—"}</td>
                        <td className="text-right tabular-nums">{r.pivot ? fmtNum(r.pivot, 2) : "—"}</td>
                        <td className={`text-right tabular-nums ${
                          r.pct_from_pivot > 8 ? "text-warning"
                          : Math.abs(r.pct_from_pivot) < 3 ? "text-pnl-up" : ""
                        }`}>{fmtNum(r.pct_from_pivot, 2)}%</td>
                        <td className={`text-right tabular-nums ${r.pct_from_20dma > 10 ? "text-warning" : ""}`}>{fmtNum(r.pct_from_20dma, 2)}%</td>
                        <td className="text-right tabular-nums">{fmtNum(r.base_depth_pct, 2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 25. OR Failure ---------------------------- */
function ORBFailurePanel() {
  const { data, isLoading } = useORBFailure();
  if (isLoading || !data) return <LoadingPanel />;
  const failed = data.rows.filter((r) => r.failure_flag).length;
  return (
    <PanelWrap>
      <HelpBlock
        what="Watches every watchlist symbol's opening-range breakout. A 'failure' is ≥2 re-entries into the OR within 30 minutes — the trapped-trader fade setup. Reversal-P is that symbol's historical hit rate for the open-close flip."
        why="Failed breakouts are where the smartest fades live. The directional trade got everyone's attention; the failure traps them and runs the stops. Knowing reversal-P lets you size the fade rather than guess."
        act="High reversal-P (>0.5) + failure_flag = take the fade with the OR midpoint as the first target. retest_count >= 4 = the original breakout is dead — flip and fade the other side."
      />
      <section className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <KPI label="Symbols scanned" value={String(data.count)} />
        <KPI label="Active failures" value={String(failed)} tone={failed > 0 ? "warning" : "neutral"} />
        <KPI label="Tradeable fades"
             value={String(data.rows.filter((r) => r.failure_flag && r.reversal_p >= 0.5).length)}
             tone="success" />
      </section>
      <Card>
        <CardHeader><CardTitle>Failure scanner</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No symbols" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">State</th>
                    <th className="text-right">OR high</th>
                    <th className="text-right">OR low</th>
                    <th className="text-right">Retests</th>
                    <th className="text-left">Failure?</th>
                    <th className="text-right">Reversal target</th>
                    <th className="text-right">Reversal P</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.symbol} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-fg-muted">{r.state.replace("_"," ")}</td>
                      <td className="text-right tabular-nums">{r.or_high ? fmtNum(r.or_high, 2) : "—"}</td>
                      <td className="text-right tabular-nums">{r.or_low ? fmtNum(r.or_low, 2) : "—"}</td>
                      <td className={`text-right tabular-nums ${r.retest_count_after_break >= 2 ? "text-warning" : ""}`}>{r.retest_count_after_break}</td>
                      <td>{r.failure_flag ? <Badge tone="warning">FAIL</Badge> : <span className="text-fg-subtle">—</span>}</td>
                      <td className="text-right tabular-nums">{r.reversal_target ? fmtNum(r.reversal_target, 2) : "—"}</td>
                      <td className={`text-right tabular-nums ${r.reversal_p >= 0.5 ? "text-pnl-up" : ""}`}>{r.reversal_p ? fmtNum(r.reversal_p * 100, 0) + "%" : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 26. Edge Ledger --------------------------- */
function EdgeLedgerPanel() {
  const { data, isLoading } = useEdgeLedger();
  if (isLoading || !data) return <LoadingPanel />;
  const t = data.totals;
  return (
    <PanelWrap>
      <HelpBlock
        what="For every closed trade, the round-trip cost (5-bps × qty × 2 spread + ₹40 brokerage) vs realised P&L. Aggregated by strategy and by symbol so cost-heavy churn is visible."
        why="Gross P&L is what you wish you made; net edge is what you actually keep. cost_drag = how much of gross goes back to the broker + bid-ask. A strategy with 80% drag is paying tuition, not making money."
        act="If a strategy's drag > 50%, switch to limit orders or fewer / bigger trades. If a symbol's net_edge is negative across 20+ trades, drop it from the playbook."
      />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Trades" value={String(t.trades)} />
        <KPI label="Gross P&amp;L" value={fmtInr(t.gross_pnl_inr ?? 0)}
             tone={(t.gross_pnl_inr ?? 0) >= 0 ? "success" : "danger"} />
        <KPI label="Cost paid" value={fmtInr(t.cost_inr ?? 0)} tone="warning" />
        <KPI label="Net edge" value={fmtInr(t.net_edge_inr ?? 0)}
             tone={(t.net_edge_inr ?? 0) >= 0 ? "success" : "danger"}
             hint={`drag ${fmtNum(t.cost_drag_pct ?? 0, 1)}% · win ${fmtNum(t.win_rate ?? 0, 0)}%`} />
      </section>
      <div className="grid md:grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle>By strategy</CardTitle></CardHeader>
          <CardContent>
            {Object.keys(data.by_strategy).length === 0 ? <EmptyState title="No closed trades yet" /> : (
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Strategy</th>
                    <th className="text-right">N</th>
                    <th className="text-right">Net edge</th>
                    <th className="text-right">Drag %</th>
                    <th className="text-right">Win %</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.by_strategy).map(([k, v]) => (
                    <tr key={k} className="border-b border-border/40">
                      <td className="py-1.5">{k}</td>
                      <td className="text-right tabular-nums">{v.trades}</td>
                      <td className={`text-right tabular-nums ${(v.net_edge_inr ?? 0) >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtInr(v.net_edge_inr ?? 0)}</td>
                      <td className={`text-right tabular-nums ${(v.cost_drag_pct ?? 0) > 50 ? "text-warning" : ""}`}>{fmtNum(v.cost_drag_pct ?? 0, 0)}%</td>
                      <td className="text-right tabular-nums">{fmtNum(v.win_rate ?? 0, 0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>By symbol · top 15</CardTitle></CardHeader>
          <CardContent>
            {Object.keys(data.by_symbol).length === 0 ? <EmptyState title="No data" /> : (
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-right">N</th>
                    <th className="text-right">Net edge</th>
                    <th className="text-right">Win %</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.by_symbol).map(([k, v]) => (
                    <tr key={k} className="border-b border-border/40">
                      <td className="py-1.5">{k}</td>
                      <td className="text-right tabular-nums">{v.trades}</td>
                      <td className={`text-right tabular-nums ${(v.net_edge_inr ?? 0) >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtInr(v.net_edge_inr ?? 0)}</td>
                      <td className="text-right tabular-nums">{fmtNum(v.win_rate ?? 0, 0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader><CardTitle>Recent trade ledger</CardTitle></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No closed trades" /> : (
            <div className="overflow-auto max-h-[480px]">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border sticky top-0 bg-surface">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">Strategy</th>
                    <th className="text-right">Qty</th>
                    <th className="text-right">Entry → Fill</th>
                    <th className="text-right">Gross</th>
                    <th className="text-right">Cost</th>
                    <th className="text-right">Net edge</th>
                    <th className="text-right">Edge (bps)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.trade_id} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-fg-muted">{r.strategy}</td>
                      <td className="text-right tabular-nums">{r.qty}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.entry, 2)} → {fmtNum(r.fill, 2)}</td>
                      <td className={`text-right tabular-nums ${r.gross_pnl_inr >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtInr(r.gross_pnl_inr)}</td>
                      <td className="text-right tabular-nums">{fmtInr(r.total_cost_inr)}</td>
                      <td className={`text-right tabular-nums ${r.net_edge_inr >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtInr(r.net_edge_inr)}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.edge_bps, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 27. Intraday Capital Rotation ------------- */
function IntradayRotationPanel() {
  const { data, isLoading } = useIntradayRotation();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="5-minute buckets from 09:15 to 15:30 IST showing deployed capital, gross exposure, realised P&L, idle %, and sector breakdown."
        why="Your worst losses live in the buckets where you were over-deployed. Your best gains live where you were idle and waited for a real signal. Seeing the day as buckets exposes the discipline gap."
        act="If peak_utilisation > 80% on a normal day, you're too aggressive. If idle_pct > 70% across most buckets, your screening is missing — widen the watchlist or look at sectors you're not covering."
      />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Capital" value={fmtInr(data.capital)} />
        <KPI label="Peak deployed" value={fmtInr(data.peak_deployed)}
             hint={`${fmtNum(data.peak_utilisation_pct, 1)}% of capital`} />
        <KPI label="Day realised" value={fmtInr(data.total_realised_pnl)}
             tone={data.total_realised_pnl >= 0 ? "success" : "danger"} />
        <KPI label="Slots" value={String(data.buckets.length)} />
      </section>
      <Card>
        <CardHeader><CardTitle>Deployed & realised across the day</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.buckets.length === 0 ? <EmptyState title="No buckets" /> : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={data.buckets}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="slot" tick={{ fill: "var(--color-fg-muted)", fontSize: 10 }} interval={5} />
                <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <ReTooltip />
                <Bar dataKey="deployed" fill="#3b82f6" />
                <Bar dataKey="realised_pnl" fill="#22c55e" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 28. Gap-Fill Probability ------------------ */
function GapFillPanel() {
  const { data, isLoading } = useGapFill();
  if (isLoading || !data) return <LoadingPanel />;
  const statusTone = (s: string): "success" | "warning" | "danger" | "neutral" =>
    s === "filled" ? "success" : s === "open" ? "warning" : "neutral";
  return (
    <PanelWrap>
      <HelpBlock
        what="For every watchlist symbol with |gap| > 0.3%, today's status (open / filled / no_gap) and the historical same-day fill rate for that symbol's recent 90 sessions."
        why="Gaps fade more often than they don't — but some symbols are 'sticky' gappers (low fill rate) while others fill almost every time. The historical_fill_p tells you which is which before you fade."
        act="Open gap + fill_p > 60% = high-conviction fade with the prior close as target. Open gap + fill_p < 40% = let it run; don't fight a sticky gap."
      />
      <Card>
        <CardHeader><CardTitle>Gap board · {data.count} symbols</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No watchlist symbols" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-right">Prev close</th>
                    <th className="text-right">Open</th>
                    <th className="text-right">Gap %</th>
                    <th className="text-left">Status</th>
                    <th className="text-right">Historical fill</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.symbol} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-right tabular-nums">{r.prev_close ? fmtNum(r.prev_close, 2) : "—"}</td>
                      <td className="text-right tabular-nums">{r.open ? fmtNum(r.open, 2) : "—"}</td>
                      <td className={`text-right tabular-nums ${Math.abs(r.gap_pct) >= 0.5 ? "text-warning" : ""}`}>{r.gap_pct ? `${fmtNum(r.gap_pct, 2)}%` : "—"}</td>
                      <td><Badge tone={statusTone(r.status)}>{r.status.replace("_"," ")}</Badge></td>
                      <td className={`text-right tabular-nums ${r.historical_fill_p > 0.6 ? "text-pnl-up" : r.historical_fill_p < 0.4 && r.historical_fill_p > 0 ? "text-pnl-down" : ""}`}>
                        {r.historical_fill_p ? `${fmtNum(r.historical_fill_p * 100, 0)}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 29. Second 5-min -------------------------- */
function Second5MinPanel() {
  const { data, isLoading } = useSecond5Min();
  if (isLoading || !data) return <LoadingPanel />;
  const tagTone = (t: string): "success" | "warning" | "danger" | "neutral" =>
    t === "TREND_CONFIRMED" ? "success"
    : t === "FAIL_DAY" ? "danger"
    : t === "WEAKENING" ? "warning"
    : "neutral";
  return (
    <PanelWrap>
      <HelpBlock
        what="At 09:25 IST, the 09:20-09:25 bar gets compared to the 09:15-09:20 bar — continuation (same direction, breaks extreme), reversal (opposite, breaks open), consolidation (inside), or weak (same but no break)."
        why="The second 5-min is the trade-confirmation of the first. Continuation on rising volume = strongest 'go' signal of the day. Reversal = the first 5-min trapped traders; fade it."
        act="Continuation + vol_ratio > 1.5 = take the breakout playbook full size. Reversal = take the fade trade. Consolidation = wait for the 09:30 break and re-classify."
      />
      <Card>
        <CardHeader><CardTitle>Second 5-min classification</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No watchlist data" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">Classification</th>
                    <th className="text-left">Day type</th>
                    <th className="text-right">Vol ratio</th>
                    <th className="text-right">2nd-bar close</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.symbol} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td>{r.classification.replace("_", " ")}</td>
                      <td><Badge tone={tagTone(r.day_type_tag)}>{r.day_type_tag}</Badge></td>
                      <td className={`text-right tabular-nums ${r.vol_ratio >= 1.5 ? "text-pnl-up" : ""}`}>{r.vol_ratio ? `${fmtNum(r.vol_ratio, 2)}×` : "—"}</td>
                      <td className="text-right tabular-nums">{r.second_bar ? fmtNum(r.second_bar.c, 2) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 30. Vol Regime Strip ---------------------- */
function VolRegimePanel() {
  const [symbol, setSymbol] = React.useState("HDFCBANK");
  const [draft, setDraft] = React.useState("HDFCBANK");
  const { data, isLoading } = useVolRegime(symbol);
  const submit = (e: React.FormEvent) => { e.preventDefault(); setSymbol(draft.trim().toUpperCase()); };

  const regimeTone = (r: string): "success" | "warning" | "danger" | "neutral" =>
    r === "TREND" ? "success"
    : r === "SHOCK" ? "danger"
    : r === "DEAD" ? "warning"
    : "neutral";

  return (
    <PanelWrap>
      <HelpBlock
        what="Per-minute regime classification (TREND / CHOP / DEAD / SHOCK) for a single symbol, with the annualised realised vol and trades-per-second of each bar."
        why="A breakout in a CHOP regime fails; a mean-reversion play in a TREND regime gets steamrolled. Knowing the regime ALONG the day stops you taking the wrong shape of trade for the tape."
        act="TREND = take direction signals at full size. CHOP = scalp ranges or skip. DEAD = stand aside, your edge needs liquidity. SHOCK = cut size 50%, wait one bar for vol to normalise."
      />
      <form onSubmit={submit} className="flex items-center gap-2">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               placeholder="Symbol"
               className="h-9 px-2 w-48 bg-surface border border-border rounded-sm text-body-sm" />
        <button type="submit" className="h-9 px-3 bg-accent text-accent-fg rounded-sm text-body-sm">Load</button>
      </form>
      {isLoading || !data ? <LoadingPanel /> : (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="Current regime" value={data.current_regime} tone={regimeTone(data.current_regime)} />
            <KPI label="Vol (ann %)" value={`${fmtNum(data.current_vol_ann ?? 0, 2)}%`} />
            <KPI label="Trades/sec" value={fmtNum(data.current_tps ?? 0, 1)} />
            <KPI label="Bars" value={String(data.bar_count)} />
          </section>
          <Card>
            <CardHeader><CardTitle>{symbol} — vol & regime</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
            <CardContent>
              {data.series.length === 0 ? <EmptyState title="No bars yet" /> : (
                <ResponsiveContainer width="100%" height={240}>
                  <RLineChart data={data.series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="t" hide />
                    <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="realised_vol_ann" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
                  </RLineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 31. Sector RRG ---------------------------- */
function SectorRRGPanel() {
  const { data, isLoading } = useSectorRRG();
  if (isLoading || !data) return <LoadingPanel />;
  const quadTone = (q: string): "success" | "warning" | "danger" | "neutral" =>
    q === "LEADING" ? "success"
    : q === "WEAKENING" ? "warning"
    : q === "LAGGING" ? "danger"
    : q === "IMPROVING" ? "neutral"
    : "neutral";
  return (
    <PanelWrap>
      <HelpBlock
        what="Relative Rotation Graph: each NIFTY sector's 14-day relative strength vs NIFTY 50 (X) and the rate-of-change of that RS (Y). Quadrant labels: LEADING / WEAKENING / LAGGING / IMPROVING."
        why="Sectors rotate clockwise through the quadrants. Buying LEADING and selling LAGGING is the textbook play, but the real edge is catching IMPROVING (about to lead) and avoiding WEAKENING (about to lag)."
        act="Trade constituents from LEADING sectors long. Start scaling INTO IMPROVING for early rotation. Avoid LAGGING entirely. Short opportunities live in WEAKENING."
      />
      <Card>
        <CardHeader><CardTitle>Sector quadrants</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No sector data (yfinance unavailable?)" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Sector</th>
                    <th className="text-left">Quadrant</th>
                    <th className="text-right">RS ratio</th>
                    <th className="text-right">RS momentum</th>
                    <th className="text-left">Tail (last 10)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.sector} className="border-b border-border/40">
                      <td className="py-1.5">{r.sector.replace("NIFTY_", "")}</td>
                      <td><Badge tone={quadTone(r.quadrant)}>{r.quadrant}</Badge></td>
                      <td className={`text-right tabular-nums ${r.rs_ratio >= 100 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtNum(r.rs_ratio, 2)}</td>
                      <td className={`text-right tabular-nums ${r.rs_momentum > 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtNum(r.rs_momentum, 2)}</td>
                      <td>
                        {r.tail.length > 0 ? (
                          <span className="text-fg-subtle text-caption">
                            {r.tail.map((p) => `${p.rs_ratio.toFixed(0)}`).join("→")}
                          </span>
                        ) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 32. Sector Dispersion --------------------- */
function SectorDispersionPanel() {
  const { data, isLoading } = useSectorDispersion();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Per-sector cross-sectional dispersion of constituent % change today + the top-3 leaders and bottom-3 laggards inside each sector."
        why="High dispersion (>1.5%) means stocks INSIDE a sector are diverging — you can trade the strongest long vs weakest short and harvest the spread. Low dispersion (<0.5%) means sector beta is dominant; pick the sector ETF instead of individual names."
        act="Sort by dispersion (top of table) — those are today's stock-picker sectors. Pair the leader with the laggard for a delta-neutral trade. If everything's low dispersion, take the index trade."
      />
      <Card>
        <CardHeader><CardTitle>Dispersion + leaders/laggards</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No sector data" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Sector</th>
                    <th className="text-right">N</th>
                    <th className="text-right">Median %</th>
                    <th className="text-right">Dispersion</th>
                    <th className="text-left">Leaders</th>
                    <th className="text-left">Laggards</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.sector} className="border-b border-border/40">
                      <td className="py-1.5">{r.sector.replace("NIFTY_", "")}</td>
                      <td className="text-right tabular-nums">{r.cohort_size}</td>
                      <td className={`text-right tabular-nums ${r.median_pct >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtNum(r.median_pct, 2)}%</td>
                      <td className={`text-right tabular-nums ${r.dispersion_pct >= 1.5 ? "text-pnl-up" : ""}`}>{fmtNum(r.dispersion_pct, 2)}%</td>
                      <td className="text-fg-subtle text-caption">
                        {r.leaders.map((l) => `${l.symbol} +${l.pct}%`).join(" · ")}
                      </td>
                      <td className="text-fg-subtle text-caption">
                        {r.laggards.map((l) => `${l.symbol} ${l.pct}%`).join(" · ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 33. Tape Speed ---------------------------- */
function TapeSpeedPanel() {
  const [symbol, setSymbol] = React.useState("HDFCBANK");
  const [draft, setDraft] = React.useState("HDFCBANK");
  const { data, isLoading } = useTapeSpeed(symbol);
  const submit = (e: React.FormEvent) => { e.preventDefault(); setSymbol(draft.trim().toUpperCase()); };
  const tone = (s: string): "success" | "warning" | "danger" | "neutral" =>
    s === "hot" ? "success" : s === "shock" ? "danger" : s === "cold" ? "warning" : "neutral";
  return (
    <PanelWrap>
      <HelpBlock
        what="Per-minute turnover (₹/min) and trades-per-sec for one symbol vs its 20-day same-time-of-day baseline. Hot tape ≥ 1.5×, cold ≤ 0.5×, shock > 2.5×."
        why="A breakout on COLD tape means nobody believes it — it'll fail. A move on HOT tape has institutional flow behind it. Knowing the tape state at the moment of entry is the difference between a winning and losing scalp."
        act="Take breakouts only when tape ≥ 1.5× baseline. SHOCK = cut size 50% and wait one minute. COLD = your entry won't follow through; skip."
      />
      <form onSubmit={submit} className="flex items-center gap-2">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               className="h-9 px-2 w-48 bg-surface border border-border rounded-sm text-body-sm" />
        <button type="submit" className="h-9 px-3 bg-accent text-accent-fg rounded-sm text-body-sm">Load</button>
      </form>
      {isLoading || !data ? <LoadingPanel /> : (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="State" value={data.current_state} tone={tone(data.current_state)} />
            <KPI label="₹/min" value={fmtInr(data.current_rupees_per_min ?? 0)}
                 hint={`baseline ${fmtInr(data.baseline_rupees_per_min)}`} />
            <KPI label="Trades/sec" value={fmtNum(data.current_tps ?? 0, 1)} />
            <KPI label="Vol/min %" value={fmtNum(data.current_realised_vol_pm ?? 0, 3)} />
          </section>
          <Card>
            <CardHeader><CardTitle>{symbol} — tape speed</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
            <CardContent>
              {data.series.length === 0 ? <EmptyState title="No bars yet" /> : (
                <ResponsiveContainer width="100%" height={240}>
                  <RLineChart data={data.series}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="t" hide />
                    <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="ratio_to_baseline" stroke="#22c55e" dot={false} strokeWidth={1.5} />
                  </RLineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 34. News Shocks --------------------------- */
function NewsShockPanel() {
  const qc = useQueryClient();
  const { data, isLoading } = useNewsShock();
  const [draft, setDraft] = React.useState({ symbol: "", minutes: 15, reason: "" });
  const [pending, setPending] = React.useState(false);
  const [feedback, setFeedback] = React.useState<string | null>(null);

  const submitPause = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!draft.symbol.trim()) return;
    setPending(true); setFeedback(null);
    try {
      const r = await pauseSymbol({
        symbol: draft.symbol.trim().toUpperCase(),
        minutes: Math.max(1, Number(draft.minutes) || 15),
        reason: draft.reason.trim() || "manual",
      });
      if (r.error) setFeedback(`error: ${r.error}`);
      else setFeedback(`paused ${r.symbol} until ${r.re_entry_at.slice(11, 19)} UTC`);
      setDraft((d) => ({ ...d, symbol: "" }));
      await qc.invalidateQueries({ queryKey: ["cockpits", "news-shock"] });
    } catch (err) {
      setFeedback(err instanceof Error ? err.message : "request failed");
    } finally {
      setPending(false);
    }
  };

  const doUnpause = async (sym: string) => {
    setPending(true); setFeedback(null);
    try {
      const r = await unpauseSymbol(sym);
      setFeedback(`unpaused ${r.symbol}${r.was_paused ? "" : " (was already clear)"}`);
      await qc.invalidateQueries({ queryKey: ["cockpits", "news-shock"] });
    } catch (err) {
      setFeedback(err instanceof Error ? err.message : "request failed");
    } finally {
      setPending(false);
    }
  };

  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Real-time NSE corporate-action + LULD/circuit/halt monitor for every held + watchlisted symbol, plus a manual circuit-breaker — pause a symbol for N minutes and @RiskGuard will skip new entries on it."
        why="A halted stock is a position you can neither size nor exit. The pause list is the safety net when you spot a shock before the automated feed catches up."
        act="Pause a symbol the moment you see a bad headline. The cooldown auto-expires; you can unpause manually if the situation clears. Default 15 minutes is a fair starting window."
      />
      <section className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <KPI label="Active shocks" value={String(data.events.length)} tone={data.events.length > 0 ? "danger" : "neutral"} />
        <KPI label="Paused symbols" value={String(data.active_pause_count ?? 0)}
             tone={(data.active_pause_count ?? 0) > 0 ? "warning" : "neutral"} />
        <KPI label="Coverage" value={String(data.coverage_symbols.length)}
             hint={`data source: ${data.data_source}`} />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>Circuit-breaker — pause / unpause</CardTitle>
          <CardDescription>
            Default cooldown {data.default_cooldown_min ?? 15} min. Pauses live in cache and auto-expire.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <form onSubmit={submitPause} className="grid grid-cols-2 sm:grid-cols-5 gap-2 items-end">
            <Field label="Symbol">
              <input value={draft.symbol}
                     onChange={(e) => setDraft({ ...draft, symbol: e.target.value })}
                     placeholder="HDFCBANK"
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm" />
            </Field>
            <Field label="Minutes">
              <input type="number" min={1} step="any" value={draft.minutes}
                     onChange={(e) => setDraft({ ...draft, minutes: Number(e.target.value) })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <Field label="Reason (optional)">
              <input value={draft.reason}
                     onChange={(e) => setDraft({ ...draft, reason: e.target.value })}
                     placeholder="circuit hit"
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm" />
            </Field>
            <div className="sm:col-span-2 flex gap-2">
              <button type="submit" disabled={pending}
                      className="h-9 px-3 bg-warning text-bg rounded-sm text-body-sm font-medium disabled:opacity-50">
                <Pause className="h-3.5 w-3.5 inline mr-1" /> Pause
              </button>
              {feedback ? <span className="text-caption text-fg-muted self-center">{feedback}</span> : null}
            </div>
          </form>

          {(data.paused_symbols ?? []).length === 0 ? (
            <p className="text-body-sm text-fg-subtle">No symbols paused.</p>
          ) : (
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr>
                  <th className="text-left py-1.5">Symbol</th>
                  <th className="text-left">Reason</th>
                  <th className="text-left">Re-entry at</th>
                  <th className="text-right">Minutes</th>
                  <th className="text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {(data.paused_symbols ?? []).map((p) => (
                  <tr key={p.symbol} className="border-b border-border/40">
                    <td className="py-1.5 font-mono">{p.symbol}</td>
                    <td className="text-fg-muted">{p.reason}</td>
                    <td className="text-fg-muted">{p.re_entry_at.slice(11, 19)} UTC</td>
                    <td className="text-right tabular-nums">{p.minutes}</td>
                    <td className="text-right">
                      <button onClick={() => doUnpause(p.symbol)} disabled={pending}
                              className="h-7 px-2 text-caption bg-surface-2 border border-border rounded-sm hover:bg-surface-3 disabled:opacity-50">
                        Unpause
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Live shocks · {data.events.length}</CardTitle>
          <CardDescription>{data.note}</CardDescription>
        </CardHeader>
        <CardContent>
          {data.events.length === 0 ? (
            <EmptyState
              title="No active shocks"
              description={`Monitoring ${data.coverage_symbols.length} symbols.`}
            />
          ) : (
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr>
                  <th className="text-left py-1.5">Symbol</th>
                  <th className="text-left">Severity</th>
                  <th className="text-left">Source</th>
                  <th className="text-left">Headline</th>
                  <th className="text-left">Action</th>
                </tr>
              </thead>
              <tbody>
                {data.events.map((e, i) => (
                  <tr key={i} className="border-b border-border/40">
                    <td className="py-1.5">{e.symbol}</td>
                    <td><Badge tone={e.severity === "critical" ? "danger" : e.severity === "warning" ? "warning" : "neutral"}>{e.severity}</Badge></td>
                    <td className="text-fg-muted">{e.source ?? "—"}</td>
                    <td>{e.headline ?? "—"}</td>
                    <td>{e.flatten_recommendation ? <Badge tone="danger">FLATTEN</Badge> : <span className="text-fg-subtle">hold</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 35. FII/DII Flow -------------------------- */
function FIIDIIFlowPanel() {
  const { data, isLoading } = useFIIDIIFlow();
  if (isLoading || !data) return <LoadingPanel />;
  const regimeTone = (r: string): "success" | "warning" | "danger" | "neutral" =>
    r.includes("BULLISH") || r === "ACCUMULATING" ? "success"
    : r.includes("BEARISH") || r === "DISTRIBUTING" ? "danger"
    : r === "MIXED" ? "warning"
    : "neutral";
  return (
    <PanelWrap>
      <HelpBlock
        what="Daily FII + DII cash buy/sell, FII index-futures net OI change, and FII index-options net premium, overlaid on NIFTY 50 closing series for context. Regime classifier tags 5d + 20d rolling sums."
        why="When FIIs sell into rallies for 5+ sessions while NIFTY makes new highs, distribution is happening — and the rally usually breaks within 10 sessions. Conversely, FII buying through a dip is the bottom signal that beats every chart pattern."
        act="If FII cash regime = DISTRIBUTING + NIFTY up 5+ days, halve long exposure. If ACCUMULATING through a drawdown, add longs aggressively."
      />
      {data.regimes ? (
        <Card>
          <CardHeader><CardTitle>Flow regimes</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {Object.entries(data.regimes).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between p-2 rounded-sm border border-border bg-surface">
                  <span className="text-caption text-fg-subtle">{k.replace(/_/g, " ")}</span>
                  <Badge tone={regimeTone(v)}>{v.replace(/_/g, " ")}</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      ) : null}
      <Card>
        <CardHeader><CardTitle>NIFTY {data.days}-day overlay</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.nifty_close.length === 0 ? <EmptyState title="NIFTY series unavailable" /> : (
            <ResponsiveContainer width="100%" height={240}>
              <RLineChart data={data.nifty_close}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fill: "var(--color-fg-muted)", fontSize: 10 }} />
                <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} domain={["auto","auto"]} />
                <ReTooltip />
                <Line type="monotone" dataKey="close" stroke="#3b82f6" dot={false} strokeWidth={2} />
              </RLineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 36. Depth Proxy --------------------------- */
function DepthImbalancePanel() {
  const { data, isLoading } = useDepthImbalance();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="Volume vs 20-day median proxy for level-2 depth (until the WS DEPTH feed lands). Iceberg-likely flag fires when today's volume is ≥ 5× baseline."
        why="A stock trading 5× baseline volume has hidden buyers or sellers — somebody big is in the tape. That's the moment to either ride alongside (matches your direction) or get out of the way (against you)."
        act="Iceberg + your position is profitable + same direction = trail more aggressively, you have wind in the sails. Iceberg + against you = exit, you're not winning that fight."
      />
      <Card>
        <CardHeader><CardTitle>Volume ratio scan</CardTitle><CardDescription>{data.note}</CardDescription></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No data" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-right">Today vol</th>
                    <th className="text-right">Baseline</th>
                    <th className="text-right">Ratio</th>
                    <th className="text-left">Iceberg?</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.symbol} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-right tabular-nums">{r.today_volume.toLocaleString("en-IN")}</td>
                      <td className="text-right tabular-nums">{r.baseline_volume.toLocaleString("en-IN")}</td>
                      <td className={`text-right tabular-nums ${r.volume_ratio >= 5 ? "text-warning" : ""}`}>{fmtNum(r.volume_ratio, 2)}×</td>
                      <td>{r.iceberg_flag ? <Badge tone="warning">ICEBERG?</Badge> : <span className="text-fg-subtle">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 37. Earnings overlay ---------------------- */
function EarningsOverlayPanel() {
  const { data, isLoading } = useEarningsOverlay();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <HelpBlock
        what="For every open position, the next 30 days of corporate events — earnings date, ex-div date, consensus EPS, average post-earnings gap %."
        why="Holding a position into earnings is binary risk — the chart and the thesis don't matter, only the print does. Knowing the date forces a decision: keep, hedge, or close before the event."
        act="3 days before earnings: decide keep / hedge / close. Hedge via a protective long put if delta < 1.5× cost. Close if the post-earnings gap distribution is wider than your stop."
      />
      <Card>
        <CardHeader>
          <CardTitle>Open positions · {data.count}</CardTitle>
          <CardDescription>{data.note}</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No open positions" /> : (
            <div className="overflow-auto">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">Side</th>
                    <th className="text-right">Qty</th>
                    <th className="text-left">Earnings date</th>
                    <th className="text-left">Ex-div date</th>
                    <th className="text-right">Days to event</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.trade_id} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td>{r.side}</td>
                      <td className="text-right tabular-nums">{r.qty}</td>
                      <td className="text-fg-muted">{r.earnings_date ?? "—"}</td>
                      <td className="text-fg-muted">{r.ex_div_date ?? "—"}</td>
                      <td className={`text-right tabular-nums ${r.days_to_event !== null && r.days_to_event <= 3 ? "text-warning" : ""}`}>
                        {r.days_to_event ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 38. Reset / Seed -------------------------- */
const RESET_OPTIONS = [
  { id: "journal",   label: "TradeJournal",       blurb: "Legacy equity + option trade rows" },
  { id: "straddles", label: "StraddlePosition",   blurb: "Active short straddles" },
  { id: "snapshots", label: "PortfolioSnapshot",  blurb: "Legacy + v2 capital snapshots" },
  { id: "audit",     label: "AuditLog",           blurb: "Every agent / risk-engine decision" },
  { id: "signals",   label: "SignalLog",          blurb: "Live-screener signals" },
  { id: "watchlist", label: "WatchlistEntry",     blurb: "Legacy watchlist (be careful!)" },
  { id: "orders",    label: "v2 Order + Outbox",  blurb: "Outbox-pattern order queue" },
  { id: "runs",      label: "AgentRun + Step",    blurb: "Agent console run history" },
  { id: "positions", label: "v2 Position",        blurb: "v2 positions ledger" },
  { id: "cache",     label: "Django cache",       blurb: "Cockpit broker-call cache (always cleared)" },
] as const;

function ResetPanel() {
  const qc = useQueryClient();
  const [picked, setPicked] = React.useState<Set<string>>(new Set());
  const [capital, setCapital] = React.useState<string>("");
  const [noReseed, setNoReseed] = React.useState(false);
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const toggle = (id: string) => {
    const s = new Set(picked);
    s.has(id) ? s.delete(id) : s.add(id);
    setPicked(s);
  };
  const selectAll = () => setPicked(new Set(RESET_OPTIONS.map((o) => o.id)));
  const clearAll = () => setPicked(new Set());

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    await runReset(Array.from(picked), capital ? Number(capital) : undefined);
  };

  const doNuke = async () => {
    const cap = capital ? Number(capital) : 500_000;
    if (!confirm(
      "⚠️  NUKE: This wipes EVERY ROW from BOTH databases (v2 + legacy).\n\n" +
      "Schema + migrations are preserved.\n" +
      (noReseed
        ? "Re-seed disabled — you'll need to create a user before logging back in.\n"
        : `Re-seeds smoke@alphadesk.local / smoke-1234 with ₹${cap.toLocaleString("en-IN")} capital.\n`) +
      "\nThere is no undo. Continue?"
    )) return;
    await runReset(["nuke"], cap);
  };

  const runReset = async (flags: string[], capNum: number | undefined) => {
    setError(null); setResult(null);
    if (flags.length === 0 && capNum === undefined) {
      setError("Pick at least one thing to reset, set a capital seed, or hit NUKE.");
      return;
    }
    if (capNum !== undefined && (!Number.isFinite(capNum) || capNum <= 0)) {
      setError("Capital must be a positive number.");
      return;
    }
    if (!flags.includes("nuke")) {
      const message =
        flags.includes("journal") || flags.includes("straddles")
          ? `This deletes ${flags.length} table${flags.length === 1 ? "" : "s"} of trading data. There is no undo. Continue?`
          : `Apply reset to ${flags.length} target${flags.length === 1 ? "" : "s"}${capNum ? ` + capital ₹${capNum.toLocaleString("en-IN")}` : ""}?`;
      if (!confirm(message)) return;
    }

    setPending(true);
    try {
      const r = await resetTradingData({
        flags, capital: capNum,
        no_reseed: flags.includes("nuke") ? noReseed : undefined,
      });
      if (r.error) setError(r.error);
      else setResult(r.summary || "Done.");
      await qc.invalidateQueries({ queryKey: ["cockpits"] });
    } catch (e) {
      setError(e instanceof Error ? e.message : "request failed");
    } finally {
      setPending(false);
    }
  };

  return (
    <PanelWrap>
      <HelpBlock
        what="Wipe trading data (journal, positions, snapshots, agent runs, cache) so you can start a fresh paper session and watch new trades flow in."
        why="The cockpits accumulate state across runs — stale entries pollute the views. A clean slate makes it obvious which trade is the one you just placed."
        act="Tick the tables to wipe, optionally seed today's capital, then Reset. The Django cache is always flushed so the cockpits don't show pre-reset numbers."
      />
      <Card>
        <CardHeader>
          <CardTitle>Reset trading data</CardTitle>
          <CardDescription>
            Mirrors <code>python manage.py reset_trading_data --confirm</code>.
            Destructive — there is no undo.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-4">
            <div className="flex items-center gap-2">
              <button type="button" onClick={selectAll}
                      className="h-8 px-3 text-body-sm bg-surface-2 border border-border rounded-sm hover:bg-surface-3">
                Select all
              </button>
              <button type="button" onClick={clearAll}
                      className="h-8 px-3 text-body-sm bg-surface-2 border border-border rounded-sm hover:bg-surface-3">
                Clear
              </button>
              <span className="text-body-sm text-fg-subtle ml-2">
                {picked.size} selected
              </span>
            </div>

            <div className="grid sm:grid-cols-2 gap-2">
              {RESET_OPTIONS.map((o) => (
                <label key={o.id}
                       className={cn(
                         "flex items-start gap-3 p-3 rounded-md border cursor-pointer",
                         "border-border bg-surface hover:bg-surface-2 transition-colors",
                         picked.has(o.id) && "border-accent/60 bg-accent/5",
                       )}>
                  <input
                    type="checkbox"
                    checked={picked.has(o.id)}
                    onChange={() => toggle(o.id)}
                    className="mt-0.5"
                  />
                  <div>
                    <div className="font-mono text-body-sm">{o.label}</div>
                    <div className="text-caption text-fg-subtle">{o.blurb}</div>
                  </div>
                </label>
              ))}
            </div>

            <div className="flex items-end gap-3 flex-wrap">
              <Field label="Seed capital (optional, INR)">
                <input
                  type="number" min={1} step="any"
                  value={capital}
                  onChange={(e) => setCapital(e.target.value)}
                  placeholder="e.g. 500000"
                  className="w-48 h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums"
                />
              </Field>
              <button
                type="submit"
                disabled={pending}
                className="h-9 px-4 bg-pnl-down text-white rounded-sm text-body-sm font-medium disabled:opacity-50"
              >
                {pending ? "Resetting…" : "Reset"}
              </button>
            </div>

            {error ? (
              <div className="text-body-sm text-pnl-down flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" /> {error}
              </div>
            ) : null}
            {result ? (
              <pre className="text-caption font-mono whitespace-pre-wrap p-3 bg-surface-2 border border-border rounded-sm">{result}</pre>
            ) : null}
          </form>
        </CardContent>
      </Card>

      <Card className="border-pnl-down/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-pnl-down" />
            Nuclear option — full DB wipe
          </CardTitle>
          <CardDescription>
            Calls Django's <code>flush</code> on every configured database
            (v2 Postgres + legacy SQLite). Schema and migrations stay; every
            row is deleted. Auto re-seeds smoke@alphadesk.local /
            smoke-1234 so you can log back in immediately.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="flex items-center gap-2 text-body-sm">
            <input type="checkbox" checked={noReseed}
                   onChange={(e) => setNoReseed(e.target.checked)} />
            <span>Skip auto re-seed (you'll need to <code>createsuperuser</code> yourself)</span>
          </label>
          <button
            type="button"
            onClick={doNuke}
            disabled={pending}
            className="h-10 px-5 bg-pnl-down text-white rounded-sm text-body-sm font-semibold disabled:opacity-50 hover:bg-pnl-down/90"
          >
            {pending ? "Nuking…" : "🧨 Nuke backend DB"}
          </button>
          <p className="text-caption text-fg-subtle">
            Capital seed (from the field above) defaults to ₹500,000 if blank.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>CLI equivalent</CardTitle>
          <CardDescription>Same operation from a terminal.</CardDescription>
        </CardHeader>
        <CardContent>
          <pre className="text-caption font-mono p-3 bg-surface-2 border border-border rounded-sm overflow-x-auto">
{`# Soft reset: trading tables only, keep watchlist, seed 500k capital
python manage.py reset_trading_data --all --keep-watchlist --capital 500000 --confirm

# Just clear today's agent runs + Django cache
python manage.py reset_trading_data --runs --cache --confirm

# Wipe trade journal only (paranoid mode)
python manage.py reset_trading_data --journal --confirm

# NUCLEAR: flush every row from both databases, re-seed smoke user
python manage.py reset_trading_data --nuke --confirm

# Nuke without re-seeding (you'll need to make a user)
python manage.py reset_trading_data --nuke --no-reseed --confirm`}
          </pre>
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 39. Stock RRG ----------------------------- */
function StockRRGPanel() {
  const [draft, setDraft] = React.useState("");
  const [symbols, setSymbols] = React.useState<string | undefined>(undefined);
  const { data, isLoading } = useStockRRG(symbols);
  const submit = (e: React.FormEvent) => { e.preventDefault(); setSymbols(draft.trim() || undefined); };

  const quadTone = (q: string): "success" | "warning" | "danger" | "neutral" =>
    q === "LEADING" ? "success"
    : q === "WEAKENING" ? "warning"
    : q === "LAGGING" ? "danger"
    : q === "IMPROVING" ? "neutral"
    : "neutral";

  return (
    <PanelWrap>
      <HelpBlock
        what="Relative Rotation Graph at the stock level: each open + watchlisted name's 14-day RS vs NIFTY 50 (X) and the rate-of-change of that RS (Y). Weekly samples, 8-week tails."
        why="Sector RRG tells you where the wind is blowing; stock RRG tells you which boat in the regatta is fastest. Buying LEADING stocks from LEADING sectors is the high-conviction pair-trade."
        act="Add longs to LEADING names. Start scaling INTO IMPROVING for early rotation. Trim WEAKENING. Avoid LAGGING (or short the weakest)."
      />
      <form onSubmit={submit} className="flex items-center gap-2">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               placeholder="Comma-separated symbols (blank = open + watchlist)"
               className="h-9 px-2 flex-1 max-w-md bg-surface border border-border rounded-sm text-body-sm" />
        <button type="submit" className="h-9 px-3 bg-accent text-accent-fg rounded-sm text-body-sm">Scan</button>
      </form>
      {isLoading || !data ? <LoadingPanel /> : (
        <Card>
          <CardHeader>
            <CardTitle>Stock RRG · {data.count} name{data.count === 1 ? "" : "s"}</CardTitle>
            <CardDescription>{data.note}</CardDescription>
          </CardHeader>
          <CardContent>
            {data.rows.length === 0 ? <EmptyState title="No data" description="Need open positions or watchlist entries." /> : (
              <div className="overflow-auto">
                <table className="w-full text-body-sm">
                  <thead className="text-fg-subtle border-b border-border">
                    <tr>
                      <th className="text-left py-1.5">Symbol</th>
                      <th className="text-left">Quadrant</th>
                      <th className="text-right">RS ratio</th>
                      <th className="text-right">RS momentum</th>
                      <th className="text-left">Tail (last 8w)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={r.symbol} className="border-b border-border/40">
                        <td className="py-1.5">{r.symbol}</td>
                        <td><Badge tone={quadTone(r.quadrant)}>{r.quadrant}</Badge></td>
                        <td className={`text-right tabular-nums ${r.rs_ratio >= 100 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtNum(r.rs_ratio, 2)}</td>
                        <td className={`text-right tabular-nums ${r.rs_momentum > 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtNum(r.rs_momentum, 2)}</td>
                        <td>
                          {r.tail.length > 0 ? (
                            <span className="text-fg-subtle text-caption">
                              {r.tail.map((p) => p.rs_ratio.toFixed(0)).join("→")}
                            </span>
                          ) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 40. Partial Fill -------------------------- */
function PartialFillPanel() {
  const { data, isLoading } = usePartialFill();
  if (isLoading || !data) return <LoadingPanel />;
  const t = data.totals;
  return (
    <PanelWrap>
      <HelpBlock
        what="For every closed trade: fill ratio (filled / requested), signed slippage in bps, queue_score ∈ [0,1] (1 = filled at mid, 0 = chased the far side), and execution-quality flags."
        why="Execution quality is the silent edge killer. A strategy that backtests at +5R but consistently chases the market (queue_score < 0.3) pays it all back in spread + impact."
        act="Strategy avg_queue_score < 0.5 = stop using market orders; switch to limit-fragments. Symbols with chronic 'low_fill_ratio' = liquidity-starved; drop them from the playbook."
      />
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Trades scored" value={String(t.trades)} />
        <KPI label="Avg queue score" value={fmtNum(t.avg_queue_score, 2)}
             tone={t.avg_queue_score >= 0.7 ? "success" : t.avg_queue_score >= 0.5 ? "neutral" : "warning"} />
        <KPI label="Avg fill ratio" value={fmtNum(t.avg_fill_ratio, 2)}
             tone={t.avg_fill_ratio >= 0.9 ? "success" : t.avg_fill_ratio >= 0.7 ? "neutral" : "warning"} />
        <KPI label="Quality flags" value={`${t.high_slippage_count + t.low_fill_count + t.chased_count}`}
             hint={`slippage ${t.high_slippage_count} · low fill ${t.low_fill_count} · chased ${t.chased_count}`}
             tone={(t.high_slippage_count + t.chased_count) > 0 ? "warning" : "neutral"} />
      </section>
      <div className="grid md:grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle>By strategy</CardTitle></CardHeader>
          <CardContent>
            {Object.keys(data.by_strategy).length === 0 ? <EmptyState title="No closed trades yet" /> : (
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Strategy</th>
                    <th className="text-right">N</th>
                    <th className="text-right">Avg queue</th>
                    <th className="text-right">Median queue</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.by_strategy).map(([k, v]) => (
                    <tr key={k} className="border-b border-border/40">
                      <td className="py-1.5">{k}</td>
                      <td className="text-right tabular-nums">{v.count}</td>
                      <td className={`text-right tabular-nums ${(v.avg_queue_score ?? 0) < 0.5 ? "text-warning" : ""}`}>
                        {fmtNum(v.avg_queue_score ?? 0, 2)}
                      </td>
                      <td className="text-right tabular-nums">{fmtNum(v.median_queue_score ?? 0, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>By symbol · top 15</CardTitle></CardHeader>
          <CardContent>
            {Object.keys(data.by_symbol).length === 0 ? <EmptyState title="No data" /> : (
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-right">N</th>
                    <th className="text-right">Avg queue</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.by_symbol).map(([k, v]) => (
                    <tr key={k} className="border-b border-border/40">
                      <td className="py-1.5">{k}</td>
                      <td className="text-right tabular-nums">{v.count}</td>
                      <td className={`text-right tabular-nums ${(v.avg_queue_score ?? 0) < 0.5 ? "text-warning" : ""}`}>
                        {fmtNum(v.avg_queue_score ?? 0, 2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader><CardTitle>Recent trades · execution detail</CardTitle></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No closed trades" /> : (
            <div className="overflow-auto max-h-[480px]">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border sticky top-0 bg-surface">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">Strategy</th>
                    <th className="text-right">Qty (req/fill)</th>
                    <th className="text-right">Slippage (bps)</th>
                    <th className="text-right">Queue</th>
                    <th className="text-left">Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.trade_id} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td className="text-fg-muted">{r.strategy}</td>
                      <td className="text-right tabular-nums">{r.qty_requested} / {r.qty_filled}</td>
                      <td className={`text-right tabular-nums ${Math.abs(r.slippage_bps) > 25 ? "text-warning" : ""}`}>
                        {fmtNum(r.slippage_bps, 1)}
                      </td>
                      <td className={`text-right tabular-nums ${r.queue_score < 0.5 ? "text-warning" : "text-pnl-up"}`}>
                        {fmtNum(r.queue_score, 2)}
                      </td>
                      <td className="text-caption">
                        {r.flags.length === 0 ? <span className="text-fg-subtle">—</span>
                          : r.flags.map((f) => <Badge key={f} tone="warning">{f}</Badge>)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 41. Intraday Sector Heatmap (5m) ---------- */
function IntradaySectorHeatmapPanel() {
  const { data, isLoading } = useIntradaySectorHeatmap();
  if (isLoading || !data) return <LoadingPanel />;

  const cellBg = (disp: number, n: number) => {
    if (n === 0) return "transparent";
    const intensity = Math.min(1, disp / 2.0);
    return `rgba(245, 158, 11, ${(intensity * 0.7).toFixed(2)})`;
  };

  return (
    <PanelWrap>
      <HelpBlock
        what="Sector × 5-min slot matrix of cross-sectional dispersion (std-dev of constituent % moves) and per-slot leader/laggard within each sector. Hot cells = stock-picking opportunity in that sector at that minute."
        why="Sector dispersion changes intraday. A sector that was a tight herd at 10:00 can fragment by 11:30 (news, earnings, single-name moves). Catching that switch is when long/short pair trades print."
        act="Hot cells (>1.5 dispersion) = open a long on the slot's leader + short on the laggard; close before the cell cools. Cold cells = trade the sector ETF or skip the cohort."
      />
      <Card>
        <CardHeader>
          <CardTitle>{data.sector_count} sectors × {data.slot_count} slots</CardTitle>
          <CardDescription>{data.note}</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? <EmptyState title="No 5-min data yet (market closed?)" /> : (
            <div className="overflow-auto">
              <table className="text-caption border-collapse">
                <thead>
                  <tr>
                    <th className="sticky left-0 bg-surface px-2 py-1 text-left text-fg-subtle font-normal">Sector \ Slot</th>
                    {data.slots.map((s) => (
                      <th key={s} className="px-1 py-1 text-fg-subtle font-normal" title={s}>
                        {s.slice(0, 5)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.sector} className="border-b border-border/30">
                      <th className="sticky left-0 bg-surface px-2 py-1 text-right text-fg-subtle font-normal whitespace-nowrap">
                        {r.sector.replace("NIFTY_", "")}
                      </th>
                      {r.cells.map((c) => (
                        <td key={c.slot}
                            title={`${c.slot}  n=${c.n}  disp=${c.dispersion}  median=${c.median}%  leader=${c.leader}%  laggard=${c.laggard}%`}
                            className="px-1 py-1 text-center tabular-nums"
                            style={{ background: cellBg(c.dispersion, c.n) }}>
                          {c.n > 0 ? fmtNum(c.dispersion, 1) : "·"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 42. Backtester Simulator ------------------ */
function BacktesterSimPanel() {
  const qc = useQueryClient();
  const [form, setForm] = React.useState({
    symbol: "NIFTY",
    strategy: "directional",
    days: 180,
    splits: 4,
    mc_runs: 200,
    ruin_pct: 30,
  });
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<BTSimResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [saved, setSaved] = React.useState<string | null>(null);
  const { data: pastRuns } = usePastBacktestRuns();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPending(true); setError(null); setResult(null); setSaved(null);
    try {
      const r = await runSimulation({
        symbol: form.symbol.toUpperCase(),
        strategy: form.strategy,
        days: Number(form.days),
        splits: Number(form.splits),
        mc_runs: Number(form.mc_runs),
        ruin_pct: Number(form.ruin_pct),
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "simulation failed");
    } finally {
      setPending(false);
    }
  };

  const doSave = async () => {
    if (!result) return;
    setPending(true); setError(null);
    try {
      const r = await saveBacktestRun({
        symbol: result.symbol,
        strategy: result.strategy,
        days: Number(form.days),
        label: `${result.symbol}-${result.strategy}-${form.days}d`,
      });
      setSaved(`Saved as ${r.id} (code ${r.code_sha})`);
      await qc.invalidateQueries({ queryKey: ["cockpits", "backtest-runs"] });
    } catch (e) {
      setError(e instanceof Error ? e.message : "save failed");
    } finally {
      setPending(false);
    }
  };

  const verdictTone = (v?: string): "success" | "warning" | "danger" | "neutral" =>
    v === "ROBUST" || v === "ALIVE" ? "success"
    : v === "FRAGILE" || v === "FADING" ? "warning"
    : v === "OVERFIT" || v === "DEAD" ? "danger"
    : "neutral";

  return (
    <PanelWrap>
      <HelpBlock
        what="Single button runs the backtester lab — walk-forward + Monte-Carlo + regime stats + cost sensitivity + capacity curve + live edge drift — and lays the whole report out for the trader."
        why="Backtest equity curves lie. The lab fans the same trade list through 6 lenses (overfit decay · ruin probability · per-regime stats · cost stress · capacity wall · live drift) so you see all the failure modes at once."
        act="Run with defaults first; if walk_forward = ROBUST + ruin_probability < 5%, save the run. If any panel is red, fix the strategy before sizing live."
      />

      {/* ── Past runs — both the new lab registry AND legacy Backtest model ── */}
      {(pastRuns?.lab?.length || pastRuns?.legacy?.length) ? (
        <Card>
          <CardHeader>
            <CardTitle>Past runs</CardTitle>
            <CardDescription>
              {(pastRuns.lab?.length ?? 0)} from the lab (in-memory) · {(pastRuns.legacy?.length ?? 0)} persisted in the v2 Backtest table.
              Click any row to reload its config into the form.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-auto max-h-[320px]">
              {pastRuns.lab?.length ? (
                <>
                  <div className="text-caption uppercase tracking-wider text-fg-subtle pb-1.5">Lab registry</div>
                  <table className="w-full text-body-sm mb-4">
                    <thead className="text-fg-subtle border-b border-border">
                      <tr>
                        <th className="text-left py-1.5">Label</th>
                        <th className="text-left">Symbol</th>
                        <th className="text-left">Strategy</th>
                        <th className="text-right">Days</th>
                        <th className="text-right">N</th>
                        <th className="text-right">Win %</th>
                        <th className="text-right">Sharpe</th>
                        <th className="text-right">P&amp;L</th>
                        <th className="text-left">SHA</th>
                        <th className="text-right">When</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pastRuns.lab.map((r) => (
                        <tr key={r.id}
                            className="border-b border-border/40 hover:bg-surface-2/50 cursor-pointer"
                            onClick={() => setForm({
                              ...form,
                              symbol: r.symbol, strategy: r.strategy, days: r.days,
                            })}>
                          <td className="py-1.5 font-mono">{r.label}</td>
                          <td>{r.symbol}</td>
                          <td className="text-fg-muted">{r.strategy}</td>
                          <td className="text-right tabular-nums">{r.days}</td>
                          <td className="text-right tabular-nums">{r.summary.n}</td>
                          <td className="text-right tabular-nums">{fmtNum(r.summary.win_rate, 1)}%</td>
                          <td className="text-right tabular-nums">{fmtNum(r.summary.sharpe, 2)}</td>
                          <td className={`text-right tabular-nums ${r.summary.total_pnl >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtInr(r.summary.total_pnl)}</td>
                          <td className="text-fg-subtle font-mono text-caption">{r.code_sha}</td>
                          <td className="text-right text-fg-subtle text-caption">{fmtRel(r.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : null}
              {pastRuns.legacy?.length ? (
                <>
                  <div className="text-caption uppercase tracking-wider text-fg-subtle pb-1.5">Persisted (v2 Backtest table)</div>
                  <table className="w-full text-body-sm">
                    <thead className="text-fg-subtle border-b border-border">
                      <tr>
                        <th className="text-left py-1.5">Run ID</th>
                        <th className="text-left">From</th>
                        <th className="text-left">To</th>
                        <th className="text-left">Status</th>
                        <th className="text-right">Metrics</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pastRuns.legacy.map((r) => (
                        <tr key={r.id} className="border-b border-border/40">
                          <td className="py-1.5 font-mono text-caption">{r.id.slice(0, 8)}…</td>
                          <td>{r.from_date}</td>
                          <td>{r.to_date}</td>
                          <td><Badge tone={
                            r.status === "done" ? "success"
                            : r.status === "failed" ? "danger"
                            : r.status === "running" ? "warning"
                            : "neutral"
                          }>{r.status}</Badge></td>
                          <td className="text-right text-fg-muted text-caption">
                            {Object.entries(r.metrics || {}).slice(0, 3)
                              .map(([k, v]) => `${k}=${typeof v === "number" ? fmtNum(v, 2) : v}`)
                              .join(" · ") || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : null}
            </div>
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Run a simulation</CardTitle>
          <CardDescription>Backend: 6 backtester endpoints fan out in parallel; report renders below.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="grid grid-cols-2 sm:grid-cols-6 gap-3 items-end">
            <Field label="Symbol">
              <input value={form.symbol}
                     onChange={(e) => setForm({ ...form, symbol: e.target.value })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm" />
            </Field>
            <Field label="Strategy">
              <select value={form.strategy}
                      onChange={(e) => setForm({ ...form, strategy: e.target.value })}
                      className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm">
                <option value="directional">directional</option>
                <option value="pyramid">pyramid</option>
                <option value="short_straddle">short_straddle</option>
                <option value="vertical_spread">vertical_spread</option>
              </select>
            </Field>
            <Field label="Days lookback">
              <input type="number" min={30} max={400} value={form.days}
                     onChange={(e) => setForm({ ...form, days: Number(e.target.value) })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <Field label="WF splits">
              <input type="number" min={2} max={10} value={form.splits}
                     onChange={(e) => setForm({ ...form, splits: Number(e.target.value) })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <Field label="MC runs">
              <input type="number" min={50} max={2000} value={form.mc_runs}
                     onChange={(e) => setForm({ ...form, mc_runs: Number(e.target.value) })}
                     className="w-full h-9 px-2 bg-surface border border-border rounded-sm text-body-sm tabular-nums" />
            </Field>
            <button type="submit" disabled={pending}
                    className="h-9 px-4 bg-accent text-accent-fg rounded-sm text-body-sm font-medium disabled:opacity-50">
              {pending ? "Running…" : "▶ Run simulation"}
            </button>
          </form>
          {error ? (
            <div className="mt-3 text-body-sm text-pnl-down flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" /> {error}
            </div>
          ) : null}
          {saved ? (
            <div className="mt-3 text-body-sm text-pnl-up flex items-center gap-2">
              <Check className="h-4 w-4" /> {saved}
            </div>
          ) : null}
        </CardContent>
      </Card>

      {result ? (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="Walk-forward verdict"
                 value={result.walk_forward?.verdict ?? "—"}
                 tone={verdictTone(result.walk_forward?.verdict)}
                 hint={`decay ${fmtNum(result.walk_forward?.decay_score ?? 0, 2)}`} />
            <KPI label="Ruin probability"
                 value={`${fmtNum((result.monte_carlo?.ruin_probability ?? 0) * 100, 1)}%`}
                 tone={(result.monte_carlo?.ruin_probability ?? 0) > 0.05 ? "danger" : "success"}
                 hint={`${result.monte_carlo?.runs ?? 0} MC paths`} />
            <KPI label="Capacity wall"
                 value={result.capacity?.capacity_wall_inr ? fmtInr(result.capacity.capacity_wall_inr) : "∞"}
                 tone={result.capacity?.capacity_wall_inr ? "warning" : "neutral"}
                 hint="size beyond which edge dies" />
            <KPI label="Live drift verdict"
                 value={result.edge_drift?.verdict ?? "—"}
                 tone={verdictTone(result.edge_drift?.verdict)}
                 hint={`ratio ${fmtNum(result.edge_drift?.drift_ratio ?? 0, 2)}`} />
          </section>

          {/* Walk-forward folds */}
          {result.walk_forward?.folds ? (
            <Card>
              <CardHeader>
                <CardTitle>Walk-forward folds</CardTitle>
                <CardDescription>
                  IS vs OOS expectancy per fold. {result.walk_forward.note}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={result.walk_forward.folds.map((f: any) => ({
                    fold: `F${f.fold}`,
                    is: f.is.expectancy, oos: f.oos.expectancy,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="fold" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <ReTooltip />
                    <Bar dataKey="is" fill="#3b82f6" name="In-sample" />
                    <Bar dataKey="oos" fill="#22c55e" name="Out-of-sample" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          ) : null}

          {/* Monte-Carlo bands */}
          {result.monte_carlo?.bands ? (
            <Card>
              <CardHeader>
                <CardTitle>Monte-Carlo equity bands</CardTitle>
                <CardDescription>
                  p5 / p50 / p95 across {result.monte_carlo.runs} bootstrap paths.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={240}>
                  <RLineChart data={result.monte_carlo.bands}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="trade_idx" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="p05" stroke="#ef4444" dot={false} strokeWidth={1} strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="p50" stroke="#3b82f6" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="p95" stroke="#22c55e" dot={false} strokeWidth={1} strokeDasharray="3 3" />
                  </RLineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          ) : null}

          {/* Regime + cost-sensitivity side by side */}
          <div className="grid md:grid-cols-2 gap-4">
            {result.regime_stats?.by_regime ? (
              <Card>
                <CardHeader><CardTitle>Per-regime stats</CardTitle></CardHeader>
                <CardContent>
                  <table className="w-full text-body-sm">
                    <thead className="text-fg-subtle border-b border-border">
                      <tr>
                        <th className="text-left py-1.5">Regime</th>
                        <th className="text-right">N</th>
                        <th className="text-right">Win %</th>
                        <th className="text-right">Expectancy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.regime_stats.by_regime).map(([k, v]: [string, any]) => (
                        <tr key={k} className="border-b border-border/40">
                          <td className="py-1.5">{k}</td>
                          <td className="text-right tabular-nums">{v.n}</td>
                          <td className="text-right tabular-nums">{fmtNum(v.win_rate, 1)}%</td>
                          <td className={`text-right tabular-nums ${v.expectancy >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>
                            {fmtNum(v.expectancy, 2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            ) : null}
            {result.cost_sensitivity?.scenarios ? (
              <Card>
                <CardHeader>
                  <CardTitle>Cost sensitivity</CardTitle>
                  <CardDescription>Net edge as broker fee scales ×0.5/×1/×2/×3.</CardDescription>
                </CardHeader>
                <CardContent>
                  <table className="w-full text-body-sm">
                    <thead className="text-fg-subtle border-b border-border">
                      <tr>
                        <th className="text-left py-1.5">Scenario</th>
                        <th className="text-right">Cost / trade</th>
                        <th className="text-right">Total P&L</th>
                        <th className="text-right">Sharpe</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.cost_sensitivity.scenarios).map(([k, v]: [string, any]) => (
                        <tr key={k} className="border-b border-border/40">
                          <td className="py-1.5">{k}</td>
                          <td className="text-right tabular-nums">{fmtInr(v.cost_per_trade_inr)}</td>
                          <td className={`text-right tabular-nums ${v.total_pnl >= 0 ? "text-pnl-up" : "text-pnl-down"}`}>{fmtInr(v.total_pnl)}</td>
                          <td className="text-right tabular-nums">{fmtNum(v.sharpe, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            ) : null}
          </div>

          {/* Capacity points */}
          {result.capacity?.points ? (
            <Card>
              <CardHeader>
                <CardTitle>Capacity curve</CardTitle>
                <CardDescription>Expectancy per trade vs deployed size (sqrt-impact model).</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <RLineChart data={result.capacity.points}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="size_inr" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="expectancy_per_trade_r" stroke="#22c55e" dot={true} strokeWidth={2} />
                  </RLineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          ) : null}

          <div className="flex items-center gap-3">
            <button onClick={doSave} disabled={pending}
                    className="h-9 px-4 bg-pnl-up text-bg rounded-sm text-body-sm font-medium disabled:opacity-50">
              💾 Save to registry
            </button>
            <span className="text-caption text-fg-subtle">
              Saving stamps the run with the current git SHA so future comparisons are diffable.
            </span>
          </div>
        </>
      ) : null}
    </PanelWrap>
  );
}

/* ------------------------------ misc -------------------------------------- */
function safeDiv(n: number, d: number): number {
  return d > 0 ? n / d : 0;
}
function fmtCountdown(seconds: number): string {
  if (seconds <= 0) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}
