/**
 * Options Dashboard — the trader's primary cockpit for options workflows.
 *
 * Single-screen layout, designed for 1440+ but reflows below. Pulls live data
 * from /market-data/pulse/, /positions/, /options-positions/, /risk/, and the
 * events firehose. Streaming LTPs come over /ws/ticks/.
 *
 * Sections:
 *   - Header strip:      live indices + VIX + capital + mode
 *   - Strategy launcher: 5 strategy quick-fire cards (left rail)
 *   - TV alerts feed:    recent autofire-eligible webhook signals (left rail)
 *   - Live positions:    open OptionsPosition rows w/ MTM + greeks
 *   - Greeks + gates:    aggregated portfolio greeks + risk-engine traffic light
 *   - Event firehose:    options-tagged events stream
 *
 * Strategies are sourced from a local catalogue, not the backend, because the
 * dashboard renders one "preset" tile per UX intent — some map 1:1 to a
 * registered strategy plugin (`short_straddle`), others (`vertical_spread`)
 * are scaffolded with a disabled CTA until the plugin lands.
 */
import * as React from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  Bot,
  CircleDot,
  Flame,
  Gauge,
  Layers,
  PlayCircle,
  Radio,
  Shield,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Wallet,
  Zap,
} from "lucide-react";

import {
  usePositions,
  useRiskAlerts,
  useRiskOverview,
  useSystemStatus,
  useAuditFeed,
  useOptionsChain,
  useExpiries,
  useStartAgentRun,
  usePortfolios,
  useBrokerLinks,
  pickDefaultPortfolio,
  parseDrfError,
  type OptionPosition,
  type RiskAlert,
} from "@/lib/v2";
import { useMarketPulse } from "@/lib/market-pulse";
import { connect } from "@/lib/ws";
import { clsPnl, cn, fmtInr, fmtPct } from "@/lib/utils";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/Select";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription, SheetBody } from "@/components/ui/Sheet";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/Tooltip";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";

/* ───────────────────────── shared state helpers ───────────────────────── */

function LoadingRow({ count = 1 }: { count?: number }) {
  return (
    <div className="space-y-1.5">
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} className="h-4 w-full" />
      ))}
    </div>
  );
}

function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="text-xs text-rose-400 bg-rose-500/10 border border-rose-500/30 rounded px-3 py-2 flex items-start gap-2">
      <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <p className="font-medium">Couldn't load</p>
        <p className="text-fg-muted line-clamp-2">{message}</p>
      </div>
      {onRetry && (
        <Button variant="ghost" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

/* ───────────────────────── strategy catalogue ───────────────────────── */

type StrategyPreset = {
  id: string;
  name: string;
  blurb: string;
  bias: "UP" | "DOWN" | "RANGE" | "ANY";
  legs: number;
  /** Backend entry-point name registered in pyproject.toml. null = coming soon. */
  pluginName: string | null;
  /** Optional config payload passed to the plugin (e.g. mode override). */
  pluginConfig?: Record<string, unknown>;
  icon: React.ComponentType<{ className?: string }>;
  accent: "emerald" | "rose" | "violet" | "amber" | "sky";
};

const STRATEGIES: StrategyPreset[] = [
  {
    id: "bull_put",
    name: "Bull Put Spread",
    blurb: "Sell ATM-ish PE, buy lower PE. Theta + delta long. Defined risk.",
    bias: "UP",
    legs: 2,
    pluginName: "vertical_spread",
    pluginConfig: { mode: "BULL_PUT" },
    icon: TrendingUp,
    accent: "emerald",
  },
  {
    id: "bear_call",
    name: "Bear Call Spread",
    blurb: "Sell ATM-ish CE, buy higher CE. Theta + delta short. Defined risk.",
    bias: "DOWN",
    legs: 2,
    pluginName: "vertical_spread",
    pluginConfig: { mode: "BEAR_CALL" },
    icon: TrendingDown,
    accent: "rose",
  },
  {
    id: "iron_condor",
    name: "Iron Condor",
    blurb: "Bull put + bear call. Pure theta. Range-bound conviction.",
    bias: "RANGE",
    legs: 4,
    pluginName: "vertical_spread",
    pluginConfig: { mode: "IRON_CONDOR" },
    icon: Layers,
    accent: "violet",
  },
  {
    id: "short_straddle",
    name: "Short Straddle",
    blurb: "Sell ATM CE + PE. Naked. Highest premium, highest gamma risk.",
    bias: "RANGE",
    legs: 2,
    pluginName: "short_straddle",
    icon: Flame,
    accent: "amber",
  },
  {
    id: "pyramid",
    name: "Pyramid CE/PE",
    blurb: "Momentum pyramiding on one leg. Aggressive, intraday.",
    bias: "ANY",
    legs: 1,
    pluginName: null,
    icon: Sparkles,
    accent: "sky",
  },
];

const ACCENT_BG: Record<StrategyPreset["accent"], string> = {
  emerald: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
  rose: "bg-rose-500/10 text-rose-400 border-rose-500/30",
  violet: "bg-violet-500/10 text-violet-400 border-violet-500/30",
  amber: "bg-amber-500/10 text-amber-400 border-amber-500/30",
  sky: "bg-sky-500/10 text-sky-400 border-sky-500/30",
};

/* ───────────────────────── onboarding banner ───────────────────────── */

/**
 * Dismissable checklist that surfaces only when something prevents trading:
 * no portfolio, no broker, or no TradingView link. Each item links to the
 * relevant page. Stores its dismissed state per-user in localStorage so we
 * don't nag once they've consciously decided to skip it.
 */
function OnboardingChecklist() {
  const { data: portfolios, isLoading: pfLoading } = usePortfolios();
  const { data: brokers, isLoading: brLoading } = useBrokerLinks();
  const [dismissed, setDismissed] = React.useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return localStorage.getItem("options-onboarding-dismissed") === "1";
  });

  if (dismissed || pfLoading || brLoading) return null;
  const hasPortfolio = (portfolios?.length ?? 0) > 0;
  const hasBroker = (brokers ?? []).some((b) => b.status === "active");
  // If everything's set up, don't show the banner.
  if (hasPortfolio && hasBroker) return null;

  const onDismiss = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("options-onboarding-dismissed", "1");
    }
    setDismissed(true);
  };

  return (
    <Card className="border-sky-500/30 bg-sky-500/5">
      <CardContent className="py-3 px-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-fg flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-sky-400" />
              Get the Options Desk fully wired
            </p>
            <p className="text-xs text-fg-muted mt-0.5">
              You can still trade against the paper chain right now. To get live broker quotes and
              alert-driven autofire, finish these steps:
            </p>
            <ol className="mt-2 space-y-1 text-xs">
              <li className="flex items-center gap-2">
                <span className={cn(
                  "w-4 h-4 rounded-full flex items-center justify-center text-[10px]",
                  hasPortfolio ? "bg-emerald-500/20 text-emerald-400" : "bg-surface-2 text-fg-muted",
                )}>
                  {hasPortfolio ? "✓" : "1"}
                </span>
                <span className={cn(hasPortfolio && "text-fg-muted line-through")}>
                  Have a portfolio
                </span>
                {!hasPortfolio && (
                  <Link to="/dashboard" className="text-sky-400 hover:underline">Create →</Link>
                )}
              </li>
              <li className="flex items-center gap-2">
                <span className={cn(
                  "w-4 h-4 rounded-full flex items-center justify-center text-[10px]",
                  hasBroker ? "bg-emerald-500/20 text-emerald-400" : "bg-surface-2 text-fg-muted",
                )}>
                  {hasBroker ? "✓" : "2"}
                </span>
                <span className={cn(hasBroker && "text-fg-muted line-through")}>
                  Connect a broker (Angel / Fyers / Zerodha)
                </span>
                {!hasBroker && (
                  <Link to="/brokers" className="text-sky-400 hover:underline">Link →</Link>
                )}
              </li>
              <li className="flex items-center gap-2">
                <span className="w-4 h-4 rounded-full flex items-center justify-center text-[10px] bg-surface-2 text-fg-muted">
                  3
                </span>
                <span>Optional — set up TradingView alerts for autofire</span>
                <Link to="/tradingview" className="text-sky-400 hover:underline">Configure →</Link>
              </li>
            </ol>
          </div>
          <button
            onClick={onDismiss}
            className="text-fg-muted hover:text-fg text-xs"
            aria-label="Dismiss"
          >
            Dismiss
          </button>
        </div>
      </CardContent>
    </Card>
  );
}

/* ───────────────────────── header strip ───────────────────────── */

function QuoteChip({ label, last, pct, stale }: {
  label: string;
  last: number | null;
  pct: number | null;
  stale?: boolean;
}) {
  const tone = pct == null ? "text-fg-muted" : pct >= 0 ? "text-emerald-400" : "text-rose-400";
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-surface-2 border border-border">
      <span className="text-[10px] uppercase tracking-wider text-fg-muted font-medium">{label}</span>
      <span className="font-mono text-sm tabular-nums text-fg">
        {last != null ? last.toLocaleString("en-IN", { maximumFractionDigits: 2 }) : "—"}
      </span>
      {pct != null && (
        <span className={cn("font-mono text-xs tabular-nums", tone)}>
          {pct >= 0 ? "+" : ""}{pct.toFixed(2)}%
        </span>
      )}
      {stale && <Badge tone="warning">stale</Badge>}
    </div>
  );
}

function HeaderStrip() {
  const { data: pulse } = useMarketPulse();
  const { data: risk } = useRiskOverview();
  const { data: sys } = useSystemStatus();
  const indicesIn = pulse?.quotes.indices_in ?? [];
  const vol = pulse?.quotes.vol ?? [];

  // Pick out NIFTY, BANKNIFTY, SENSEX, India VIX
  const pick = (sym: string, group: typeof indicesIn) =>
    group.find((q) => q.symbol.toUpperCase().includes(sym));
  const nifty = pick("NSEI", indicesIn) ?? pick("NIFTY", indicesIn);
  const bnf = pick("NSEBANK", indicesIn) ?? pick("BANKNIFTY", indicesIn);
  const sensex = pick("BSESN", indicesIn) ?? pick("SENSEX", indicesIn);
  const vix = pick("VIX", vol) ?? pick("VIX", indicesIn);

  const isLive = sys?.trading_mode === "live";
  const isPaused = sys?.ai_paused;
  const sessionOpen = pulse?.is_market_open;

  return (
    <div className="flex flex-wrap items-center gap-2 pb-3 border-b border-border">
      <QuoteChip label="NIFTY" last={nifty?.last ?? null} pct={nifty?.change_pct ?? null} stale={nifty?.stale} />
      <QuoteChip label="BANKNIFTY" last={bnf?.last ?? null} pct={bnf?.change_pct ?? null} stale={bnf?.stale} />
      <QuoteChip label="SENSEX" last={sensex?.last ?? null} pct={sensex?.change_pct ?? null} stale={sensex?.stale} />
      <QuoteChip label="VIX" last={vix?.last ?? null} pct={vix?.change_pct ?? null} stale={vix?.stale} />

      <div className="ml-auto flex items-center gap-2">
        {risk && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-surface-2 border border-border">
                <Wallet className="w-3.5 h-3.5 text-fg-muted" />
                <span className="font-mono text-sm tabular-nums">{fmtInr(risk.capital)}</span>
                <span className={cn("font-mono text-xs", clsPnl(-risk.daily_loss))}>
                  {risk.daily_loss > 0 ? "−" : "+"}{fmtInr(Math.abs(risk.daily_loss))}
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              Capital · today's P&L · daily loss budget {fmtPct(risk.daily_loss_limit_pct / 100)}
            </TooltipContent>
          </Tooltip>
        )}

        <Badge tone={isLive ? "danger" : "neutral"} dot>{isLive ? "LIVE" : "PAPER"}</Badge>
        {sessionOpen
          ? <Badge tone="success" dot>Market open</Badge>
          : <Badge tone="neutral">Market closed</Badge>
        }
        {isPaused && <Badge tone="warning" dot>AI paused</Badge>}
      </div>
    </div>
  );
}

/* ───────────────────────── strategy launcher ───────────────────────── */

function StrategyCard({ preset, onClick }: { preset: StrategyPreset; onClick: () => void }) {
  const Icon = preset.icon;
  const enabled = preset.pluginName != null;
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full text-left p-3 rounded-md border transition-colors",
        "bg-surface-2 border-border hover:border-border-strong",
        enabled ? "cursor-pointer" : "opacity-60 cursor-not-allowed",
      )}
      disabled={!enabled}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className={cn("p-1.5 rounded border", ACCENT_BG[preset.accent])}>
          <Icon className="w-3.5 h-3.5" />
        </span>
        <span className="text-sm font-medium text-fg flex-1">{preset.name}</span>
        <Badge tone="neutral">{preset.legs}L</Badge>
      </div>
      <p className="text-xs text-fg-muted leading-snug mb-2">{preset.blurb}</p>
      <div className="flex items-center justify-between">
        <Badge tone={preset.bias === "UP" ? "success" : preset.bias === "DOWN" ? "danger" : "neutral"}>
          bias: {preset.bias}
        </Badge>
        {!enabled && <Badge tone="warning">soon</Badge>}
      </div>
    </button>
  );
}

function StrategyLauncher({ onPick }: { onPick: (p: StrategyPreset) => void }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Zap className="w-3.5 h-3.5" /> Strategy Launcher
        </CardTitle>
        <CardDescription className="text-xs">
          Click to configure & fire. Bias dictates structure; RiskGuard validates.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {STRATEGIES.map((s) => (
          <StrategyCard key={s.id} preset={s} onClick={() => onPick(s)} />
        ))}
      </CardContent>
    </Card>
  );
}

/* ───────────────────────── TV alerts rail ───────────────────────── */

function TVAlertsRail() {
  const { data: audit, isLoading, isError, error, refetch } = useAuditFeed(15);
  const tvEvents = (audit ?? []).filter(
    (e) => e.type.toLowerCase().includes("tradingview") || e.type.toLowerCase().includes("signal"),
  ).slice(0, 6);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Radio className="w-3.5 h-3.5" /> TradingView Alerts
        </CardTitle>
        <CardDescription className="text-xs">
          Inbound webhook signals — autofire-eligible appear in green.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-1.5 px-3 pb-3">
        {isLoading ? (
          <LoadingRow count={3} />
        ) : isError ? (
          <ErrorBox message={parseDrfError(error).message} onRetry={() => refetch()} />
        ) : tvEvents.length === 0 ? (
          <div className="py-2 text-center space-y-1">
            <p className="text-xs text-fg-muted">No recent alerts</p>
            <Link to="/tradingview" className="text-[11px] text-sky-400 hover:underline">
              Set up a webhook →
            </Link>
          </div>
        ) : (
          tvEvents.map((e, i) => {
            const autofired = e.detail?.toLowerCase().includes("autofired");
            return (
              <div
                key={i}
                className="flex items-center gap-2 text-xs px-2 py-1.5 rounded bg-surface border border-border"
              >
                <CircleDot className={cn("w-3 h-3 shrink-0", autofired ? "text-emerald-400" : "text-fg-muted")} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="font-medium text-fg">{e.symbol || "—"}</span>
                    <span className="text-fg-muted truncate">{e.detail}</span>
                  </div>
                  <span className="text-[10px] text-fg-muted">{new Date(e.time).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}

/* ───────────────────────── live option chain ───────────────────────── */

function LiveChainCard({ underlying }: { underlying: string }) {
  const { data, isLoading, isError, error, refetch } = useOptionsChain({
    underlying, strikes_window: 6,
  });
  const rows = data?.rows ?? [];
  const isFallback = data?.is_fallback === true;
  const isLive = !!data && !isFallback && (data.source === "angel_one" || data.source === "fyers" || data.source === "zerodha");
  const failedSources = (data?.attempted_sources ?? []).filter((s) => !s.ok);

  return (
    <Card>
      <CardHeader className="pb-2 flex-row items-center justify-between">
        <div className="min-w-0">
          <CardTitle className="text-sm flex items-center gap-2">
            <Layers className="w-3.5 h-3.5" /> Live Option Chain
          </CardTitle>
          <CardDescription className="text-xs">
            {isLoading ? (
              <Skeleton className="h-3 w-48" />
            ) : data ? (
              <>
                <span className="text-fg">{data.underlying}</span> · expiry{" "}
                <span className="text-fg">{data.expiry || "—"}</span>
                {data.spot > 0 && (
                  <> · spot{" "}
                    <span className="font-mono text-fg">
                      {data.spot.toLocaleString("en-IN", { maximumFractionDigits: 2 })}
                    </span>
                  </>
                )}
                {data.vix != null && data.vix > 0 && <> · VIX {data.vix.toFixed(2)}</>}
                {data.pcr_oi != null && (
                  <> · PCR <span className="font-mono">{data.pcr_oi.toFixed(3)}</span></>
                )}
              </>
            ) : (
              "—"
            )}
          </CardDescription>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Badge tone={isLive ? "success" : "warning"} dot>
                  {data?.source ?? (isLoading ? "…" : "—")}
                </Badge>
              </span>
            </TooltipTrigger>
            <TooltipContent>
              {isLive
                ? `Live data from ${data?.source}`
                : isFallback
                  ? "Broker unavailable — showing paper-synth chain. Connect a broker for live quotes."
                  : "No data source attempted"}
            </TooltipContent>
          </Tooltip>
          <FreshnessIndicator timestamp={data?.fetched_at ?? null} variant="muted" compact />
        </div>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        {isLoading ? (
          <div className="p-4">
            <LoadingRow count={5} />
          </div>
        ) : isError ? (
          <div className="p-3">
            <ErrorBox
              message={parseDrfError(error).message}
              onRetry={() => refetch()}
            />
          </div>
        ) : rows.length === 0 ? (
          <EmptyState
            icon={<Layers />}
            title="No chain rows"
            description={
              failedSources.length
                ? `Tried: ${failedSources.map((s) => `${s.name} (${s.error ?? "failed"})`).join(", ")}`
                : "The broker returned an empty chain. Try a different underlying or expiry."
            }
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-fg-muted border-b border-border">
                  <th className="text-right py-1.5 px-2">CE OI</th>
                  <th className="text-right py-1.5 px-2">CE Δ</th>
                  <th className="text-right py-1.5 px-2">CE bid</th>
                  <th className="text-right py-1.5 px-2 font-medium text-fg">LTP</th>
                  <th className="text-center py-1.5 px-2 font-semibold text-fg border-x border-border">Strike</th>
                  <th className="text-left py-1.5 px-2 font-medium text-fg">LTP</th>
                  <th className="text-left py-1.5 px-2">PE bid</th>
                  <th className="text-left py-1.5 px-2">PE Δ</th>
                  <th className="text-left py-1.5 px-2">PE OI</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const isAtm = r.strike === data?.atm_strike;
                  return (
                    <tr
                      key={r.strike}
                      className={cn(
                        "border-b border-border",
                        isAtm && "bg-amber-500/10",
                      )}
                    >
                      <td className="text-right tabular-nums px-2 py-1 text-fg-muted">
                        {r.ce ? r.ce.oi.toLocaleString("en-IN") : "—"}
                      </td>
                      <td className="text-right tabular-nums px-2 py-1 text-emerald-400">
                        {r.ce ? r.ce.delta.toFixed(2) : "—"}
                      </td>
                      <td className="text-right tabular-nums px-2 py-1 text-fg-muted">
                        {r.ce ? r.ce.bid.toFixed(2) : "—"}
                      </td>
                      <td className="text-right tabular-nums px-2 py-1 font-mono font-medium text-emerald-400">
                        {r.ce ? r.ce.ltp.toFixed(2) : "—"}
                      </td>
                      <td className={cn(
                        "text-center font-mono font-semibold tabular-nums px-2 py-1 border-x border-border",
                        isAtm ? "text-amber-400" : "text-fg",
                      )}>
                        {r.strike}
                      </td>
                      <td className="text-left tabular-nums px-2 py-1 font-mono font-medium text-rose-400">
                        {r.pe ? r.pe.ltp.toFixed(2) : "—"}
                      </td>
                      <td className="text-left tabular-nums px-2 py-1 text-fg-muted">
                        {r.pe ? r.pe.bid.toFixed(2) : "—"}
                      </td>
                      <td className="text-left tabular-nums px-2 py-1 text-rose-400">
                        {r.pe ? r.pe.delta.toFixed(2) : "—"}
                      </td>
                      <td className="text-left tabular-nums px-2 py-1 text-fg-muted">
                        {r.pe ? r.pe.oi.toLocaleString("en-IN") : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {isFallback && (
          <div className="px-3 py-2 border-t border-amber-500/30 bg-amber-500/5 text-xs text-amber-400 flex items-center justify-between gap-2">
            <span>
              Showing synthesised chain — connect a broker for live quotes.
            </span>
            <Link to="/brokers" className="font-medium underline underline-offset-2 hover:no-underline">
              Connect →
            </Link>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ───────────────────────── live positions table ───────────────────────── */

type EnrichedOption = OptionPosition & { combined_ltp: number; combined_sold: number };

function LivePositionsTable({ live }: { live: boolean }) {
  const { data: positions, isLoading, isError, error, refetch } = usePositions();
  const rows: OptionPosition[] = positions?.options ?? [];

  const enriched: EnrichedOption[] = rows.map((p) => ({
    ...p,
    combined_ltp: p.ce_current + p.pe_current,
    combined_sold: p.ce_sell + p.pe_sell,
  }));

  const columns: Column<EnrichedOption>[] = [
    {
      key: "underlying",
      header: "Underlying",
      render: (p) => (
        <div className="flex flex-col">
          <span className="font-medium text-fg">{p.underlying}</span>
          <span className="text-xs text-fg-muted">
            {p.strike ? `${p.strike} ATM` : `CE ${p.ce_strike} / PE ${p.pe_strike}`}
          </span>
        </div>
      ),
    },
    {
      key: "expiry",
      header: "Expiry",
      render: (p) => (
        <div className="flex flex-col">
          <span className="text-sm text-fg">{p.expiry}</span>
          <span className={cn("text-xs", p.dte <= 1 ? "text-rose-400" : "text-fg-muted")}>
            {p.dte} DTE
          </span>
        </div>
      ),
    },
    {
      key: "lots",
      header: "Size",
      render: (p) => (
        <span className="font-mono text-sm tabular-nums">
          {p.lots}×{p.lot_size}
        </span>
      ),
    },
    {
      key: "combined_sold",
      header: "Sold @",
      render: (p) => (
        <span className="font-mono text-sm tabular-nums text-fg-muted">
          ₹{p.combined_sold.toFixed(2)}
        </span>
      ),
    },
    {
      key: "combined_ltp",
      header: "Now",
      render: (p) => {
        const pct = p.combined_sold > 0 ? (p.combined_ltp / p.combined_sold) * 100 : 0;
        const tone = pct < 70 ? "text-emerald-400" : pct < 130 ? "text-fg" : "text-rose-400";
        return (
          <div className="flex flex-col">
            <span className={cn("font-mono text-sm tabular-nums", tone)}>₹{p.combined_ltp.toFixed(2)}</span>
            <span className="text-[10px] text-fg-muted tabular-nums">{pct.toFixed(0)}% of sold</span>
          </div>
        );
      },
    },
    {
      key: "net_delta",
      header: "Δ",
      render: (p) => (
        <span className={cn("font-mono text-sm tabular-nums",
          Math.abs(p.net_delta) > 0.5 ? "text-amber-400" : "text-fg-muted")}>
          {p.net_delta >= 0 ? "+" : ""}{p.net_delta.toFixed(2)}
        </span>
      ),
    },
    {
      key: "pnl_inr",
      header: "P&L",
      render: (p) => (
        <span className={cn("font-mono text-sm font-medium tabular-nums", clsPnl(p.pnl_inr))}>
          {p.pnl_inr >= 0 ? "+" : "−"}{fmtInr(Math.abs(p.pnl_inr))}
        </span>
      ),
    },
    {
      key: "status",
      header: "",
      render: () => (
        <Button variant="ghost" size="sm">Manage</Button>
      ),
    },
  ];

  return (
    <Card>
      <CardHeader className="pb-2 flex-row items-center justify-between">
        <div>
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="w-3.5 h-3.5" /> Open Options Positions
            <Badge tone="neutral">{enriched.length}</Badge>
          </CardTitle>
          <CardDescription className="text-xs">
            Combined premium · Δ · live P&L · click Manage for actions.
          </CardDescription>
        </div>
        <Badge tone={live ? "success" : "neutral"} dot>
          {live ? "live ticks" : "polling"}
        </Badge>
      </CardHeader>
      <CardContent className="px-0 pt-0">
        {isLoading ? (
          <div className="p-4"><LoadingRow count={3} /></div>
        ) : isError ? (
          <div className="p-3">
            <ErrorBox message={parseDrfError(error).message} onRetry={() => refetch()} />
          </div>
        ) : enriched.length === 0 ? (
          <EmptyState
            icon={<Activity />}
            title="No open option positions"
            description="Fire a strategy from the launcher to start trading."
          />
        ) : (
          <DataTable<EnrichedOption>
            columns={columns}
            rows={enriched}
            rowKey={(r) => String(r.id)}
          />
        )}
      </CardContent>
    </Card>
  );
}

/* ───────────────────────── greeks + risk strip ───────────────────────── */

function GreeksRiskStrip() {
  const { data: positions } = usePositions();
  const { data: risk } = useRiskOverview();
  const { data: alerts } = useRiskAlerts();

  const optionsOpen = positions?.options ?? [];
  const netDelta = optionsOpen.reduce((s, p) => s + (p.net_delta ?? 0), 0);
  const totalSold = optionsOpen.reduce((s, p) => s + (p.ce_sell + p.pe_sell) * p.lots * p.lot_size, 0);
  const totalNow = optionsOpen.reduce((s, p) => s + (p.ce_current + p.pe_current) * p.lots * p.lot_size, 0);
  const totalTheta = totalSold > 0 ? (totalSold - totalNow) : 0; // crude theta proxy (gain since entry)

  const riskTone =
    risk?.status === "GREEN" ? "success" :
    risk?.status === "YELLOW" ? "warning" : "danger";
  const criticalAlerts = (alerts ?? []).filter((a: RiskAlert) => a.severity === "critical");

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Shield className="w-3.5 h-3.5" /> Greeks & Risk Gates
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-2">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-fg-muted mb-1">Net Δ</div>
            <div className={cn("font-mono text-lg tabular-nums",
              Math.abs(netDelta) > 0.5 ? "text-amber-400" : "text-fg")}>
              {netDelta >= 0 ? "+" : ""}{netDelta.toFixed(2)}
            </div>
            <div className="text-[10px] text-fg-muted">across {optionsOpen.length} legs</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-fg-muted mb-1">Premium sold</div>
            <div className="font-mono text-lg tabular-nums text-fg">{fmtInr(totalSold)}</div>
            <div className="text-[10px] text-fg-muted">notional at risk</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-fg-muted mb-1">Theta captured</div>
            <div className={cn("font-mono text-lg tabular-nums", clsPnl(totalTheta))}>
              {totalTheta >= 0 ? "+" : "−"}{fmtInr(Math.abs(totalTheta))}
            </div>
            <div className="text-[10px] text-fg-muted">decay since entry</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-fg-muted mb-1">Capital used</div>
            <div className="font-mono text-lg tabular-nums text-fg">
              {risk ? fmtPct(risk.capital_deployed_pct / 100) : "—"}
            </div>
            <div className="text-[10px] text-fg-muted">
              {risk && fmtInr(risk.capital_deployed)} deployed
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-fg-muted mb-1">RiskGuard</div>
            <Badge tone={riskTone} dot>{risk?.status ?? "—"}</Badge>
            {criticalAlerts.length > 0 && (
              <div className="text-[10px] text-rose-400 mt-1 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                {criticalAlerts.length} critical
              </div>
            )}
          </div>
        </div>
        {criticalAlerts.length > 0 && (
          <div className="mt-3 border-t border-border pt-3 space-y-1">
            {criticalAlerts.map((a, i) => (
              <div key={i} className="flex items-start gap-2 text-xs">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
                <span className="text-fg">{a.message}</span>
                <span className="text-fg-muted">— {a.action}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ───────────────────────── event firehose ───────────────────────── */

function EventFirehose() {
  const { data: audit } = useAuditFeed(40);
  // Filter to options-tagged events: type matches straddle|option|verticals|pyramid
  const events = (audit ?? []).filter((e) =>
    /straddle|option|vertical|pyramid|greek|theta/i.test(e.type + " " + e.detail),
  ).slice(0, 12);

  return (
    <Card>
      <CardHeader className="pb-2 flex-row items-center justify-between">
        <CardTitle className="text-sm flex items-center gap-2">
          <Bot className="w-3.5 h-3.5" /> Options Activity
        </CardTitle>
        <FreshnessIndicator timestamp={null} variant="muted" compact />
      </CardHeader>
      <CardContent className="px-3 pb-3">
        {events.length === 0 ? (
          <p className="text-xs text-fg-muted py-4 text-center">
            No recent options activity — fire a strategy or wait for a TV alert.
          </p>
        ) : (
          <div className="space-y-1.5 max-h-[280px] overflow-y-auto">
            {events.map((e, i) => (
              <div
                key={i}
                className="flex items-start gap-2 text-xs px-2 py-1.5 rounded bg-surface border border-border"
              >
                <span className="text-fg-muted tabular-nums shrink-0 mt-0.5">
                  {new Date(e.time).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <Badge tone="neutral">{e.type}</Badge>
                    {e.symbol && <span className="font-medium text-fg">{e.symbol}</span>}
                  </div>
                  <p className="text-fg-muted mt-0.5 line-clamp-2">{e.detail}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ───────────────────────── configure sheet ───────────────────────── */

function ConfigureStrategySheet({
  preset,
  open,
  onClose,
}: {
  preset: StrategyPreset | null;
  open: boolean;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const { data: portfolios, isLoading: pfLoading } = usePortfolios();
  const startRun = useStartAgentRun();

  // Form state
  const [underlying, setUnderlying] = React.useState<string>("NIFTY");
  const [expiry, setExpiry] = React.useState<string>("");   // "" = backend picks nearest
  const [lots, setLots] = React.useState<number>(1);
  const [autoProfitTake, setAutoProfitTake] = React.useState(true);
  const [reentry, setReentry] = React.useState(false);
  const [riskCapPct, setRiskCapPct] = React.useState<number>(0.08);

  // Error state
  const [errorMsg, setErrorMsg] = React.useState<string>("");
  const [fieldErrors, setFieldErrors] = React.useState<Record<string, string[]>>({});

  // Expiry list — refreshes on underlying change.
  const { data: expiriesData, isLoading: expiriesLoading } =
    useExpiries(underlying, 12, open);
  const expiries = expiriesData?.expiries ?? [];

  // Reset expiry when underlying changes — the previously-selected one
  // may not exist in the new underlying's chain.
  React.useEffect(() => {
    setExpiry("");
  }, [underlying]);

  // Live chain preview — wider window (15) so the 80/60 selector has
  // room to walk for normal-vol regimes where the target strike sits
  // 3–5 strikes OTM. Cache-shares with the dashboard's LiveChainCard
  // when the dashboard's window matches.
  const { data: chain, isLoading: chainLoading, isError: chainError } =
    useOptionsChain({
      underlying,
      expiry: expiry || undefined,
      strikes_window: 15,
      enabled: open,
    });

  React.useEffect(() => {
    setErrorMsg("");
    setFieldErrors({});
  }, [preset?.id]);

  if (!preset) return null;

  const enabled = preset.pluginName != null;
  const defaultPortfolio = pickDefaultPortfolio(portfolios, "paper");
  const blockReason: string | null = !enabled
    ? `Plugin "${preset.id}" isn't registered on the backend yet`
    : pfLoading
      ? null
      : !defaultPortfolio
        ? "You need a portfolio before firing. Create one on Dashboard."
        : null;

  /* Compute the strike preview from the live chain.
   *
   * Replaces the earlier "return null silently" with a richer return
   * type so the operator always sees WHY the preview failed:
   *   - { kind: "ok", sell, buy, credit, width, side, atm_pe_ltp, targets }
   *   - { kind: "no-side", reason }                  (preset doesn't preview)
   *   - { kind: "no-chain" }                         (chain is loading / errored)
   *   - { kind: "no-atm", reason }                   (missing ATM data)
   *   - { kind: "no-strikes", reason, walked }       (80/60 rule found no match)
   */
  const SELL_PCT = 0.80;
  const BUY_PCT = 0.60;

  type StrikeSide = "PE" | "CE";
  type StrikeRow = NonNullable<typeof chain>["rows"][number];
  type WalkedStrike = { strike: number; price: number; side: StrikeSide };
  type Preview =
    | { kind: "ok"; sell: StrikeRow; buy: StrikeRow; credit: number; width: number;
        side: StrikeSide; atm_pe_ltp: number; atm_ce_ltp: number;
        target_sell: number; target_buy: number }
    | { kind: "no-side"; reason: string }
    | { kind: "no-chain" }
    | { kind: "no-atm"; reason: string }
    | { kind: "no-strikes"; reason: string; walked: WalkedStrike[];
        target_sell: number; target_buy: number };

  const previewStrikes: Preview = (() => {
    if (!chain) return { kind: "no-chain" };

    const isPutSide = preset.id === "bull_put" || preset.id === "iron_condor";
    const isCallSide = preset.id === "bear_call";
    if (!isPutSide && !isCallSide) {
      return { kind: "no-side", reason: `${preset.name} isn't a vertical spread — no preview available.` };
    }

    // Resolve ATM: prefer backend's hint, else nearest strike to spot.
    let atm = chain.atm_strike ?? undefined;
    if (atm == null && chain.rows.length > 0 && chain.spot > 0) {
      atm = chain.rows
        .slice()
        .sort((a, b) => Math.abs(a.strike - chain.spot) - Math.abs(b.strike - chain.spot))[0]
        ?.strike;
    }
    if (atm == null) {
      return { kind: "no-atm", reason: "Chain returned no ATM hint and no strikes near spot." };
    }

    const atmRow = chain.rows.find((r) => r.strike === atm);
    const atm_pe_ltp = atmRow?.pe?.ltp ?? 0;
    const atm_ce_ltp = atmRow?.ce?.ltp ?? 0;
    const target_sell = isPutSide ? atm_pe_ltp * SELL_PCT : atm_ce_ltp * SELL_PCT;
    const target_buy = isPutSide ? atm_pe_ltp * BUY_PCT : atm_ce_ltp * BUY_PCT;

    if (target_sell <= 0) {
      return {
        kind: "no-atm",
        reason: `ATM ${isPutSide ? "PE" : "CE"} has zero LTP — either market is closed or it's expiry day at the close.`,
      };
    }

    // Walk OTM strikes from the ATM outward.
    const otmRows = isPutSide
      ? chain.rows.filter((r) => r.strike < atm! && r.pe && (r.pe.ltp > 0))
                  .sort((a, b) => b.strike - a.strike)
      : chain.rows.filter((r) => r.strike > atm! && r.ce && (r.ce.ltp > 0))
                  .sort((a, b) => a.strike - b.strike);

    const walked: WalkedStrike[] = otmRows.map((r) => ({
      strike: r.strike,
      price: (isPutSide ? r.pe?.ltp : r.ce?.ltp) ?? 0,
      side: isPutSide ? "PE" : "CE",
    }));

    if (otmRows.length === 0) {
      return {
        kind: "no-strikes",
        reason: "Chain has no priced strikes on the required side. Widen the strikes window or wait for refresh.",
        walked, target_sell, target_buy,
      };
    }

    const sell = otmRows.find((r) => {
      const price = isPutSide ? r.pe?.ltp : r.ce?.ltp;
      return (price ?? 0) > 0 && (price ?? 0) <= target_sell;
    });
    if (!sell) {
      return {
        kind: "no-strikes",
        reason: `No strike at-or-under the 80% target (₹${target_sell.toFixed(2)}). Walked ${walked.length} strikes — closest was ₹${walked[0]?.price.toFixed(2)}. Try widening the chain (backend uses strikes_window=15 here).`,
        walked, target_sell, target_buy,
      };
    }

    const buy = otmRows.find((r) => {
      const sameSideStrike = r.strike;
      if (isPutSide) {
        if (sameSideStrike >= sell.strike) return false;
      } else {
        if (sameSideStrike <= sell.strike) return false;
      }
      const price = isPutSide ? r.pe?.ltp : r.ce?.ltp;
      return (price ?? 0) > 0 && (price ?? 0) <= target_buy;
    });
    if (!buy) {
      return {
        kind: "no-strikes",
        reason: `Found sell leg at ${sell.strike} but no buy leg at-or-under the 60% target (₹${target_buy.toFixed(2)}). The chain may not extend far enough — widen the window.`,
        walked, target_sell, target_buy,
      };
    }

    const sellQuote = isPutSide ? sell.pe : sell.ce;
    const buyQuote = isPutSide ? buy.pe : buy.ce;
    if (!sellQuote || !buyQuote) {
      return { kind: "no-strikes", reason: "Selected legs missing quotes", walked, target_sell, target_buy };
    }
    // Broker-truthful credit: sell at bid (with ltp fallback), buy at ask.
    const sellPrice = sellQuote.bid > 0 ? sellQuote.bid : sellQuote.ltp;
    const buyPrice = buyQuote.ask > 0 ? buyQuote.ask : buyQuote.ltp;
    const credit = sellPrice - buyPrice;
    const width = isPutSide ? sell.strike - buy.strike : buy.strike - sell.strike;

    return {
      kind: "ok",
      sell, buy, credit, width,
      side: isPutSide ? "PE" : "CE",
      atm_pe_ltp, atm_ce_ltp, target_sell, target_buy,
    };
  })();

  const lotSize = underlying === "NIFTY" ? 75 : underlying === "BANKNIFTY" ? 30 : 20;
  const previewOk = previewStrikes.kind === "ok" ? previewStrikes : null;
  const maxProfit = previewOk ? previewOk.credit * lotSize * lots : 0;
  const maxLoss = previewOk
    ? Math.max(0, previewOk.width - previewOk.credit) * lotSize * lots
    : 0;
  const capital = 500_000;
  const riskCapInr = capital * riskCapPct;
  const exceedsCap = maxLoss > 0 && maxLoss > riskCapInr;

  const onFire = async () => {
    setErrorMsg("");
    setFieldErrors({});
    if (!enabled || !defaultPortfolio || !preset.pluginName) return;
    try {
      // Build a config payload matching the strategy's JSONSchema.
      // NB: `allow_reentry` is intentionally NOT sent — the schema doesn't
      // declare it as a known field and the lifecycle module reads it from
      // its own constants. Future: extend the schema if it becomes
      // operator-tunable.
      const config: Record<string, unknown> = {
        ...(preset.pluginConfig ?? {}),
        underlying,
        lots,
        risk_cap_pct: riskCapPct,
        profit_take_pct: autoProfitTake ? 0.70 : 0.95,
      };
      // Only pass expiry when the user actively picked one; otherwise let
      // the backend resolve the nearest weekly via the broker.
      if (expiry) config.expiry = expiry;
      const run = await startRun.mutateAsync({
        strategy_name: preset.pluginName,
        portfolio: defaultPortfolio.id,
        config,
      });
      onClose();
      navigate(`/agents/${run.id}`);
    } catch (err: unknown) {
      const parsed = parseDrfError(err);
      setErrorMsg(parsed.message);
      setFieldErrors(parsed.fields);
    }
  };

  const Icon = preset.icon;
  return (
    <Sheet open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent width="640px">
        <SheetHeader>
          <div className="flex items-start gap-3">
            <span className={cn("p-2 rounded border shrink-0", ACCENT_BG[preset.accent])}>
              <Icon className="w-5 h-5" />
            </span>
            <div className="flex-1 min-w-0">
              <SheetTitle>{preset.name}</SheetTitle>
              <SheetDescription>{preset.blurb}</SheetDescription>
              <div className="flex flex-wrap gap-1.5 mt-2">
                <Badge tone={preset.bias === "UP" ? "success" : preset.bias === "DOWN" ? "danger" : "neutral"}>
                  bias: {preset.bias}
                </Badge>
                <Badge tone="neutral">{preset.legs} legs</Badge>
                <Badge tone="neutral">defined-risk</Badge>
                <Badge tone={enabled ? "success" : "warning"} dot>
                  {enabled ? preset.pluginName : "plugin pending"}
                </Badge>
              </div>
            </div>
          </div>
        </SheetHeader>

        <SheetBody className="space-y-5">
          {/* ── Trade setup ─────────────────────────────────────── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-3 flex items-center gap-1.5">
              <Activity className="w-3 h-3" /> Trade Setup
            </h3>
            <div className="grid grid-cols-3 gap-4 p-4 rounded-md border border-border bg-surface-2">
              <div>
                <label className="text-xs font-medium text-fg block mb-1.5">Underlying</label>
                <Select value={underlying} onValueChange={setUnderlying}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="NIFTY">NIFTY 50</SelectItem>
                    <SelectItem value="BANKNIFTY">BANK NIFTY</SelectItem>
                    <SelectItem value="SENSEX">SENSEX</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-[11px] text-fg-muted mt-1">
                  Lot size: <span className="font-mono">{lotSize}</span>
                </p>
              </div>
              <div>
                <label className="text-xs font-medium text-fg block mb-1.5">Expiry</label>
                <Select
                  value={expiry || "__auto__"}
                  onValueChange={(v) => setExpiry(v === "__auto__" ? "" : v)}
                  disabled={expiriesLoading}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Loading…" />
                  </SelectTrigger>
                  <SelectContent>
                    {/* Radix forbids empty-string Item values; use a sentinel
                        that maps back to "" (auto) at change time. */}
                    <SelectItem value="__auto__">
                      <span className="flex items-center gap-2">
                        <span>Nearest weekly</span>
                        <Badge tone="neutral">auto</Badge>
                      </span>
                    </SelectItem>
                    {expiries.length === 0 && !expiriesLoading && (
                      <div className="px-3 py-2 text-xs text-fg-muted">
                        No expiries available
                      </div>
                    )}
                    {expiries.map((e) => {
                      const dteLabel = e.dte === 0 ? "today" : `${e.dte} DTE`;
                      return (
                        <SelectItem key={e.expiry} value={e.expiry}>
                          <span className="flex items-center justify-between gap-3 w-full">
                            <span className="font-mono">{e.expiry}</span>
                            <span className="flex items-center gap-1.5">
                              <span className="text-fg-muted text-[11px]">{dteLabel}</span>
                              <Badge tone={e.is_monthly ? "success" : "neutral"}>
                                {e.is_monthly ? "monthly" : "weekly"}
                              </Badge>
                            </span>
                          </span>
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>
                <p className="text-[11px] text-fg-muted mt-1">
                  {(() => {
                    if (!expiry) return "Auto = nearest weekly";
                    const e = expiries.find((x) => x.expiry === expiry);
                    if (!e) return "Picked: " + expiry;
                    return `${e.weekday} · ${e.is_monthly ? "monthly" : "weekly"}`;
                  })()}
                </p>
              </div>
              <div>
                <label className="text-xs font-medium text-fg block mb-1.5">Lots</label>
                <input
                  type="number"
                  value={lots}
                  min={1}
                  max={50}
                  onChange={(e) => setLots(Math.max(1, Number(e.target.value) || 1))}
                  className="w-full text-sm px-3 py-2 rounded bg-surface border border-border focus:border-sky-500 focus:outline-none"
                />
                <p className="text-[11px] text-fg-muted mt-1">
                  Position: <span className="font-mono">{(lots * lotSize).toLocaleString("en-IN")}</span> sh/leg
                </p>
              </div>
            </div>
          </section>

          {/* ── Strike preview ──────────────────────────────────── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-3 flex items-center gap-2">
              <Layers className="w-3 h-3" /> Strike Preview
              {chain?.source && (
                <Badge tone={chain.source === "paper" ? "warning" : "success"} dot>{chain.source}</Badge>
              )}
              {chain && (
                <span className="font-normal text-fg-muted">
                  · {chain.underlying} {chain.expiry || "—"}
                  {chain.spot > 0 && <> · spot <span className="font-mono text-fg">{chain.spot.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span></>}
                  {chain.atm_strike && <> · ATM <span className="font-mono text-fg">{chain.atm_strike}</span></>}
                </span>
              )}
            </h3>

            {chainLoading || previewStrikes.kind === "no-chain" ? (
              <div className="p-4 rounded-md border border-border bg-surface-2">
                <LoadingRow count={3} />
              </div>
            ) : chainError ? (
              <ErrorBox message="Couldn't load option chain. Backend may be unreachable." />
            ) : previewStrikes.kind === "no-side" ? (
              <div className="p-4 rounded-md border border-border bg-surface-2 text-xs text-fg-muted">
                {previewStrikes.reason}
              </div>
            ) : previewStrikes.kind === "no-atm" ? (
              <div className="p-4 rounded-md border border-amber-500/30 bg-amber-500/5 text-xs text-amber-400 space-y-1">
                <p className="font-medium flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> Can't anchor the preview</p>
                <p className="text-fg-muted">{previewStrikes.reason}</p>
              </div>
            ) : previewStrikes.kind === "no-strikes" ? (
              <div className="p-4 rounded-md border border-amber-500/30 bg-amber-500/5 text-xs space-y-2">
                <p className="font-medium text-amber-400 flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3" /> 80/60 rule found no match
                </p>
                <p className="text-fg-muted leading-relaxed">{previewStrikes.reason}</p>
                <div className="grid grid-cols-2 gap-2 pt-1 text-fg-muted">
                  <div>Target sell ≤ <span className="font-mono text-fg">₹{previewStrikes.target_sell.toFixed(2)}</span></div>
                  <div>Target buy ≤ <span className="font-mono text-fg">₹{previewStrikes.target_buy.toFixed(2)}</span></div>
                </div>
                {previewStrikes.walked.length > 0 && (
                  <div className="pt-2 border-t border-border">
                    <p className="text-[10px] uppercase tracking-wider text-fg-muted mb-1">
                      Chain walk (first 6 OTM strikes)
                    </p>
                    <ul className="font-mono text-[11px] text-fg-muted space-y-0.5">
                      {previewStrikes.walked.slice(0, 6).map((w) => {
                        const passesSell = w.price <= previewStrikes.target_sell;
                        const passesBuy = w.price <= previewStrikes.target_buy;
                        return (
                          <li key={w.strike} className="flex items-center gap-2">
                            <span className="text-fg">{w.strike}</span>
                            <span>{w.side}</span>
                            <span>₹{w.price.toFixed(2)}</span>
                            {passesBuy ? (
                              <Badge tone="success">buy ok</Badge>
                            ) : passesSell ? (
                              <Badge tone="neutral">sell ok</Badge>
                            ) : (
                              <Badge tone="warning">over target</Badge>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 rounded-md border border-border bg-surface-2 space-y-3">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-fg-muted">Sell {previewStrikes.side}</p>
                    <p className="font-mono text-base mt-0.5">
                      <span className="text-rose-400">{previewStrikes.sell.strike}</span>
                      <span className="text-fg-muted"> @ </span>
                      <span className="text-fg">
                        ₹{(() => {
                          const q = previewStrikes.side === "PE" ? previewStrikes.sell.pe : previewStrikes.sell.ce;
                          return ((q?.bid && q.bid > 0 ? q.bid : q?.ltp) ?? 0).toFixed(2);
                        })()}
                      </span>
                    </p>
                    <p className="text-[10px] text-fg-muted mt-0.5">
                      target ≤ ₹{previewStrikes.target_sell.toFixed(2)} (80% of ATM)
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-fg-muted">Buy {previewStrikes.side}</p>
                    <p className="font-mono text-base mt-0.5">
                      <span className="text-emerald-400">{previewStrikes.buy.strike}</span>
                      <span className="text-fg-muted"> @ </span>
                      <span className="text-fg">
                        ₹{(() => {
                          const q = previewStrikes.side === "PE" ? previewStrikes.buy.pe : previewStrikes.buy.ce;
                          return ((q?.ask && q.ask > 0 ? q.ask : q?.ltp) ?? 0).toFixed(2);
                        })()}
                      </span>
                    </p>
                    <p className="text-[10px] text-fg-muted mt-0.5">
                      target ≤ ₹{previewStrikes.target_buy.toFixed(2)} (60% of ATM)
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-3 pt-3 border-t border-border">
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-fg-muted">Width</p>
                    <p className="font-mono text-sm text-fg">{previewStrikes.width} pts</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-fg-muted">Max profit</p>
                    <p className="font-mono text-sm text-emerald-400">+{fmtInr(maxProfit)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-fg-muted">Max loss</p>
                    <p className={cn("font-mono text-sm", exceedsCap ? "text-rose-400" : "text-fg")}>
                      −{fmtInr(maxLoss)}
                    </p>
                  </div>
                </div>
                <p className="text-[11px] text-fg-muted leading-relaxed">
                  Selector uses the <span className="font-medium text-fg">80% / 60%</span> rule on the live chain.
                  Final strikes are recomputed by the backend at fire time using fresh quotes.
                </p>
              </div>
            )}
          </section>

          {/* ── Risk profile ────────────────────────────────────── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-3 flex items-center gap-1.5">
              <Shield className="w-3 h-3" /> Risk Profile
            </h3>
            <div className="p-4 rounded-md border border-border bg-surface-2 space-y-3">
              <label className="text-xs font-medium text-fg block">Max loss per trade</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { val: 0.02, label: "2%", sub: "tight", inr: capital * 0.02 },
                  { val: 0.08, label: "8%", sub: "options", inr: capital * 0.08 },
                  { val: 0.15, label: "15%", sub: "max", inr: capital * 0.15 },
                ].map((opt) => {
                  const active = riskCapPct === opt.val;
                  return (
                    <button
                      key={opt.val}
                      onClick={() => setRiskCapPct(opt.val)}
                      className={cn(
                        "p-3 rounded border text-left transition-colors",
                        active
                          ? "border-sky-500 bg-sky-500/10"
                          : "border-border bg-surface hover:border-border-strong",
                      )}
                    >
                      <p className={cn("font-mono font-medium text-base", active ? "text-sky-400" : "text-fg")}>
                        {opt.label}
                      </p>
                      <p className="text-[10px] text-fg-muted uppercase">{opt.sub}</p>
                      <p className="text-[11px] text-fg-muted mt-1 font-mono">{fmtInr(opt.inr)}</p>
                    </button>
                  );
                })}
              </div>
              {exceedsCap && (
                <div className="text-xs flex items-start gap-2 text-rose-400 bg-rose-500/5 border border-rose-500/30 rounded px-3 py-2">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  <span>
                    Max loss <span className="font-mono">{fmtInr(maxLoss)}</span> exceeds the{" "}
                    <span className="font-mono">{(riskCapPct * 100).toFixed(0)}%</span> cap (
                    <span className="font-mono">{fmtInr(riskCapInr)}</span>). Reduce lots or pick a higher cap.
                  </span>
                </div>
              )}
            </div>
          </section>

          {/* ── Lifecycle rules ─────────────────────────────────── */}
          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-fg-muted mb-3 flex items-center gap-1.5">
              <Activity className="w-3 h-3" /> Lifecycle Rules
            </h3>
            <div className="p-4 rounded-md border border-border bg-surface-2 space-y-3">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoProfitTake}
                  onChange={(e) => setAutoProfitTake(e.target.checked)}
                  className="mt-0.5 rounded"
                />
                <div className="flex-1">
                  <p className="text-sm text-fg">Auto-close at 70% of max profit</p>
                  <p className="text-[11px] text-fg-muted leading-relaxed">
                    Lock in <span className="font-mono text-fg">{fmtInr(maxProfit * 0.70)}</span> instead of holding for the full{" "}
                    <span className="font-mono text-fg">{fmtInr(maxProfit)}</span>.
                  </p>
                </div>
              </label>
              <label className="flex items-start gap-3 cursor-not-allowed opacity-70">
                <input type="checkbox" defaultChecked disabled className="mt-0.5 rounded" />
                <div className="flex-1">
                  <p className="text-sm text-fg">Hard stop at 1.5× credit</p>
                  <p className="text-[11px] text-fg-muted">Always on. Caps damage on a flash move.</p>
                </div>
              </label>
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={reentry}
                  onChange={(e) => setReentry(e.target.checked)}
                  className="mt-0.5 rounded"
                />
                <div className="flex-1">
                  <p className="text-sm text-fg">Re-enter after profit-take</p>
                  <p className="text-[11px] text-fg-muted">
                    Open a fresh spread at the new ATM if bias holds and momentum is intact.
                  </p>
                </div>
              </label>
            </div>
          </section>

          {/* ── Block + error surfaces ──────────────────────────── */}
          {blockReason && (
            <div className="bg-amber-500/10 border border-amber-500/30 text-amber-400 text-sm rounded-md px-4 py-3 flex items-start gap-3">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-medium">Can't fire yet</p>
                <p className="text-fg-muted text-xs mt-0.5">{blockReason}</p>
              </div>
              {!defaultPortfolio && portfolios && portfolios.length === 0 && (
                <Link to="/dashboard" className="text-amber-400 underline text-xs whitespace-nowrap">
                  Open dashboard →
                </Link>
              )}
            </div>
          )}

          {errorMsg && (
            <div className="bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm rounded-md px-4 py-3 space-y-2">
              <p className="font-medium flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" /> Backend rejected the trade
              </p>
              <p className="text-fg-muted text-xs">{errorMsg}</p>
              {Object.keys(fieldErrors).length > 0 && (
                <ul className="ml-4 list-disc text-fg-muted text-xs space-y-0.5">
                  {Object.entries(fieldErrors).map(([k, msgs]) => (
                    <li key={k}>
                      <code className="text-fg bg-surface-2 px-1 rounded">{k}</code> — {msgs.join(", ")}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </SheetBody>

        {/* ── Sticky footer ──────────────────────────────────────── */}
        <div className="border-t border-border px-5 py-3 bg-surface flex items-center justify-between gap-3 sticky bottom-0">
          <div className="text-xs text-fg-muted truncate">
            Routes to <code className="font-mono text-fg">paper</code> ·{" "}
            <span className="font-mono text-fg">{defaultPortfolio?.name ?? "—"}</span>
          </div>
          <div className="flex gap-2 shrink-0">
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
            <Button
              disabled={!!blockReason || startRun.isPending}
              onClick={onFire}
            >
              <PlayCircle className="w-4 h-4 mr-2" />
              {startRun.isPending ? "Firing…" : "Fire paper trade"}
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}

/* ───────────────────────── page ───────────────────────── */

export function OptionsDashboardPage() {
  const navigate = useNavigate();
  const [picked, setPicked] = React.useState<StrategyPreset | null>(null);
  const [sheetOpen, setSheetOpen] = React.useState(false);
  const [wsLive, setWsLive] = React.useState(false);

  // Keep the connection-pill honest so the trader knows ticks are live.
  React.useEffect(() => {
    const ws = connect(
      "/ws/ticks/",
      () => {},
      {
        onOpen: () => setWsLive(true),
        onClose: () => setWsLive(false),
        onError: () => setWsLive(false),
      },
    );
    return () => ws.close();
  }, []);

  const onPick = (p: StrategyPreset) => {
    setPicked(p);
    setSheetOpen(true);
  };

  return (
    <div className="p-4 space-y-4">
      {/* page heading */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-fg flex items-center gap-2">
            <Gauge className="w-5 h-5 text-amber-400" />
            Options Desk
          </h1>
          <p className="text-sm text-fg-muted mt-0.5">
            One screen for every options strategy — live market, open positions, greeks, gates, activity.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={() => navigate("/tradingview")}>
          <Radio className="w-3.5 h-3.5 mr-1.5" />
          TV alerts
        </Button>
      </div>

      <HeaderStrip />

      <OnboardingChecklist />

      {/* main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-[300px_1fr] gap-4">
        {/* left rail */}
        <div className="space-y-4">
          <StrategyLauncher onPick={onPick} />
          <TVAlertsRail />
        </div>

        {/* main content */}
        <div className="space-y-4 min-w-0">
          <LiveChainCard underlying="NIFTY" />
          <LivePositionsTable live={wsLive} />
          <GreeksRiskStrip />
          <EventFirehose />
        </div>
      </div>

      <ConfigureStrategySheet
        preset={picked}
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
      />
    </div>
  );
}

export default OptionsDashboardPage;
