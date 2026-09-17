/**
 * Setup — Stage 5 of The Cascade.
 *
 * The desk has a name (from Stage 4 shortlist) and wants to know:
 * "would this trade actually clear right now, and if not, why not?"
 *
 * This page answers that without firing an agent or executing anything.
 * It builds a deterministic ATR-anchored plan (1.5×ATR stop, 3×ATR target,
 * 1%-of-capital sizing) and runs the production 10-criterion @RiskGuard.
 * The breakdown card shows EVERY criterion (pass + fail) so the operator
 * can see the full picture, not just the first failure.
 *
 *  Layout
 *  ──────────────────────────────────────────────────────────────────
 *   [ Header — symbol · BUY/SELL toggle · refresh · breadcrumb-back ]
 *   ┌─ Plan ─────────────┐  ┌─ Verdict + Risk breakdown ────────┐
 *   │ entry / SL / target│  │ ✓ APPROVED · "Approved (regime OK)"│
 *   │ qty · risk · R:R   │  │  10 rows · pass/fail per criterion │
 *   │ market context     │  └────────────────────────────────────┘
 *   └────────────────────┘
 *   [ Errors / data warnings (if any) ]
 */
import * as React from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft, ArrowDownRight, ArrowUpRight, Check, RefreshCcw, ShieldAlert, ShieldCheck, Tag, X,
} from "lucide-react";

import {
  WATCHLIST_KIND_META, useWatchlistsBySymbol,
} from "@/lib/v2";

import {
  criterionTone,
  useMarketPulse,
  useSetupPreview,
  type SetupCriterion,
  type SetupPayload,
  type SetupPlan,
} from "@/lib/market-pulse";
import { cn, fmtDateTime, fmtInr, fmtNum, fmtPct, fmtRel } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { OpButton } from "@/features/ops/OpButton";
import { LivePositionGuard, TrackRecordCard } from "./TrackRecordCard";
import { TrackSetupButton } from "./TrackSetupButton";

export function SetupPage() {
  const { symbol = "" } = useParams();
  const navigate = useNavigate();
  const [side, setSide] = React.useState<"BUY" | "SELL">("BUY");

  const { data: pulse } = useMarketPulse();
  const isOpen = pulse?.is_market_open ?? false;

  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } =
    useSetupPreview(symbol, { side, isOpen });

  if (!symbol) {
    return (
      <div className="px-6 py-6 max-w-[800px] mx-auto">
        <EmptyState
          title="No symbol"
          description="Open a name from the Shortlist to preview a setup."
          action={<Button onClick={() => navigate("/shortlist")}>Go to Shortlist</Button>}
        />
      </div>
    );
  }

  if (isLoading) return <SetupLoading symbol={symbol} />;
  if (isError) {
    return (
      <div className="px-6 py-6 max-w-[800px] mx-auto">
        <EmptyState
          title={`Couldn't load setup for ${symbol}`}
          description={(error as Error)?.message ?? "Unknown error"}
          action={<Button onClick={() => refetch()}>Try again</Button>}
        />
      </div>
    );
  }
  if (!data) return null;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <SetupHeader
        data={data}
        side={side}
        setSide={setSide}
        onRefresh={() => refetch()}
        refreshing={isFetching}
        updatedAt={dataUpdatedAt}
      />

      <LivePositionGuard symbol={data.symbol} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PlanCard data={data} />
        <RiskBreakdownCard data={data} />
      </div>

      <TrackRecordCard symbol={data.symbol} lastPrice={data.market.last} />

      {data.errors.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Data warnings</CardTitle>
            <CardDescription>
              Non-fatal issues from the data port — the verdict above is
              still authoritative, but some inputs may be thin.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="list-disc pl-5 text-body-sm text-fg-muted space-y-1">
              {data.errors.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/* ================================================================== */
/* Header — title · BUY/SELL toggle · refresh · back-link              */
/* ================================================================== */
function SetupHeader({
  data, side, setSide, onRefresh, refreshing, updatedAt,
}: {
  data: SetupPayload;
  side: "BUY" | "SELL";
  setSide: (s: "BUY" | "SELL") => void;
  onRefresh: () => void;
  refreshing: boolean;
  updatedAt: number;
}) {
  return (
    <header className="space-y-3">
      <div className="flex items-center gap-2 text-caption text-fg-subtle">
        <Link
          to="/shortlist"
          className="inline-flex items-center gap-1 hover:text-fg-muted"
        >
          <ArrowLeft className="h-3 w-3" />
          Shortlist
        </Link>
        <span>·</span>
        <span className="uppercase tracking-wider">Stage 5 · SETUP</span>
      </div>

      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <h1 className="text-h1 text-fg font-mono">{data.symbol}</h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            Deterministic plan (1.5×ATR stop · 3×ATR target · 1% risk sizing)
            run through the production 10-criterion @RiskGuard.  Nothing here
            executes — this is "would-this-clear?", not "send-it".
          </p>
          <WatchlistBadges symbol={data.symbol} />
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <SideToggle side={side} setSide={setSide} />
          <RegimePill data={data} />
          {/* Generation time of THIS plan (server as_of), so a saved setup
              has a fixed anchor to measure performance from. The client
              fetch time is secondary, shown as the relative "· refreshed". */}
          <span
            className="text-caption text-fg-subtle"
            title={`Setup generated ${fmtDateTime(data.as_of)} · refreshed ${fmtRel(new Date(updatedAt).toISOString())} ago`}
          >
            Generated {fmtDateTime(data.as_of)}
          </span>

          {/* Non-execute action: add to the "Tracked Setups" watchlist +
              snapshot the plan. Nothing here sends an order. */}
          <TrackSetupButton data={data} />

          {/* Send the symbol to the planner CLI — read the rationale + RiskGuard
              verdict in the streamed log. Doesn't auto-refetch the deterministic
              setup view (different sources). */}
          <OpButton
            command="run_trading_agent"
            defaultArgs={`"Plan a ${side} trade for ${data.symbol}"`}
            label="Ask planner"
            description={`Run the LLM planner for ${data.symbol} — see the rationale + risk verdict in the log.`}
          />

          <Button
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            aria-label="Refresh setup"
            disabled={refreshing}
          >
            <RefreshCcw className={cn("h-4 w-4", refreshing && "animate-spin")} />
          </Button>
        </div>
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* Watchlist badges                                                    */
/* ------------------------------------------------------------------ */

/** "In: <list>, <list>" row that surfaces which of the operator's
 *  TradingView watchlists contain this symbol — manual or auto-resolved.
 *  Renders nothing when the symbol isn't in any watchlist (no empty row,
 *  no clutter). Each badge links to /tradingview so the operator can jump
 *  to the list itself. */
function WatchlistBadges({ symbol }: { symbol: string }) {
  const { data: lists = [], isLoading } = useWatchlistsBySymbol(symbol);
  if (isLoading || lists.length === 0) return null;
  return (
    <div className="mt-2 flex items-center gap-1.5 flex-wrap">
      <Tag className="h-3 w-3 text-fg-subtle" aria-hidden />
      <span className="text-caption text-fg-subtle">In:</span>
      {lists.map((wl) => {
        const meta = WATCHLIST_KIND_META[wl.kind];
        return (
          <Link key={wl.id} to="/tradingview" title={meta.blurb}>
            <Badge tone={wl.is_auto ? "brand" : "neutral"}>
              {wl.name}
              {wl.is_auto && <span className="ml-1 text-caption opacity-75">· auto</span>}
            </Badge>
          </Link>
        );
      })}
    </div>
  );
}

function SideToggle({
  side, setSide,
}: { side: "BUY" | "SELL"; setSide: (s: "BUY" | "SELL") => void }) {
  return (
    <div
      role="group"
      aria-label="Trade direction"
      className="inline-flex rounded-sm border border-border bg-surface overflow-hidden"
    >
      {(["BUY", "SELL"] as const).map((s) => {
        const active = s === side;
        return (
          <button
            key={s}
            type="button"
            onClick={() => setSide(s)}
            className={cn(
              "px-3 h-8 text-body-sm inline-flex items-center gap-1",
              "transition-[background-color,color] duration-120",
              active
                ? s === "BUY"
                  ? "bg-pnl-up/15 text-pnl-up"
                  : "bg-pnl-down/15 text-pnl-down"
                : "text-fg-muted hover:bg-surface-2 hover:text-fg",
            )}
            aria-pressed={active}
          >
            {s === "BUY" ? <ArrowUpRight className="h-3.5 w-3.5" /> :
              <ArrowDownRight className="h-3.5 w-3.5" />}
            {s}
          </button>
        );
      })}
    </div>
  );
}

function RegimePill({ data }: { data: SetupPayload }) {
  const r = data.regime;
  const tone = r.tradeable ? "success" : "danger";
  const label = r.tradeable ? `Regime ${r.vol ?? "ok"}` : "Regime blocked";
  return (
    <Badge tone={tone} dot title={r.summary || ""}>
      {label}
    </Badge>
  );
}

/* ================================================================== */
/* Plan card                                                            */
/* ================================================================== */
function PlanCard({ data }: { data: SetupPayload }) {
  const plan = data.plan;
  if (!plan) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Proposed plan</CardTitle>
          <CardDescription>
            Could not build a plan — insufficient market data for {data.symbol}.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Market data={data} />
        </CardContent>
      </Card>
    );
  }

  const sideTone = plan.side === "BUY" ? "success" : "danger";
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Badge tone={sideTone}>{plan.side}</Badge>
              <span className="font-mono">{plan.symbol}</span>
            </CardTitle>
            <CardDescription>
              ATR-anchored entry · {plan.quantity} qty · {plan.confidence.toFixed(2)} conf
            </CardDescription>
          </div>
          <Badge tone="neutral">R:R {plan.risk_reward_ratio.toFixed(2)}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-3">
          <PriceCell label="Entry" value={plan.entry_price} mono accent="fg" />
          <PriceCell label="Stop" value={plan.stop_loss} mono accent="down" />
          <PriceCell label="Target" value={plan.target} mono accent="up" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KvCell label="Quantity" value={fmtNum(plan.quantity)} />
          <KvCell label="Risk / share" value={`₹${plan.risk_per_share.toFixed(2)}`} />
          <KvCell label="Risk total" value={fmtInr(plan.risk_amount)} />
          <KvCell label="Reward total" value={fmtInr(plan.reward_amount)} />
        </div>
        {plan.notes.length > 0 && (
          <ul className="list-disc pl-5 text-caption text-fg-muted space-y-0.5">
            {plan.notes.map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        )}
        <Market data={data} />
      </CardContent>
    </Card>
  );
}

function PriceCell({
  label, value, mono, accent,
}: {
  label: string;
  value: number | null;
  mono?: boolean;
  accent?: "fg" | "up" | "down";
}) {
  const color =
    accent === "up" ? "text-pnl-up" :
    accent === "down" ? "text-pnl-down" : "text-fg";
  return (
    <div className="rounded-sm border border-border/60 p-3 bg-surface-2/30">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">
        {label}
      </div>
      <div className={cn("text-h3", color, mono && "font-mono tabular")}>
        {value == null ? "—" : value.toFixed(2)}
      </div>
    </div>
  );
}

function KvCell({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-sm border border-border/60 p-2">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">
        {label}
      </div>
      <div className="text-body font-mono tabular text-fg">{value}</div>
    </div>
  );
}

function Market({ data }: { data: SetupPayload }) {
  const m = data.market;
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t border-border/60">
      <KvCell label="Last" value={m.last == null ? "—" : m.last.toFixed(2)} />
      <KvCell label="ATR (14)" value={m.atr == null ? "—" : m.atr.toFixed(2)} />
      <KvCell label="ATR %" value={m.atr_pct == null ? "—" : `${m.atr_pct.toFixed(2)}%`} />
      <KvCell label="Window Δ" value={fmtPct(m.change_pct, 2)} />
    </div>
  );
}

/* ================================================================== */
/* Risk-breakdown card — verdict + 10 criterion rows                    */
/* ================================================================== */
function RiskBreakdownCard({ data }: { data: SetupPayload }) {
  const { approved, reason, criteria } = data.risk;
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              {approved ? (
                <ShieldCheck className="h-5 w-5 text-pnl-up" aria-hidden />
              ) : (
                <ShieldAlert className="h-5 w-5 text-pnl-down" aria-hidden />
              )}
              {approved ? "Approved by @RiskGuard" : "Rejected by @RiskGuard"}
            </CardTitle>
            <CardDescription>{reason}</CardDescription>
          </div>
          <Badge tone={approved ? "success" : "danger"}>
            {approved ? "PASS" : "FAIL"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <ul className="divide-y divide-border/60">
          {criteria.map((c) => (
            <CriterionRow key={c.key} c={c} />
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function CriterionRow({ c }: { c: SetupCriterion }) {
  const tone = criterionTone(c.severity, c.passed);
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span
        aria-hidden
        className={cn(
          "mt-0.5 inline-flex h-5 w-5 items-center justify-center rounded-xs",
          c.passed ? "bg-pnl-up/15 text-pnl-up" : "bg-pnl-down/15 text-pnl-down",
        )}
      >
        {c.passed ? <Check className="h-3.5 w-3.5" /> : <X className="h-3.5 w-3.5" />}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-body-sm text-fg">{c.label}</span>
          <Badge tone={tone}>{c.passed ? "ok" : c.severity}</Badge>
        </div>
        <div className="text-caption text-fg-muted font-mono">{c.detail}</div>
      </div>
    </li>
  );
}

/* ================================================================== */
/* Loading skeleton                                                     */
/* ================================================================== */
function SetupLoading({ symbol }: { symbol: string }) {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <div className="flex items-center gap-2 text-caption text-fg-subtle">
        <Link to="/shortlist" className="hover:text-fg-muted">Shortlist</Link>
        <span>·</span>
        <span className="uppercase tracking-wider">Stage 5 · {symbol}</span>
      </div>
      <Skeleton className="h-12 w-80" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Skeleton className="h-72 w-full" />
        <Skeleton className="h-72 w-full" />
      </div>
    </div>
  );
}

// Side-effect: keep an unused-import guard from biting if `SetupPlan` ever
// stops being referenced — it will when we extend the plan card.
export type _SetupPlanRef = SetupPlan;
