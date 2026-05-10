/**
 * What's Happening Today — the homepage of AlphaDesk.
 *
 * This is the trader's morning read. Before any agent runs, before any order,
 * the operator lands here and asks "is this a day to trade, and if so what
 * kind?" That's The Cascade Stage 1 (REGIME) and Stage 2 (CONTEXT).
 *
 * Everything on this page is derived from /api/v1/market-data/pulse/. The
 * backend classifier is deterministic and shares its vol/trend gates with
 * @RiskGuard and @OptionsStrategist — so what the operator sees here is
 * exactly what the agents will act on.
 *
 *   Layout
 *   ────────────────────────────────────────────────────────────────
 *   [ Regime banner — tradeable? + one-line summary                 ]
 *   [ Agent guidance panel — directional / straddle go-no-go        ]
 *
 *   [ India VIX (big) ] [ NIFTY ] [ BANKNIFTY ] [ USDINR ]
 *
 *   [ India indices row ]       [ Commodities row            ]
 *   [ Global indices row ]      [ FX + Rates row             ]
 *
 *   [ Sector heatmap — 11 NSE sector indices ranked by % move  ]
 *
 *   [ Errors / freshness footer                                ]
 */
import * as React from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Gauge, RefreshCcw, TrendingDown, TrendingUp } from "lucide-react";

import {
  useMarketPulse,
  type Quote,
  type Sector,
  type PulsePayload,
  volTone,
  trendTone,
  toneColor,
  verdictTone,
  VOL_LABEL,
  TREND_LABEL,
  TONE_LABEL,
  phaseLabel,
} from "@/lib/market-pulse";
import { cn, fmtNum, fmtPct, fmtRel } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

export function MarketPulsePage() {
  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } =
    useMarketPulse();

  if (isLoading) return <PulseLoading />;
  if (isError) return <PulseError error={error as Error} onRetry={() => refetch()} />;
  if (!data) return null;

  const indiaQuotes = data.quotes.indices_in ?? [];
  const globalQuotes = data.quotes.indices_global ?? [];
  const vixQuote = (data.quotes.vol ?? []).find((q) => q.symbol === "INDIAVIX");
  const commodities = data.quotes.commodities ?? [];
  const fx = data.quotes.fx ?? [];
  const rates = data.quotes.rates ?? [];

  const nifty = indiaQuotes.find((q) => q.symbol === "NIFTY");
  const bankNifty = indiaQuotes.find((q) => q.symbol === "BANKNIFTY");
  const usdinr = fx.find((q) => q.symbol === "USDINR");

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      {/* ────────── Header ────────── */}
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">
            Stage 1 · REGIME — What's happening today
          </p>
          <h1 className="text-h1 text-fg">
            {greetingLine(data)}
          </h1>
          <p className="text-body-sm text-fg-muted mt-1 flex items-center gap-2 flex-wrap">
            <Badge tone={data.is_market_open ? "success" : "neutral"} dot>
              {phaseLabel(data.session_phase)}
            </Badge>
            <span aria-hidden>·</span>
            <span>Last update {fmtRel(data.as_of)}</span>
            <span aria-hidden>·</span>
            <span>Auto-refresh {data.is_market_open ? "15s" : "60s"}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            leading={<RefreshCcw className={cn("h-4 w-4", isFetching && "animate-spin")} />}
            onClick={() => refetch()}
            disabled={isFetching}
          >
            Refresh
          </Button>
        </div>
      </header>

      {/* ────────── Regime banner ────────── */}
      <RegimeBanner data={data} />

      {/* ────────── Agent guidance ────────── */}
      <GuidancePanel data={data} />

      {/* ────────── Headline row: VIX + NIFTY + BANKNIFTY + USDINR ────────── */}
      <section
        aria-label="Headline quotes"
        className="grid grid-cols-2 sm:grid-cols-4 gap-3"
      >
        <VixCard quote={vixQuote} volTier={data.regime.vol} />
        <QuoteCard quote={nifty} highlight />
        <QuoteCard quote={bankNifty} />
        <QuoteCard quote={usdinr} inverted />
      </section>

      {/* ────────── Two-column: Commodities + India indices ────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <QuoteSection
          title="India indices"
          description="Broad-market tape — what the index operator is watching."
          quotes={indiaQuotes}
        />
        <QuoteSection
          title="Commodities"
          description="Crude, gold, gas, copper — inputs to the whole rotation story."
          quotes={commodities}
        />
      </div>

      {/* ────────── Global + FX/Rates ────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <QuoteSection
          title="Global indices"
          description="Overnight / same-day bias into the Indian open."
          quotes={globalQuotes}
        />
        <QuoteSection
          title="FX & rates"
          description="Currency + US yields — the macro backdrop."
          quotes={[...fx, ...rates]}
        />
      </div>

      {/* ────────── Sector heatmap ────────── */}
      <SectorHeatmap sectors={data.sectors} />

      {/* ────────── Errors footer ────────── */}
      {data.errors.length > 0 && (
        <div
          role="alert"
          className="rounded-md border border-warn/40 bg-warn/5 p-3 text-body-sm text-fg-muted flex gap-2 items-start"
        >
          <AlertTriangle className="h-4 w-4 text-warn mt-0.5 shrink-0" aria-hidden />
          <div>
            <div className="font-medium text-fg">Data source reported issues</div>
            <ul className="list-disc ml-5 mt-1">
              {data.errors.map((e, i) => (
                <li key={i} className="font-mono text-caption">{e}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <p className="text-caption text-fg-subtle text-right">
        Pulse served at {new Date(dataUpdatedAt).toLocaleTimeString("en-IN")} ·
        source: yfinance (fallback to Angel One when broker linked)
      </p>
    </div>
  );
}

/* ================================================================== */
/* Regime banner — the single most important thing on the page         */
/* ================================================================== */
function RegimeBanner({ data }: { data: PulsePayload }) {
  const r = data.regime;
  const TrendIcon =
    r.trend === "up" ? TrendingUp :
    r.trend === "down" ? TrendingDown :
    Gauge;

  return (
    <Card
      className={cn(
        "border-l-4",
        r.tradeable ? "border-l-pnl-up/70" : "border-l-pnl-down/70",
      )}
    >
      <CardContent className="p-5">
        <div className="flex items-start gap-4 flex-wrap">
          <div
            className={cn(
              "h-12 w-12 rounded-md flex items-center justify-center shrink-0",
              r.tradeable ? "bg-pnl-up/10 text-pnl-up" : "bg-pnl-down/10 text-pnl-down",
            )}
            aria-hidden
          >
            <TrendIcon className="h-6 w-6" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Badge tone={r.tradeable ? "success" : "danger"} dot>
                {r.tradeable ? "Tradeable" : "Stand down"}
              </Badge>
              <Badge tone={volTone(r.vol)}>VOL · {VOL_LABEL[r.vol]}</Badge>
              <Badge tone={trendTone(r.trend)}>{TREND_LABEL[r.trend]}</Badge>
              <Badge tone={toneColor(r.global_tone)}>
                GLOBAL · {TONE_LABEL[r.global_tone]}
              </Badge>
            </div>
            <p className="text-body text-fg leading-snug">{r.summary}</p>
            <dl className="grid grid-cols-3 gap-4 mt-3 text-body-sm">
              <RegimeStat
                label="India VIX"
                value={r.vix != null ? fmtNum(r.vix, 2) : "—"}
              />
              <RegimeStat
                label="NIFTY gap"
                value={fmtPct(r.nifty_gap_pct, 2)}
                pnl={r.nifty_gap_pct}
              />
              <RegimeStat
                label="S&P change"
                value={fmtPct(r.sp500_change_pct, 2)}
                pnl={r.sp500_change_pct}
              />
            </dl>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function RegimeStat({
  label, value, pnl,
}: { label: string; value: string; pnl?: number | null }) {
  const cls =
    pnl == null ? "text-fg" :
    pnl > 0 ? "text-pnl-up" :
    pnl < 0 ? "text-pnl-down" : "text-fg";
  return (
    <div>
      <dt className="text-caption uppercase tracking-wider text-fg-subtle">{label}</dt>
      <dd className={cn("font-mono tabular text-body font-semibold mt-0.5", cls)}>{value}</dd>
    </div>
  );
}

/* ================================================================== */
/* Agent guidance panel — what the desk will (not) do                   */
/* ================================================================== */
function GuidancePanel({ data }: { data: PulsePayload }) {
  const g = data.guidance;
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>Agent guidance</CardTitle>
            <CardDescription>
              What the desk's agents will act on in this regime —
              derived from the same vol/trend gates as @RiskGuard.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border">
          <GuidanceCell
            agent="@DirectionalTrader"
            role="Plan BUY/SELL equity intraday"
            verdict={g.directional}
          />
          <GuidanceCell
            agent="@OptionsStrategist"
            role="Short straddle / iron condor lifecycle"
            verdict={g.straddle}
          />
        </div>
        <ul className="border-t border-border px-5 py-3 space-y-1">
          {g.reasons.length === 0 ? (
            <li className="text-body-sm text-fg-muted">
              No explicit reasons returned — the regime is mid-spectrum.
            </li>
          ) : (
            g.reasons.map((r, i) => (
              <li
                key={i}
                className="text-body-sm text-fg-muted flex gap-2 items-start"
              >
                <span className="text-fg-subtle font-mono">{String(i + 1).padStart(2, "0")}</span>
                <span>{r}</span>
              </li>
            ))
          )}
        </ul>
      </CardContent>
    </Card>
  );
}

function GuidanceCell({
  agent, role, verdict,
}: { agent: string; role: string; verdict: "favored" | "neutral" | "avoid" }) {
  const label = {
    favored: "Favored",
    neutral: "Neutral — size small",
    avoid: "Avoid",
  }[verdict];
  return (
    <div className="px-5 py-4 flex items-center justify-between gap-3">
      <div className="min-w-0">
        <div className="text-body-sm font-mono text-accent">{agent}</div>
        <div className="text-caption text-fg-subtle">{role}</div>
      </div>
      <Badge tone={verdictTone(verdict)} dot>
        {label}
      </Badge>
    </div>
  );
}

/* ================================================================== */
/* VIX card — headline because VIX gates everything downstream         */
/* ================================================================== */
function VixCard({ quote, volTier }: { quote: Quote | undefined; volTier: string }) {
  const pnl = quote?.change_pct ?? null;
  return (
    <Card className="col-span-2 sm:col-span-1">
      <CardContent className="px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-caption uppercase tracking-wider text-fg-subtle">
            India VIX
          </span>
          <Badge tone={volTone(volTier as any)}>{VOL_LABEL[volTier as keyof typeof VOL_LABEL]}</Badge>
        </div>
        <div className="flex items-baseline gap-2 mt-1.5">
          <span className="font-mono tabular text-num-lg text-fg">
            {quote?.last != null ? fmtNum(quote.last, 2) : "—"}
          </span>
          <span
            className={cn(
              "font-mono tabular text-body-sm",
              pnl == null ? "text-fg-muted" :
              pnl > 0 ? "text-pnl-down" : "text-pnl-up", /* VIX up = bad for longs */
            )}
          >
            {fmtPct(pnl, 2)}
          </span>
        </div>
        <p className="text-caption text-fg-subtle mt-0.5">
          {gateLabel(volTier)}
        </p>
      </CardContent>
    </Card>
  );
}

function gateLabel(vol: string): string {
  switch (vol) {
    case "extreme": return "Halt — no new options";
    case "high":    return "No fresh straddles";
    case "elevated":return "Size down";
    case "normal":  return "Straddle gate: OK";
    case "low":     return "Premium thin";
    case "complacent": return "Watch for vol spike";
    default:        return "—";
  }
}

/* ================================================================== */
/* Quote card — one tile per instrument                                 */
/* ================================================================== */
function QuoteCard({
  quote, highlight, inverted,
}: {
  quote: Quote | undefined;
  highlight?: boolean;
  /** For USDINR etc — weaker rupee is "risk-off", render with flipped colour. */
  inverted?: boolean;
}) {
  const pnl = quote?.change_pct ?? null;
  const cls =
    pnl == null ? "text-fg-muted" :
    (inverted ? pnl < 0 : pnl > 0) ? "text-pnl-up" :
    (inverted ? pnl > 0 : pnl < 0) ? "text-pnl-down" : "text-fg-muted";
  return (
    <Card
      className={cn(
        highlight && "ring-1 ring-accent/30",
      )}
    >
      <CardContent className="px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-caption uppercase tracking-wider text-fg-subtle truncate">
            {quote?.label ?? "—"}
          </span>
          {quote?.source && quote.source !== "yfinance" && (
            <Badge tone="info">{quote.source}</Badge>
          )}
        </div>
        <div className="flex items-baseline gap-2 mt-1.5">
          <span className="font-mono tabular text-num-lg text-fg">
            {quote?.last != null ? fmtNum(quote.last, 2) : "—"}
          </span>
          <span className={cn("font-mono tabular text-body-sm", cls)}>
            {fmtPct(pnl, 2)}
          </span>
        </div>
        <p className="text-caption text-fg-subtle mt-0.5 font-mono tabular">
          {quote?.day_low != null && quote?.day_high != null
            ? `${fmtNum(quote.day_low, 2)} – ${fmtNum(quote.day_high, 2)}`
            : quote?.prev_close != null
              ? `prev ${fmtNum(quote.prev_close, 2)}`
              : "—"}
        </p>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Quote section — grouped list of instruments                          */
/* ================================================================== */
function QuoteSection({
  title, description, quotes,
}: {
  title: string;
  description: string;
  quotes: Quote[];
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        {quotes.length === 0 ? (
          <EmptyState
            className="m-4"
            title="No data"
            description="The data provider returned no quotes for this group. Try refreshing."
          />
        ) : (
          <ul className="divide-y divide-border/60">
            {quotes.map((q) => (
              <QuoteRow key={q.symbol} quote={q} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function QuoteRow({ quote }: { quote: Quote }) {
  const pnl = quote.change_pct;
  const pnlCls =
    pnl == null ? "text-fg-muted" :
    pnl > 0 ? "text-pnl-up" :
    pnl < 0 ? "text-pnl-down" : "text-fg-muted";
  return (
    <li className="px-5 py-2.5 flex items-center gap-3">
      <div className="flex-1 min-w-0">
        <div className="text-body-sm text-fg truncate">{quote.label}</div>
        <div className="text-caption text-fg-subtle font-mono">
          {quote.symbol}
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className="font-mono tabular text-body-sm text-fg">
          {quote.last != null ? fmtNum(quote.last, 2) : "—"}
        </div>
        <div className={cn("font-mono tabular text-caption", pnlCls)}>
          {fmtPct(pnl, 2)}
        </div>
      </div>
    </li>
  );
}

/* ================================================================== */
/* Sector heatmap — ranked tiles coloured by % change                   */
/* ================================================================== */
function SectorHeatmap({ sectors }: { sectors: Sector[] }) {
  const valid = sectors.filter((s) => s.change_pct != null);
  const hasData = valid.length > 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>Sector heatmap</CardTitle>
            <CardDescription>
              NSE sector indices, ranked by day move. Use the leaders to
              decide which watchlist to pull into Stage 3 (Rotation).
            </CardDescription>
          </div>
          {hasData && (
            <Badge tone="neutral">
              {valid.length} of {sectors.length} reporting
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {!hasData ? (
          <EmptyState
            title="No sector data"
            description="The sector indices provider returned nothing. Try again in a minute."
          />
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
            {sectors.map((s) => (
              <SectorTile key={s.key} sector={s} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SectorTile({ sector }: { sector: Sector }) {
  const pct = sector.change_pct;
  // Tailwind's JIT can't see dynamically-interpolated opacity, so we hit the
  // CSS vars directly.  --pnl-up / --pnl-down are stored as three
  // space-separated RGB components (see globals.css) — perfect for the
  // modern rgb() / alpha syntax.
  const alpha =
    pct == null ? 0 :
    0.08 + Math.min(1, Math.abs(pct) / 2.5) * 0.28; /* 2.5% move = saturated */
  const style: React.CSSProperties = pct == null ? {} : {
    backgroundColor: pct > 0
      ? `rgb(var(--pnl-up) / ${alpha})`
      : pct < 0
        ? `rgb(var(--pnl-down) / ${alpha})`
        : undefined,
  };
  // Deep-link to the Stage 3 drill-in.  The rotation page scrolls to the
  // matching card via `useLocation().hash` — so clicking a tile lands the
  // operator directly on that sector's leaders + laggards list.
  return (
    <Link
      to={`/rotation#${sector.key}`}
      aria-label={`Open ${sector.label} in sector rotation`}
      className={cn(
        "rounded-sm border border-border/60 p-3 flex flex-col justify-between gap-1",
        "min-h-[78px] transition-[border-color,transform] duration-120",
        "hover:border-accent/50 hover:-translate-y-[1px]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
      )}
      style={style}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-body-sm font-medium text-fg truncate">{sector.label}</span>
        <span className="text-caption text-fg-subtle font-mono">#{sector.rank}</span>
      </div>
      <div className="flex items-baseline justify-between">
        <span className="text-caption text-fg-subtle font-mono tabular">
          {sector.last != null ? fmtNum(sector.last, 0) : "—"}
        </span>
        <span
          className={cn(
            "font-mono tabular text-body-sm",
            pct == null ? "text-fg-muted" :
            pct > 0 ? "text-pnl-up" :
            pct < 0 ? "text-pnl-down" : "text-fg-muted",
          )}
        >
          {fmtPct(pct, 2)}
        </span>
      </div>
    </Link>
  );
}

/* ================================================================== */
/* Loading + error states                                               */
/* ================================================================== */
function PulseLoading() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <Skeleton className="h-12 w-96" />
      <Skeleton className="h-28 w-full" />
      <Skeleton className="h-36 w-full" />
      <div className="grid grid-cols-4 gap-3">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
      <Skeleton className="h-64 w-full" />
    </div>
  );
}

function PulseError({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="px-6 py-12 max-w-2xl mx-auto">
      <EmptyState
        icon={<AlertTriangle />}
        title="Market pulse failed to load"
        description={error?.message ?? "Unknown error"}
        action={<Button onClick={onRetry}>Try again</Button>}
      />
    </div>
  );
}

/* ================================================================== */
/* Copy helpers                                                         */
/* ================================================================== */
function greetingLine(data: PulsePayload): React.ReactNode {
  const phase = data.session_phase;
  const tradeable = data.regime.tradeable;
  const accent = (txt: string) => <span className="text-accent">{txt}</span>;

  if (phase === "weekend") return <>Markets closed — {accent("plan the week")}.</>;
  if (phase === "pre-open") return <>Pre-open — {accent("read the tape")} before 9:15.</>;
  if (phase === "post-close")
    return <>Close done — {accent("review and reset")} for tomorrow.</>;

  return tradeable ? (
    <>Markets open — {accent("regime is tradeable")}.</>
  ) : (
    <>Markets open — {accent("but stand down")} today.</>
  );
}
