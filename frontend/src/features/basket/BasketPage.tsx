/**
 * Morning Basket — intraday equity + ATM options momentum play.
 *
 * Shows: market mood → equity signals → options signal → execution status.
 * Cross-links: Scanner (signal source) + Backtester (validation).
 */
import { Link } from "react-router-dom";
import { BarChart3, RefreshCcw, TrendingUp } from "lucide-react";

import {
  useMarketPulse,
  useBasketStatus,
  phaseTone,
  type BasketSignal,
  type BasketPayload,
} from "@/lib/market-pulse";
import { cn, fmtNum } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { KPI } from "@/components/ui/KPI";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

const MOOD_TONE = {
  BULLISH: "success" as const,
  BEARISH: "danger" as const,
  NEUTRAL: "neutral" as const,
};

export function BasketPage() {
  const { data: pulse } = useMarketPulse();
  const { data, isLoading, isError, error, refetch, isFetching } =
    useBasketStatus({ isOpen: pulse?.is_market_open ?? false });

  if (isLoading) return <BasketLoading />;
  if (isError) return <BasketError error={error as Error} onRetry={() => refetch()} />;
  if (!data) return null;

  const md = data.mood_details;
  const equitySignals = data.signals.filter((s) => s.leg_type === "equity");
  const optionSignals = data.signals.filter((s) => s.leg_type === "option");

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Morning Basket</p>
          <h1 className="text-h1 text-fg">Basket Strategy</h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            Intraday equity + ATM index options. Mood → signals → scale-in execution.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCcw className={cn("h-4 w-4", isFetching && "animate-spin")} />
          </Button>
          <Link to="/swing-scanner">
            <Button variant="ghost" size="sm" leading={<TrendingUp className="h-4 w-4" />}>Scanner</Button>
          </Link>
          <Link to="/backtester?tab=basket">
            <Button variant="ghost" size="sm" leading={<BarChart3 className="h-4 w-4" />}>Backtest</Button>
          </Link>
        </div>
      </header>

      {/* Market Mood */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                Market Mood
                <Badge tone={MOOD_TONE[data.mood as keyof typeof MOOD_TONE] ?? "neutral"} dot>
                  {data.mood}
                </Badge>
              </CardTitle>
              <CardDescription>{md?.confidence ? `${(md.confidence * 100).toFixed(0)}% confidence` : ""}</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="A/D Ratio" value={md?.ad_ratio} valueFormat="num"
                 hint={md ? `${md.advance} up / ${md.decline} down` : ""} />
            <KPI label="NIFTY" value={md?.nifty_spot} valueFormat="num"
                 hint={md ? `Gap: ${md.gap_pct >= 0 ? "+" : ""}${md.gap_pct.toFixed(2)}%` : ""} />
            <KPI label="VIX" value={md?.vix} valueFormat="num"
                 hint={md?.vix_tier ?? ""} />
            <KPI label="Confidence" value={md?.confidence ? md.confidence * 100 : 0} valueFormat="num"
                 hint="%" />
          </div>
          {md?.reasons && md.reasons.length > 0 && (
            <div className="flex gap-1.5 flex-wrap">
              {md.reasons.map((r, i) => (
                <span key={i} className="inline-flex items-center rounded-xs px-1.5 py-0.5 text-caption text-fg-muted border border-border/60 bg-surface-2">
                  {r}
                </span>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Execution plan summary */}
      {data.signals.length > 0 && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-4 flex-wrap text-body-sm">
              <span className="text-fg-subtle">Execution:</span>
              <Badge tone="neutral">T1 50% MARKET</Badge>
              <Badge tone="neutral">T2 30% LIMIT</Badge>
              <Badge tone="neutral">T3 20% confirm</Badge>
              <span className="text-fg-subtle">|</span>
              <span className="text-fg-muted">
                Pyramid at +1R · Trail at 5 EMA · EOD close 15:15
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Signals */}
      {data.signals.length > 0 ? (
        <div className="space-y-3">
          {/* Equity legs */}
          {equitySignals.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-body">Equity Signals ({equitySignals.length})</CardTitle>
                <CardDescription>OK cycle phases + 5 EMA / BB momentum confirmed</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {equitySignals.map((s) => (
                    <SignalRow key={s.symbol} signal={s} />
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Options leg */}
          {optionSignals.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-body">Options Signal</CardTitle>
                <CardDescription>ATM {data.mood === "BULLISH" ? "CE" : "PE"} on {optionSignals[0]?.option_symbol?.split(/\d/)[0] || "NIFTY"}</CardDescription>
              </CardHeader>
              <CardContent>
                {optionSignals.map((s) => (
                  <div key={s.symbol} className="flex items-center justify-between gap-4 py-2">
                    <div className="flex items-center gap-3">
                      <Badge tone={s.option_type === "CE" ? "success" : "danger"}>
                        {s.option_type} {s.strike}
                      </Badge>
                      <span className="font-mono text-body-sm text-fg">{s.symbol}</span>
                      {s.expiry && <span className="text-caption text-fg-subtle">exp {s.expiry}</span>}
                    </div>
                    <div className="flex items-center gap-4 text-body-sm font-mono">
                      <span className="text-fg">₹{fmtNum(s.entry_price, 2)}</span>
                      <span className="text-pnl-down">SL ₹{fmtNum(s.stoploss, 2)}</span>
                      <span className="text-fg-subtle">risk ₹{fmtNum(s.risk_points, 2)}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      ) : (
        <EmptyState
          title={data.mood === "NEUTRAL" ? "No basket today" : "No signals"}
          description={
            data.mood === "NEUTRAL"
              ? "Market mood is neutral — sitting out."
              : "No equity candidates passed momentum filters."
          }
        />
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

function SignalRow({ signal: s }: { signal: BasketSignal }) {
  const riskPct = s.entry_price > 0 ? (s.risk_points / s.entry_price * 100) : 0;
  const qualityScore = Math.round(s.confluence * 100);
  const qualityColor = qualityScore >= 70 ? "text-pnl-up" : qualityScore >= 50 ? "text-fg" : "text-fg-muted";

  return (
    <div className="rounded-sm border border-border/60 p-3 space-y-2">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2 min-w-0">
          <Badge tone={s.side === "BUY" ? "success" : "danger"}>{s.side}</Badge>
          <Link to={`/setup/${encodeURIComponent(s.symbol)}`}
                className="font-mono text-body text-fg font-semibold hover:text-accent">
            {s.symbol}
          </Link>
          <Badge tone={phaseTone(s.phase)}>{s.phase}</Badge>
        </div>
        <div className={cn("text-body font-mono font-semibold", qualityColor)}>
          {qualityScore}%
        </div>
      </div>
      <div className="grid grid-cols-4 gap-3 text-caption">
        <div>
          <div className="text-fg-subtle uppercase tracking-wider">Entry</div>
          <div className="font-mono text-body-sm text-fg">{fmtNum(s.entry_price, 2)}</div>
        </div>
        <div>
          <div className="text-fg-subtle uppercase tracking-wider">Stop Loss</div>
          <div className="font-mono text-body-sm text-pnl-down">{fmtNum(s.stoploss, 2)}</div>
        </div>
        <div>
          <div className="text-fg-subtle uppercase tracking-wider">Risk</div>
          <div className="font-mono text-body-sm text-fg">{fmtNum(s.risk_points, 2)} ({riskPct.toFixed(1)}%)</div>
        </div>
        <div>
          <div className="text-fg-subtle uppercase tracking-wider">Target (2R)</div>
          <div className="font-mono text-body-sm text-pnl-up">
            {fmtNum(s.entry_price + s.risk_points * 2, 2)}
          </div>
        </div>
      </div>
      {/* Risk bar visualization */}
      <div className="h-1.5 rounded-full bg-surface-2 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-pnl-down via-fg-subtle to-pnl-up"
          style={{ width: `${Math.min(qualityScore, 100)}%` }}
        />
      </div>
    </div>
  );
}

function BasketLoading() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <Skeleton className="h-12 w-64" />
      <Skeleton className="h-40 w-full" />
      <Skeleton className="h-32 w-full" />
    </div>
  );
}

function BasketError({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="px-6 py-6 max-w-[800px] mx-auto">
      <EmptyState
        title="Couldn't load basket"
        description={error?.message ?? "Unknown error"}
        action={<Button onClick={onRetry}>Try again</Button>}
      />
    </div>
  );
}
