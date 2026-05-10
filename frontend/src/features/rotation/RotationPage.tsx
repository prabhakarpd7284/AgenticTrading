/**
 * Sector Rotation — Stage 3 of The Cascade.
 *
 * The trader's flow: Pulse tells you *whether* to trade; Rotation tells you
 * *what sector* is leading today and who the leaders / laggards are inside
 * that sector.  Feeds into Stage 4 (shortlist) where specific tradeable
 * names surface with their setups.
 *
 * Layout
 * ────────────────────────────────────────────────────────────────
 *  [ Header — as-of + refresh + errors                           ]
 *  [ Ranking strip — top-to-bottom list of sectors, % move       ]
 *     └─ each sector has leaders (green), laggards (red), breadth
 *
 * Anchor-scroll: `/rotation#NIFTY_IT` jumps to the IT sector card so the
 * operator can deep-link from the pulse heatmap tile.
 */
import * as React from "react";
import { useLocation } from "react-router-dom";
import { AlertTriangle, RefreshCcw, TrendingDown, TrendingUp } from "lucide-react";

import {
  useMarketPulse,
  useSectorRotation,
  type SectorRotation,
  type StockMove,
} from "@/lib/market-pulse";
import { cn, fmtNum, fmtPct, fmtRel } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

export function RotationPage() {
  // Pulse is the source of truth for "is the market open", so we keep the
  // rotation poll cadence in sync with it rather than duplicating the clock.
  const { data: pulse } = useMarketPulse();
  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } =
    useSectorRotation({ isOpen: pulse?.is_market_open ?? false });

  // Scroll to the sector card targeted by the URL hash (e.g. `#NIFTY_IT`).
  const loc = useLocation();
  React.useEffect(() => {
    if (!data || !loc.hash) return;
    const id = loc.hash.replace(/^#/, "");
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [data, loc.hash]);

  if (isLoading) return <RotationLoading />;
  if (isError) return <RotationError error={error as Error} onRetry={() => refetch()} />;
  if (!data) return null;

  const reporting = data.sectors.filter((s) => s.change_pct != null).length;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">
            Stage 3 · ROTATION — Who's leading today
          </p>
          <h1 className="text-h1 text-fg">Sector Rotation</h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            NSE sector indices ranked by day move, with the top-3 leaders
            and worst-3 laggards inside each.  Use this to pick the sector
            before you pick the symbol.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone="neutral">{reporting} of {data.sectors.length} reporting</Badge>
          <span className="text-caption text-fg-subtle">
            Updated {fmtRel(new Date(dataUpdatedAt).toISOString())}
          </span>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => refetch()}
            aria-label="Refresh rotation"
            disabled={isFetching}
          >
            <RefreshCcw className={cn("h-4 w-4", isFetching && "animate-spin")} />
          </Button>
        </div>
      </header>

      {data.sectors.length === 0 ? (
        <EmptyState
          title="No sectors to show"
          description="The data provider returned nothing.  Try again in a minute."
        />
      ) : (
        <div className="space-y-3">
          {data.sectors.map((s) => (
            <SectorCard key={s.key} sector={s} />
          ))}
        </div>
      )}

      {data.errors.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-warn" />
              Provider warnings
            </CardTitle>
            <CardDescription>
              Some symbols failed to resolve.  Cells fall back to "—".
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
/* One sector card — header + leaders + laggards + breadth              */
/* ================================================================== */
function SectorCard({ sector }: { sector: SectorRotation }) {
  const pct = sector.change_pct;
  const tone =
    pct == null ? "neutral" :
    pct > 0.25 ? "success" :
    pct < -0.25 ? "danger" : "neutral";

  const hasLeaders = sector.leaders.length > 0;
  const hasLaggards = sector.laggards.length > 0;
  const hasAny = hasLeaders || hasLaggards;

  return (
    <Card id={sector.key}>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-caption text-fg-subtle font-mono">#{sector.rank}</span>
            <CardTitle>{sector.label}</CardTitle>
            <Badge tone={tone}>
              {pct == null ? "—" :
                pct > 0 ? <><TrendingUp className="h-3 w-3 mr-1 inline" />{fmtPct(pct)}</> :
                pct < 0 ? <><TrendingDown className="h-3 w-3 mr-1 inline" />{fmtPct(pct)}</> :
                fmtPct(pct)}
            </Badge>
          </div>
          <CardDescription className="mt-1">
            {sector.last != null ? `Last ${fmtNum(sector.last, 0)} · ` : ""}
            <span className="text-pnl-up">{sector.breadth.up} up</span>
            {" · "}
            <span className="text-pnl-down">{sector.breadth.down} down</span>
            {" · "}
            <span className="text-fg-muted">{sector.breadth.flat} flat</span>
          </CardDescription>
        </div>
      </CardHeader>
      <CardContent>
        {!hasAny ? (
          <p className="text-body-sm text-fg-muted">
            No tracked constituents for this sector yet.
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <StockList title="Leaders" moves={sector.leaders} />
            <StockList title="Laggards" moves={sector.laggards} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StockList({
  title, moves,
}: { title: string; moves: StockMove[] }) {
  if (moves.length === 0) {
    return (
      <div className="rounded-sm border border-border/60 p-3">
        <div className="text-caption uppercase tracking-wider text-fg-subtle mb-2">{title}</div>
        <p className="text-body-sm text-fg-muted">—</p>
      </div>
    );
  }

  return (
    <div className="rounded-sm border border-border/60 p-3">
      <div className="text-caption uppercase tracking-wider text-fg-subtle mb-2">
        {title}
      </div>
      <ul className="space-y-1.5">
        {moves.map((m) => {
          const pct = m.change_pct;
          const color =
            pct == null ? "text-fg-muted" :
            pct > 0 ? "text-pnl-up" :
            pct < 0 ? "text-pnl-down" : "text-fg-muted";
          return (
            <li key={m.symbol} className="flex items-center justify-between gap-3">
              <span className="text-body-sm font-medium text-fg truncate">{m.symbol}</span>
              <span className="flex items-baseline gap-3 font-mono tabular">
                <span className="text-caption text-fg-subtle">
                  {m.last != null ? fmtNum(m.last, 1) : "—"}
                </span>
                <span className={cn("text-body-sm", color)}>
                  {fmtPct(pct, 2)}
                </span>
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/* ================================================================== */
/* Loading / error states                                               */
/* ================================================================== */
function RotationLoading() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <Skeleton className="h-12 w-80" />
      <div className="space-y-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-40 w-full" />
        ))}
      </div>
    </div>
  );
}

function RotationError({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="px-6 py-6 max-w-[800px] mx-auto">
      <EmptyState
        title="Couldn't load sector rotation"
        description={error?.message ?? "Unknown error"}
        action={<Button onClick={onRetry}>Try again</Button>}
      />
    </div>
  );
}
