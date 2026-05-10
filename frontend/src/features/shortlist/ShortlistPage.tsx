/**
 * Shortlist — Stage 4 of The Cascade.
 *
 * Rotation (Stage 3) tells us *which sector* is hot; Shortlist turns that
 * into the 10-15 *tradeable names* the desk will actually watch.  Every
 * candidate carries a 0-100 confluence score and a stack of human-readable
 * "reasons" pills so the operator can audit *why* a name made the cut
 * before handing it to Stage 5 (@DirectionalTrader).
 *
 * Layout
 * ────────────────────────────────────────────────────────────────
 *  [ Header — hot sectors + refresh + filtered-out count           ]
 *  [ Candidate cards, sorted desc by score — each shows:
 *      symbol + sector + score badge
 *      KPI row: change · ATR% · turnover · rel vol · 52w
 *      reasons pills (sector #n, leader, aligned, ATR, …)
 *  ]
 *  [ Collapsed "Filtered out" section — why a name didn't pass     ]
 */
import * as React from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, ArrowRight, ChevronDown, ChevronUp, ListChecks, RefreshCcw } from "lucide-react";

import {
  useMarketPulse,
  useShortlist,
  scoreTone,
  type RejectedCandidate,
  type StockCandidate,
} from "@/lib/market-pulse";
import { cn, fmtNum, fmtPct, fmtRel } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

export function ShortlistPage() {
  const { data: pulse } = useMarketPulse();
  const { data, isLoading, isError, error, refetch, isFetching, dataUpdatedAt } =
    useShortlist({ isOpen: pulse?.is_market_open ?? false });

  if (isLoading) return <ShortlistLoading />;
  if (isError) return <ShortlistError error={error as Error} onRetry={() => refetch()} />;
  if (!data) return null;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <p className="text-caption uppercase tracking-wider text-fg-subtle">
            Stage 4 · SHORTLIST — Tradeable names today
          </p>
          <h1 className="text-h1 text-fg">Shortlist</h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            Names drawn from the hot sectors, filtered on F&amp;O-eligibility,
            ATR, and turnover, then ranked by a confluence score.  This feeds
            directly into Stage 5 (@DirectionalTrader).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone="neutral">
            {data.candidates.length} candidates
          </Badge>
          {data.filtered_out.length > 0 && (
            <Badge tone="warning">
              {data.filtered_out.length} filtered out
            </Badge>
          )}
          <span className="text-caption text-fg-subtle">
            Updated {fmtRel(new Date(dataUpdatedAt).toISOString())}
          </span>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => refetch()}
            aria-label="Refresh shortlist"
            disabled={isFetching}
          >
            <RefreshCcw className={cn("h-4 w-4", isFetching && "animate-spin")} />
          </Button>
        </div>
      </header>

      {data.hot_sectors.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap text-body-sm text-fg-muted">
          <ListChecks className="h-4 w-4 text-fg-subtle" aria-hidden />
          <span className="text-caption uppercase tracking-wider text-fg-subtle">
            Hot sectors
          </span>
          {data.hot_sectors.map((s) => (
            <Badge key={s} tone="brand">{s.replace("NIFTY_", "")}</Badge>
          ))}
        </div>
      )}

      {data.candidates.length === 0 ? (
        <EmptyState
          title="No shortlist candidates"
          description={
            data.errors.length > 0
              ? `Upstream data hiccup: ${data.errors[0]}`
              : "Hot sectors produced no names that cleared the hard filters."
          }
        />
      ) : (
        <div className="space-y-3">
          {data.candidates.map((c) => (
            <CandidateCard key={c.symbol} c={c} />
          ))}
        </div>
      )}

      {data.filtered_out.length > 0 && (
        <FilteredOut rows={data.filtered_out} />
      )}

      {data.errors.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-warn" />
              Provider warnings
            </CardTitle>
            <CardDescription>
              Data source reported non-fatal issues.  Candidates with missing
              metrics were skipped, not ranked.
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
/* Candidate card — score + metrics + reasons                           */
/* ================================================================== */
function CandidateCard({ c }: { c: StockCandidate }) {
  const tone = scoreTone(c.score);
  const dayChangeColor =
    c.change_pct == null ? "text-fg-muted" :
    c.change_pct > 0 ? "text-pnl-up" :
    c.change_pct < 0 ? "text-pnl-down" : "text-fg-muted";

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <CardTitle className="font-mono">
              <Link
                to={`/setup/${encodeURIComponent(c.symbol)}`}
                className="hover:text-accent focus-visible:outline-none focus-visible:underline"
                aria-label={`Open setup preview for ${c.symbol}`}
              >
                {c.symbol}
              </Link>
            </CardTitle>
            <Badge tone="neutral">
              {c.sector_label} · #{c.sector_rank}
            </Badge>
            {c.is_leader && <Badge tone="brand">Sector leader</Badge>}
          </div>
          <CardDescription className="mt-1">
            {c.last != null ? `Last ${fmtNum(c.last, 1)}` : "Last —"}
            {" · "}
            <span className={dayChangeColor}>{fmtPct(c.change_pct, 2)}</span>
          </CardDescription>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <Badge tone={tone}>Score {c.score.toFixed(1)}</Badge>
          <Link
            to={`/setup/${encodeURIComponent(c.symbol)}`}
            className="inline-flex items-center gap-1 text-caption text-fg-subtle hover:text-fg"
          >
            Setup <ArrowRight className="h-3 w-3" aria-hidden />
          </Link>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        <Metrics c={c} />
        {c.reasons.length > 0 && (
          <div className="flex gap-1.5 flex-wrap">
            {c.reasons.map((r, i) => (
              <span
                key={i}
                className="inline-flex items-center rounded-xs px-1.5 py-0.5 text-caption text-fg-muted border border-border/60 bg-surface-2"
              >
                {r}
              </span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Metrics({ c }: { c: StockCandidate }) {
  const cells: Array<[string, string, string?]> = [
    ["ATR %",       c.atr_pct != null      ? `${c.atr_pct.toFixed(2)}%` : "—"],
    ["Turnover",    c.turnover_cr != null  ? `₹${c.turnover_cr.toFixed(1)}cr` : "—"],
    ["Rel vol",     c.rel_volume != null   ? `${c.rel_volume.toFixed(2)}x` : "—"],
    ["52w pos",     c.range_52w_pos != null ? `${(c.range_52w_pos * 100).toFixed(0)}%` : "—"],
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {cells.map(([label, value]) => (
        <div key={label} className="rounded-sm border border-border/60 p-2">
          <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
          <div className="text-body font-mono tabular text-fg">{value}</div>
        </div>
      ))}
    </div>
  );
}

/* ================================================================== */
/* Filtered-out rejected candidates — collapsed drawer                  */
/* ================================================================== */
function FilteredOut({ rows }: { rows: RejectedCandidate[] }) {
  const [open, setOpen] = React.useState(false);
  return (
    <Card>
      <CardHeader
        className="cursor-pointer select-none"
        onClick={() => setOpen((o) => !o)}
        role="button"
        aria-expanded={open}
      >
        <CardTitle className="flex items-center gap-2">
          {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          Filtered out ({rows.length})
        </CardTitle>
        <CardDescription>
          Names from the hot sectors that missed a hard filter.  Expand to see why.
        </CardDescription>
      </CardHeader>
      {open && (
        <CardContent>
          <ul className="divide-y divide-border/60">
            {rows.map((r) => (
              <li
                key={r.symbol}
                className="flex items-center justify-between gap-3 py-2"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-mono text-body-sm text-fg">{r.symbol}</span>
                  <span className="text-caption text-fg-subtle">{r.sector_label}</span>
                </div>
                <div className="flex gap-1.5 flex-wrap justify-end">
                  {r.reject_reasons.map((reason, i) => (
                    <Badge key={i} tone="danger">{reason}</Badge>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </CardContent>
      )}
    </Card>
  );
}

/* ================================================================== */
/* Loading / error states                                               */
/* ================================================================== */
function ShortlistLoading() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1200px] mx-auto">
      <Skeleton className="h-12 w-80" />
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-36 w-full" />
        ))}
      </div>
    </div>
  );
}

function ShortlistError({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="px-6 py-6 max-w-[800px] mx-auto">
      <EmptyState
        title="Couldn't load shortlist"
        description={error?.message ?? "Unknown error"}
        action={<Button onClick={onRetry}>Try again</Button>}
      />
    </div>
  );
}
