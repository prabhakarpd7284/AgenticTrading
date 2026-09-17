/**
 * Track record — gives the Setup page a memory.
 *
 * Stage 5 answers "would this clear right now?".  It used to be amnesiac: it
 * never told you that you already hold the name, or how the last few trades on
 * it actually went.  This module closes the Cascade → Feedback loop right at
 * the decision point, reusing the same data + chart modal the Monthly review
 * already uses — no new execution surface, just context.
 *
 *   • LivePositionGuard — a warning strip when an open position exists for the
 *     symbol today (don't plan a fresh entry blind to what you're holding).
 *   • TrackRecordCard   — stats + recent closed trades (each opens the shared
 *     TradeChartModal) + a screener-signal summary line.
 */
import * as React from "react";
import { Link } from "react-router-dom";
import { History, LineChart, TriangleAlert } from "lucide-react";

import {
  useTrades, usePositions, useSymbolSignals, useSavedSetups,
  type Trade, type SavedSetup,
} from "@/lib/v2";
import type { PositionLeg } from "@/lib/monthly";
import { TradeChartModal } from "@/features/monthly/TradeChartModal";
import { cn, clsPnl, fmtDateTime, fmtInr, fmtNum, fmtRel } from "@/lib/utils";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

const MAX_ROWS = 8;

const REASON_TEXT: Record<string, string> = {
  SL_HIT: "Stopped", TARGET_HIT: "Target", TRAIL: "Trailed", EOD: "Squared off", MANUAL: "Closed",
};
const REASON_TONE: Record<string, "success" | "danger" | "neutral"> = {
  TARGET_HIT: "success", SL_HIT: "danger", EOD: "neutral", MANUAL: "neutral", TRAIL: "neutral",
};

/** Map a /trades row into the PositionLeg shape the chart modal consumes. */
function tradeToLeg(t: Trade): PositionLeg {
  return {
    id: t.id,
    symbol: t.symbol,
    side: t.side,
    quantity: t.quantity,
    entry_price: t.entry_price,
    exit_price: t.exit_price,
    entry_date: t.trade_date,
    exit_date: t.closed_at,
    target_price: t.target,
    stop_price: t.stop_loss,
    pnl: t.pnl ?? 0,
    status: t.status === "CLOSED" ? "CLOSED" : "OPEN",
    close_reason: t.close_reason || undefined,
    source: t.source,
  };
}

/* ================================================================== */
/* Live-position guard                                                 */
/* ================================================================== */

/** Warning strip shown only when an open position exists for this symbol
 *  today. Renders nothing otherwise (no empty row). Links to /positions. */
export function LivePositionGuard({ symbol }: { symbol: string }) {
  const { data } = usePositions();
  const pos = (data?.equity ?? []).find(
    (p) => p.symbol.toUpperCase() === symbol.toUpperCase(),
  );
  if (!pos) return null;

  const px = pos.fill_price ?? pos.entry_price;
  return (
    <Link
      to="/positions"
      className={cn(
        "flex items-center gap-3 rounded-sm border border-warn/40 bg-warn/10 px-4 py-2.5",
        "text-body-sm transition-colors hover:bg-warn/15",
      )}
    >
      <TriangleAlert className="h-4 w-4 shrink-0 text-warn" aria-hidden />
      <span className="text-fg">
        You already hold this name —{" "}
        <Badge tone={pos.side === "BUY" ? "success" : "danger"}>{pos.side}</Badge>{" "}
        <span className="font-mono tabular">{fmtNum(pos.quantity)}</span> @{" "}
        <span className="font-mono tabular">{fmtNum(px, 2)}</span>
      </span>
      {pos.pnl != null && (
        <span className={cn("ml-auto font-mono tabular", clsPnl(pos.pnl))}>
          {pos.pnl >= 0 ? "+" : ""}{fmtInr(pos.pnl)}
        </span>
      )}
      <span className={cn("text-caption text-fg-subtle", pos.pnl == null && "ml-auto")}>
        Positions →
      </span>
    </Link>
  );
}

/* ================================================================== */
/* Track-record card                                                   */
/* ================================================================== */

export function TrackRecordCard({
  symbol, lastPrice,
}: { symbol: string; lastPrice?: number | null }) {
  const { data, isLoading } = useTrades({ symbol, limit: 50 });
  const signals = useSymbolSignals(symbol);
  const saved = useSavedSetups(symbol);

  const trades = data?.results ?? [];
  const legs = React.useMemo(() => trades.map(tradeToLeg), [trades]);
  const [openIdx, setOpenIdx] = React.useState<number | null>(null);

  // Stats over trades that actually settled (have a realized P&L).
  const settled = trades.filter((t) => t.pnl != null);
  const wins = settled.filter((t) => (t.pnl ?? 0) > 0).length;
  const netPnl = settled.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const winRate = settled.length ? Math.round((wins / settled.length) * 100) : null;
  const tradeCount = data?.count ?? trades.length;
  const last = trades[0]; // endpoint orders by -created_at

  const sigCount = signals.data?.count ?? 0;
  const savedSetups = saved.data ?? [];
  const hasHistory = tradeCount > 0 || sigCount > 0 || savedSetups.length > 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5 text-fg-muted" aria-hidden />
              Track record
            </CardTitle>
            <CardDescription>
              Past trades &amp; signals for{" "}
              <span className="font-mono">{symbol}</span> — click any row to
              replay it on a chart. Context only; nothing here executes.
            </CardDescription>
          </div>
          {netPnl !== 0 && settled.length > 0 && (
            <Badge tone={netPnl > 0 ? "success" : "danger"}>
              {netPnl > 0 ? "+" : ""}{fmtInr(netPnl)} net
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading ? (
          <Skeleton className="h-40 w-full" />
        ) : !hasHistory ? (
          <EmptyState
            title="No history yet"
            description={`No trades or screener signals on record for ${symbol}.`}
          />
        ) : (
          <>
            {/* Stat strip */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Trades" value={fmtNum(tradeCount)} />
              <Stat label="Win rate" value={winRate == null ? "—" : `${winRate}%`} />
              <Stat
                label="Net P&L"
                value={settled.length ? fmtInr(netPnl) : "—"}
                cls={settled.length ? clsPnl(netPnl) : undefined}
              />
              <Stat
                label="Last traded"
                value={last ? `${fmtRel(last.created_at)} ago` : "—"}
              />
            </div>

            {/* Saved setups — snapshots you chose to track; did you act? */}
            {savedSetups.length > 0 && (
              <SavedSetupsSection
                setups={savedSetups}
                trades={trades}
                lastPrice={lastPrice ?? null}
                onViewTrade={(i) => setOpenIdx(i)}
              />
            )}

            {/* Recent trades — click to open the shared chart modal */}
            {trades.length > 0 && (
              <div className="space-y-1">
                <SectionLabel>Trade history</SectionLabel>
                <div className="divide-y divide-border/60 border-t border-border/60">
                  {trades.slice(0, MAX_ROWS).map((t, i) => (
                    <TradeRow key={t.id} t={t} onView={() => setOpenIdx(i)} />
                  ))}
                </div>
              </div>
            )}

            {/* Signal summary line */}
            <SignalLine
              count={sigCount}
              loading={signals.isLoading}
              latest={signals.data?.latest ?? null}
              converted={tradeCount}
            />
          </>
        )}
      </CardContent>

      {/* Lives at card level so ←/→ can page through every trade. */}
      <TradeChartModal legs={legs} index={openIdx} onIndexChange={setOpenIdx} />
    </Card>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-caption uppercase tracking-wider text-fg-subtle">
      {children}
    </div>
  );
}

/* ================================================================== */
/* Saved setups — "did I act on this, and how is it doing?"            */
/* ================================================================== */

/** The trade (if any) that this snapshot turned into: same side, opened at
 *  or after the setup was generated. Returns its index in the trades array
 *  (which aligns with `legs`) so the row can open the chart modal. */
function matchedTradeIndex(setup: SavedSetup, trades: Trade[]): number {
  const gen = Date.parse(setup.generated_at);
  let best = -1;
  let bestT = Infinity;
  trades.forEach((t, i) => {
    if (t.side !== setup.side) return;
    const created = Date.parse(t.created_at);
    if (!Number.isFinite(created) || created < gen) return;
    if (created < bestT) { bestT = created; best = i; }
  });
  return best;
}

/** Signed move from planned entry in the trade's favour (+ = good), as %. */
function favorablePct(setup: SavedSetup, last: number | null): number | null {
  if (last == null || setup.entry_price == null || setup.entry_price === 0) return null;
  const raw = (last - setup.entry_price) / setup.entry_price * 100;
  return setup.side === "BUY" ? raw : -raw;
}

function SavedSetupsSection({
  setups, trades, lastPrice, onViewTrade,
}: {
  setups: SavedSetup[];
  trades: Trade[];
  lastPrice: number | null;
  onViewTrade: (i: number) => void;
}) {
  // Newest snapshot first.
  const sorted = React.useMemo(
    () => [...setups].sort((a, b) => Date.parse(b.generated_at) - Date.parse(a.generated_at)),
    [setups],
  );
  return (
    <div className="space-y-1">
      <SectionLabel>Saved setups · {setups.length}</SectionLabel>
      <div className="divide-y divide-border/60 border-t border-border/60">
        {sorted.map((s) => {
          const tradeIdx = matchedTradeIndex(s, trades);
          return (
            <SavedSetupRow
              key={s.id}
              s={s}
              tradeIdx={tradeIdx}
              trade={tradeIdx >= 0 ? trades[tradeIdx] : null}
              lastPrice={lastPrice}
              onViewTrade={onViewTrade}
            />
          );
        })}
      </div>
    </div>
  );
}

function SavedSetupRow({
  s, tradeIdx, trade, lastPrice, onViewTrade,
}: {
  s: SavedSetup;
  tradeIdx: number;
  trade: Trade | null;
  lastPrice: number | null;
  onViewTrade: (i: number) => void;
}) {
  const taken = tradeIdx >= 0 && trade != null;
  const fav = favorablePct(s, lastPrice);

  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-0.5 py-2.5 text-body-sm">
      {/* Plan + generation time */}
      <div className="flex items-center gap-2 min-w-0 flex-wrap">
        <Badge tone={s.side === "BUY" ? "success" : "danger"}>{s.side}</Badge>
        <span className="font-mono tabular text-fg-muted">
          @{fmtNum(s.entry_price, 2)}
        </span>
        <span className="text-caption text-fg-subtle">gen {fmtDateTime(s.generated_at)}</span>
        {!s.risk_approved && <Badge tone="warning">was blocked</Badge>}
      </div>

      {/* Outcome — taken (real P&L, opens chart) or not-taken (live vs entry) */}
      {taken ? (
        <button
          type="button"
          onClick={() => onViewTrade(tradeIdx)}
          title="Open the trade this setup turned into"
          className="justify-self-end inline-flex items-center gap-1.5 text-right hover:underline"
        >
          <Badge tone="brand">Taken</Badge>
          <span className={cn("font-mono tabular", clsPnl(trade!.pnl))}>
            {trade!.pnl == null
              ? (trade!.status === "CLOSED" ? "—" : "open")
              : `${trade!.pnl >= 0 ? "+" : ""}${fmtInr(trade!.pnl)}`}
          </span>
        </button>
      ) : (
        <span className="justify-self-end text-right text-caption text-fg-subtle whitespace-nowrap">
          Not taken
          {lastPrice != null && (
            <>
              {" · now "}
              <span className="font-mono tabular text-fg-muted">{fmtNum(lastPrice, 2)}</span>
              {fav != null && (
                <span className={cn("font-mono tabular", clsPnl(fav))}>
                  {" "}({fav >= 0 ? "+" : ""}{fav.toFixed(2)}% vs entry)
                </span>
              )}
            </>
          )}
        </span>
      )}
    </div>
  );
}

function Stat({ label, value, cls }: { label: string; value: React.ReactNode; cls?: string }) {
  return (
    <div className="rounded-sm border border-border/60 p-2">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn("text-body font-mono tabular text-fg", cls)}>{value}</div>
    </div>
  );
}

function TradeRow({ t, onView }: { t: Trade; onView: () => void }) {
  const closed = t.status === "CLOSED";
  const reason = REASON_TEXT[t.close_reason] ?? (closed ? "Closed" : "Open");
  const tone = REASON_TONE[t.close_reason] ?? "neutral";
  return (
    <button
      type="button"
      onClick={onView}
      title="Replay this trade on a chart"
      className={cn(
        "grid w-full grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 px-1 py-2.5 text-left",
        "text-body-sm transition-colors hover:bg-surface-2/40 md:grid-cols-[1fr_auto_auto_auto]",
      )}
    >
      {/* Symbol · side · source */}
      <span className="flex items-center gap-2 min-w-0">
        <Badge tone={t.side === "BUY" ? "success" : "danger"}>{t.side}</Badge>
        <span className="text-caption text-fg-subtle">
          {new Date(`${t.trade_date}T00:00:00`).toLocaleDateString(undefined, {
            day: "2-digit", month: "short",
          })}
        </span>
        {t.source && t.source !== "intraday" && (
          <span className="text-caption text-fg-subtle capitalize">{t.source}</span>
        )}
      </span>

      {/* Entry → exit */}
      <span className="font-mono tabular text-fg-muted text-right whitespace-nowrap">
        {fmtNum(t.entry_price, 2)}
        {t.exit_price != null && (
          <>
            {" → "}
            <span className="text-fg">{fmtNum(t.exit_price, 2)}</span>
          </>
        )}
        <span className="text-fg-subtle"> · {t.quantity}q</span>
      </span>

      {/* Outcome */}
      <span className="hidden md:block text-right">
        <Badge tone={tone}>{reason}</Badge>
      </span>

      {/* P&L */}
      <span
        className={cn(
          "font-mono tabular text-right whitespace-nowrap col-start-2 md:col-start-auto",
          clsPnl(t.pnl),
        )}
      >
        {t.pnl == null ? "—" : `${t.pnl >= 0 ? "+" : ""}${fmtInr(t.pnl)}`}
      </span>
    </button>
  );
}

function SignalLine({
  count, loading, latest, converted,
}: {
  count: number;
  loading: boolean;
  latest: { ts: string; side?: string; source?: string } | null;
  converted: number;
}) {
  if (loading) return <Skeleton className="h-5 w-64" />;
  if (count === 0) return null;
  return (
    <div className="flex items-center gap-2 text-caption text-fg-subtle">
      <LineChart className="h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>
        Screener fired <span className="text-fg-muted">{fmtNum(count)}</span>{" "}
        signal{count === 1 ? "" : "s"}
        {latest?.side ? ` (latest ${latest.side})` : ""} ·{" "}
        <span className="text-fg-muted">{fmtNum(converted)}</span> became trade
        {converted === 1 ? "" : "s"}
        {latest ? ` · last ${fmtRel(latest.ts)} ago` : ""}
      </span>
    </div>
  );
}
