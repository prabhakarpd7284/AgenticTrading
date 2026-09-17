/**
 * TradeChartModal — visual review of a single derived trade.
 *
 * Opened from the Monthly trade table's "View chart" action. Shows the trade on
 * a TradingView lightweight-charts candlestick chart with:
 *   • neutral candles so the trade overlays own the colour,
 *   • horizontal Entry / Stop / Target / Exit price lines,
 *   • VERTICAL lines at the entry + exit *times* with embedded labels (the trade
 *     timeline), and the held window tinted by the P&L sign.
 * A plain-language summary + a 👍/👎 + note feedback control round it out. ← / →
 * page through the table.
 */
import * as React from "react";
import {
  ColorType, createChart, LineStyle,
  type SeriesMarker, type Time, type UTCTimestamp,
} from "lightweight-charts";
import { Calendar, ChevronLeft, ChevronRight, ThumbsDown, ThumbsUp } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { cn, fmtInr, fmtNum } from "@/lib/utils";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/Dialog";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import type { PositionLeg } from "@/lib/monthly";

interface ChartCandle { t: string; o: number; h: number; l: number; c: number; v: number }
interface ChartPayload {
  trade_id: string; symbol: string; side: "BUY" | "SELL";
  interval: string; source: string; strategy: string;
  entry_price: number; stop: number | null; target: number | null;
  exit_price: number | null; close_reason: string; pnl: number;
  reasoning: string; trade_date: string;
  entry_ts: string | null; exit_ts: string | null;
  candles: ChartCandle[];
}
interface FeedbackPayload { vote: "up" | "down" | null; note: string }

const OUTCOME_TONE: Record<string, "success" | "danger" | "neutral"> = {
  TARGET_HIT: "success", SL_HIT: "danger", EOD: "neutral", MANUAL: "neutral", TRAIL: "neutral",
};
const REASON_TEXT: Record<string, string> = {
  SL_HIT: "Stopped out", TARGET_HIT: "Target hit", TRAIL: "Trailing stop",
  EOD: "Squared off", MANUAL: "Closed",
};

const CHART_H = "h-[300px] sm:h-[420px]";
// IST is a fixed +5:30 (no DST). lightweight-charts renders UTCTimestamps as
// UTC, so intraday stamps are shifted to read in IST wall-clock.
const IST_OFFSET_SEC = 5.5 * 3600;

/** A theme token (e.g. "--accent") → comma-rgb lightweight-charts can parse. */
function tokenRgb(name: string): string {
  if (typeof window === "undefined") return "#888888";
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v ? `rgb(${v.split(/\s+/).join(", ")})` : "#888888";
}

/** "2026-05-29" → "Fri, 29 May 2026". */
function fmtDay(iso: string): string {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, {
    weekday: "short", day: "2-digit", month: "short", year: "numeric",
  });
}

/** Concise one-line description of the trade. */
function actionLine(leg: PositionLeg): string {
  const act = leg.side === "BUY" ? "Bought" : "Shorted";
  const reason = REASON_TEXT[leg.close_reason ?? ""] ?? "Closed";
  const tail = leg.exit_price != null
    ? ` → exited @ ${fmtNum(leg.exit_price, 2)} · ${reason}`
    : "";
  return `${act} ${leg.quantity} ${leg.symbol} @ ${fmtNum(leg.entry_price, 2)}${tail}`;
}

/** Holding period from the resolved entry/exit timestamps. */
function holdLabel(data?: ChartPayload): string | null {
  if (!data?.entry_ts || !data?.exit_ts) return null;
  const ms = Date.parse(data.exit_ts) - Date.parse(data.entry_ts);
  if (!Number.isFinite(ms) || ms < 0) return null;
  if (data.interval === "1d") return `${Math.max(1, Math.round(ms / 86_400_000))}d`;
  const hrs = ms / 3_600_000;
  if (hrs >= 1) return `${hrs.toFixed(hrs < 10 ? 1 : 0)}h`;
  return `${Math.max(1, Math.round(ms / 60_000))}m`;
}

export function TradeChartModal({
  legs, index, onIndexChange,
}: {
  legs: PositionLeg[];
  /** Index into `legs` of the open trade, or null when closed. */
  index: number | null;
  onIndexChange: (i: number | null) => void;
}) {
  const open = index != null;
  const leg = index != null ? legs[index] : null;
  const hasPrev = index != null && index > 0;
  const hasNext = index != null && index < legs.length - 1;

  // Timeframe lives here (not in the keyed ModalBody) so the 1H/1D choice
  // persists as you page between trades.
  const [tf, setTf] = React.useState<"1d" | "1h">("1d");

  // ← / → page through the table. Ignored while typing in the feedback note.
  React.useEffect(() => {
    if (!open || index == null) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT") return;
      if (e.key === "ArrowRight" && index < legs.length - 1) {
        e.preventDefault();
        onIndexChange(index + 1);
      } else if (e.key === "ArrowLeft" && index > 0) {
        e.preventDefault();
        onIndexChange(index - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, index, legs.length, onIndexChange]);

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onIndexChange(null); }}>
      <DialogContent className="w-[min(96vw,980px)]">
        <DialogTitle className="sr-only">Trade chart — {leg?.symbol}</DialogTitle>
        {open && leg && index != null && (
          <ModalBody
            key={leg.id}
            leg={leg}
            position={{ current: index + 1, total: legs.length }}
            tf={tf}
            onTf={setTf}
            onPrev={hasPrev ? () => onIndexChange(index - 1) : undefined}
            onNext={hasNext ? () => onIndexChange(index + 1) : undefined}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function ModalBody({
  leg, position, tf, onTf, onPrev, onNext,
}: {
  leg: PositionLeg;
  position: { current: number; total: number };
  tf: "1d" | "1h";
  onTf: (tf: "1d" | "1h") => void;
  onPrev?: () => void;
  onNext?: () => void;
}) {
  const qc = useQueryClient();

  // Swing trades default to daily candles; offer a 1h view too. The choice is
  // owned by the parent so it survives trade-to-trade navigation.
  const isSwing = leg.source === "swing";

  const { data, isLoading, isError } = useQuery<ChartPayload>({
    queryKey: ["trade-chart", leg.id, isSwing ? tf : "5m"],
    queryFn: () =>
      api.get(`trades/${leg.id}/chart/${isSwing ? `?interval=${tf}` : ""}`).then((r) => r.data),
    staleTime: 5 * 60_000,
  });

  const fb = useQuery<FeedbackPayload>({
    queryKey: ["trade-feedback", leg.id],
    queryFn: () => api.get(`trades/${leg.id}/feedback/`).then((r) => r.data),
    staleTime: 60_000,
  });

  const [note, setNote] = React.useState("");
  React.useEffect(() => { if (fb.data?.note) setNote(fb.data.note); }, [fb.data?.note]);

  const submit = useMutation({
    mutationFn: (vote: "up" | "down") =>
      api.post(`trades/${leg.id}/feedback/`, { vote, note }).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["trade-feedback", leg.id] }),
  });
  const vote = fb.data?.vote ?? null;

  const sideTone = leg.side === "BUY" ? "success" : "danger";
  const outcomeTone = OUTCOME_TONE[leg.close_reason ?? ""] ?? "neutral";
  const pnlCls = leg.pnl > 0 ? "text-pnl-up" : leg.pnl < 0 ? "text-pnl-down" : "text-fg";

  return (
    <div className="space-y-4">
      {/* Navigation */}
      <div className="flex items-center justify-between gap-2 pr-6">
        <span className="text-caption text-fg-muted tabular-nums">
          Trade {position.current} / {position.total}
        </span>
        <div className="flex items-center gap-1">
          <span className="hidden sm:inline text-caption text-fg-muted mr-1">
            <kbd className="font-mono">←</kbd> <kbd className="font-mono">→</kbd> to move
          </span>
          <Button variant="ghost" size="icon" onClick={onPrev} disabled={!onPrev}
            aria-label="Previous trade">
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={onNext} disabled={!onNext}
            aria-label="Next trade">
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Header — symbol + badges + date, P&L on the right */}
      <div className="flex items-start justify-between gap-3 pr-6">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono text-h3 text-fg">{leg.symbol}</span>
            <Badge tone={sideTone}>{leg.side}</Badge>
            {leg.close_reason && (
              <Badge tone={outcomeTone}>{leg.close_reason.replace("_", " ")}</Badge>
            )}
            {leg.source && <Badge tone="neutral">{leg.source}</Badge>}
          </div>
          <div className="flex items-center gap-1.5 text-body-sm text-fg-muted mt-1">
            <Calendar className="h-3.5 w-3.5" aria-hidden /> {fmtDay(leg.entry_date)}
          </div>
        </div>
        <div className={cn("text-right font-mono tabular shrink-0", pnlCls)}>
          <div className="text-caption text-fg-muted">Realised P&amp;L</div>
          <div className="text-h3">{leg.pnl > 0 ? "+" : ""}{fmtInr(leg.pnl)}</div>
        </div>
      </div>

      {/* Action + stats */}
      <div className="rounded-sm border border-border/60 bg-surface-2/60 px-3 py-2.5 space-y-2">
        <p className="text-body-sm text-fg">{actionLine(leg)}</p>
        <TradeStats leg={leg} data={data} />
      </div>

      {/* Timeframe toggle — swing trades can switch between daily and 1h */}
      {isSwing && (
        <div className="flex items-center justify-end gap-2">
          <span className="text-caption text-fg-subtle">Timeframe</span>
          <div className="inline-flex rounded-sm border border-border/60 overflow-hidden">
            {(["1d", "1h"] as const).map((iv) => (
              <button
                key={iv}
                type="button"
                onClick={() => onTf(iv)}
                className={cn(
                  "px-2.5 py-1 text-caption transition-colors",
                  tf === iv
                    ? "bg-accent/20 text-accent font-semibold"
                    : "text-fg-subtle hover:bg-surface-2",
                )}
              >
                {iv.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chart */}
      {isLoading && <Skeleton className={cn("w-full", CHART_H)} />}
      {isError && (
        <div className={cn("grid place-items-center text-body-sm text-fg-muted", CHART_H)}>
          Couldn’t load candles for this trade.
        </div>
      )}
      {data && data.candles.length === 0 && (
        <div className={cn("grid place-items-center text-body-sm text-fg-muted", CHART_H)}>
          No candles available for this session.
        </div>
      )}
      {data && data.candles.length > 0 && <TradeChart data={data} />}

      {/* Level legend */}
      {data && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-caption">
          <Level token="--accent" label="Entry" value={data.entry_price} />
          {data.stop != null && <Level token="--pnl-down" label="Stop" value={data.stop} />}
          {data.target != null && <Level token="--pnl-up" label="Target" value={data.target} />}
          {data.exit_price != null && <Level token="--warn" label="Exit" value={data.exit_price} />}
        </div>
      )}

      {/* Feedback */}
      <div className="border-t border-border/60 pt-3 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-caption uppercase tracking-wider text-fg-subtle">
            Was this a good trade?
          </span>
          {vote && (
            <span className="text-caption text-fg-muted">
              You voted {vote === "up" ? "👍" : "👎"}
            </span>
          )}
        </div>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          onBlur={() => { if (vote && note !== (fb.data?.note ?? "")) submit.mutate(vote); }}
          placeholder="Optional note — what was right or wrong about this setup? (saved with your rating)"
          rows={2}
          className="w-full resize-none rounded-sm border border-border/60 bg-surface-2 px-2.5 py-1.5 text-body-sm text-fg placeholder:text-fg-subtle focus:outline-none focus:ring-1 focus:ring-accent/50"
        />
        <div className="flex items-center gap-2">
          <Button
            size="sm" variant={vote === "up" ? "primary" : "secondary"}
            onClick={() => submit.mutate("up")} disabled={submit.isPending}
          >
            <ThumbsUp className="mr-1.5 h-3.5 w-3.5" /> Good
          </Button>
          <Button
            size="sm" variant={vote === "down" ? "primary" : "secondary"}
            onClick={() => submit.mutate("down")} disabled={submit.isPending}
          >
            <ThumbsDown className="mr-1.5 h-3.5 w-3.5" /> Bad
          </Button>
        </div>
      </div>
    </div>
  );
}

/**
 * Outcome stats. Risk / R:R are shown only when the stop sits on the correct
 * side of entry — a trailed stop on a winner can end up the "wrong" side, where
 * a risk figure would be meaningless.
 */
function TradeStats({ leg, data }: { leg: PositionLeg; data?: ChartPayload }) {
  const long = leg.side === "BUY";
  const qty = leg.quantity;
  const capital = leg.entry_price * qty;
  const ret = capital ? (leg.pnl / capital) * 100 : 0;
  const captured = leg.exit_price != null
    ? (long ? leg.exit_price - leg.entry_price : leg.entry_price - leg.exit_price)
    : null;

  let riskRs: number | null = null;
  let rr: number | null = null;
  if (leg.stop_price != null) {
    const riskPts = long ? leg.entry_price - leg.stop_price : leg.stop_price - leg.entry_price;
    if (riskPts > 0) {
      riskRs = riskPts * qty;
      if (leg.target_price != null) {
        const rewardPts = long
          ? leg.target_price - leg.entry_price
          : leg.entry_price - leg.target_price;
        if (rewardPts > 0) rr = rewardPts / riskPts;
      }
    }
  }
  const held = holdLabel(data);

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-caption">
      <Metric label="Return" value={`${ret >= 0 ? "+" : ""}${ret.toFixed(2)}%`} tone={ret} />
      {captured != null && (
        <Metric
          label="Captured"
          value={`${captured >= 0 ? "+" : ""}${fmtNum(captured, 2)} pts`}
          tone={captured}
        />
      )}
      {rr != null && <Metric label="R:R" value={`${rr.toFixed(2)}:1`} />}
      {riskRs != null && <Metric label="Risk" value={fmtInr(riskRs)} />}
      {held && <Metric label="Held" value={held} />}
      <Metric label="Capital" value={fmtInr(capital, { compact: true })} />
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: number }) {
  const cls = tone == null ? "text-fg" : tone >= 0 ? "text-pnl-up" : "text-pnl-down";
  return (
    <span className="text-fg-subtle">
      {label} <span className={cn("font-mono tabular", cls)}>{value}</span>
    </span>
  );
}

function Level({ token, label, value }: { token: string; label: string; value: number }) {
  return (
    <span className="flex items-center gap-1.5 text-fg-subtle">
      <span className="inline-block h-2 w-3 rounded-xs" style={{ background: `rgb(var(${token}))` }} />
      {label} <span className="font-mono text-fg">{fmtNum(value, 2)}</span>
    </span>
  );
}

/**
 * Candlestick chart with the trade highlighted: neutral candles, horizontal
 * level lines, a P&L-tinted held window, and VERTICAL entry/exit lines with
 * embedded labels (the trade timeline).
 */
function TradeChart({ data }: { data: ChartPayload }) {
  const elRef = React.useRef<HTMLDivElement>(null);
  const [coords, setCoords] = React.useState<{ e: number | null; x: number | null; w: number }>(
    { e: null, x: null, w: 0 },
  );
  const LV = React.useMemo(() => ({
    entry: tokenRgb("--accent"), stop: tokenRgb("--pnl-down"),
    target: tokenRgb("--pnl-up"), exit: tokenRgb("--warn"),
  }), []);

  React.useEffect(() => {
    const el = elRef.current;
    if (!el || data.candles.length === 0) return;
    const daily = data.interval === "1d";

    const chart = createChart(el, {
      width: el.clientWidth,
      height: el.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: "rgba(0, 0, 0, 0)" },
        textColor: "rgb(160, 166, 174)",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(148,163,184,0.07)" },
        horzLines: { color: "rgba(148,163,184,0.07)" },
      },
      rightPriceScale: { borderColor: "rgba(148,163,184,0.2)" },
      timeScale: {
        borderColor: "rgba(148,163,184,0.2)",
        timeVisible: !daily, secondsVisible: false,
      },
      crosshair: { mode: 0 },
      handleScale: false,
      handleScroll: false,
    });

    const series = chart.addCandlestickSeries({
      upColor: "rgba(148,163,184,0.20)", downColor: "rgba(148,163,184,0.50)",
      wickUpColor: "rgba(148,163,184,0.65)", wickDownColor: "rgba(148,163,184,0.65)",
      borderUpColor: "rgba(148,163,184,0.65)", borderDownColor: "rgba(148,163,184,0.65)",
      borderVisible: true, lastValueVisible: false, priceLineVisible: false,
    });

    const toTime = (t: string) =>
      daily
        ? (t.slice(0, 10) as unknown as UTCTimestamp)
        : ((Math.floor(Date.parse(t) / 1000) + IST_OFFSET_SEC) as UTCTimestamp);

    series.setData(
      data.candles.map((c) => ({
        time: toTime(c.t), open: c.o, high: c.h, low: c.l, close: c.c,
      })),
    );

    const line = (
      price: number | null, color: string, title: string,
      width: 1 | 2, dashed: boolean, axisLabel: boolean,
    ) => {
      if (price == null) return;
      series.createPriceLine({
        price, color, lineWidth: width,
        lineStyle: dashed ? LineStyle.Dashed : LineStyle.Solid,
        axisLabelVisible: axisLabel, title,
      });
    };
    line(data.stop, LV.stop, "Stop", 1, true, false);
    line(data.target, LV.target, "Target", 1, true, false);
    line(data.entry_price, LV.entry, "Entry", 2, false, true);
    line(data.exit_price, LV.exit, "Exit", 2, false, true);

    // Small on-candle markers in addition to the vertical lines.
    const long = data.side === "BUY";
    const markers: SeriesMarker<Time>[] = [];
    if (data.entry_ts) {
      markers.push({
        time: toTime(data.entry_ts), position: long ? "belowBar" : "aboveBar",
        color: LV.entry, shape: long ? "arrowUp" : "arrowDown", text: "Entry",
      });
    }
    if (data.exit_ts) {
      markers.push({
        time: toTime(data.exit_ts), position: long ? "aboveBar" : "belowBar",
        color: LV.exit, shape: "circle", text: "Exit",
      });
    }
    const k = (t: Time) => (typeof t === "number" ? t : String(t));
    markers.sort((a, b) => (k(a.time) < k(b.time) ? -1 : k(a.time) > k(b.time) ? 1 : 0));
    series.setMarkers(markers);

    chart.timeScale().fitContent();

    // Project entry/exit times to pixel x for the vertical-line overlay.
    const recompute = () => {
      const ts = chart.timeScale();
      const e = data.entry_ts ? ts.timeToCoordinate(toTime(data.entry_ts)) : null;
      const x = data.exit_ts ? ts.timeToCoordinate(toTime(data.exit_ts)) : null;
      setCoords({ e: e as number | null, x: x as number | null, w: el.clientWidth });
    };
    recompute();
    chart.timeScale().subscribeVisibleTimeRangeChange(recompute);

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
      recompute();
    });
    ro.observe(el);
    return () => { ro.disconnect(); chart.remove(); };
  }, [data, LV]);

  const tint = data.pnl >= 0 ? "rgba(16,185,129,0.10)" : "rgba(244,63,94,0.10)";
  const reason = REASON_TEXT[data.close_reason] ?? data.close_reason;

  return (
    <div className={cn("relative w-full", CHART_H)}>
      <div ref={elRef} className="absolute inset-0" />
      {/* Held window */}
      {coords.e != null && coords.x != null && (
        <div
          className="absolute top-0 bottom-[26px] pointer-events-none"
          style={{
            left: Math.min(coords.e, coords.x),
            width: Math.max(2, Math.abs(coords.x - coords.e)),
            background: tint,
          }}
          aria-hidden
        />
      )}
      {/* Entry timeline line + label */}
      {coords.e != null && (
        <VMark x={coords.e} containerW={coords.w} color={LV.entry} row={0}
          title="Entry" detail={`${data.strategy || data.side} · ${fmtNum(data.entry_price, 2)}`} />
      )}
      {/* Exit timeline line + label */}
      {coords.x != null && data.exit_price != null && (
        <VMark x={coords.x} containerW={coords.w} color={LV.exit} row={1}
          title="Exit" detail={`${reason} · ${fmtNum(data.exit_price, 2)}`} />
      )}
    </div>
  );
}

/** A vertical line at a time coordinate with an embedded label. */
function VMark({
  x, containerW, color, row, title, detail,
}: {
  x: number;
  containerW: number;
  color: string;
  row: number;
  title: string;
  detail: string;
}) {
  const nearRight = x > containerW - 160;
  const pos = nearRight ? { right: containerW - x + 4 } : { left: x + 4 };
  return (
    <>
      <div
        className="absolute top-0 bottom-[26px] w-px pointer-events-none"
        style={{ left: x, background: color, opacity: 0.85 }}
        aria-hidden
      />
      <div
        className="absolute pointer-events-none whitespace-nowrap rounded-xs px-1.5 py-0.5 text-[10px] leading-tight"
        style={{
          top: 4 + row * 20, ...pos,
          color, borderColor: color, borderWidth: 1, borderStyle: "solid",
          background: "rgb(var(--surface))",
        }}
      >
        <span className="font-semibold">{title}</span> {detail}
      </div>
    </>
  );
}
