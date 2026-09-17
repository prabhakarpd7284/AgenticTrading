import { ArrowDown, ArrowUp } from "lucide-react";
import { cn, fmtInr, fmtPct, clsPnl } from "@/lib/utils";
import { Skeleton } from "./Skeleton";

/**
 * KPI — headline number for Dashboard / Portfolio.
 *
 *  label         short descriptor (caption tone)
 *  value         the primary number
 *  valueFormat   "inr" | "pct" | "num" (defaults to num with tabular digits)
 *  delta         optional signed change (used for arrow + colour)
 *  deltaFormat   "pct" | "inr" — how the delta renders
 *  hint          optional secondary row (e.g. "vs yesterday")
 *  live          if true, wraps value in aria-live="polite" (announces ticks)
 *  loading       render skeleton
 */
export interface KPIProps {
  label: string;
  value?: number;
  valueFormat?: "inr" | "pct" | "num";
  delta?: number;
  deltaFormat?: "pct" | "inr";
  hint?: string;
  live?: boolean;
  loading?: boolean;
  className?: string;
}

export function KPI({
  label, value, valueFormat = "num", delta, deltaFormat = "pct",
  hint, live, loading, className,
}: KPIProps) {
  const rendered =
    value === undefined ? "—" :
    valueFormat === "inr" ? fmtInr(value) :
    valueFormat === "pct" ? fmtPct(value) :
    new Intl.NumberFormat("en-IN").format(value);

  return (
    <div className={cn("rounded-md border border-border bg-surface px-4 py-3", className)}>
      <div className="flex items-center justify-between">
        <span className="text-caption uppercase tracking-wider text-fg-subtle">{label}</span>
        {delta !== undefined && <DeltaChip delta={delta} format={deltaFormat} />}
      </div>

      {loading ? (
        <Skeleton className="h-8 w-28 mt-2" />
      ) : (
        <div
          className={cn("font-mono tabular text-num-lg mt-1.5 text-fg")}
          aria-live={live ? "polite" : undefined}
        >
          {rendered}
        </div>
      )}

      {hint && <p className="text-caption text-fg-subtle mt-0.5">{hint}</p>}
    </div>
  );
}

function DeltaChip({ delta, format }: { delta: number; format: "pct" | "inr" }) {
  const sign = delta > 0 ? "+" : delta < 0 ? "−" : "";
  // Build the magnitude string WITHOUT a sign — DeltaChip prepends its own
  // glyph (+ / −) below, so re-running through fmtPct (which always stamps
  // its own "+" for positives) would double the sign on negatives, e.g.
  // "−+0.89%". Render an unsigned magnitude here instead.
  const magnitude =
    format === "pct"
      ? `${Math.abs(delta).toFixed(2)}%`
      : fmtInr(Math.abs(delta));
  // Concatenate sign + magnitude into ONE string so it lands as a single
  // text node — RTL `getByText` cannot match across adjacent text nodes
  // produced by `{sign}{magnitude}` interpolation.
  const label = `${sign}${magnitude}`;
  const Icon = delta >= 0 ? ArrowUp : ArrowDown;
  return (
    <span className={cn(
      "inline-flex items-center gap-0.5 rounded-xs border px-1.5 py-0.5 text-caption font-mono tabular",
      clsPnl(delta),
      delta >= 0 ? "border-pnl-up/25 bg-pnl-up/10" : "border-pnl-down/25 bg-pnl-down/10",
    )}>
      <Icon className="h-3 w-3" aria-hidden />
      <span className="sr-only">{delta >= 0 ? "up" : "down"}</span>
      {label}
    </span>
  );
}
