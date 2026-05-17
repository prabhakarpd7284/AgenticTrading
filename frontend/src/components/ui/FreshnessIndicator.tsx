import * as React from "react";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "./Tooltip";

/**
 * Visual "data freshness" pill.
 *
 * Trading decisions are time-sensitive — a quote 90 seconds old is not the
 * same as a quote 5 seconds old. This component takes a timestamp (number ms
 * or ISO string) and renders a colour-coded "Updated Xs ago" badge. The
 * colour follows three thresholds:
 *
 *   age < freshMs       → green  (fresh)
 *   age < staleMs       → yellow (getting old, glance OK)
 *   age >= staleMs      → red    (stale — verify before acting)
 *
 * Defaults (5s fresh, 90s stale) suit live broker ticks. Pass `freshMs` /
 * `staleMs` to soften the gates for slower data sources (e.g. monthly
 * reports rebuild on a 5-minute cadence).
 *
 * Re-renders itself once per second so "Updated 5s ago" actually counts up
 * — without that, a user looking at a quiet page would see the badge frozen
 * even though time has clearly passed. Re-render is cheap; this is only on
 * pages the operator is actively looking at.
 */

export interface FreshnessIndicatorProps {
  /** Source timestamp. ms-since-epoch (TanStack's `dataUpdatedAt`) or ISO. */
  timestamp: number | string | Date | null | undefined;
  /** Below this age (ms) shows green. Default 5_000. */
  freshMs?: number;
  /** Below this age (ms) shows yellow. At/above is red. Default 90_000. */
  staleMs?: number;
  /** Label prefix. Default "Updated". */
  label?: string;
  /** Render compact (icon-only with tooltip). */
  compact?: boolean;
  /** "muted" suppresses the badge background so it can sit inline in a header. */
  variant?: "badge" | "muted";
  className?: string;
}

function toMs(t: FreshnessIndicatorProps["timestamp"]): number | null {
  if (t == null) return null;
  if (typeof t === "number") return t;
  if (t instanceof Date) return t.getTime();
  const parsed = new Date(t).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

function formatAge(ms: number): string {
  const secs = Math.max(0, Math.floor(ms / 1000));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86_400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86_400)}d ago`;
}

function formatExact(ms: number): string {
  try {
    return new Date(ms).toLocaleString("en-IN", { hour12: false });
  } catch {
    return new Date(ms).toISOString();
  }
}

export function FreshnessIndicator({
  timestamp,
  freshMs = 5_000,
  staleMs = 90_000,
  label = "Updated",
  compact = false,
  variant = "badge",
  className,
}: FreshnessIndicatorProps) {
  const ms = toMs(timestamp);
  // Force a re-render so the "Xs ago" string keeps counting up while the
  // operator stares at a quiet page. Half-second cadence so the seconds
  // counter is smooth without being wasteful.
  const [, tick] = React.useState(0);
  React.useEffect(() => {
    if (ms == null) return;
    const id = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, [ms]);

  if (ms == null) {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1 text-caption text-fg-subtle",
          variant === "badge" &&
            "rounded-xs border border-border/60 bg-surface-2 px-1.5 py-0.5",
          className,
        )}
        title="No timestamp"
      >
        <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-fg-subtle" />
        {label} —
      </span>
    );
  }

  const age = Math.max(0, Date.now() - ms);
  const tier: "fresh" | "stale" | "old" =
    age < freshMs ? "fresh" : age < staleMs ? "stale" : "old";

  const toneCls =
    tier === "fresh"
      ? "text-pnl-up"
      : tier === "stale"
        ? "text-warn"
        : "text-pnl-down";
  const dotCls =
    tier === "fresh"
      ? "bg-pnl-up"
      : tier === "stale"
        ? "bg-warn"
        : "bg-pnl-down";
  const borderCls =
    variant === "badge"
      ? tier === "fresh"
        ? "border-pnl-up/30 bg-pnl-up/5"
        : tier === "stale"
          ? "border-warn/30 bg-warn/5"
          : "border-pnl-down/30 bg-pnl-down/5"
      : "";

  const ageStr = formatAge(age);
  const tooltipText = `${label} ${ageStr} (${formatExact(ms)})`;

  const body = compact ? (
    <span
      aria-label={tooltipText}
      className={cn(
        "inline-flex items-center gap-1 rounded-xs",
        variant === "badge" && cn("border px-1.5 py-0.5 text-caption", borderCls, toneCls),
        variant === "muted" && cn("text-caption", toneCls),
        className,
      )}
    >
      <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", dotCls)} />
    </span>
  ) : (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-caption font-mono tabular",
        variant === "badge" && cn("rounded-xs border px-1.5 py-0.5", borderCls),
        toneCls,
        className,
      )}
    >
      <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", dotCls)} />
      {label} {ageStr}
    </span>
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild>{body}</TooltipTrigger>
      <TooltipContent>{tooltipText}</TooltipContent>
    </Tooltip>
  );
}
