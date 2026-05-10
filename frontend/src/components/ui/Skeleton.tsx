import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Skeleton — shimmer placeholder. Respects prefers-reduced-motion
 * (falls back to a flat tinted block without animation).
 */
export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden
      className={cn(
        "rounded-xs shimmer animate-shimmer motion-reduce:animate-none",
        "motion-reduce:bg-surface-2",
        className,
      )}
      {...props}
    />
  );
}
