import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeStyles = cva(
  "inline-flex items-center gap-1 rounded-xs px-1.5 py-0.5 text-caption font-medium uppercase tracking-wider border",
  {
    variants: {
      tone: {
        neutral: "bg-surface-2 text-fg-muted border-border",
        brand:   "bg-accent/15 text-accent border-accent/30",
        success: "bg-pnl-up/15 text-pnl-up border-pnl-up/30",
        warning: "bg-warn/15 text-warn border-warn/30",
        danger:  "bg-pnl-down/15 text-pnl-down border-pnl-down/30",
        info:    "bg-info/15 text-info border-info/30",
      },
      dot: { true: "pl-1", false: "" },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeStyles> {}

export function Badge({ className, tone, dot, children, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeStyles({ tone, dot }), className)} {...props}>
      {dot && <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}
