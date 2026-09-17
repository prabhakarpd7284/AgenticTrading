import * as React from "react";
import { cn } from "@/lib/utils";

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      role="status"
      className={cn(
        "flex flex-col items-center text-center gap-3 py-12 px-6",
        "rounded-md border border-dashed border-border bg-surface/50",
        className,
      )}
    >
      {icon && (
        <div className="text-fg-subtle [&>svg]:h-8 [&>svg]:w-8" aria-hidden>
          {icon}
        </div>
      )}
      <div className="space-y-1.5 max-w-sm">
        <h3 className="text-h3 text-fg">{title}</h3>
        {description && <p className="text-body-sm text-fg-muted">{description}</p>}
      </div>
      {action}
    </div>
  );
}
