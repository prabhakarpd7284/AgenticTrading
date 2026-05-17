import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "./Button";
import { cn } from "@/lib/utils";

export interface ErrorStateProps {
  title?: string;
  description?: string;
  onRetry?: () => void;
  supportHref?: string;
  className?: string;
}

export function ErrorState({
  title = "Something went wrong.",
  description = "The request failed. You can try again or contact support if the problem persists.",
  onRetry, supportHref = "mailto:support@alphadesk.app", className,
}: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center text-center gap-3 py-10 px-6",
        "rounded-md border border-danger/30 bg-pnl-down/5",
        className,
      )}
    >
      <AlertTriangle className="h-8 w-8 text-danger" aria-hidden />
      <div className="space-y-1.5 max-w-md">
        <h3 className="text-h3 text-fg">{title}</h3>
        <p className="text-body-sm text-fg-muted">{description}</p>
      </div>
      <div className="flex gap-2">
        {onRetry && (
          <Button variant="secondary" size="sm" onClick={onRetry} leading={<RefreshCw className="h-4 w-4" />}>
            Try again
          </Button>
        )}
        <Button variant="ghost" size="sm" asChild>
          <a href={supportHref}>Contact support</a>
        </Button>
      </div>
    </div>
  );
}
