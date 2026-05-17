/**
 * OpButton — a trigger button that opens a wide modal containing the
 * OpRunPanel. The primitive for embedding a contextual CLI runner in any
 * page (Monthly's "Refresh signal outcomes", Swing Scanner's "Re-scan
 * now", Setup's "Plan a trade for <symbol>", etc.).
 *
 *   <OpButton
 *     command="enrich_signals"
 *     defaultArgs="--all"
 *     label="Refresh signal outcomes"
 *     description="Backfill EOD outcomes — feeds the Monthly numbers."
 *     onSuccess={() => qc.invalidateQueries({ queryKey: ["monthly"] })}
 *   />
 */
import * as React from "react";
import * as DialogPrim from "@radix-ui/react-dialog";
import { Terminal, X } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button, type ButtonProps } from "@/components/ui/Button";
import { Dialog, DialogTrigger } from "@/components/ui/Dialog";
import { OpRunPanel, type OpRunPanelProps } from "@/features/ops/OpRunPanel";

export interface OpButtonProps
  extends Omit<OpRunPanelProps, "onSuccess">,
    Pick<ButtonProps, "variant" | "size"> {
  /** Visible label on the trigger button. Defaults to the command name. */
  label?: string;
  /** Optional icon on the trigger button. Defaults to a terminal icon. */
  icon?: React.ReactNode;
  /** Whether to render the trigger as an icon-only button. */
  iconOnly?: boolean;
  /** Caller can pass a custom child to use as the trigger instead of the button. */
  children?: React.ReactNode;
  /** Fired on exit 0 — typically a query invalidation. */
  onSuccess?: () => void;
  /** Controlled-open mode. If not given, the button manages its own state. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function OpButton(props: OpButtonProps) {
  const {
    label, icon, iconOnly, children, variant = "secondary", size = "sm",
    open, onOpenChange,
    command, description, defaultArgs, dangerous, onSuccess, onError, hideHelp,
  } = props;

  const trigger = children ? (
    <DialogTrigger asChild>{children}</DialogTrigger>
  ) : iconOnly ? (
    <DialogTrigger asChild>
      <Button variant={variant} size="icon" aria-label={label ?? command}>
        {icon ?? <Terminal className="size-4" />}
      </Button>
    </DialogTrigger>
  ) : (
    <DialogTrigger asChild>
      <Button variant={variant} size={size}>
        {icon ?? <Terminal className="mr-1.5 size-4" />}
        {label ?? command}
      </Button>
    </DialogTrigger>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {trigger}
      <DialogPrim.Portal>
        <DialogPrim.Overlay
          className={cn(
            "fixed inset-0 z-dialog bg-black/60 backdrop-blur-[2px]",
            "data-[state=open]:animate-fade-in motion-reduce:animate-none",
          )}
        />
        <DialogPrim.Content
          className={cn(
            "fixed left-1/2 top-1/2 z-dialog flex -translate-x-1/2 -translate-y-1/2 flex-col",
            "w-[min(95vw,820px)] h-[min(85vh,640px)]",
            "rounded-lg border border-border bg-surface p-5 shadow-lg",
            "data-[state=open]:animate-slide-up motion-reduce:animate-none",
            "focus:outline-none",
          )}
        >
          <DialogPrim.Title className="sr-only">{label ?? command}</DialogPrim.Title>
          <DialogPrim.Description className="sr-only">
            {description ?? `Run the ${command} management command.`}
          </DialogPrim.Description>
          <DialogPrim.Close
            aria-label="Close"
            className="absolute right-3 top-3 inline-flex h-7 w-7 items-center justify-center rounded text-fg-muted hover:bg-surface-2 hover:text-fg"
          >
            <X className="size-4" aria-hidden />
          </DialogPrim.Close>

          <OpRunPanel
            command={command}
            description={description}
            defaultArgs={defaultArgs}
            dangerous={dangerous}
            hideHelp={hideHelp}
            onSuccess={onSuccess}
            onError={onError}
          />
        </DialogPrim.Content>
      </DialogPrim.Portal>
    </Dialog>
  );
}
