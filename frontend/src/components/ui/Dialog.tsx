import * as React from "react";
import * as DialogPrim from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Dialog — Radix-based modal with focus trap and Esc handling.
 * Title + Description are required for a11y; pass them as children components.
 */
export const Dialog        = DialogPrim.Root;
export const DialogTrigger = DialogPrim.Trigger;
export const DialogClose   = DialogPrim.Close;

export function DialogContent({
  className, children, ...props
}: React.ComponentProps<typeof DialogPrim.Content>) {
  return (
    <DialogPrim.Portal>
      <DialogPrim.Overlay className={cn(
        "fixed inset-0 z-dialog bg-black/60 backdrop-blur-[2px]",
        "data-[state=open]:animate-fade-in motion-reduce:animate-none",
      )} />
      <DialogPrim.Content
        {...props}
        className={cn(
          "fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-dialog",
          "w-[min(92vw,480px)] max-h-[85vh] overflow-auto",
          "rounded-lg border border-border bg-surface p-5 shadow-lg",
          "data-[state=open]:animate-slide-up motion-reduce:animate-none",
          "focus:outline-none",
          className,
        )}
      >
        {children}
        <DialogPrim.Close
          aria-label="Close"
          className="absolute right-3 top-3 inline-flex h-7 w-7 items-center justify-center rounded-xs text-fg-muted hover:bg-surface-2 hover:text-fg"
        >
          <X className="h-4 w-4" aria-hidden />
        </DialogPrim.Close>
      </DialogPrim.Content>
    </DialogPrim.Portal>
  );
}

export const DialogTitle = ({ className, ...p }: React.ComponentProps<typeof DialogPrim.Title>) =>
  <DialogPrim.Title className={cn("text-h2 text-fg", className)} {...p} />;

export const DialogDescription = ({ className, ...p }: React.ComponentProps<typeof DialogPrim.Description>) =>
  <DialogPrim.Description className={cn("text-body-sm text-fg-muted mt-1", className)} {...p} />;
