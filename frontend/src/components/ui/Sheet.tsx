import * as React from "react";
import * as DialogPrim from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Sheet — right-edge slide-in panel for "inspect alongside" workflows.
 *
 * Built on Radix Dialog (focus management + Escape + ARIA out of the box)
 * but rendered as a fixed right-edge panel instead of a centred modal.
 * The Root is non-modal (`modal={false}`) so the page beneath stays
 * interactive — the operator can click another row in the rail while
 * the Sheet is open and the panel just re-renders with new content.
 *
 * Use this for: inspecting an event, viewing a position's details,
 * editing a strategy's params — anywhere the user wants to see and act
 * on something without losing their place in the parent view.
 *
 * For a true modal (focus trap, backdrop, block-everything-else) use
 * Dialog instead.
 */
export function Sheet({
  open, onOpenChange, children,
}: {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: React.ReactNode;
}) {
  // modal={false} ⇒ no overlay, no focus trap, no blocking outside clicks.
  // Escape and our own Close button still trigger onOpenChange(false).
  return (
    <DialogPrim.Root open={open} onOpenChange={onOpenChange} modal={false}>
      {children}
    </DialogPrim.Root>
  );
}

export const SheetTrigger = DialogPrim.Trigger;
export const SheetClose   = DialogPrim.Close;

export function SheetContent({
  className, children, width = "560px", ...props
}: React.ComponentProps<typeof DialogPrim.Content> & {
  /** Sheet width — CSS length. Defaults to 560px (comfortable for JSON + cross-links). */
  width?: string;
}) {
  // pointer-events-auto: Radix Dialog's Content gets pointer-events:none when
  // modal={false} so clicks pass through to the page beneath; we re-enable on
  // the panel itself so its own children receive their clicks.
  return (
    <DialogPrim.Portal>
      <DialogPrim.Content
        {...props}
        // onInteractOutside fires when the user clicks anywhere outside the
        // Sheet. With modal=false Radix would close on outside-click by
        // default; we suppress that so clicking another rail row updates
        // the panel content rather than dismissing it. Operator dismisses
        // via Esc or the explicit Close affordance.
        onInteractOutside={(e) => e.preventDefault()}
        // PointerDownOutside fires for pointer events specifically — also
        // suppress so toolbar/header clicks don't dismiss the panel.
        onPointerDownOutside={(e) => e.preventDefault()}
        style={{ width, ...(props.style ?? {}) }}
        className={cn(
          "fixed right-0 top-0 bottom-0 z-dialog pointer-events-auto",
          "flex flex-col",
          "border-l border-border bg-surface shadow-2xl",
          "data-[state=open]:animate-slide-in-right motion-reduce:animate-none",
          "focus:outline-none",
          className,
        )}
      >
        {children}
        <DialogPrim.Close
          aria-label="Close panel"
          className="absolute right-3 top-3 inline-flex h-7 w-7 items-center justify-center rounded-xs text-fg-muted hover:bg-surface-2 hover:text-fg"
        >
          <X className="h-4 w-4" aria-hidden />
        </DialogPrim.Close>
      </DialogPrim.Content>
    </DialogPrim.Portal>
  );
}

export function SheetHeader({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "px-5 py-4 border-b border-border shrink-0",
        className,
      )}
      {...p}
    />
  );
}

export function SheetBody({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex-1 min-h-0 overflow-auto px-5 py-4", className)} {...p} />;
}

export function SheetFooter({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "px-5 py-3 border-t border-border shrink-0 flex items-center justify-end gap-2",
        className,
      )}
      {...p}
    />
  );
}

export const SheetTitle = ({ className, ...p }: React.ComponentProps<typeof DialogPrim.Title>) =>
  <DialogPrim.Title className={cn("text-h2 text-fg", className)} {...p} />;

export const SheetDescription = ({ className, ...p }: React.ComponentProps<typeof DialogPrim.Description>) =>
  <DialogPrim.Description className={cn("text-body-sm text-fg-muted mt-1", className)} {...p} />;
