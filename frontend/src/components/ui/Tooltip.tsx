import * as React from "react";
import * as TooltipPrim from "@radix-ui/react-tooltip";
import { cn } from "@/lib/utils";

/**
 * Wrap the app in <TooltipProvider delayDuration={250}> once.
 */
export const TooltipProvider = TooltipPrim.Provider;
export const Tooltip         = TooltipPrim.Root;
export const TooltipTrigger  = TooltipPrim.Trigger;

export function TooltipContent({
  className, sideOffset = 6, ...props
}: React.ComponentProps<typeof TooltipPrim.Content>) {
  return (
    <TooltipPrim.Portal>
      <TooltipPrim.Content
        sideOffset={sideOffset}
        {...props}
        className={cn(
          "z-tooltip rounded-xs bg-surface-2 border border-border-strong",
          "px-2 py-1 text-caption text-fg shadow",
          "data-[state=delayed-open]:animate-fade-in motion-reduce:animate-none",
          className,
        )}
      >
        {props.children}
        <TooltipPrim.Arrow className="fill-surface-2" />
      </TooltipPrim.Content>
    </TooltipPrim.Portal>
  );
}
