/**
 * Select — accessible dropdown using Radix Select, themed against the
 * AlphaDesk design tokens.
 *
 * The native <select> element is a UX dead-end on dark themes:
 *   - the dropdown panel inherits the OS chrome, not our tokens
 *   - the chevron is OS-default (a thin grey caret) which can disappear
 *     against a dark trigger
 *   - the hover/selected state in the option list is OS-blue, fighting
 *     the rest of the UI
 *
 * This component renders a Radix Select with our `bg-surface`,
 * `border-border-strong`, `text-fg` tokens throughout, so it looks
 * identical on light and dark themes and works with the keyboard
 * (Space/Enter to open, ↑/↓ to navigate, Esc to close, type-ahead).
 *
 * API mirrors a controlled native select:
 *   <Select value={mode} onValueChange={setMode}>
 *     <SelectTrigger placeholder="Pick a mode" />
 *     <SelectContent>
 *       <SelectItem value="BULL_PUT">Bull Put</SelectItem>
 *       <SelectItem value="BEAR_CALL">Bear Call</SelectItem>
 *     </SelectContent>
 *   </Select>
 */
import * as React from "react";
import * as SelectPrim from "@radix-ui/react-select";
import { Check, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";


export const Select = SelectPrim.Root;
export const SelectValue = SelectPrim.Value;
export const SelectGroup = SelectPrim.Group;
export const SelectLabel = ({ className, ...p }: React.ComponentProps<typeof SelectPrim.Label>) => (
  <SelectPrim.Label
    {...p}
    className={cn("px-2 py-1.5 text-[10px] uppercase tracking-wider text-fg-muted", className)}
  />
);
export const SelectSeparator = ({ className, ...p }: React.ComponentProps<typeof SelectPrim.Separator>) => (
  <SelectPrim.Separator {...p} className={cn("my-1 h-px bg-border", className)} />
);


/* ───────── Trigger ─────────
 * Looks exactly like the Input field but acts as a button.
 * Shows the selected value (or placeholder) on the left, chevron on the right.
 */
export const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrim.Trigger>,
  React.ComponentProps<typeof SelectPrim.Trigger> & { placeholder?: string }
>(({ className, placeholder, children, ...props }, ref) => (
  <SelectPrim.Trigger
    ref={ref}
    className={cn(
      "flex items-center justify-between gap-2 w-full h-9 px-3",
      "rounded-sm border bg-surface text-body text-fg",
      "border-border hover:border-border-strong",
      "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-1 focus-visible:ring-offset-bg",
      "transition-[box-shadow,border-color] duration-120",
      "data-[placeholder]:text-fg-subtle",
      "disabled:cursor-not-allowed disabled:opacity-50",
      className,
    )}
    {...props}
  >
    {/* When children are supplied, render them verbatim — caller controls the
        formatting (e.g. multi-line). Otherwise show the selected value via
        SelectValue with the optional placeholder. */}
    {children ?? <SelectPrim.Value placeholder={placeholder} />}
    <SelectPrim.Icon asChild>
      <ChevronDown className="h-4 w-4 text-fg-muted shrink-0" aria-hidden />
    </SelectPrim.Icon>
  </SelectPrim.Trigger>
));
SelectTrigger.displayName = "SelectTrigger";


/* ───────── Content (popup panel) ─────────
 * Portalled so it isn't clipped by Sheet's overflow. Animates in via
 * Radix's data-state attributes. Max-height + auto-scroll keeps long
 * expiry lists usable.
 */
export const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrim.Content>,
  React.ComponentProps<typeof SelectPrim.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrim.Portal>
    <SelectPrim.Content
      ref={ref}
      position={position}
      sideOffset={4}
      className={cn(
        "z-50 min-w-[10rem] overflow-hidden",
        "rounded-sm border border-border-strong bg-surface",
        "shadow-lg shadow-black/40",
        "data-[state=open]:animate-in data-[state=closed]:animate-out",
        "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
        "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
        // Match trigger width when popper-positioned so the panel doesn't
        // get narrower than the field it dropped from.
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=top]:-translate-y-1 " +
          "w-[var(--radix-select-trigger-width)] max-h-[var(--radix-select-content-available-height)]",
        className,
      )}
      {...props}
    >
      <SelectPrim.ScrollUpButton className="flex items-center justify-center h-6 cursor-default text-fg-muted">
        <ChevronUp className="h-3.5 w-3.5" />
      </SelectPrim.ScrollUpButton>
      <SelectPrim.Viewport className="p-1">
        {children}
      </SelectPrim.Viewport>
      <SelectPrim.ScrollDownButton className="flex items-center justify-center h-6 cursor-default text-fg-muted">
        <ChevronDown className="h-3.5 w-3.5" />
      </SelectPrim.ScrollDownButton>
    </SelectPrim.Content>
  </SelectPrim.Portal>
));
SelectContent.displayName = "SelectContent";


/* ───────── Item ─────────
 * One row in the panel. Hover + keyboard focus both use `data-[highlighted]`
 * (Radix sets it for whichever input device is active), so colour stays
 * consistent across mouse + keyboard navigation.
 */
export const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrim.Item>,
  React.ComponentProps<typeof SelectPrim.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrim.Item
    ref={ref}
    className={cn(
      "relative flex items-center select-none rounded-xs cursor-default",
      "h-8 pl-7 pr-2 text-body-sm text-fg",
      "data-[highlighted]:bg-surface-2 data-[highlighted]:text-fg data-[highlighted]:outline-none",
      "data-[state=checked]:font-medium",
      "data-[disabled]:opacity-50 data-[disabled]:pointer-events-none",
      className,
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrim.ItemIndicator>
        <Check className="h-3.5 w-3.5 text-accent" />
      </SelectPrim.ItemIndicator>
    </span>
    <SelectPrim.ItemText>{children}</SelectPrim.ItemText>
  </SelectPrim.Item>
));
SelectItem.displayName = "SelectItem";
