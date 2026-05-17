import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { Loader2 } from "lucide-react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

/**
 * Button — primary interactive element.
 *
 * Variants:
 *   primary     — high-emphasis brand action (one per view ideally)
 *   secondary   — neutral action; pairs with primary
 *   ghost       — low-emphasis; toolbar / inline
 *   destructive — irreversible action (close position, delete)
 *   link        — inline anchor-styled action (no padding, underline-on-hover)
 *
 * Sizes: sm (h-8) / md (h-9) / lg (h-10) / icon (square h-9).
 *
 * Accessibility:
 *   • focus-visible ring uses `shadow-glow` (token), 3:1 against bg
 *   • disabled buttons are non-interactive AND get aria-disabled
 *   • loading sets aria-busy and replaces leading slot with spinner
 *   • use `asChild` to render as <a> or <Link> while keeping styles
 */
const button = cva(
  [
    "inline-flex items-center justify-center gap-2 select-none whitespace-nowrap",
    "font-medium rounded-sm",
    "transition-[background-color,color,box-shadow,transform] duration-120 ease-out",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-1 focus-visible:ring-offset-bg",
    "disabled:opacity-50 disabled:pointer-events-none",
    "active:translate-y-px motion-reduce:active:translate-y-0",
  ].join(" "),
  {
    variants: {
      variant: {
        primary:     "bg-accent text-accent-fg hover:bg-accent-hover active:bg-accent-pressed",
        secondary:   "bg-surface-2 text-fg hover:bg-surface-3 border border-border",
        ghost:       "bg-transparent text-fg-muted hover:text-fg hover:bg-surface-2",
        destructive: "bg-danger text-white hover:brightness-110",
        link:        "bg-transparent text-accent hover:underline underline-offset-4 px-0 h-auto",
      },
      size: {
        sm:   "h-8 px-3 text-body-sm",
        md:   "h-9 px-4 text-body-sm",
        lg:   "h-10 px-5 text-body",
        icon: "h-9 w-9 p-0",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof button> {
  asChild?: boolean;
  loading?: boolean;
  leading?: React.ReactNode;
  trailing?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant, size, asChild, loading, leading, trailing, children, disabled, ...props },
    ref,
  ) => {
    const Comp: any = asChild ? Slot : "button";
    const isDisabled = disabled || loading;

    // Radix Slot uses React.Children.only — it can only forward a SINGLE
    // child element. Wrapping leading/trailing/spinner inside the slotted
    // child would change its tag (e.g. <a>). So when asChild is set we
    // render the consumer's child verbatim and skip the affordances.
    // The variant + size + interaction classes still flow through via
    // Slot's className merge.
    const inner = asChild ? (
      children
    ) : (
      <>
        {loading ? (
          <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden />
        ) : (
          leading
        )}
        {children}
        {!loading && trailing}
      </>
    );

    return (
      <Comp
        ref={ref}
        className={cn(button({ variant, size }), className)}
        aria-busy={loading || undefined}
        aria-disabled={isDisabled || undefined}
        disabled={isDisabled}
        {...props}
      >
        {inner}
      </Comp>
    );
  },
);
Button.displayName = "Button";
