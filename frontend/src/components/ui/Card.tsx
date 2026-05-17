import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Card composition — content container with soft elevation.
 *
 * Anatomy: Card > CardHeader > (CardTitle + CardDescription) + CardContent + CardFooter.
 * Use `interactive` when the whole card is a link/button target — adds hover + focus ring.
 */
export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
  as?: keyof JSX.IntrinsicElements;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, interactive, as: As = "div", ...props }, ref) => {
    // `As` is a union of every JSX intrinsic — TS chokes trying to merge
    // HTMLDivElement props with SVG element props ("Expression produces a
    // union type that is too complex to represent"). Cast the component to
    // a permissive type so JSX type-checks against the resolved tag at
    // runtime instead of the union.
    const Tag = As as React.ElementType;
    return (
      <Tag
        ref={ref as React.Ref<HTMLDivElement>}
        className={cn(
          "rounded-md border border-border bg-surface",
          "shadow-sm",
          interactive &&
            "cursor-pointer transition-[box-shadow,transform,border-color] duration-120 " +
              "hover:border-border-strong hover:shadow-md " +
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
          className,
        )}
        {...props}
      />
    );
  },
);
Card.displayName = "Card";

export const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...p }, ref) => (
    <div ref={ref} className={cn("flex flex-col gap-1 px-5 pt-5 pb-3", className)} {...p} />
  ),
);
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...p }, ref) => (
    <h3 ref={ref} className={cn("text-body font-semibold tracking-tight text-fg", className)} {...p} />
  ),
);
CardTitle.displayName = "CardTitle";

export const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...p }, ref) => (
  <p ref={ref} className={cn("text-body-sm text-fg-muted", className)} {...p} />
));
CardDescription.displayName = "CardDescription";

export const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...p }, ref) => <div ref={ref} className={cn("px-5 py-4", className)} {...p} />,
);
CardContent.displayName = "CardContent";

export const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...p }, ref) => (
    <div
      ref={ref}
      className={cn("flex items-center gap-2 px-5 pt-3 pb-5 border-t border-border/60", className)}
      {...p}
    />
  ),
);
CardFooter.displayName = "CardFooter";

/** @deprecated use CardContent instead */
export const CardBody = CardContent;
