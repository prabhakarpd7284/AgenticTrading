import * as React from "react";
import * as LabelPrim from "@radix-ui/react-label";
import { cn } from "@/lib/utils";

/**
 * Input — accessible text field with leading/trailing slots, label, hint, and error.
 *
 * Always use a Label. If you must hide it, pass `labelSrOnly`.
 * Error message is linked via aria-describedby; aria-invalid flips on error.
 */
export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  labelSrOnly?: boolean;
  hint?: string;
  error?: string;
  leading?: React.ReactNode;
  trailing?: React.ReactNode;
  optional?: boolean;
  /** Width modifier for the container — defaults to w-full */
  containerClassName?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      id,
      label,
      labelSrOnly,
      hint,
      error,
      leading,
      trailing,
      optional,
      className,
      containerClassName,
      required,
      ...props
    },
    ref,
  ) => {
    const autoId = React.useId();
    const inputId = id ?? autoId;
    const hintId = `${inputId}-hint`;
    const errId = `${inputId}-err`;
    const describedBy =
      [error ? errId : null, hint ? hintId : null].filter(Boolean).join(" ") || undefined;

    return (
      <div className={cn("w-full", containerClassName)}>
        {label && (
          <LabelPrim.Root
            htmlFor={inputId}
            className={cn(
              "mb-1.5 inline-flex items-center gap-1.5 text-body-sm text-fg",
              labelSrOnly && "sr-only",
            )}
          >
            {label}
            {optional && <span className="text-caption text-fg-subtle">(optional)</span>}
            {required && <span className="text-danger" aria-hidden>*</span>}
          </LabelPrim.Root>
        )}

        <div
          className={cn(
            "flex items-center gap-2 h-9 px-3 rounded-sm border bg-surface",
            "focus-within:ring-2 focus-within:ring-accent/60 focus-within:ring-offset-1 focus-within:ring-offset-bg",
            "transition-[box-shadow,border-color] duration-120",
            error ? "border-danger" : "border-border hover:border-border-strong",
          )}
        >
          {leading && <span className="text-fg-muted [&>svg]:h-4 [&>svg]:w-4" aria-hidden>{leading}</span>}
          <input
            id={inputId}
            ref={ref}
            aria-invalid={!!error || undefined}
            aria-describedby={describedBy}
            required={required}
            className={cn(
              "flex-1 bg-transparent outline-none",
              "text-body text-fg placeholder:text-fg-subtle",
              "disabled:cursor-not-allowed disabled:opacity-50",
              className,
            )}
            {...props}
          />
          {trailing && <span className="text-fg-muted [&>svg]:h-4 [&>svg]:w-4" aria-hidden>{trailing}</span>}
        </div>

        {hint && !error && (
          <p id={hintId} className="mt-1 text-caption text-fg-subtle">{hint}</p>
        )}
        {error && (
          <p id={errId} role="alert" className="mt-1 text-caption text-danger">{error}</p>
        )}
      </div>
    );
  },
);
Input.displayName = "Input";
