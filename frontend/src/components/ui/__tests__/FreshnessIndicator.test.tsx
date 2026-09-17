import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TooltipProvider } from "@/components/ui/Tooltip";
import { FreshnessIndicator } from "../FreshnessIndicator";

/**
 * FreshnessIndicator drives "should I trust this number" decisions for the
 * trader. Its invariants:
 *  - colour follows three tiers (fresh / stale / old) using --pnl-up / --warn
 *    / --pnl-down so the trader doesn't have to read the seconds counter.
 *  - the seconds counter renders a relative age string ("Xs ago").
 *  - missing timestamps degrade to an em-dash (never crash, never lie).
 */
function withProvider(ui: React.ReactNode) {
  return <TooltipProvider>{ui}</TooltipProvider>;
}

describe("<FreshnessIndicator />", () => {
  it("renders 'fresh' tier when within freshMs", () => {
    const now = Date.now();
    render(withProvider(
      <FreshnessIndicator
        timestamp={now - 2_000}
        freshMs={5_000}
        staleMs={30_000}
      />,
    ));
    const node = screen.getByText(/Updated 2s ago/);
    expect(node).toBeInTheDocument();
    expect(node.className).toMatch(/text-pnl-up/);
  });

  it("renders 'stale' tier with warn colour between freshMs and staleMs", () => {
    const now = Date.now();
    render(withProvider(
      <FreshnessIndicator
        timestamp={now - 20_000}
        freshMs={5_000}
        staleMs={60_000}
      />,
    ));
    const node = screen.getByText(/Updated 20s ago/);
    expect(node.className).toMatch(/text-warn/);
  });

  it("renders 'old' tier with danger colour past staleMs", () => {
    const now = Date.now();
    render(withProvider(
      <FreshnessIndicator
        timestamp={now - 120_000}
        freshMs={5_000}
        staleMs={60_000}
      />,
    ));
    const node = screen.getByText(/Updated 2m ago/);
    expect(node.className).toMatch(/text-pnl-down/);
  });

  it("renders an em-dash when no timestamp is provided", () => {
    render(withProvider(<FreshnessIndicator timestamp={null} />));
    expect(screen.getByText(/Updated —/)).toBeInTheDocument();
  });

  it("respects the custom label", () => {
    const now = Date.now();
    render(withProvider(
      <FreshnessIndicator
        timestamp={now - 1_000}
        label="Last tick"
      />,
    ));
    expect(screen.getByText(/Last tick 1s ago/)).toBeInTheDocument();
  });
});
