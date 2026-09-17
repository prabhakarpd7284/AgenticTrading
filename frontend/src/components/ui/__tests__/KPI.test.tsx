import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { KPI } from "../KPI";

/**
 * KPI is the first thing a user sees on the dashboard. Its invariants:
 *  - INR values use the en-IN grouping (lakhs / crores).
 *  - Positive / negative deltas render the correct sign glyph AND colour class
 *    (colour alone is never enough — WCAG non-visual cues).
 *  - `live` flips on aria-live so screen readers announce tick updates.
 */
describe("<KPI />", () => {
  it("formats an INR value with the en-IN grouping", () => {
    render(<KPI label="Capital" value={500_000} valueFormat="inr" />);
    // Intl output: "₹5,00,000" — lakhs separator, not thousands.
    expect(screen.getByText(/₹5,00,000/)).toBeInTheDocument();
  });

  it("renders a positive delta with the up arrow and pnl-up class", () => {
    render(
      <KPI label="Day P&L" value={12_340} valueFormat="inr" delta={1.42} />,
    );
    const chip = screen.getByText(/\+1\.42%/);
    expect(chip).toBeInTheDocument();
    expect(chip.className).toMatch(/text-pnl-up/);
    expect(screen.getByText(/up/i)).toBeInTheDocument(); // sr-only text
  });

  it("renders a negative delta with the minus sign and pnl-down class", () => {
    render(
      <KPI label="Day P&L" value={-12_340} valueFormat="inr" delta={-0.89} />,
    );
    const chip = screen.getByText(/−0\.89%/);
    expect(chip).toBeInTheDocument();
    expect(chip.className).toMatch(/text-pnl-down/);
    expect(screen.getByText(/down/i)).toBeInTheDocument();
  });

  it("renders a dash when value is undefined", () => {
    render(<KPI label="Sharpe" />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("sets aria-live='polite' on the value when live=true", () => {
    render(<KPI label="Tick" value={125.4} valueFormat="num" live />);
    const el = screen.getByText(/125/);
    expect(el).toHaveAttribute("aria-live", "polite");
  });

  it("omits aria-live when live is not set (default)", () => {
    render(<KPI label="Tick" value={125.4} valueFormat="num" />);
    const el = screen.getByText(/125/);
    expect(el).not.toHaveAttribute("aria-live");
  });

  it("renders a skeleton instead of the value when loading", () => {
    const { container } = render(
      <KPI label="Loading" value={42} loading />,
    );
    // Skeleton renders a div with the animate-shimmer class (per design system).
    const skeleton = container.querySelector(".animate-shimmer");
    expect(skeleton).not.toBeNull();
    // The raw value text should not appear.
    expect(screen.queryByText("42")).not.toBeInTheDocument();
  });
});
