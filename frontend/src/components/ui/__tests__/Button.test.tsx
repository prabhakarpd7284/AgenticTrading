import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "../Button";

/**
 * Button is the most used component in the app — every page touches it.
 * The tests lean into the invariants that, if they slipped, would silently
 * degrade UX or accessibility for every downstream feature.
 */
describe("<Button />", () => {
  it("renders with the default primary variant", () => {
    render(<Button>Save</Button>);
    const btn = screen.getByRole("button", { name: /save/i });
    expect(btn).toBeInTheDocument();
    expect(btn.className).toMatch(/bg-accent/);
  });

  it("applies the destructive variant classes", () => {
    render(<Button variant="destructive">Close position</Button>);
    const btn = screen.getByRole("button", { name: /close position/i });
    expect(btn.className).toMatch(/bg-danger/);
  });

  it("sets aria-busy and disables the button while loading", () => {
    render(<Button loading>Place order</Button>);
    const btn = screen.getByRole("button", { name: /place order/i });
    expect(btn).toHaveAttribute("aria-busy", "true");
    expect(btn).toBeDisabled();
  });

  it("shows the spinner and hides the trailing icon while loading", () => {
    render(
      <Button loading trailing={<span data-testid="trailing" />}>
        Submit
      </Button>,
    );
    // trailing must not render while loading
    expect(screen.queryByTestId("trailing")).not.toBeInTheDocument();
    // spinner (Loader2) has the animate-spin class
    const svg = document.querySelector("svg.animate-spin");
    expect(svg).not.toBeNull();
  });

  it("does not fire onClick when disabled", async () => {
    const onClick = vi.fn();
    render(
      <Button disabled onClick={onClick}>
        Go
      </Button>,
    );
    await userEvent.click(screen.getByRole("button", { name: /go/i }));
    expect(onClick).not.toHaveBeenCalled();
  });

  it("renders as a child element when asChild is set", () => {
    render(
      <Button asChild>
        <a href="/dashboard">Go to desk</a>
      </Button>,
    );
    const link = screen.getByRole("link", { name: /go to desk/i });
    expect(link).toHaveAttribute("href", "/dashboard");
    // It should still carry button-variant classes via Slot merge.
    expect(link.className).toMatch(/bg-accent/);
  });

  it("merges custom className with variant classes", () => {
    render(
      <Button className="w-full">
        Full width
      </Button>,
    );
    const btn = screen.getByRole("button", { name: /full width/i });
    expect(btn.className).toMatch(/w-full/);
    expect(btn.className).toMatch(/bg-accent/);
  });
});
