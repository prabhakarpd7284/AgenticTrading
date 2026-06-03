import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../Select";

/**
 * The themed Select replaced native <select> elements in the Options Desk's
 * configure modal — the native ones were invisible on dark mode (no
 * matching tokens) and their dropdown UX was OS-chrome. These tests lock
 * in the theming + interaction behaviour we depend on.
 */
function ControlledSelect({ initial = "" }: { initial?: string } = {}) {
  const [v, setV] = (require("react") as typeof import("react")).useState(initial);
  return (
    <Select value={v || "auto"} onValueChange={(x) => setV(x === "auto" ? "" : x)}>
      <SelectTrigger>
        <SelectValue placeholder="Pick one" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="auto">Auto</SelectItem>
        <SelectItem value="NIFTY">NIFTY</SelectItem>
        <SelectItem value="BANKNIFTY">BANK NIFTY</SelectItem>
      </SelectContent>
    </Select>
  );
}

describe("<Select />", () => {
  it("renders trigger with placeholder when nothing is selected", () => {
    render(<ControlledSelect />);
    const trigger = screen.getByRole("combobox");
    expect(trigger).toBeInTheDocument();
    // Trigger should use our themed surface bg + border, not the OS default.
    expect(trigger.className).toMatch(/bg-surface/);
    expect(trigger.className).toMatch(/border-border/);
  });

  it("opens the panel on click and shows items themed against our tokens", async () => {
    const user = userEvent.setup();
    render(<ControlledSelect />);
    await user.click(screen.getByRole("combobox"));
    // Radix portals the content — query by role.
    const items = await screen.findAllByRole("option");
    expect(items.map((i) => i.textContent)).toEqual([
      "Auto", "NIFTY", "BANK NIFTY",
    ]);
    // Items should have the highlighted-state class so hover/keyboard
    // navigation uses our surface-2 colour, not OS blue.
    expect(items[0].className).toMatch(/data-\[highlighted\]:bg-surface-2/);
  });

  it("selecting an item closes the panel and updates the trigger label", async () => {
    const user = userEvent.setup();
    render(<ControlledSelect />);
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: /bank nifty/i }));
    // Trigger now shows the selected label.
    expect(screen.getByRole("combobox")).toHaveTextContent(/bank nifty/i);
  });
});
