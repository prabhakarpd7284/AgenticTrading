import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SchemaForm, schemaErrors } from "../SchemaForm";
import type { JsonSchema } from "../useStrategies";

/** The JSON-Schema renderer used for built-in strategy templates. */
const SCHEMA: JsonSchema = {
  type: "object",
  required: ["strike"],
  properties: {
    underlying: { type: "string", enum: ["NIFTY", "BANKNIFTY"], default: "NIFTY" },
    strike: { type: "integer" },
    risk_pct: { type: "number", minimum: 0.1, maximum: 5, default: 1, description: "risk per trade" },
    dry_run: { type: "boolean", default: true },
    symbols: { type: "array", items: { type: "string" } },
    expiry: { type: ["string", "null"], description: "DDMMMYYYY" },
    only_strategy: { type: "string", description: "Run only this strategy" },
  },
};

describe("SchemaForm", () => {
  it("renders a control per schema field with descriptions", () => {
    render(<SchemaForm schema={SCHEMA} value={{}} onChange={() => undefined} />);
    expect(screen.getByLabelText("Underlying").tagName).toBe("SELECT");
    expect(screen.getByLabelText(/Strike/)).toHaveAttribute("type", "number");
    expect(screen.getByLabelText("Risk pct")).toHaveAttribute("max", "5");
    expect(screen.getByText("risk per trade")).toBeInTheDocument();
    expect(screen.getByLabelText("Dry run")).toHaveAttribute("type", "checkbox");
    expect(screen.getByLabelText("Symbols")).toHaveAttribute("placeholder", "comma-separated");
  });

  it("emits typed values", () => {
    const onChange = vi.fn();
    render(<SchemaForm schema={SCHEMA} value={{}} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Risk pct"), { target: { value: "0.5" } });
    expect(onChange).toHaveBeenLastCalledWith({ risk_pct: 0.5 });
    fireEvent.change(screen.getByLabelText("Underlying"), { target: { value: "BANKNIFTY" } });
    expect(onChange).toHaveBeenLastCalledWith({ underlying: "BANKNIFTY" });
    const syms = screen.getByLabelText("Symbols");
    fireEvent.change(syms, { target: { value: "infy, tcs" } });
    fireEvent.blur(syms);
    expect(onChange).toHaveBeenLastCalledWith({ symbols: ["infy", "tcs"] });
  });

  it("validates required and min/max, and shows errors", () => {
    expect(schemaErrors(SCHEMA, { risk_pct: 9 })).toEqual([
      { path: "strike", message: "required" },
      { path: "risk_pct", message: "maximum 5" },
    ]);
    render(<SchemaForm schema={SCHEMA} value={{}} onChange={() => undefined}
                       errors={[{ path: "risk_pct", message: "maximum 5" }]} />);
    expect(screen.getByRole("alert")).toHaveTextContent("maximum 5");
  });

  it("flags broken sub-strategies on the field that selects them", () => {
    render(<SchemaForm schema={SCHEMA} value={{}} onChange={() => undefined}
                       broken={{ "EMA Crossover Trend": "needs 35 bars of 15m" }} />);
    expect(screen.getByText(/is broken — needs 35 bars of 15m/)).toBeInTheDocument();
  });
});
