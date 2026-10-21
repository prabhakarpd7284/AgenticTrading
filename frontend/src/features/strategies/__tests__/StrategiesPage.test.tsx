import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/Tooltip";
import { TEMPLATES, row } from "./fixtures";

/** Strategies list (R4): stage, last result with provenance, what's next, templates, delete guard. */

let responses: Record<string, unknown> = {};
const post = vi.fn();
const del = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => (url in responses
      ? Promise.resolve({ data: responses[url] }) : Promise.reject(new Error(`unmocked ${url}`)))),
    post: (...a: unknown[]) => post(...a),
    delete: (...a: unknown[]) => del(...a),
    patch: vi.fn(),
  },
}));

import { StrategiesPage } from "../StrategiesPage";

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider><MemoryRouter><StrategiesPage /></MemoryRouter></TooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  post.mockReset();
  del.mockReset();
  responses = {
    "/strategies/instances/": [
      row(),
      row({ id: "s2", name: "Claude directional", kind: "plugin", strategy_name: "directional",
            template_key: "plugin.directional", stage: "draft", backtest: "forward_paper_only",
            backtest_reason: "LLM", last_backtest: null, last_result: null,
            next_step: { code: "forward_paper_only", title: "Forward paper test only", detail: "LLM decisions", action: null },
            can_delete: false, delete_blocker: "it is in the paper stage — archive it instead" }),
    ],
    "/strategies/templates/": TEMPLATES,
    "/rag/knowledge/": { count: 0, results: [] },
  };
});

describe("StrategiesPage", () => {
  it("lists strategies with stage, provenance and the next step", async () => {
    renderPage();
    expect(await screen.findByText("ORB")).toBeInTheDocument();
    expect(screen.getByText("backtested")).toBeInTheDocument();
    expect(screen.getByTestId("provenance-badge")).toHaveTextContent("real · 5m · 19 sess · n=12 (too few)");
    expect(screen.getByText("Next: get more trades")).toBeInTheDocument();
    const llm = screen.getByText("Claude directional").closest("tr")!;
    expect(within(llm).getByText("Forward paper test only", { selector: "span" })).toBeInTheDocument();
    expect(within(llm).queryByRole("button", { name: "Backtest" })).toBeNull();
  });

  it("new strategy: templates are labelled honestly and plugins get a real param form", async () => {
    post.mockResolvedValue({ data: { id: "new1" } });
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: "New strategy" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Decisions come from the Claude planner — a backtest number would be fake.")).toBeInTheDocument();
    expect(within(dialog).getByText(/Broken: BB Squeeze Breakout never fire/)).toBeInTheDocument();
    fireEvent.click(within(dialog).getByText("Intraday screener (8 setups)"));
    const risk = await within(dialog).findByLabelText("Max risk pct");
    fireEvent.change(risk, { target: { value: "9" } });
    expect(within(dialog).getByRole("alert")).toHaveTextContent("maximum 5");
    expect(within(dialog).getByRole("button", { name: "Create" })).toBeDisabled();
    fireEvent.change(risk, { target: { value: "0.5" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/strategies/instances/", {
      name: "Intraday screener (8 setups)", template_key: "plugin.intraday_screener", params: { max_risk_pct: 0.5 },
    }));
  });

  it("delete is blocked for a strategy with paper history and offers archive instead", async () => {
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: "Delete Claude directional" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/Can't delete: it is in the paper stage/)).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Delete" })).toBeNull();
    post.mockResolvedValue({ data: row({ id: "s2", archived: true }) });
    fireEvent.click(within(dialog).getByRole("button", { name: "Archive instead" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/strategies/instances/s2/archive/", {}));
    expect(del).not.toHaveBeenCalled();
  });
});
