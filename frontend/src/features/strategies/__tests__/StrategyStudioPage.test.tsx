import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/Tooltip";
import { BACKTEST_DONE, SPEC, TRADES, detail } from "./fixtures";

/** Strategy Studio page (R4): what's next, rule builder + explain, versions, Test tab report. */

let responses: Record<string, unknown> = {};
const post = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => (url in responses
      ? Promise.resolve({ data: responses[url] }) : Promise.reject(new Error(`unmocked ${url}`)))),
    post: (...a: unknown[]) => post(...a),
    patch: vi.fn(),
    delete: vi.fn(),
  },
}));

import { StrategyStudioPage } from "../StrategyStudioPage";

function renderAt(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes><Route path="/strategies/:id" element={<StrategyStudioPage />} /></Routes>
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}

const VALID = { valid: true, errors: [], warnings: [], spec: SPEC, spec_hash: "abcdef1234567890",
                explain: ["Buys when close crosses above opening-range high (15 min)."], live_gate: { passes: true, min_rr: 1.5, message: "" } };

beforeEach(() => {
  post.mockReset();
  post.mockImplementation((url: string) => {
    if (url === "/strategies/spec/validate/") return Promise.resolve({ data: VALID });
    return Promise.reject(new Error(`unmocked POST ${url}`));
  });
  responses = {
    "/strategies/instances/s1/": detail(),
    "/strategies/instances/s1/versions/": [{ id: "v1", version: 1, spec_hash: "abcdef1234567890", note: "created",
      created_at: "2026-10-21T09:00:00Z", created_by: "me@x", definition: SPEC, backtests: 1 }],
    "/strategies/backtests/?instance=s1": [{ ...BACKTEST_DONE, metrics: BACKTEST_DONE.metrics }],
    "/strategies/backtests/bt1/": BACKTEST_DONE,
    "/strategies/backtests/bt1/trades/": TRADES,
    "/strategies/universe/": { intraday: { symbols: ["RELIANCE"], indices: [], first: "2026-04-20", last: "2026-10-21", source: "" },
                               daily: { symbol_count: 2900, first: "2026-04-20", last: "2026-10-21", source: "" },
                               research_oos: { from: "2026-09-01", to: "2026-10-19" } },
  };
});

describe("StrategyStudioPage", () => {
  it("starts with the stage and what's next, and explains the rules", async () => {
    renderAt("/strategies/s1");
    expect(await screen.findByTestId("next-step")).toHaveTextContent("Next: get more trades");
    expect(screen.getByText("backtested")).toBeInTheDocument();
    expect(screen.getAllByTestId("rule-row")).toHaveLength(2);
    expect(await screen.findByTestId("explain")).toHaveTextContent("crosses above opening-range high");
    expect(screen.getByRole("button", { name: "Save as v2" })).toBeDisabled();       // nothing changed yet
  });

  it("an edit becomes a new version", async () => {
    post.mockImplementation((url: string) => {
      if (url === "/strategies/spec/validate/") return Promise.resolve({ data: VALID });
      if (url === "/strategies/instances/s1/versions/") {
        return Promise.resolve({ data: { created: true, instance: detail({ current_version: 2 }), version: {} } });
      }
      return Promise.reject(new Error(url));
    });
    renderAt("/strategies/s1");
    fireEvent.change(await screen.findByLabelText("Target value"), { target: { value: "3" } });
    const save = screen.getByRole("button", { name: "Save as v2" });
    await waitFor(() => expect(save).toBeEnabled());
    fireEvent.click(save);
    await waitFor(() => expect(post).toHaveBeenCalledWith("/strategies/instances/s1/versions/",
      expect.objectContaining({ spec: expect.objectContaining({ exits: expect.objectContaining({ target: { kind: "r", value: 3 } }) }) })));
  });

  it("Test tab shows the report with provenance, IS/OOS and trades", async () => {
    renderAt("/strategies/s1?tab=test&backtest=bt1");
    const report = await screen.findByTestId("backtest-report");
    expect(within(report).getAllByTestId("provenance-badge")[0]).toHaveTextContent("n=12 (too few)");
    expect(within(report).getByText(/Only 12 trade\(s\)/)).toBeInTheDocument();
    expect(within(report).getByRole("table", { name: "In-sample vs out-of-sample" })).toHaveTextContent("Out-of-sample");
    expect(await within(report).findByText("SL hit")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run backtest (v1)" })).toBeInTheDocument();
  });

  it("runs a backtest from the Test tab", async () => {
    post.mockImplementation((url: string) => {
      if (url === "/strategies/backtests/") return Promise.resolve({ data: { ...BACKTEST_DONE, id: "bt1" } });
      return Promise.resolve({ data: VALID });
    });
    renderAt("/strategies/s1?tab=test");
    fireEvent.click(await screen.findByRole("button", { name: "Run backtest (v1)" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/strategies/backtests/", expect.objectContaining({
      instance: "s1", version: 1, to_date: "2026-10-21", from_date: "2026-09-21", symbols: null, capital: 500000,
    })));
  });

  it("versions tab lists versions", async () => {
    renderAt("/strategies/s1?tab=versions");
    expect(await screen.findByText("first version")).toBeInTheDocument();
    expect(screen.getByText("created")).toBeInTheDocument();
  });

  it("a forward-paper-only strategy says why it can't be backtested", async () => {
    responses["/strategies/instances/s1/"] = detail({
      kind: "plugin", strategy_name: "directional", backtest: "forward_paper_only",
      backtest_reason: "Decisions come from the Claude planner.", definition: { universe: "NIFTY50" },
      schema: { type: "object", properties: { universe: { type: "string", enum: ["NIFTY50", "NIFTY100"] } } },
      next_step: { code: "forward_paper_only", title: "Forward paper test only", detail: "LLM", action: null },
    });
    renderAt("/strategies/s1?tab=test");
    expect(await screen.findByTestId("not-backtestable")).toHaveTextContent("Claude planner");
  });
});
