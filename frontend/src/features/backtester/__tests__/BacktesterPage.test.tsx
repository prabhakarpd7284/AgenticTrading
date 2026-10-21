import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/Tooltip";
import { BACKTEST_DONE, TRADES, detail, row } from "@/features/strategies/__tests__/fixtures";

/** Backtester (R4): any user strategy can be backtested; the three legacy modes stay reachable. */

let responses: Record<string, unknown> = {};
const post = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => (url in responses
      ? Promise.resolve({ data: responses[url] }) : Promise.reject(new Error(`unmocked ${url}`)))),
    post: (...a: unknown[]) => post(...a),
  },
}));

import { BacktesterPage } from "../BacktesterPage";

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider><MemoryRouter><BacktesterPage /></MemoryRouter></TooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  post.mockReset();
  responses = {
    "/strategies/instances/": [row(), row({ id: "s2", name: "LLM", backtest: "forward_paper_only" })],
    "/strategies/instances/s1/": detail(),
    "/strategies/universe/": { intraday: { symbols: [], indices: [], first: "2026-04-20", last: "2026-10-21", source: "" },
                               daily: { symbol_count: 0, first: null, last: "2026-10-21", source: "" },
                               research_oos: { from: "2026-09-01", to: "2026-10-19" } },
    "/strategies/backtests/bt1/": BACKTEST_DONE,
    "/strategies/backtests/bt1/trades/": TRADES,
  };
});

describe("BacktesterPage", () => {
  it("keeps the legacy modes and defaults to Your strategies", async () => {
    renderPage();
    for (const t of ["Your strategies", "Daily Swing", "Intraday Multi-TF", "Morning Basket"]) {
      expect(screen.getByRole("tab", { name: t })).toBeInTheDocument();
    }
    const select = await screen.findByLabelText("Strategy");
    expect(await screen.findByRole("option", { name: /LLM — Forward paper test only/ })).toBeDisabled();
    expect(select).toBeInTheDocument();
  });

  it("backtests a user strategy and shows the report", async () => {
    post.mockResolvedValue({ data: { ...BACKTEST_DONE } });
    renderPage();
    const select = await screen.findByLabelText("Strategy");
    await screen.findByRole("option", { name: "ORB (v1)" });
    fireEvent.change(select, { target: { value: "s1" } });
    fireEvent.click(await screen.findByRole("button", { name: "Run backtest (v1)" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/strategies/backtests/", expect.objectContaining({ instance: "s1" })));
    expect(await screen.findByTestId("backtest-report")).toBeInTheDocument();
  });
});
