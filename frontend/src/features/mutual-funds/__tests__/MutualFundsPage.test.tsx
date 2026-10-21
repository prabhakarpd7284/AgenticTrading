import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";

/**
 * /mutual-funds page: empty state (no CAS imported) and populated state,
 * against a mocked `api` — no backend involved.
 */

type Fixture = Record<string, unknown>;
let responses: Fixture = {};

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => {
      const key = Object.keys(responses).find((k) => url.startsWith(k));
      return key ? Promise.resolve({ data: responses[key] }) : Promise.reject(new Error(`unmocked ${url}`));
    }),
    post: vi.fn(() => Promise.resolve({ data: { count: 0, notes: [] } })),
  },
}));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import { MutualFundsPage } from "../MutualFundsPage";

const STATUS_FRESH = {
  latest_nav_date: "2026-10-16",
  max_nav_date: "2026-10-18",
  last_pull_at: "2026-10-16T18:20:00Z",
  stale_cutoff_date: "2026-10-13",
  stale_after_trading_days: 3,
  catalog_is_stale: false,
  scheme_count: 14390,
  held_scheme_count: 0,
  stale_holdings_count: 0,
  stale_holdings: [],
};

const EMPTY_SUMMARY = {
  holding_count: 0, invested: 0, current_value: 0, gain: 0, gain_pct: null,
  gain_basis_invested: 0, unvalued_count: 0, missing_cost_count: 0, allocation: [],
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/mutual-funds"]}>
        <MutualFundsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("MutualFundsPage", () => {
  beforeEach(() => {
    responses = {};
  });

  it("shows the CAS import empty state when there are no holdings", async () => {
    responses = {
      "/mutual-funds/holdings/": { results: [], summary: EMPTY_SUMMARY, include_closed: false },
      "/mutual-funds/status/": STATUS_FRESH,
      "/mutual-funds/advisor/findings/": { latest_run_at: null, count: 0, results: [] },
    };
    renderPage();

    expect(await screen.findByText("No mutual fund holdings yet")).toBeInTheDocument();
    expect(screen.getByText(/import_cas_statement/)).toHaveTextContent(
      '--tenant istpduckymomo --portfolio "Mutual Funds" --dry-run',
    );
    expect(await screen.findByText("NAV as of 2026-10-16")).toBeInTheDocument();
    expect(screen.getByText(/No advisor findings yet/)).toBeInTheDocument();
    expect(screen.queryByText("Current value")).not.toBeInTheDocument();
    // Portfolio is the default tab; the Screener tab exists but is not mounted yet.
    expect(screen.getByRole("tab", { name: "Portfolio" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Screener" })).toBeInTheDocument();
  });

  it("renders KPIs, holdings, allocation and estimate-labelled findings", async () => {
    responses = {
      "/mutual-funds/holdings/": {
        include_closed: false,
        results: [
          {
            id: 1, portfolio_id: "p", portfolio_name: "Mutual Funds", folio_number: "12345/67",
            source: "cas", scheme_code: "100001", scheme_name: "Acme Bluechip Fund - Regular Plan - Growth",
            amc_name: "Acme Mutual Fund", plan: "Regular Plan", option: "Growth Option",
            category: "Equity Scheme - Large Cap Fund", is_direct: false, units: 10,
            avg_cost_per_unit: 50, invested: 500, latest_nav: 100, nav_date: "2026-10-16",
            nav_is_stale: false, current_value: 1000, gain: 500, gain_pct: 100,
          },
        ],
        summary: {
          ...EMPTY_SUMMARY, holding_count: 1, invested: 500, current_value: 1000, gain: 500,
          gain_pct: 100, gain_basis_invested: 500,
          allocation: [{ category: "Equity Scheme - Large Cap Fund", value: 1000, pct: 100 }],
        },
      },
      "/mutual-funds/status/": { ...STATUS_FRESH, held_scheme_count: 1, stale_holdings_count: 1 },
      "/mutual-funds/advisor/findings/": {
        latest_run_at: "2026-10-17T04:00:00Z", count: 1,
        results: [{
          id: 9, ts: "2026-10-17T04:00:00Z", kind: "direct_vs_regular", severity: "warn",
          text: "Acme Bluechip is a regular plan. ESTIMATED drag ~INR 8/yr.", is_estimate: true, payload: {},
        }],
      },
    };
    renderPage();

    expect(await screen.findByText("Acme Bluechip Fund - Regular Plan - Growth")).toBeInTheDocument();
    expect(screen.getByText("Current value")).toBeInTheDocument();
    expect(screen.queryByText("No mutual fund holdings yet")).not.toBeInTheDocument();

    const alloc = screen.getByRole("list", { name: "Allocation by category" });
    expect(within(alloc).getByText("Equity Scheme - Large Cap Fund")).toBeInTheDocument();
    expect(within(alloc).getByText("100.0%")).toBeInTheDocument();

    const findings = await screen.findByRole("list", { name: "Advisor findings" });
    expect(within(findings).getByText("Direct vs regular")).toBeInTheDocument();
    expect(within(findings).getByText("Estimate")).toBeInTheDocument();
    expect(within(findings).getByText(/ESTIMATE from an assumed/)).toBeInTheDocument();

    // A stale held scheme turns the freshness badge into a warning.
    expect(await screen.findByText(/1 held scheme\(s\) stale/)).toBeInTheDocument();
  });
});
