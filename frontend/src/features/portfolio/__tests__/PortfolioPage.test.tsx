import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/Tooltip";

/**
 * /portfolio (UX redesign R2): Overview renders "not connected" instead of 0,
 * tabs are URL-addressable and embed the existing pages, Attribution has an
 * honest empty state.
 */

let responses: Record<string, unknown> = {};

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => {
      const key = Object.keys(responses).find((k) => url === k);
      return key ? Promise.resolve({ data: responses[key] }) : Promise.reject(new Error(`unmocked ${url}`));
    }),
  },
}));

// The embedded pages have their own tests; stub them here.
vi.mock("@/features/positions/PositionsPage", () => ({ PositionsPage: () => <div>positions-page</div> }));
vi.mock("@/features/options/OptionsDashboardPage", () => ({ default: () => <div>options-page</div> }));
vi.mock("@/features/mutual-funds/MutualFundsPage", () => ({ MutualFundsPage: () => <div>mf-page</div> }));
vi.mock("@/features/monthly/MonthlyPage", () => ({ MonthlyPage: () => <div>monthly-page</div> }));

import { PortfolioPage } from "../PortfolioPage";

const FRESH = { as_of: "2026-10-21T17:58:54Z", age_seconds: 49, stale: false, source: "PortfolioSnapshot" };
const NONE = { as_of: null, age_seconds: null, stale: false, source: "no open positions" };

const OVERVIEW = {
  as_of: "2026-10-21T18:00:00Z",
  mode: "paper",
  portfolios: [
    { id: "p1", name: "Default", mode: "paper", capital: 500000, role: "trading" },
    { id: "p2", name: "Mutual Funds", mode: "paper", capital: 0, role: "mutual_funds" },
  ],
  total: { value: 500000, invested: 0, cash: 500000, capital: 500000, trading_value: 500000,
           mutual_funds_value: null, components_included: ["equity", "options"], components_missing: ["mutual_funds"] },
  pnl: {
    day: { realized: null, unrealized_open: null, total: null },
    week: { realized: null, unrealized_open: null, total: null },
    month: { realized: null, unrealized_open: null, total: null },
    since_start: { realized: null, unrealized_open: null, total: null },
    status: "empty", label: "no trades yet — P&L starts with the first fill",
  },
  components: {
    equity: { status: "empty", label: "no open equity positions", open_count: 0, pending_orders: 0, cost: null,
              market_value: null, unrealized_pnl: null, open_risk: null, positions_without_stop: 0, unpriced: 0,
              positions: [], freshness: NONE },
    options: { status: "empty", label: "no open option positions", open_count: 0, unrealized_pnl: null,
               long_premium_paid: null, short_premium_sold: null, net_liquidation: null,
               positions_without_stop: 0, positions: [], freshness: NONE },
    mutual_funds: { status: "not_connected", label: "no CAS imported yet",
                    reason: "No mutual-fund holdings on file — import a CAS statement to connect them.",
                    holding_count: 0, current_value: null, invested: null, gain: null,
                    catalog_nav_date: "2026-10-16", nav_is_stale: null, stale_holdings: 0, allocation: [],
                    freshness: { ...FRESH, stale: true, source: "AMFI NAV catalog" } },
  },
  allocation: {
    by_asset_class: [
      { key: "equity", label: "Equity", value: null, status: "empty", pct: null },
      { key: "options", label: "Options", value: null, status: "empty", pct: null },
      { key: "mutual_funds", label: "Mutual funds", value: null, status: "not_connected", pct: null },
      { key: "cash", label: "Cash", value: 500000, status: "ok", pct: 100 },
    ],
    by_strategy: [], by_sector: [],
  },
  risk: { open_risk: null, open_risk_pct_of_capital: null, positions_without_stop: 0, margin_used_book: null,
          margin_used_broker: 0, margin_available_broker: 0,
          daily_loss: { used: null, limit: 15000, limit_pct_of_capital: 3, pct_used: null,
                        basis: "today's realized + open mark-to-market" },
          note: "nothing open — no risk on the book" },
  history: { status: "ok", basis: "daily close per portfolio", snapshot_count: 2000, first_date: "2026-10-17",
             points: [{ date: "2026-10-21", equity: 500000, realized_cum: 0, unrealized: 0 }],
             max_drawdown: { abs: 0, pct: 0, peak_date: null, trough_date: null, note: "on daily closes" },
             freshness: FRESH },
  broker_reconciliation: {
    status: "ok", label: "angel_one active", positions: 0, holdings: 0, margin_used: 0, margin_available: 0,
    freshness: FRESH, book_positions: 0, discrepancy: null,
    links: [
      { broker: "angel_one", link_status: "active",
        snapshot: { fetched_at: "2026-10-21T17:59:25Z", ok: true, error: "", positions: 0, holdings: 0,
                    margin_used: 0, margin_available: 0, freshness: FRESH } },
      { broker: "zerodha", link_status: "expired", snapshot: null },
    ],
  },
  freshness: {},
};

const ATTRIBUTION_EMPTY = {
  as_of: "2026-10-21T18:00:00Z", since: "2025-11-01", status: "empty",
  label: "no closed trades in this window yet", trade_count: 0, total_pnl: null,
  by_strategy: [], by_symbol: [], by_month: [],
};

function Where() {
  return <div data-testid="where">{useLocation().pathname}</div>;
}

function renderAt(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/portfolio" element={<><PortfolioPage /><Where /></>} />
            <Route path="/portfolio/:tab" element={<><PortfolioPage /><Where /></>} />
          </Routes>
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  responses = {
    "/portfolio/overview/": OVERVIEW,
    "/portfolio/attribution/": ATTRIBUTION_EMPTY,
  };
});

describe("PortfolioPage", () => {
  it("shows totals, and missing components as not connected (never 0)", async () => {
    renderAt("/portfolio");
    const headline = await screen.findByRole("region", { name: "Headline numbers" });
    expect(within(headline).getByText("Total value")).toBeInTheDocument();
    expect(within(headline).getAllByText("no trades yet").length).toBe(3);
    expect(screen.getByRole("note")).toHaveTextContent(/exclude mutual funds \(not connected\)/);
    const mf = screen.getByTestId("component-mutual-funds");
    expect(within(mf).getByText("not connected")).toBeInTheDocument();
    expect(within(mf).queryByText("₹0")).not.toBeInTheDocument();
    expect(within(screen.getByTestId("component-equity")).getByText("nothing held")).toBeInTheDocument();
    expect(screen.getByText("Held at broker (reconciliation)")).toBeInTheDocument();
    expect(screen.getByText("expired")).toBeInTheDocument();
  });

  it("switches tabs through the URL and embeds the existing pages", async () => {
    renderAt("/portfolio/positions");
    expect(await screen.findByText("positions-page")).toBeInTheDocument();
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Mutual Funds" }));
    expect(await screen.findByText("mf-page")).toBeInTheDocument();
    expect(screen.getByTestId("where")).toHaveTextContent("/portfolio/mutual-funds");
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Overview" }));
    expect(screen.getByTestId("where")).toHaveTextContent(/^\/portfolio$/);
  });

  it("attribution shows an honest empty state", async () => {
    renderAt("/portfolio/attribution");
    expect(await screen.findByText("No closed trades yet")).toBeInTheDocument();
    expect(screen.getByText(/Nothing closed since 2025-11-01/)).toBeInTheDocument();
  });

  it("attribution groups render when there are trades", async () => {
    responses["/portfolio/attribution/"] = {
      ...ATTRIBUTION_EMPTY, status: "ok", label: "", trade_count: 2, total_pnl: 250,
      by_strategy: [{ key: "intraday_screener", pnl: 250, trades: 2, wins: 1, losses: 1, win_rate: 50 }],
      by_symbol: [{ key: "TCS", pnl: 250, trades: 2, wins: 1, losses: 1, win_rate: 50 }],
      by_month: [{ key: "2026-10", pnl: 250, trades: 2, wins: 1, losses: 1, win_rate: 50 }],
    };
    renderAt("/portfolio/attribution");
    expect(await screen.findByText("intraday_screener")).toBeInTheDocument();
    expect(screen.getByText("TCS")).toBeInTheDocument();
    expect(screen.getByText("2026-10")).toBeInTheDocument();
  });
});
