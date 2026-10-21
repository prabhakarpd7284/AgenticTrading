import { describe, it, expect, vi } from "vitest";
import { render, screen, within, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

/** Screener tab: default category, ranking table, sorting, breakdown, missing-data badges. */

const CATS = [
  { category: "Equity Scheme - Flexi Cap Fund", count: 46, as_of: "2026-10-16" },
  { category: "Equity Scheme - Large Cap Fund", count: 34, as_of: "2026-10-16" },
];

function row(rank: number, code: string, name: string, score: number, cagr3: number) {
  return {
    rank, scheme_code: code, scheme_name: name, amc_name: `${name} AMC`, score, partial_score: false,
    components: { consistency: score, risk_adjusted: 60, drawdown: 40, cost: null, size_age: 80 },
    cagr_1y: 0.1, cagr_3y: cagr3, cagr_5y: 0.15, rolling_3y_median: 0.14, rolling_3y_min: 0.02,
    rolling_3y_pct_above_cat_median: 70, rolling_3y_windows: 60, volatility: 0.14, max_drawdown: -0.25,
    worst_1y: -0.2, sharpe: 0.8, sortino: 1.1, excess_cagr_3y_vs_benchmark: 0.01, benchmark_name: "NIFTY 500",
    history_years: 8, last_nav_date: "2026-10-16", quality_flags: rank === 2 ? ["data_issue_in_3y_window"] : [],
    expense_ratio_pct: null, aum_cr: null,
  };
}

const RESULT = {
  category: "Equity Scheme - Flexi Cap Fund",
  as_of: "2026-10-16",
  rows: [row(1, "122639", "Parag Parikh Flexi Cap", 81, 0.18), row(2, "118955", "HDFC Flexi Cap", 74, 0.22)],
  universe_count: 46, eligible_count: 30, excluded: { "history < 3.0y": 16 },
  weights_used: { consistency: 35.29, risk_adjusted: 29.41, drawdown: 23.53, size_age: 11.76 },
  notes: ["TER not sourced yet (AMFI's TER page has no verified feed) - component dropped, weights renormalised."],
  index_check: {
    category: "Equity Scheme - Flexi Cap Fund", basis: "tri", reference: "NIFTY 500 TRI (gross)", windows: 100,
    verdict: "no", text: "The median active fund did NOT beat the passive alternative.",
    pct_windows_median_active_beat: 31, pct_windows_recent_3y: 20, median_gap_pp: -1.1,
    funds_beating_reference_majority: "9/40", first_window_end: "2017-01-31", last_window_end: "2026-10-31",
  },
};

const get = vi.fn((url: string, _cfg?: unknown) => {
  if (url.startsWith("/mutual-funds/screener/")) {
    const withCat = url.includes("category=");
    return Promise.resolve({
      data: {
        available_categories: CATS, personal_use_note: "Personal-use screen.", metrics_as_of: "2026-10-16",
        result: withCat ? RESULT : null,
      },
    });
  }
  if (url.includes("/nav-history/")) {
    return Promise.resolve({
      data: {
        scheme_code: "118955", scheme_name: "HDFC Flexi Cap", category: "x", is_active: true, count: 2,
        first_date: "2026-10-15", last_date: "2026-10-16",
        navs: [{ date: "2026-10-15", nav: 10 }, { date: "2026-10-16", nav: 11 }],
        benchmark: { index_name: "NIFTY 500", is_tri: true, points: [{ date: "2026-10-15", value: 10 }] },
      },
    });
  }
  return Promise.reject(new Error(`unmocked ${url}`));
});

vi.mock("@/lib/api", () => ({ api: { get: (url: string, cfg?: unknown) => get(url, cfg) } }));

import { MFScreener } from "../MFScreener";

function renderIt() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MFScreener />
    </QueryClientProvider>,
  );
}

describe("MFScreener", () => {
  it("defaults to Flexi Cap and shows the ranking with missing-data badges", async () => {
    renderIt();
    expect(await screen.findByText("Parag Parikh Flexi Cap")).toBeInTheDocument();
    expect(get.mock.calls.some(([u]) => String(u).includes("category=Equity+Scheme+-+Flexi+Cap+Fund"))).toBe(true);

    const fresh = screen.getByTestId("screener-freshness");
    expect(within(fresh).getByText("Metrics as of 2026-10-16")).toBeInTheDocument();
    expect(within(fresh).getByText("TER / AUM data missing")).toBeInTheDocument();
    expect(screen.getByTestId("screener-meta")).toHaveTextContent("30 of 46 funds eligible");
    expect(screen.getByText(/TER not sourced yet/)).toBeInTheDocument();
    expect(screen.getByText("Data issue (3y)")).toBeInTheDocument();

    const ic = screen.getByTestId("index-check");
    expect(within(ic).getByText("Index fund wins")).toBeInTheDocument();
    expect(within(ic).getByText("Gross TRI fallback")).toBeInTheDocument();
  });

  it("sorts by a column and shows the score breakdown for a clicked fund", async () => {
    renderIt();
    await screen.findByText("Parag Parikh Flexi Cap");
    const table = screen.getByRole("table");
    const names = () => within(table).getAllByText(/Flexi Cap$/).map((e) => e.textContent);
    expect(names()).toEqual(["Parag Parikh Flexi Cap", "HDFC Flexi Cap"]);

    fireEvent.click(within(table).getByText("3y CAGR"));          // desc on first click
    expect(names()).toEqual(["HDFC Flexi Cap", "Parag Parikh Flexi Cap"]);

    fireEvent.click(within(table).getByText("HDFC Flexi Cap"));
    const breakdown = await screen.findByRole("list", { name: "Score breakdown" });
    expect(within(breakdown).getByText("Cost (TER)")).toBeInTheDocument();
    expect(within(breakdown).getByText("n/a")).toBeInTheDocument();
    expect(screen.getByText(/Historical ranges only/)).toBeInTheDocument();
  });
});
