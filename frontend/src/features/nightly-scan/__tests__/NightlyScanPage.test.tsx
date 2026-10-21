import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";

/**
 * /nightly-scan page: empty state, populated verdict table (expand a row →
 * headlines with links + levels), cooldown status — against a mocked `api`.
 */

type Fixture = Record<string, unknown>;
let responses: Fixture = {};

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => {
      const key = Object.keys(responses).find((k) => url.startsWith(k));
      return key ? Promise.resolve({ data: responses[key] }) : Promise.reject(new Error(`unmocked ${url}`));
    }),
  },
}));

import { NightlyScanPage } from "../NightlyScanPage";

const COOLDOWN = {
  as_of: "2026-10-21",
  rules: { loss_days: 3, cooldown_trading_days: 2, reduced_risk_multiplier: 0.5, costs_deducted: true },
  results: [{
    portfolio_id: "p1", portfolio: "Default", mode: "paper", as_of: "2026-10-21",
    status: "cooldown", loss_streak: 0, trigger_date: "2026-10-20", cooldown_until: "2026-10-22",
    risk_multiplier: 0, reason: "3 consecutive loss days", history: [
      { date: "2026-10-16", net_pnl: -100, kind: "loss", in_cooldown: false },
    ],
  }],
};

const row = (over: Record<string, unknown>) => ({
  symbol: "INFY", data_date: "2026-10-20",
  levels: { data_date: "2026-10-20", cpr: { pp: 1500, tc: 1502, bc: 1498, width_pct: 0.27, class: "normal" },
            floor: { r1: 1510, r2: 1520, r3: 1530, s1: 1490, s2: 1480, s3: 1470 },
            camarilla: { h3: 1512, h4: 1525, l3: 1488 } },
  cpr_width_pct: 0.27, cpr_class: "normal", two_day_relationship: "higher",
  camarilla_relationship: "overlapping_higher", bias: "bullish",
  sentiment_score: 0.62, sentiment_label: "strong_positive", sentiment_weight: 1.2,
  headline_count: 4, scored_count: 4, event_tomorrow: false, event_purpose: "",
  allow_long: true, allow_short: false, allowed: "long",
  reasons: ["strong positive sentiment (+0.62) → no shorts"],
  headlines: [{ id: 1, title: "Infosys Q2 profit beats estimates", url: "https://example.com/infy",
                source: "google_news", publisher: "Mint", kind: "news",
                published_at: "2026-10-20T10:00:00Z", score: 0.8, materiality: "results" }],
  ...over,
});

const SCAN = {
  running: false,
  thresholds: { STRONG_THRESHOLD: 0.5 },
  scan: {
    id: 1, for_date: "2026-10-21", session_date: "2026-10-20", status: "partial",
    started_at: "2026-10-20T15:30:00Z", finished_at: "2026-10-20T15:40:00Z",
    timings: { total: 600, news: 120 }, telegram_sent: true, error: "",
    sources: { google_news: { ok: true, seen: 300, new: 120, secs: 70 },
               livemint_markets: { ok: false, seen: 0, new: 0, errors: ["timeout"] } },
    summary: {
      universe: 3, pivot_source: "pivot_boss",
      market: { score: 0.12, label: "neutral", headlines: 30, scored: 30, breadth: 1 },
      headlines_total: 10, headlines_scored: 9, sentiment_available: true,
      skip: [{ symbol: "INFY", allowed: "long", reasons: [] }, { symbol: "HDFCBANK", allowed: "none", reasons: [] }],
      unscored: [], degraded: ["sources: livemint_markets"],
    },
    rows: [
      row({}),
      row({ symbol: "HDFCBANK", sentiment_label: "strong_negative", sentiment_score: -0.7, allowed: "none",
            allow_long: false, allow_short: false, event_tomorrow: true, event_purpose: "Financial Results on 2026-10-21",
            headlines: [] }),
      row({ symbol: "TCS", sentiment_label: "neutral", sentiment_score: 0.01, allowed: "both",
            allow_short: true, reasons: [], headlines: [] }),
    ],
  },
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/nightly-scan"]}>
        <NightlyScanPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("NightlyScanPage", () => {
  beforeEach(() => {
    responses = {};
  });

  it("shows the empty state when no scan has run", async () => {
    responses = {
      "/nightly-scan/latest/": { scan: null, running: false, thresholds: {} },
      "/trading-rules/cooldown/": { ...COOLDOWN, results: [] },
    };
    renderPage();
    expect(await screen.findByText("No nightly scan yet")).toBeInTheDocument();
    expect(screen.getByText(/run_nightly_scan/)).toBeInTheDocument();
    expect(await screen.findByText("No trading portfolios.")).toBeInTheDocument();
  });

  it("renders verdicts, cooldown and expands a row into headlines + levels", async () => {
    responses = { "/nightly-scan/latest/": SCAN, "/trading-rules/cooldown/": COOLDOWN };
    renderPage();

    const infy = await screen.findByTestId("row-INFY");
    expect(within(infy).getByText("Long only")).toBeInTheDocument();
    expect(within(infy).getByText("Strong +")).toBeInTheDocument();
    expect(within(infy).getByText("0.27%")).toBeInTheDocument();
    const hdfc = screen.getByTestId("row-HDFCBANK");
    expect(within(hdfc).getByText("Skip")).toBeInTheDocument();
    expect(within(hdfc).getByText("Event")).toBeInTheDocument();

    expect(screen.getByText("partial")).toBeInTheDocument();
    expect(screen.getByText("Telegram sent")).toBeInTheDocument();
    expect(await screen.findByText("Cooldown until 2026-10-22")).toBeInTheDocument();

    fireEvent.click(infy);
    const link = screen.getByRole("link", { name: /Infosys Q2 profit beats estimates/ });
    expect(link).toHaveAttribute("href", "https://example.com/infy");
    expect(screen.getByText("TC 1502.00")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Restricted only" }));
    expect(screen.queryByTestId("row-TCS")).not.toBeInTheDocument();
    expect(screen.getByTestId("row-HDFCBANK")).toBeInTheDocument();
  });
});
