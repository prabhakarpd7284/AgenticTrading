import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/Tooltip";

/**
 * Today (UX redesign R3): Action Board cards with risk pre-check + honest
 * "unproven" track record, informative empty states, approve flow, and
 * Market Pulse collapsed by default.
 */

let responses: Record<string, unknown> = {};
const post = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => {
      const key = Object.keys(responses).find((k) => url === k);
      return key ? Promise.resolve({ data: responses[key] }) : Promise.reject(new Error(`unmocked ${url}`));
    }),
    post: (...args: unknown[]) => post(...args),
  },
}));

vi.mock("@/features/market-pulse/MarketPulsePage", () => ({ MarketPulsePage: () => <div>pulse-page</div> }));

import { TodayPage } from "../TodayPage";

const STAT = { n: 0, wins: 0, losses: 0, win_rate: null, avg_r: null, net_pnl: null, label: "n=0 · unproven", basis: "" };
const TRACK = {
  strategy: "directional", stage: "unproven — forward paper test", proven: false, cards_total: 2, first_date: "2026-10-23",
  paper: { ...STAT, open: 0, approved_cards: 0 },
  missed: { ...STAT, cards: 0, entered: 0, no_entry: 0, no_data: 0 },
  blocked: { ...STAT, cards: 0, entered: 0, no_entry: 0, no_data: 0 },
  summary: "n=0 · unproven paper trades",
};

function card(over: Record<string, unknown> = {}) {
  return {
    id: "c1", date: "2026-10-23", kind: "BUY", symbol: "INFY", strategy: "directional", strategy_version: "1.0.0",
    stage: "unproven — forward paper test", horizon: "intraday", side: "BUY", entry: 1500, entry_zone: [1500, 1500],
    stoploss: 1490, target: 1520, suggested_qty: 33, capital_required: 49500, risk_at_stake: 330, risk_multiplier: 1,
    risk_check: { approved: true, reason: "Approved", failed_criterion: "", criteria: {}, gates: {} },
    risk_passed: true, confidence: 0.7, reasons: ["Opening range break above PP", "Early move +1.35%", "Nightly scan: both allowed"],
    source_refs: { run_receipt_id: "r1" }, shadow: false, blocked_reason: "", planned_as_of: "2026-10-23T04:05:00Z",
    expires_at: "2026-10-23T09:30:00Z", status: "proposed", snoozed_until: null, decided_at: null, decision_note: "",
    order: null, outcome: null, ...over,
  };
}

const RECEIPT = {
  id: "r1", status: "success", trigger: "beat", started_at: "2026-10-23T04:05:00Z", finished_at: "2026-10-23T04:07:00Z",
  error: "", skipped: null, as_of: "2026-10-23T04:05:00Z", replay: false, candidates: 49, fresh: 49, planner_calls: 10,
  planned: 2, passed_risk: 1, cards: 2, dropped: {}, telegram_sent: true, warnings: [],
};

function today(over: Record<string, unknown> = {}) {
  return {
    date: "2026-10-23", now: "2026-10-23T05:00:00Z",
    strategy: { name: "directional", stage: "unproven — forward paper test", mode: "paper" },
    board: { built: true, is_trading_day: true, scheduled_for: "2026-10-23T04:05:00Z", schedule_label: "09:35 IST",
             expires_at: "2026-10-23T09:30:00Z", due: true, receipt: RECEIPT, planner_cap: 10 },
    counts: { total: 2, open: 1, passed_risk: 1, by_status: {} },
    can_decide: true,
    cards: [
      card(),
      card({ id: "c2", symbol: "TCS", side: "SELL", status: "rejected", risk_passed: false,
             risk_check: { approved: false, reason: "Risk:Reward ratio 1.20 below minimum 1.5",
                           failed_criterion: "risk_reward", criteria: {}, gates: {} } }),
    ],
    track_record: { directional: TRACK },
    ...over,
  };
}

const OVERVIEW = {
  total: { value: 500000 }, pnl: { day: { total: null } },
  risk: { daily_loss: { used: 0, limit: 15000, limit_pct_of_capital: 3, pct_used: 0, basis: "" } },
  components: { equity: { open_count: 1 }, options: { open_count: 0 } },
};

const STATUS = {
  as_of: "", mode: { mode: "PAPER", reason: "", env_mode: "paper" },
  data: { state: "live", reason: "", market_open: true,
          brokers: [{ broker: "zerodha", status: "expired", last_refreshed_at: null, last_error: "" }],
          candles: {} },
  jobs: { ok: 3, total: 4, warn: 0, failed: 1, missed: 0, pending: 0, untracked: 0, health: "red", label: "",
          problems: [{ key: "nightly-scan", label: "Nightly scan", state: "failed", due_at: "x", receipt_id: "r9" }] },
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider>
        <MemoryRouter><TodayPage /></MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  post.mockReset();
  responses = {
    "/actions/today/": today(),
    "/portfolio/overview/": OVERVIEW,
    "/runs/status/": STATUS,
    "/trading-rules/cooldown/": { as_of: "2026-10-23", results: [] },
  };
});

describe("TodayPage", () => {
  it("shows cards with risk pre-check, reasons and the unproven track record", async () => {
    renderPage();
    const infy = await screen.findByTestId("action-card-INFY");
    expect(within(infy).getByText("risk pre-check: would pass")).toBeInTheDocument();
    expect(within(infy).getByText("Opening range break above PP")).toBeInTheDocument();
    expect(within(infy).getByText("₹330")).toBeInTheDocument();
    const tcs = screen.getByTestId("action-card-TCS");
    expect(within(tcs).getByText(/would fail — reward:risk/)).toBeInTheDocument();
    expect(within(tcs).queryByRole("button", { name: "Approve" })).toBeNull();
    expect(screen.getByTestId("stage-badge")).toHaveTextContent("unproven — forward paper test");
    expect(screen.getByTestId("track-record")).toHaveTextContent("n=0 · unproven");
    expect(screen.getByTestId("board-status")).toHaveTextContent("49 candidates");
  });

  it("approves through a confirm step", async () => {
    post.mockResolvedValue({ data: card({ status: "executed" }) });
    renderPage();
    const infy = await screen.findByTestId("action-card-INFY");
    fireEvent.click(within(infy).getByRole("button", { name: "Approve" }));
    fireEvent.click(within(infy).getByRole("button", { name: "Confirm paper order" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/actions/c1/approve/", {}));
  });

  it("skips with a reason and snoozes", async () => {
    post.mockResolvedValue({ data: card() });
    renderPage();
    const infy = await screen.findByTestId("action-card-INFY");
    fireEvent.click(within(infy).getByRole("button", { name: "30m" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/actions/c1/snooze/", { minutes: 30 }));
    fireEvent.click(within(infy).getByRole("button", { name: "Skip" }));
    fireEvent.change(within(infy).getByLabelText("Skip reason"), { target: { value: "results week" } });
    fireEvent.click(within(infy).getByRole("button", { name: "Confirm skip" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/actions/c1/skip/", { reason: "results week" }));
  });

  it("explains an empty board that ran", async () => {
    responses["/actions/today/"] = today({ cards: [], board: { ...today().board,
      receipt: { ...RECEIPT, planned: 0, passed_risk: 0 } } });
    renderPage();
    expect(await screen.findByText(/No actions today — board built .*49 candidates, 0 planned, 0 passed risk/))
      .toBeInTheDocument();
    expect(screen.getByText("View the board receipt").closest("a")).toHaveAttribute("href", "/runs/r1");
  });

  it("says when the board has not been built yet", async () => {
    responses["/actions/today/"] = today({ cards: [], board: { ...today().board, built: false, due: false, receipt: null } });
    renderPage();
    expect(await screen.findByText("Board not built yet")).toBeInTheDocument();
    expect(screen.getByText(/runs at 09:35 IST/)).toBeInTheDocument();
  });

  it("lists what needs attention and keeps market context collapsed", async () => {
    renderPage();
    const list = await screen.findByTestId("attention-list");
    expect(within(list).getByText("Nightly scan: failed").closest("a")).toHaveAttribute("href", "/runs/r9");
    expect(within(list).getByText(/zerodha link expired — re-login needed/)).toBeInTheDocument();
    expect(screen.queryByText("pulse-page")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Market context/ }));
    expect(screen.getByText("pulse-page")).toBeInTheDocument();
    expect(screen.getByText("Open Market Pulse").closest("a")).toHaveAttribute("href", "/pulse");
    expect(await screen.findByTestId("portfolio-snapshot")).toHaveTextContent("Open positions");
  });
});
