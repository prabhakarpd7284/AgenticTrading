import { describe, it, expect, beforeEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/Tooltip";

/**
 * /runs page + StatusStrip + RunButton against a mocked `api`:
 * schedule colours, receipt list, receipt detail with checks, rerun flow.
 */

type Fixture = Record<string, unknown>;
let responses: Fixture = {};
const post = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string, cfg?: { params?: Record<string, unknown> }) => {
      if (url === "/runs/receipts/" && cfg?.params?.celery_task_id) {
        return Promise.resolve({ data: { results: [{ ...RECEIPT, id: "r-new", status: "success" }] } });
      }
      const key = Object.keys(responses).find((k) => url === k || url.startsWith(k));
      return key ? Promise.resolve({ data: responses[key] }) : Promise.reject(new Error(`unmocked ${url}`));
    }),
    post: (...a: unknown[]) => post(...a),
  },
}));

import { RunsPage } from "../RunsPage";
import { StatusStrip } from "@/components/layout/StatusStrip";

const RECEIPT = {
  id: "r1", kind: "import", name: "pull_amfi_nav", scope: "system", task_name: "apps.mutual_funds.tasks.pull_amfi_nav",
  schedule_key: "", expected_at: null, trigger: "manual", status: "success",
  started_at: "2026-10-21T13:52:00Z", finished_at: "2026-10-21T13:53:02Z", duration_seconds: 61.8,
  outputs: { rows_parsed: 14397, navs_created: 6, links: [{ label: "Mutual funds", href: "/mutual-funds" }] },
  warnings: [], error: "", check_counts: { pass: 1, warn: 1, fail: 0, info: 2 },
  detail_kind: "", detail_id: "", rerunnable: true, celery_task_id: "",
};

const DETAIL = {
  ...RECEIPT,
  inputs: {}, data_provenance: { source: "AMFI NAVAll.txt" }, previous: null, detail: null,
  rerun_note: "Idempotent AMFI download (~1 min).",
  checks: [
    { name: "rows_parsed", status: "pass", message: "14397 scheme/NAV rows parsed" },
    { name: "nav_fresh", status: "warn", message: "catalog NAV date 2026-10-16 (expected 2026-10-19; 1 business day(s) behind)" },
  ],
};

const SCHEDULE = {
  date: "2026-10-21",
  summary: { ok: 1, total: 2, warn: 0, failed: 0, missed: 1, pending: 1, untracked: 1, health: "red",
             label: "1 of 2 scheduled jobs OK" },
  jobs: [],
  slots: [
    { key: "mf-nav-pull-morning", label: "AMFI NAV pull (morning)", task: "t", receipt_name: "pull_amfi_nav",
      due_at: "2026-10-21T02:00:00Z", window_end: "", schedule: "daily · 07:30 IST", rerunnable: true,
      state: "untracked", reason: "before run receipts were enabled", receipt: null },
    { key: "daily-swing-scan", label: "Swing scan (Oliver Kell)", task: "t", receipt_name: "pipeline.swing_scan",
      due_at: "2026-10-21T02:50:00Z", window_end: "", schedule: "Mon–Fri · 08:20 IST", rerunnable: true,
      state: "missed", reason: "no run recorded in its window", receipt: null },
    { key: "daily-cooldown-refresh", label: "Cooldown refresh", task: "t", receipt_name: "cooldown_refresh",
      due_at: "2026-10-21T11:15:00Z", window_end: "", schedule: "Mon–Fri · 16:45 IST", rerunnable: true,
      state: "ok", receipt: { id: "r2", status: "success", started_at: "2026-10-21T11:15:01Z", finished_at: null,
                              duration_seconds: 0.1, trigger: "beat", warn_checks: 0, error: "" } },
    { key: "nightly-scan", label: "Nightly scan", task: "t", receipt_name: "nightly_scan",
      due_at: "2026-10-21T15:30:00Z", window_end: "", schedule: "daily · 21:00 IST", rerunnable: true,
      state: "pending", reason: "not due yet", receipt: null },
  ],
};

const STATUS = {
  as_of: "2026-10-21T14:00:00Z",
  mode: { mode: "PAPER", reason: "TRADING_MODE=paper", env_mode: "paper" },
  data: { state: "stale", reason: "1m candles end 2026-10-19; expected 2026-10-21", market_open: false,
          brokers: [], candles: { "1m": { name: "candles_fresh", status: "warn", message: "" },
                                  "1d": { name: "candles_fresh", status: "warn", message: "" } } },
  jobs: { ...SCHEDULE.summary, problems: [{ key: "daily-swing-scan", label: "Swing scan", state: "missed",
                                            due_at: "", receipt_id: null }] },
};

function renderAt(path: string, ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TooltipProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/runs" element={ui} />
            <Route path="/runs/:receiptId" element={ui} />
          </Routes>
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  post.mockReset();
  responses = {
    "/runs/schedule/": SCHEDULE,
    "/runs/receipts/r1/": DETAIL,
    "/runs/receipts/r-new/": { ...DETAIL, id: "r-new" },
    "/runs/receipts/": { results: [RECEIPT], count: 1, names: ["pull_amfi_nav"] },
    "/runs/status/": STATUS,
  };
});

describe("RunsPage", () => {
  it("shows today's schedule with a state per slot", async () => {
    renderAt("/runs", <RunsPage />);
    expect(await screen.findByText("Swing scan (Oliver Kell)")).toBeInTheDocument();
    expect(within(screen.getByTestId("slot-daily-swing-scan")).getByText("missed")).toBeInTheDocument();
    expect(within(screen.getByTestId("slot-daily-cooldown-refresh")).getByText("OK")).toBeInTheDocument();
    expect(within(screen.getByTestId("slot-nightly-scan")).getByText("pending")).toBeInTheDocument();
    expect(within(screen.getByTestId("slot-mf-nav-pull-morning")).getByText("not tracked")).toBeInTheDocument();
    expect(screen.getByText(/1 of 2 scheduled jobs OK/)).toBeInTheDocument();
  });

  it("lists receipts and opens one with its checks, outputs and links", async () => {
    renderAt("/runs/r1", <RunsPage />);
    const receipt = await screen.findByTestId("run-receipt");
    expect(within(receipt).getByText("nav_fresh")).toBeInTheDocument();
    expect(within(receipt).getByText(/1 business day\(s\) behind/)).toBeInTheDocument();
    expect(within(receipt).getByText("14397")).toBeInTheDocument();
    expect(within(receipt).getByRole("link", { name: /Open Mutual funds/ })).toHaveAttribute("href", "/mutual-funds");
    expect(screen.getAllByText("pull_amfi_nav").length).toBeGreaterThan(0);
  });

  it("reruns through confirm → queued → receipt", async () => {
    post.mockResolvedValue({ data: { name: "pull_amfi_nav", dispatched: true, celery_task_id: "c-1" } });
    renderAt("/runs/r1", <RunsPage />);
    fireEvent.click(await screen.findByRole("button", { name: /Run again/ }));
    expect(screen.getByText(/Idempotent AMFI download/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));
    await waitFor(() => expect(post).toHaveBeenCalledWith("/runs/rerun/pull_amfi_nav/"));
    await waitFor(() => expect(screen.getAllByTestId("run-receipt").length).toBe(2));
  });
});

describe("StatusStrip", () => {
  it("shows mode, data status and job health", async () => {
    renderAt("/runs", <StatusStrip />);
    expect(await screen.findByText("PAPER")).toBeInTheDocument();
    expect(screen.getByText("data stale")).toBeInTheDocument();
    expect(screen.getByText("1/2 jobs OK")).toBeInTheDocument();
    expect(screen.getByText("1/2 jobs OK").closest("a")).toHaveAttribute("href", "/runs");
  });
});
