import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";

import type { AgentEvent, AgentRun } from "@/types";

/**
 * Integration test: when @RiskGuard rejects a plan, the console must surface
 * a role="alert" banner with the reason. This is the single most important UX
 * affordance in AgentConsolePage — missing this means a user doesn't see why
 * their trade didn't go through.
 *
 * We mock `api` and `ws.connect` so the page mounts with:
 *   - one known run on the rail,
 *   - an injectable WS that we drive manually with risk-rejected events.
 */

let wsEmit: ((msg: unknown) => void) | null = null;
const wsClose = vi.fn();

vi.mock("@/lib/ws", () => ({
  connect: (_path: string, onMessage: (m: unknown) => void) => {
    wsEmit = onMessage;
    return { send: vi.fn(), close: wsClose };
  },
}));

const mockRuns: AgentRun[] = [
  {
    id: "00000000-0000-4000-a000-000000000001",
    status: "running",
    strategy_name: "Intraday Momentum",
    strategy_version: 3,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
  } as unknown as AgentRun,
];

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => {
      if (url.startsWith("/agents/runs")) {
        return Promise.resolve({ data: mockRuns });
      }
      if (url.startsWith("/agents/catalog")) {
        return Promise.resolve({ data: [] });
      }
      if (url.startsWith("/portfolios")) {
        return Promise.resolve({ data: [] });
      }
      return Promise.resolve({ data: [] });
    }),
    post: vi.fn(),
  },
}));

// sonner toasts call document APIs on mount; noop them
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import { AgentConsolePage } from "../AgentConsolePage";

function renderWithRouter() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter
        initialEntries={["/agents/00000000-0000-4000-a000-000000000001"]}
      >
        <Routes>
          <Route path="/agents/:runId" element={<AgentConsolePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AgentConsolePage — RiskGuard breach", () => {
  beforeEach(() => {
    wsEmit = null;
    wsClose.mockReset();
  });

  it("renders the risk-breach alert when the WS emits an approved=false result", async () => {
    renderWithRouter();

    // wait for the run to load + effect to mount the ws
    // "Intraday Momentum" appears in both the run rail and the page header,
    // so findByText would fail with "multiple elements". findAllByText waits
    // for at least one and accepts duplicates.
    await screen.findAllByText(/Intraday Momentum/i);
    expect(wsEmit).not.toBeNull();

    const rejected: AgentEvent = {
      seq: 7,
      node: "risk",
      type: "result",
      timestamp: new Date().toISOString(),
      payload: {
        approved: false,
        reason: "daily_loss_cap_exceeded",
      },
    } as unknown as AgentEvent;

    act(() => {
      wsEmit!(rejected);
    });

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/@RiskGuard blocked this plan/i);
    expect(alert).toHaveTextContent(/daily_loss_cap_exceeded/);

    // The "Execute (blocked)" fallback button must be disabled so no human
    // can override RiskGuard from this view.
    const blocked = screen.getByRole("button", { name: /execute \(blocked\)/i });
    expect(blocked).toBeDisabled();
  });

  it("does not render the alert for an approved plan", async () => {
    renderWithRouter();
    // "Intraday Momentum" appears in both the run rail and the page header,
    // so findByText would fail with "multiple elements". findAllByText waits
    // for at least one and accepts duplicates.
    await screen.findAllByText(/Intraday Momentum/i);

    const approved: AgentEvent = {
      seq: 7,
      node: "risk",
      type: "result",
      timestamp: new Date().toISOString(),
      payload: { approved: true, reason: null },
    } as unknown as AgentEvent;

    act(() => {
      wsEmit!(approved);
    });

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
