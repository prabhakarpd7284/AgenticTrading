import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route, useLocation } from "react-router-dom";

import type { AgentRun } from "@/types";
import type { AuditEntry, EventDetail } from "@/lib/v2";

/**
 * Legacy audit rows in the rail used to be visually-prominent but
 * non-interactive — a dead end. Now they open a right-edge Sheet
 * (non-modal so the rail stays clickable) that surfaces the full
 * Event row (payload, severity, cross-links) and offers an
 * "Open run" jump when the event ties to a workflow_run.
 *
 * This test covers:
 *   - clicking a row with an `id` opens the panel,
 *   - the panel renders the payload + cross-links from /events/{id}/,
 *   - "Open run" navigates to /agents/<workflow_run> when set,
 *   - rows without an `id` (older bridge payloads) stay inert.
 */

// The page wires a WS connection per selected run; this test doesn't drive
// any WS messages, so we just stub out connect() to a no-op handle.
vi.mock("@/lib/ws", () => ({
  connect: (_path: string, _onMessage: (m: unknown) => void) => ({
    send: vi.fn(),
    close: vi.fn(),
  }),
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

const auditWithId: AuditEntry = {
  id: 4242,
  time: "09:14:25",
  type: "RISK_REJECT",
  symbol: "HDFCBANK",
  detail: "RiskGuard blocked HDFCBANK BUY — daily loss cap exceeded",
};
const auditNoId: AuditEntry = {
  // legacy bridge row pre-id rollout — no `id` field
  time: "09:13:11",
  type: "TRADE_PLAN",
  symbol: "ICICIBANK",
  detail: "Plan drafted for ICICIBANK",
} as AuditEntry;

const mockEvent: EventDetail = {
  id: 4242,
  ts: "2026-05-18T09:14:25Z",
  type: "risk.rejected",
  severity: "warn",
  actor_kind: "risk_engine",
  actor_user: null,
  workflow_run: "11111111-1111-4111-8111-111111111111",
  step_name: "risk_check",
  trade_id: null,
  order: null,
  signal_id: null,
  payload: { reason: "daily_loss_cap_exceeded", capital_used_pct: 92.3 },
  text: "Daily loss cap exceeded — plan blocked.",
  request_id: "req-abc-123",
};

vi.mock("@/lib/api", () => ({
  api: {
    get: vi.fn((url: string) => {
      if (url.startsWith("/agents/runs")) return Promise.resolve({ data: mockRuns });
      if (url.startsWith("/agents/catalog")) return Promise.resolve({ data: [] });
      if (url.startsWith("/portfolios")) return Promise.resolve({ data: [] });
      if (url.startsWith("/events/audit")) {
        return Promise.resolve({ data: { results: [auditWithId, auditNoId] } });
      }
      if (url.startsWith("/events/4242/")) {
        return Promise.resolve({ data: mockEvent });
      }
      return Promise.resolve({ data: [] });
    }),
    post: vi.fn(),
  },
}));

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import { AgentConsolePage } from "../AgentConsolePage";

function LocationTracker({ onLoc }: { onLoc: (path: string) => void }) {
  const loc = useLocation();
  React.useEffect(() => onLoc(loc.pathname), [loc.pathname, onLoc]);
  return null;
}
import * as React from "react";

function renderPage(onLoc: (path: string) => void) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/agents/00000000-0000-4000-a000-000000000001"]}>
        <LocationTracker onLoc={onLoc} />
        <Routes>
          <Route path="/agents/:runId" element={<AgentConsolePage />} />
          <Route path="/agents" element={<AgentConsolePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AgentConsolePage — legacy audit row click", () => {
  it("opens the right-edge detail panel when clicking a row with id, shows payload, and jumps to the run", async () => {
    let lastPath = "";
    renderPage((p) => { lastPath = p; });

    // wait for the legacy audit list to appear
    const trigger = await screen.findByRole(
      "button",
      { name: /Open detail for RISK_REJECT at 09:14:25/i },
    );

    await act(async () => { fireEvent.click(trigger); });

    // Dialog title + body
    expect(await screen.findByText(/^Event detail$/)).toBeInTheDocument();
    expect(await screen.findByText(/risk.rejected/)).toBeInTheDocument();
    // payload value rendered in the JSON pre
    expect(await screen.findByText(/daily_loss_cap_exceeded/)).toBeInTheDocument();
    // step name in the cross-link list
    expect(screen.getByText(/risk_check/)).toBeInTheDocument();

    // "Open run" present because workflow_run is set on the mock event
    const open = screen.getByRole("button", { name: /open run/i });
    await act(async () => { fireEvent.click(open); });

    // Navigation should land on /agents/<workflow_run>
    expect(lastPath).toBe(`/agents/${mockEvent.workflow_run}`);
  });

  it("does not render a click target for rows that lack an id (legacy bridge rows)", async () => {
    renderPage(() => {});
    await screen.findByRole(
      "button",
      { name: /Open detail for RISK_REJECT at 09:14:25/i },
    );
    // The TRADE_PLAN row from auditNoId has no `id` and therefore no button.
    expect(
      screen.queryByRole(
        "button",
        { name: /Open detail for TRADE_PLAN at 09:13:11/i },
      ),
    ).toBeNull();
    // ...but the detail text is still visible (we don't hide the row, just
    // disable interaction)
    expect(screen.getByText(/Plan drafted for ICICIBANK/i)).toBeInTheDocument();
  });
});
