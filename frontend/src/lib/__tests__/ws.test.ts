/**
 * Regression tests for the WebSocket helper's reconnect policy.
 *
 * The bug this locks down: Channels consumers that `close()` *before*
 * `accept()` (the ops console rejecting a non-admin) produce a failed
 * handshake, which browsers report as close code 1006 — never the 4403
 * the server passed. So the 4401/4403 short-circuit didn't fire and the
 * client reconnected forever, painting a wall of "✗ websocket error"
 * lines in the run log. Callers now get `everOpened` to tell a rejected
 * handshake from a dropped connection, and can opt out of reconnect
 * entirely for one-shot sockets whose onOpen has a side effect.
 */
import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";

import { connect } from "../ws";
import { useAuthStore } from "@/stores/auth";

/** Minimal WebSocket double — records instances, lets tests drive events. */
class FakeWS {
  static instances: FakeWS[] = [];
  static OPEN = 1;

  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;

  constructor(public url: string, public protocols?: string[]) {
    FakeWS.instances.push(this);
  }

  send(data: string) { this.sent.push(data); }
  close() { this.readyState = 3; }

  /** Simulate a completed handshake. */
  accept() { this.readyState = 1; this.onopen?.(); }
  /** Simulate any close (1006 = failed handshake / abnormal). */
  drop(code = 1006) { this.readyState = 3; this.onclose?.({ code }); }
}

describe("connect()", () => {
  const realWS = globalThis.WebSocket;

  beforeEach(() => {
    FakeWS.instances = [];
    vi.useFakeTimers();
    // @ts-expect-error -- test double
    globalThis.WebSocket = FakeWS;
    useAuthStore.setState({ accessToken: "tok" } as never);
  });

  afterEach(() => {
    vi.useRealTimers();
    globalThis.WebSocket = realWS;
  });

  it("reports everOpened=false when the handshake is rejected", () => {
    const onClose = vi.fn();
    connect("/ws/ops/", () => {}, { reconnect: false, onClose });

    FakeWS.instances[0].drop(1006);

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onClose.mock.calls[0][1]).toEqual({ everOpened: false });
  });

  it("does not reconnect when reconnect:false — the old error-spam loop", () => {
    connect("/ws/ops/", () => {}, { reconnect: false });

    FakeWS.instances[0].drop(1006);
    vi.advanceTimersByTime(120_000);

    expect(FakeWS.instances).toHaveLength(1);
  });

  it("still reconnects with backoff by default", () => {
    connect("/ws/ticks/", () => {});

    FakeWS.instances[0].accept();
    FakeWS.instances[0].drop(1006);
    vi.advanceTimersByTime(1_000);

    expect(FakeWS.instances).toHaveLength(2);
  });

  it("marks everOpened once the socket has been open", () => {
    const onClose = vi.fn();
    connect("/ws/ops/", () => {}, { reconnect: false, onClose });

    FakeWS.instances[0].accept();
    FakeWS.instances[0].drop(1006);

    expect(onClose.mock.calls[0][1]).toEqual({ everOpened: true });
  });

  it("passes the JWT as the second subprotocol", () => {
    connect("/ws/ops/", () => {}, { reconnect: false });
    expect(FakeWS.instances[0].protocols).toEqual(["jwt", "tok"]);
  });
});
