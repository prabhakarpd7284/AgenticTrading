/**
 * Regression tests for the axios client.
 *
 * The backend globally wraps ModelViewSet list responses in DRF's
 * CursorPagination envelope `{ next, previous, results: [...] }`.
 * Every feature page types those endpoints as bare `T[]` and feeds them
 * straight into components that call `rows.length` / `rows.map(...)`.
 * Without auto-unwrap, clicking any nav link (Dashboard, Strategies,
 * Backtester, Brokers, Agents) crashes the page.
 *
 * This test locks in the interceptor behavior so the crash doesn't
 * regress silently.  Uses axios's built-in `adapter` override — no
 * extra dev-deps required.
 */
import { describe, expect, it, beforeEach, afterEach } from "vitest";
import type { AxiosRequestConfig, AxiosResponse } from "axios";
import { api, list } from "../api";

/** Swap in a canned response for a single test. */
function stubOnce(status: number, data: unknown) {
  const original = api.defaults.adapter;
  api.defaults.adapter = (cfg: AxiosRequestConfig) =>
    Promise.resolve<AxiosResponse>({
      data,
      status,
      statusText: "OK",
      headers: {},
      // @ts-expect-error -- request config shape from axios is loose in tests
      config: cfg,
    });
  return () => { api.defaults.adapter = original; };
}

describe("api interceptor", () => {
  let restore: () => void = () => {};
  afterEach(() => restore());

  it("unwraps DRF CursorPagination envelope to a bare array", async () => {
    restore = stubOnce(200, {
      next: null,
      previous: null,
      results: [{ id: "p1" }, { id: "p2" }],
    });
    const { data } = await api.get("/portfolios/");
    expect(Array.isArray(data)).toBe(true);
    expect(data).toEqual([{ id: "p1" }, { id: "p2" }]);
  });

  it("leaves detail responses alone (no next/previous keys)", async () => {
    restore = stubOnce(200, { results: [{ event: "x" }] });
    const { data } = await api.get("/legacy/audit/");
    expect(data).toEqual({ results: [{ event: "x" }] });
  });

  it("leaves bare-object responses alone", async () => {
    restore = stubOnce(200, { capital: 500000, daily_pnl: 0 });
    const { data } = await api.get("/legacy/portfolio/");
    expect(data).toEqual({ capital: 500000, daily_pnl: 0 });
  });

  it("list() always returns an array on empty/null bodies", async () => {
    restore = stubOnce(204, null);
    const rows = await list("/foo/");
    expect(rows).toEqual([]);
  });

  it("list() handles a paginated envelope end-to-end", async () => {
    restore = stubOnce(200, {
      next: "cursor=abc",
      previous: null,
      results: [{ a: 1 }, { a: 2 }],
    });
    const rows = await list<{ a: number }>("/bar/");
    expect(rows).toEqual([{ a: 1 }, { a: 2 }]);
  });
});
