/**
 * Tests for the user-facing v2 helpers — parseDrfError + pickDefaultPortfolio.
 *
 * These are the two places where "garbage in" surfaces as a confusing UI:
 *   - parseDrfError turns DRF's varied error shapes into a stable, readable
 *     {message, fields} pair that the Options Desk Configure sheet shows.
 *   - pickDefaultPortfolio decides which portfolio a fired trade lands in
 *     when the user hasn't explicitly chosen one.
 */
import { describe, expect, it } from "vitest";

import {
  parseDrfError,
  pickDefaultPortfolio,
  type PortfolioRow,
} from "../v2";

describe("parseDrfError", () => {
  it("returns Network error on ERR_NETWORK", () => {
    expect(parseDrfError({ code: "ERR_NETWORK" })).toEqual({
      message: "Network error — backend unreachable",
      fields: {},
    });
  });

  it("returns Timeout on ECONNABORTED", () => {
    expect(parseDrfError({ code: "ECONNABORTED" })).toEqual({
      message: "Request timed out",
      fields: {},
    });
  });

  it("handles DRF detail envelope", () => {
    const out = parseDrfError({
      response: { data: { detail: "Authentication credentials were not provided." } },
    });
    expect(out.message).toContain("Authentication credentials");
    expect(out.fields).toEqual({});
  });

  it("handles plain string body", () => {
    expect(parseDrfError({ response: { data: "no route" } })).toEqual({
      message: "no route",
      fields: {},
    });
  });

  it("handles ModelSerializer field-error envelope", () => {
    const out = parseDrfError({
      response: { data: { strategy_name: ["This field is required."] } },
    });
    expect(out.fields.strategy_name).toEqual(["This field is required."]);
    expect(out.message).toContain("strategy_name");
  });

  it("handles JSONSchema validator envelope", () => {
    // This is the shape AgentRunViewSet.create returns when the strategy's
    // params schema rejects the supplied config.
    const out = parseDrfError({
      response: {
        data: {
          config: [
            { path: ["lots"], message: "1 is less than the minimum of 1" },
            { path: ["mode"], message: "BUL is not one of ['BULL_PUT', 'BEAR_CALL']" },
          ],
        },
      },
    });
    expect(out.fields["config.lots"]).toEqual(["1 is less than the minimum of 1"]);
    expect(out.fields["config.mode"][0]).toContain("BUL");
    expect(out.message).toContain("lots");
    expect(out.message).toContain("mode");
  });

  it("falls back to axios message for non-HTTP errors", () => {
    expect(parseDrfError({ message: "TypeError: undefined" })).toEqual({
      message: "TypeError: undefined",
      fields: {},
    });
  });

  it("returns a generic Unknown error when nothing matches", () => {
    expect(parseDrfError({})).toEqual({
      message: "Unknown error",
      fields: {},
    });
  });
});


describe("pickDefaultPortfolio", () => {
  const pf = (id: string, mode: "paper" | "live"): PortfolioRow => ({
    id,
    name: `pf-${id}`,
    capital: 500_000,
    used_capital: 0,
    realized_pnl: 0,
    day_pnl: 0,
    mode,
    broker_link: null,
  });

  it("returns null when there are no portfolios", () => {
    expect(pickDefaultPortfolio([], "paper")).toBeNull();
    expect(pickDefaultPortfolio(undefined, "paper")).toBeNull();
  });

  it("prefers the paper-mode portfolio when available", () => {
    const r = pickDefaultPortfolio([pf("live-1", "live"), pf("paper-1", "paper")], "paper");
    expect(r?.id).toBe("paper-1");
  });

  it("falls back to the first portfolio when preferred mode is missing", () => {
    const r = pickDefaultPortfolio([pf("live-1", "live")], "paper");
    expect(r?.id).toBe("live-1");
  });

  it("respects prefer='live' too", () => {
    const r = pickDefaultPortfolio([pf("paper-1", "paper"), pf("live-1", "live")], "live");
    expect(r?.id).toBe("live-1");
  });
});
