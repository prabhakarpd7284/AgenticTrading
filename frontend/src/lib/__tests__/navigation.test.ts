import { describe, it, expect } from "vitest";
import { DESTINATIONS, locate } from "../navigation";

/** R0 regroup: 6 destinations, and every pre-redesign sidebar page still reachable under one. */
const OLD_PAGES = [
  "/pulse", "/rotation", "/shortlist", "/swing-scanner", "/basket", "/nightly-scan", "/dashboard",
  "/positions", "/options", "/monthly", "/mutual-funds", "/agents", "/strategies", "/backtester",
  "/pyramid", "/scalping", "/brokers", "/broker-monitor", "/tradingview", "/ops", "/pipeline",
];

describe("navigation", () => {
  it("has exactly the six destinations in order", () => {
    expect(DESTINATIONS.map((d) => d.label)).toEqual(
      ["Today", "Portfolio", "Strategies", "Lab", "Runs", "Settings"]);
  });

  it("keeps all 21 old pages as sub-items", () => {
    const all = DESTINATIONS.flatMap((d) => d.items.map((i) => i.to));
    for (const p of OLD_PAGES) expect(all).toContain(p);
    expect(OLD_PAGES).toHaveLength(21);
  });

  it("locates pages under their destination", () => {
    expect(locate("/").destination?.key).toBe("today");
    expect(locate("/mutual-funds").destination?.key).toBe("portfolio");
    expect(locate("/agents/abc").item?.label).toBe("Agent Console");
    expect(locate("/runs/123").destination?.key).toBe("runs");
    expect(locate("/lab").destination?.key).toBe("lab");
    expect(locate("/setup/INFY").destination?.key).toBe("today");
    expect(locate("/brokers").destination?.key).toBe("settings");
  });
});
