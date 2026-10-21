import { describe, it, expect, beforeEach, vi } from "vitest";

/** R2: Monthly defaults to live data; a stale "mock" choice in storage is ignored. */
describe("monthly data source", () => {
  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });

  it("is live by default", async () => {
    const m = await import("../monthly");
    expect(m.MONTHLY_MOCK_ALLOWED).toBe(false);
    expect(m.MONTHLY_USE_MOCK).toBe(false);
    expect(m.getMonthlySource()).toBe("live");
  });

  it("ignores a mock choice stored by an older build", async () => {
    localStorage.setItem("monthly_data_source", "mock");
    const m = await import("../monthly");
    expect(m.getMonthlySource()).toBe("live");
    m.setMonthlySource("mock");
    expect(m.getMonthlySource()).toBe("live");
  });
});
