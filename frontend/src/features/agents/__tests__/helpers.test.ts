import { describe, it, expect } from "vitest";

import type { AgentEvent } from "@/types";
import {
  agentForNode, inferStepKind, computeKpis, cliForStrategy,
} from "../AgentConsolePage";

/**
 * Pure-helper coverage for the Agents Console enhancements. Keeps the
 * KPI strip, StepKind icon mapping, and inline-CLI dispatch from rotting
 * silently — these power the visual scan of a running workflow.
 */

describe("agentForNode", () => {
  it("maps known nodes to their persona labels", () => {
    expect(agentForNode("planner").label).toBe("@DirectionalTrader");
    expect(agentForNode("risk").label).toBe("@RiskGuard");
    expect(agentForNode("execute").label).toBe("@Broker");
  });

  it("falls back to the raw node name for unknown nodes", () => {
    expect(agentForNode("brand_new_node").label).toBe("brand_new_node");
  });
});

describe("inferStepKind", () => {
  it("classifies the planner as an LLM step", () => {
    expect(inferStepKind("planner").kind).toBe("llm");
    expect(inferStepKind("generate_action").kind).toBe("llm");
  });

  it("classifies the risk node as a risk step", () => {
    expect(inferStepKind("risk").kind).toBe("risk");
    expect(inferStepKind("validate_action").kind).toBe("risk");
  });

  it("classifies execute as a broker step", () => {
    expect(inferStepKind("execute").kind).toBe("broker");
  });

  it("defaults unknown nodes to the deterministic-service bucket", () => {
    const k = inferStepKind("totally_new_node");
    expect(k.kind).toBe("service");
    // Unknown nodes also get a usable label (the node name itself), so the
    // a11y aria-label on the icon doesn't end up empty.
    expect(k.label).toBe("totally_new_node");
  });
});

describe("computeKpis", () => {
  const mk = (seq: number, node: string, type: AgentEvent["type"], ts?: string): AgentEvent => ({
    seq, node, type, payload: {}, ts,
  });

  it("returns zero-shaped KPIs for an empty event stream", () => {
    const k = computeKpis([]);
    expect(k).toEqual({ eventCount: 0, distinctNodes: 0, llmCalls: 0, elapsedMs: null });
  });

  it("counts events, distinct nodes, and LLM result calls (not tokens)", () => {
    const k = computeKpis([
      mk(1, "init",    "info",   "2026-05-18T09:14:00Z"),
      mk(2, "planner", "token",  "2026-05-18T09:14:01Z"),  // streamed token — NOT counted
      mk(3, "planner", "token",  "2026-05-18T09:14:01.5Z"),
      mk(4, "planner", "result", "2026-05-18T09:14:02Z"),  // counted
      mk(5, "risk",    "result", "2026-05-18T09:14:02.2Z"),
      mk(6, "execute", "result", "2026-05-18T09:14:03Z"),
    ]);
    expect(k.eventCount).toBe(6);
    expect(k.distinctNodes).toBe(4); // init, planner, risk, execute
    expect(k.llmCalls).toBe(1);       // planner result only
    expect(k.elapsedMs).toBe(3_000);  // 09:14:00 → 09:14:03
  });

  it("returns null elapsed when fewer than two events carry a timestamp", () => {
    expect(computeKpis([mk(1, "init", "info", "2026-05-18T09:14:00Z")]).elapsedMs).toBeNull();
    expect(computeKpis([mk(1, "init", "info"), mk(2, "risk", "result")]).elapsedMs).toBeNull();
  });

  it("tolerates out-of-order timestamps when computing elapsed", () => {
    // Backend ts is monotonic in practice, but the helper shouldn't crash
    // if a re-delivered event arrives late.
    const k = computeKpis([
      mk(1, "init",    "info",   "2026-05-18T09:14:05Z"),
      mk(2, "planner", "result", "2026-05-18T09:14:00Z"),
    ]);
    expect(k.elapsedMs).toBe(5_000);
  });
});

describe("cliForStrategy", () => {
  it("returns the equity CLI for directional / intraday names", () => {
    expect(cliForStrategy("directional")?.command).toBe("run_trading_agent");
    expect(cliForStrategy("Intraday Momentum")?.command).toBe("run_trading_agent");
  });

  it("returns the straddle CLI", () => {
    expect(cliForStrategy("short_straddle")?.command).toBe("manage_straddle");
  });

  it("returns the pyramid CLI with a dry-run default", () => {
    const c = cliForStrategy("pyramid_options");
    expect(c?.command).toBe("run_pyramid");
    expect(c?.args).toContain("--dry-run");
  });

  it("returns null for an unrecognised strategy (so the button hides)", () => {
    expect(cliForStrategy("unknown_strategy")).toBeNull();
    expect(cliForStrategy(undefined)).toBeNull();
  });
});
