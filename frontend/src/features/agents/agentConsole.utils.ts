import { Brain, Database, Radio, ShieldCheck, Sparkles, Wrench, type LucideIcon } from "lucide-react";
import type { AgentEvent } from "@/types";

/* =================================================================== */
/* Agent + StepKind classification                                      */
/* =================================================================== */

// Persona labels per workflow node — these are stable across the three
// workflows (equity / straddle / pyramid). Keep additions in lock-step
// with the strategy plugins' graph nodes.
const AGENT_BY_NODE: Record<string, { label: string; tone: "brand" | "info" | "warning" | "success" | "danger" }> = {
  fetch_data:       { label: "@DataAnalyst",      tone: "info"    },
  retrieve_context: { label: "@PortfolioTracker", tone: "brand"   },
  planner:          { label: "@DirectionalTrader",tone: "brand"   },
  generate_action:  { label: "@OptionsStrategist",tone: "brand"   },
  risk:             { label: "@RiskGuard",        tone: "warning" },
  validate_action:  { label: "@RiskGuard",        tone: "warning" },
  execute:          { label: "@Broker",           tone: "success" },
  journal:          { label: "@Journal",          tone: "info"    },
  init:             { label: "@System",           tone: "info"    },
};

export function agentForNode(node: string) {
  return AGENT_BY_NODE[node] ?? { label: node, tone: "info" as const };
}

// StepKind icons follow the redesign-v2 plan: 🧠 LLM, 🛡 risk, 📡 broker,
// ⚙ deterministic service, 📓 persistence. Operators learn the symbols
// fast and can scan a long timeline at a glance.
export type StepKind = "llm" | "risk" | "broker" | "service" | "persist" | "init";

export interface KindInfo {
  kind: StepKind;
  Icon: LucideIcon;
  colorCls: string;
  label: string;
}

const STEP_KIND_BY_NODE: Record<string, KindInfo> = {
  fetch_data:       { kind: "service", Icon: Wrench,   colorCls: "text-fg-muted", label: "Deterministic service" },
  retrieve_context: { kind: "service", Icon: Wrench,   colorCls: "text-fg-muted", label: "Deterministic service" },
  planner:          { kind: "llm",     Icon: Brain,    colorCls: "text-brand",    label: "LLM step" },
  generate_action:  { kind: "llm",     Icon: Brain,    colorCls: "text-brand",    label: "LLM step" },
  risk:             { kind: "risk",    Icon: ShieldCheck,colorCls: "text-warn",   label: "Risk engine" },
  validate_action:  { kind: "risk",    Icon: ShieldCheck,colorCls: "text-warn",   label: "Risk engine" },
  execute:          { kind: "broker",  Icon: Radio,    colorCls: "text-pnl-up",   label: "Broker call" },
  journal:          { kind: "persist", Icon: Database, colorCls: "text-fg-muted", label: "Persistence" },
  init:             { kind: "init",    Icon: Sparkles, colorCls: "text-fg-subtle",label: "Init" },
};

export function inferStepKind(node: string): KindInfo {
  return STEP_KIND_BY_NODE[node] ?? {
    kind: "service",
    Icon: Wrench,
    colorCls: "text-fg-subtle",
    label: node,
  };
}

/* =================================================================== */
/* KPI computation                                                      */
/* =================================================================== */
export interface KpiSummary {
  eventCount: number;
  distinctNodes: number;
  llmCalls: number;
  /** ms between first and last event with ts; null if <2 timestamped events. */
  elapsedMs: number | null;
}

export function computeKpis(events: AgentEvent[]): KpiSummary {
  const nodes = new Set<string>();
  let llmCalls = 0;
  let firstTs: number | null = null;
  let lastTs: number | null = null;
  let timestamped = 0;

  for (const ev of events) {
    nodes.add(ev.node);
    // Count distinct LLM-kind steps that produced a result (not every token).
    // A 50-token streaming planner shouldn't read as "50 LLM calls".
    if (inferStepKind(ev.node).kind === "llm" && ev.type === "result") {
      llmCalls += 1;
    }
    if (ev.ts) {
      const t = new Date(ev.ts).getTime();
      if (!Number.isNaN(t)) {
        timestamped += 1;
        if (firstTs == null || t < firstTs) firstTs = t;
        if (lastTs == null || t > lastTs) lastTs = t;
      }
    }
  }

  // "Elapsed" needs a span. With <2 timestamped events the KPI strip renders
  // "—" rather than misleading "0ms".
  const elapsedMs =
    timestamped >= 2 && firstTs != null && lastTs != null
      ? lastTs - firstTs
      : null;

  return {
    eventCount: events.length,
    distinctNodes: nodes.size,
    llmCalls,
    elapsedMs,
  };
}

/* =================================================================== */
/* Strategy → CLI mapping (for the inline OpButton)                     */
/* =================================================================== */
// Best-effort match against StrategyCatalog.name (lowercased). The
// strategy plugins live in backend/plugins/strategy_*; their names map
// onto the equivalent management commands. If we can't match, we hide
// the CLI button rather than show a useless one.
export function cliForStrategy(
  name: string | undefined,
): { command: string; args: string; description: string } | null {
  if (!name) return null;
  const n = name.toLowerCase();
  if (n.includes("directional") || n.includes("equity") || n.includes("intraday")) {
    return {
      command: "run_trading_agent",
      args: "--show-journal",
      description: "Inspect the equity directional journal or fire a new plan.",
    };
  }
  if (n.includes("straddle")) {
    return {
      command: "manage_straddle",
      args: "--list",
      description: "List, register, or close short-straddle positions.",
    };
  }
  if (n.includes("pyramid")) {
    return {
      command: "run_pyramid",
      args: "--strike 24200 --type CE --dry-run",
      description: "Pyramid backtest on intraday option candles (dry-run by default).",
    };
  }
  if (n.includes("screener")) {
    return {
      command: "run_screener",
      args: "",
      description: "Run the live intraday screener.",
    };
  }
  if (n.includes("swing") || n.includes("ok")) {
    return {
      command: "run_ok_scanner",
      args: "--actionable-only",
      description: "Oliver Kell daily/weekly cycle scan.",
    };
  }
  return null;
}
