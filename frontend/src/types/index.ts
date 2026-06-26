export interface Portfolio {
  id: string;
  name: string;
  capital: string;
  used_capital: string;
  day_pnl: string;
  realized_pnl: string;
  mode: "paper" | "live";
}

export interface Position {
  id: string;
  symbol: string;
  side: "BUY" | "SELL";
  qty: number;
  avg_price: string;
  last_ltp: string | null;
  unrealized_pnl: string;
  status: "open" | "closed";
}

export type AgentRunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface AgentRunSummaryKpis {
  realized_pnl_inr?: number;
  trades?: number;
  won?: boolean;
  total_pnl_inr?: number;
  roi_pct?: number;
  peak_lots?: number;
}

/** Light row returned by the paginated list endpoint (no config/result blobs). */
export interface AgentRunSummary {
  id: string;
  strategy_name: string;
  strategy_version: string;
  status: AgentRunStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  summary?: AgentRunSummaryKpis | null;
}

/** Full record returned by the detail (retrieve) endpoint. */
export interface AgentRun extends AgentRunSummary {
  config: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string;
}

export interface StrategySchema {
  name: string;
  version: string;
  asset_class: "equity" | "options" | "futures";
  params: Record<string, unknown>;
  required_retrievers: string[];
}

export interface AgentEvent {
  seq: number;
  node: string;
  type: "token" | "state" | "result" | "error" | "info";
  payload: Record<string, unknown>;
  /** Server-side ISO timestamp stamped at emit time. Optional — older
   *  publishers (pre-May-2026) and unit-test fixtures may omit it. */
  ts?: string;
}

export interface JournalEntry {
  id: string;
  kind: string;
  title: string;
  body: string;
  created_at: string;
  tags: string[];
  meta: Record<string, unknown>;
}
