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

export interface AgentRun {
  id: string;
  strategy_name: string;
  strategy_version: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  config: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  // Only populated on the detail endpoint (/agents/runs/{id}/) — null on lists.
  steps?: Array<{
    seq: number;
    node: string;
    event_type: "token" | "state" | "result" | "error" | "info";
    payload: Record<string, unknown>;
    created_at: string;
  }> | null;
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
