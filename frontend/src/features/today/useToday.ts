/**
 * Today destination — Action Board data hooks (UX redesign R3, plan §3–§4).
 *
 *   GET  /api/v1/actions/today/?date=        board status + cards + strategy track record
 *   POST /api/v1/actions/<id>/approve|skip|snooze/
 *   GET  /api/v1/trading-rules/cooldown/     losing-streak cooldown (Needs attention)
 *
 * Every idea is from the directional strategy, PAPER only, and labelled
 * "unproven — forward paper test" by the backend; the UI never upgrades it.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export type ActionStatus =
  | "proposed" | "approved" | "skipped" | "snoozed" | "executed" | "expired" | "rejected";

export interface RiskCheck {
  approved: boolean;
  reason: string;
  failed_criterion: string;
  criteria: Record<string, Record<string, unknown>>;
  gates: Record<string, unknown>;
  at_approve?: { approved: boolean; reason: string; failed_criterion: string } | null;
}

export interface ActionOutcome {
  status: "no_data" | "no_entry" | "stop" | "target" | "eod";
  entry_hit: boolean | null;
  entry_time?: string; exit_time?: string; exit_price?: number; r?: number | null; pnl?: number;
  note?: string;
}

export interface TodayActionCard {
  id: string;
  date: string;
  kind: string;
  symbol: string;
  strategy: string;
  strategy_version: string;
  stage: string;
  horizon: string;
  side: "BUY" | "SELL";
  entry: number;
  entry_zone: [number, number];
  stoploss: number;
  target: number;
  suggested_qty: number;
  capital_required: number;
  risk_at_stake: number;
  risk_multiplier: number;
  risk_check: RiskCheck;
  risk_passed: boolean;
  confidence: number | null;
  reasons: string[];
  source_refs: Record<string, unknown> & { run_receipt_id?: string | null };
  shadow: boolean;
  blocked_reason: string;
  planned_as_of: string;
  expires_at: string;
  status: ActionStatus;
  snoozed_until: string | null;
  decided_at: string | null;
  decision_note: string;
  order: { id: string; status: string; qty: number; price: number | null; error: string } | null;
  outcome: ActionOutcome | null;
}

export interface StatBlock {
  n: number; wins: number; losses: number; win_rate: number | null; avg_r: number | null;
  net_pnl: number | null; label: string; basis: string;
}

export interface TrackRecord {
  strategy: string;
  stage: string;
  proven: boolean;
  cards_total: number;
  first_date: string | null;
  paper: StatBlock & { open: number; approved_cards: number };
  missed: StatBlock & { cards: number; entered: number; no_entry: number; no_data: number };
  blocked: StatBlock & { cards: number; entered: number; no_entry: number; no_data: number };
  summary: string;
}

export interface BoardReceipt {
  id: string;
  status: "queued" | "running" | "success" | "partial" | "failed" | "not_run" | "skipped";
  trigger: string;
  started_at: string;
  finished_at: string | null;
  error: string;
  skipped: string | null;
  as_of: string | null;
  replay: boolean | null;
  candidates: number | null;
  fresh: number | null;
  planner_calls: number | null;
  planned: number | null;
  passed_risk: number | null;
  cards: number | null;
  dropped: Record<string, number>;
  telegram_sent: boolean | null;
  warnings: string[];
}

export interface TodayResponse {
  date: string;
  now: string;
  strategy: { name: string; stage: string; mode: string };
  board: {
    built: boolean;
    is_trading_day: boolean;
    scheduled_for: string;
    schedule_label: string;
    expires_at: string;
    due: boolean;
    receipt: BoardReceipt | null;
    planner_cap: number;
  };
  counts: { total: number; open: number; passed_risk: number; by_status: Record<ActionStatus, number> };
  can_decide: boolean;
  cards: TodayActionCard[];
  track_record: Record<string, TrackRecord>;
}

export interface CooldownRow {
  portfolio_id: string; portfolio: string; mode: string; status: "normal" | "cooldown" | "reduced";
  loss_streak: number; cooldown_until: string | null; risk_multiplier: number; reason: string;
}

export function useTodayActions(date?: string) {
  return useQuery<TodayResponse>({
    queryKey: ["actions", "today", date ?? "today"],
    queryFn: async () =>
      (await api.get<TodayResponse>("/actions/today/", { params: date ? { date } : {} })).data,
    staleTime: 15_000,
    refetchInterval: 60_000,
  });
}

export function useCooldown() {
  return useQuery<{ as_of: string; results: CooldownRow[] }>({
    queryKey: ["trading-rules", "cooldown"],
    queryFn: async () => (await api.get("/trading-rules/cooldown/")).data,
    staleTime: 120_000,
  });
}

export type Decision =
  | { id: string; decision: "approve"; note?: string }
  | { id: string; decision: "skip"; reason?: string }
  | { id: string; decision: "snooze"; minutes: number };

export function useDecideAction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, decision, ...body }: Decision) =>
      (await api.post<TodayActionCard>(`/actions/${id}/${decision}/`, body)).data,
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["actions"] });
      qc.invalidateQueries({ queryKey: ["portfolio-overview"] });
    },
  });
}

/** Human label for a RiskEngine / order-gate criterion key. */
export const CRITERION_LABEL: Record<string, string> = {
  regime: "market regime",
  fields: "order fields",
  sl_direction: "stop direction",
  target_direction: "target direction",
  capital: "capital",
  risk_per_trade: "risk per trade",
  daily_loss: "daily loss limit",
  position_size: "position size",
  risk_reward: "reward:risk",
  confidence: "confidence",
  open_positions: "max open positions",
  kill_switch: "kill switch",
  ai_paused: "AI paused",
  cooldown: "losing-streak cooldown",
  sentiment: "sentiment filter",
  notional: "notional backstop",
  size: "size",
};
