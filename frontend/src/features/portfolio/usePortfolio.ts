/**
 * Portfolio destination — data hooks (UX redesign R2, plan §5).
 *
 *   GET /api/v1/portfolio/overview/      whole-book picture (equity, options, MFs, risk, history)
 *   GET /api/v1/portfolio/attribution/   realized P&L by strategy / symbol / month (?months=)
 *
 * Honesty contract with the backend: a component with no data source comes
 * back `status: "not_connected"` with `null` numbers — never 0. `empty` means
 * connected but holding nothing.
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export type ComponentStatus = "ok" | "empty" | "not_connected";

export interface Freshness {
  as_of: string | null;
  age_seconds: number | null;
  stale: boolean;
  source: string;
  note?: string;
}

export interface EquityPositionRow {
  id: string; symbol: string; side: string; units: number; entry: number; mark: number | null;
  mark_source: string; mark_as_of: string | null; cost: number; unrealized_pnl: number | null;
  stop: number | null; risk_to_stop: number | null; strategy: string; sector: string; product: string;
}

export interface EquityComponent {
  status: ComponentStatus; label: string; open_count: number; pending_orders: number;
  cost: number | null; market_value: number | null; unrealized_pnl: number | null;
  open_risk: number | null; positions_without_stop: number; unpriced: number;
  positions: EquityPositionRow[]; freshness: Freshness;
}

export interface OptionsComponent {
  status: ComponentStatus; label: string; open_count: number; unrealized_pnl: number | null;
  long_premium_paid: number | null; short_premium_sold: number | null; net_liquidation: number | null;
  positions_without_stop: number;
  positions: { id: string; type: string; underlying: string; expiry: string; lots: number; status: string;
               open_pnl: number | null; legs: number; premium: number; strategy: string }[];
  freshness: Freshness;
}

export interface MFComponent {
  status: ComponentStatus; label: string; reason?: string; holding_count: number;
  current_value: number | null; invested: number | null; gain: number | null; gain_pct?: number | null;
  catalog_nav_date: string | null; nav_is_stale: boolean | null; stale_holdings: number;
  allocation: { category: string; value: number; pct: number }[]; freshness: Freshness;
}

export interface PnlPeriod { realized: number | null; unrealized_open: number | null; total: number | null }

export interface AllocationRow { key: string; value: number; pct: number | null }

export interface BrokerLinkRow {
  broker: string; link_status: string;
  snapshot: null | { fetched_at: string; ok: boolean; error: string; positions: number; holdings: number;
                     margin_used: number | null; margin_available: number | null; freshness: Freshness };
}

export interface PortfolioOverview {
  as_of: string;
  mode: "paper" | "live";
  portfolios: { id: string; name: string; mode: string; capital: number | null; role: "trading" | "mutual_funds" }[];
  total: {
    value: number | null; invested: number | null; cash: number | null; capital: number | null;
    trading_value: number | null; mutual_funds_value: number | null;
    components_included: string[]; components_missing: string[];
  };
  pnl: { day: PnlPeriod; week: PnlPeriod; month: PnlPeriod; since_start: PnlPeriod;
         status: "ok" | "empty"; label: string; mutual_funds_gain?: number | null };
  components: { equity: EquityComponent; options: OptionsComponent; mutual_funds: MFComponent };
  allocation: {
    by_asset_class: { key: string; label: string; value: number | null; status: string; pct: number | null }[];
    by_strategy: AllocationRow[];
    by_sector: AllocationRow[];
  };
  risk: {
    open_risk: number | null; open_risk_pct_of_capital: number | null; positions_without_stop: number;
    margin_used_book: number | null; margin_used_broker: number | null; margin_available_broker: number | null;
    daily_loss: { used: number | null; limit: number | null; limit_pct_of_capital: number;
                  pct_used: number | null; basis: string };
    note: string;
  };
  history: {
    status: "ok" | "not_connected"; label?: string; basis?: string;
    points: { date: string; equity: number; realized_cum: number; unrealized: number }[];
    snapshot_count: number; first_date?: string | null;
    max_drawdown: null | { abs: number; pct: number; peak_date: string | null; trough_date: string | null; note: string };
    freshness: Freshness;
  };
  broker_reconciliation: {
    status: "ok" | "not_connected"; label: string; links: BrokerLinkRow[]; positions: number; holdings: number;
    margin_used: number | null; margin_available: number | null; freshness: Freshness;
    book_positions?: number; discrepancy: string | null;
  };
  freshness: Record<"equity" | "options" | "mutual_funds" | "snapshots" | "broker", Freshness>;
}

export interface AttributionGroup {
  key: string; pnl: number; trades: number; wins: number; losses: number; win_rate: number | null;
}

export interface PortfolioAttribution {
  as_of: string; since: string; status: "ok" | "empty"; label: string; trade_count: number;
  total_pnl: number | null;
  by_strategy: AttributionGroup[]; by_symbol: AttributionGroup[]; by_month: AttributionGroup[];
}

export function usePortfolioOverview() {
  return useQuery<PortfolioOverview>({
    queryKey: ["portfolio-overview"],
    queryFn: async () => (await api.get<PortfolioOverview>("/portfolio/overview/")).data,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

export function usePortfolioAttribution(months = 12) {
  return useQuery<PortfolioAttribution>({
    queryKey: ["portfolio-attribution", months],
    queryFn: async () =>
      (await api.get<PortfolioAttribution>("/portfolio/attribution/", { params: { months } })).data,
    staleTime: 120_000,
  });
}
