/** Shared API fixtures for the Strategy Studio tests (shapes from apps/strategies/api/views.py). */
export const SPEC = {
  schema_version: 1,
  meta: { name: "ORB", description: "", tags: [] },
  universe: { symbols: ["RELIANCE", "INFY"] },
  timeframe: "5m",
  indicators: [],
  entry: {
    long: { all: [
      { left: { price: "close" }, op: "crosses_above", right: { ind: "orb_high", minutes: 15 } },
      { left: { price: "close" }, op: ">", right: { ind: "vwap" } },
    ] },
    short: null, time_from: "09:35", time_to: "13:00", weekdays: [0, 1, 2, 3, 4], cooldown_bars: 0,
  },
  exits: { stop: { kind: "atr", value: 1.5, atr_period: 14 }, target: { kind: "r", value: 2 }, breakeven_r: null,
           trail: { kind: "owner" }, eod_time: "15:14", max_hold_bars: null },
  sizing: { risk_pct: 0.5, max_position_pct: 10, max_positions: 3 },
  limits: { daily_loss_pct: 3, max_trades_per_day: 6, owner_cooldown: true },
  costs: { slippage_pct: 0.02, model: "auto" },
};

export const PROVENANCE = {
  data_source: "market_data.Candle 1m — Angel One SmartAPI", synthetic: false, timeframe: "5m",
  requested: { from: "2026-05-04", to: "2026-05-29" }, actual: { first: "2026-05-04", last: "2026-05-29" },
  sessions: 19, symbols_requested: 2, symbols_loaded: ["INFY", "RELIANCE"], symbols_missing: [], bars: 3024,
  gap_sessions: 0, gaps_by_symbol: {},
  split: { method: "last 30% of sessions held out", is: ["2026-05-04", "2026-05-20"], oos: ["2026-05-21", "2026-05-29"] },
  label: "IS + OOS", research_oos_overlap: false, research_note: "",
  sample: { trades: 12, oos_trades: 4, threshold: 30, too_few: true },
  costs: { model: "nse_intraday_mis", description: "NSE intraday (MIS)", as_of: "2026-10", slippage_pct_per_side: 0.02 },
  engine: "TIMELINE", definition: { kind: "spec", strategy: "spec", version: 1, hash: "abcdef1234567890" },
  signals: { detected: 30, taken_to_fill: 12, rejected: { max_positions: 3 }, dropped: {} },
  live_gate: { passes: true, min_rr: 1.5, message: "" },
  notes: ["Owner trail uses the ATR(14) of this timeframe."],
  warnings: ["Only 12 trade(s) — under 30, the metrics are anecdotes, not evidence."],
};

const M = { trades: 12, wins: 4, losses: 8, win_rate: 33.3, avg_r: -0.21, expectancy_r: -0.21, profit_factor: 0.7,
            net_pnl: -1234.5, gross_pnl: -800, costs: 434.5, max_drawdown: 2000, max_drawdown_pct: 0.4,
            too_few_trades: true, min_trades: 30 };

export const BACKTEST_DONE = {
  id: "bt1", status: "done", created_at: "2026-10-21T10:00:00Z", version: 1, from_date: "2026-05-04",
  to_date: "2026-05-29", progress: 100, progress_note: "done", error: "", instance_id: "s1", instance_name: "ORB",
  config: { symbols: ["RELIANCE", "INFY"], capital: 500000, slippage_pct: null, oos_fraction: 0.3 },
  metrics: M, split_metrics: { is: { ...M, trades: 8 }, oos: { ...M, trades: 4 }, oos_start: "2026-05-21", oos_fraction: 0.3 },
  provenance: PROVENANCE,
  equity_curve: [{ t: "2026-05-04T10:00:00+05:30", equity: 499800, drawdown: -200 },
                 { t: "2026-05-05T10:00:00+05:30", equity: 498765, drawdown: -1235 }],
  started_at: "2026-10-21T10:00:01Z", finished_at: "2026-10-21T10:00:05Z", cancel_requested: false, trade_count: 1,
};

export const TRADES = [{
  seq: 1, symbol: "RELIANCE", side: "BUY", segment: "is", entry_time: "2026-05-04T09:40:00+05:30",
  exit_time: "2026-05-04T10:10:00+05:30", entry_price: 1440, exit_price: 1432, stop: 1432.5, target: 1455, qty: 34,
  gross_pnl: -272, costs: 60, net_pnl: -332, r_multiple: -1.3, exit_reason: "SL hit", bars_held: 6, meta: {},
}];

const SUMMARY = {
  id: "bt1", status: "done", created_at: "2026-10-21T10:00:00Z", version: 1, from_date: "2026-05-04",
  to_date: "2026-05-29", progress: 100, progress_note: "done", error: "",
  metrics: M, provenance: { label: "IS + OOS", synthetic: false, timeframe: "5m", sessions: 19,
                            sample: PROVENANCE.sample, research_oos_overlap: false, data_source: "Angel" },
};

export function row(over: Record<string, unknown> = {}) {
  return {
    id: "s1", name: "ORB", description: "", kind: "spec", strategy_name: "spec", strategy_version: "1",
    template_key: "rule.orb_breakout", stage: "backtested", current_version: 1, archived: false, archived_at: null,
    created_at: "2026-10-21T09:00:00Z", updated_at: "2026-10-21T10:00:00Z", backtest: "backtest", backtest_reason: "",
    last_backtest: SUMMARY, last_result: SUMMARY,
    next_step: { code: "too_few_trades", title: "Next: get more trades",
                 detail: "The last backtest had only 12 trade(s) (< 30).", action: "backtest" },
    can_delete: true, delete_blocker: null, live_gate: { passes: true, min_rr: 1.5, message: "" }, is_enabled: false,
    ...over,
  };
}

export function detail(over: Record<string, unknown> = {}) {
  return { ...row(), definition: SPEC, version_hash: "abcdef1234567890", explain: ["Buys when close crosses above …"],
           versions: 1, ...over };
}

export const TEMPLATES = [
  { key: "rule.orb_breakout", kind: "spec", strategy_name: "spec", title: "Opening-range breakout (5m)",
    summary: "Buy a 5-minute close above the first 15 minutes' high.", backtest: "backtest", timeframe: "5m",
    spec: SPEC, schema: null, defaults: null, notes: [], broken: {}, reason: "",
    live_gate: { passes: true, min_rr: 1.5, message: "" } },
  { key: "plugin.directional", kind: "plugin", strategy_name: "directional", title: "Directional (Claude planner)",
    summary: "LLM planner.", backtest: "forward_paper_only", timeframe: null, spec: null,
    schema: { type: "object", properties: { universe: { type: "string", enum: ["NIFTY50", "NIFTY100"], default: "NIFTY50" } } },
    defaults: { universe: "NIFTY50" }, notes: [], broken: {},
    reason: "Decisions come from the Claude planner — a backtest number would be fake.",
    live_gate: { passes: null, min_rr: 1.5, message: "" } },
  { key: "plugin.intraday_screener", kind: "plugin", strategy_name: "intraday_screener", title: "Intraday screener (8 setups)",
    summary: "Replays the live screener.", backtest: "backtest", timeframe: "1m", spec: null,
    schema: { type: "object", properties: {
      max_risk_pct: { type: "number", minimum: 0.1, maximum: 5, default: 1 },
      only_strategy: { type: "string", description: "Run only this strategy" },
    } },
    defaults: { max_risk_pct: 1 }, notes: ["Exits are assumed."],
    broken: { "BB Squeeze Breakout": "needs 20 bars of 15m" }, reason: "",
    live_gate: { passes: true, min_rr: 1.5, message: "" } },
];
