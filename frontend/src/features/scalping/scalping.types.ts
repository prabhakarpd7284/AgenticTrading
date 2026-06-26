/** Wire types for the scalp simulation WebSocket + run config. */

export interface ScalpConfig {
  underlying: string;
  strike: number;
  type: "CE" | "PE";
  expiry: string;
  date: string;
  resolution: "5S" | "10S" | "15S" | "30S" | "45S" | "1";
  bin_width: number;
  window_secs: number;
  entry_threshold: number;
  capital: number;
  risk_pct: number;
  max_pyramids: number;
  lot_size: number;
  require_bias_alignment: boolean;
  allow_reverse: boolean;
  speed: number;
  dry_run: boolean;
  mode: "sim" | "live";
  place_orders: boolean;
  pullback_min_bins: number;
  reversal_bins: number;
}

export interface SessionBin {
  low: number;
  weight: number;
  pct: number;
  ticks: number;
  visits: number;
  poc: boolean;
}

export interface SessionProfile {
  bins: SessionBin[];
  poc: number;
  total: number;
}

export type Side = "FLAT" | "LONG" | "SHORT";

export interface PositionState {
  side: Side;
  lots: number;
  avg: number;
  sl: number;
  ltp: number;
  unrealized_pts: number;
  unrealized_inr: number;
  pyramids: number;
}

export interface PressureState {
  pressure: number;
  poc: number;
  vah: number;
  val: number;
  bias: string;
  bins: { low: number; weight: number }[];
}

export interface DecisionMsg {
  action: "enter_long" | "enter_short" | "add" | "exit";
  side: Side;
  price: number;
  lots: number;
  sl?: number;
  pnl_pts?: number;
  reason: string;
  seq: number;
  ts?: number;
}

export interface Kpis {
  realized_pnl_pts: number;
  realized_pnl_inr: number;
  peak_unrealized_pts: number;
  trades: number;
  entries: number;
  peak_lots: number;
  open_side: Side;
  open_lots: number;
  won: boolean;
}

export type Status = "idle" | "creating" | "connecting" | "running" | "paused" | "done" | "error";
export type ConnState = "connecting" | "live" | "reconnecting" | "closed_auth";

export interface Annotation {
  candle_ts: string;
  note: string;
}
