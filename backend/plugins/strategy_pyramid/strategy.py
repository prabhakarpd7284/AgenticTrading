"""
Greedy Aggressive Pyramiding Strategy
======================================

Captures big momentum moves by pyramiding into winners.

Core idea:
  - Enter on momentum alignment (5 EMA + BB + RSI with EMA/WMA)
  - Pyramid aggressively: risk 80% of unrealized profit on each add
  - Trail by higher lows and candle lows — let winners run
  - Worst case on any stop: keep 20% of peak unrealized profit

Indicators (all candle + math, no magic):
  - 5 EMA             — fast trend filter
  - Bollinger Bands   — volatility context + mean
  - RSI(14)           — momentum oscillator
  - RSI EMA(3)        — short-term RSI momentum
  - RSI WMA(21)       — RSI trend filter

Trailing:
  - Higher-low detection from recent candles
  - SL = max(last higher low, floor of last 3 candle lows)
  - SL only ratchets up, never down
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from trading.utils.indicators import _ema, _wma, _rsi_series, bollinger_bands


# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────

@dataclass
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    @staticmethod
    def from_raw(raw: list) -> "Candle":
        """From Angel One candle: [ts, o, h, l, c, v]."""
        return Candle(
            timestamp=str(raw[0]),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=int(raw[5]) if len(raw) > 5 else 0,
        )


@dataclass
class PyramidEntry:
    """One entry (initial or add-on) in the pyramid."""
    bar_index: int
    timestamp: str
    price: float
    lots: int
    sl_at_entry: float
    reason: str


@dataclass
class PyramidResult:
    """Complete result of a pyramiding simulation."""
    symbol: str
    entries: List[PyramidEntry] = field(default_factory=list)
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    total_lots: int = 0
    total_cost: float = 0.0
    realized_pnl: float = 0.0
    peak_unrealized: float = 0.0
    peak_lots: int = 0
    lot_size: int = 25
    log: List[str] = field(default_factory=list)

    @property
    def avg_entry(self) -> float:
        return self.total_cost / self.total_lots if self.total_lots > 0 else 0

    @property
    def pnl_per_lot(self) -> float:
        return self.exit_price - self.avg_entry if self.total_lots > 0 else 0

    @property
    def total_pnl_points(self) -> float:
        """P&L in points (per lot)."""
        if not self.entries:
            return 0
        return sum(
            (self.exit_price - e.price) * e.lots for e in self.entries
        )

    @property
    def total_pnl_rupees(self) -> float:
        return self.total_pnl_points * self.lot_size


# ──────────────────────────────────────────────
# Indicator computation (vectorized over candle list)
# ──────────────────────────────────────────────

def compute_indicators(candles: List[Candle], idx: int) -> Optional[dict]:
    """Compute all indicators up to candle[idx]. Returns None if insufficient data."""
    if idx < 21:
        return None

    closes = [c.close for c in candles[:idx + 1]]

    # 5 EMA
    ema5_series = _ema(closes, 5)
    ema5 = ema5_series[-1] if ema5_series else None

    # Bollinger Bands
    bb = bollinger_bands(closes, period=20, num_std=2.0)

    # RSI series + overlays
    rsi_vals = _rsi_series(closes, 14)
    # Filter out None values for EMA/WMA computation
    rsi_clean = [v for v in rsi_vals if v is not None]

    rsi_current = rsi_clean[-1] if rsi_clean else 50.0

    # RSI EMA(3) — short-term RSI momentum
    rsi_ema3 = None
    if len(rsi_clean) >= 3:
        rsi_ema_series = _ema(rsi_clean, 3)
        rsi_ema3 = rsi_ema_series[-1] if rsi_ema_series else None

    # RSI WMA(21) — RSI trend filter
    rsi_wma21 = None
    if len(rsi_clean) >= 21:
        rsi_wma_series = _wma(rsi_clean, 21)
        rsi_wma21 = rsi_wma_series[-1] if rsi_wma_series else None

    return {
        "ema5": round(ema5, 2) if ema5 else None,
        "bb_upper": bb["upper"],
        "bb_middle": bb["middle"],
        "bb_lower": bb["lower"],
        "bb_bandwidth": bb["bandwidth"],
        "bb_squeeze": bb["squeeze"],
        "rsi": round(rsi_current, 2),
        "rsi_ema3": round(rsi_ema3, 2) if rsi_ema3 else None,
        "rsi_wma21": round(rsi_wma21, 2) if rsi_wma21 else None,
    }


# ──────────────────────────────────────────────
# Higher-low detection
# ──────────────────────────────────────────────

def find_higher_low(candles: List[Candle], idx: int, lookback: int = 5) -> Optional[float]:
    """
    Detect the most recent higher low in the last `lookback` candles.

    A higher low forms when:
      candle[i].low > candle[i-1].low AND candle[i-1].low < candle[i-2].low
    (i.e., candle[i-1] was a pivot low, and the next low is higher)

    Returns the pivot low price, or None.
    """
    start = max(2, idx - lookback)
    best_hl = None
    for i in range(start, idx + 1):
        if (candles[i - 1].low < candles[i - 2].low
                and candles[i - 1].low < candles[i].low):
            # candle[i-1] is a pivot low, and current candle confirms it
            pivot = candles[i - 1].low
            if best_hl is None or pivot > best_hl:
                best_hl = pivot
    return best_hl


def floor_of_recent_lows(candles: List[Candle], idx: int, n: int = 3) -> float:
    """Lowest low of last n COMPLETED candles (excludes current bar)."""
    end = idx  # exclusive — don't include current bar's low
    start = max(0, end - n)
    if start >= end:
        return candles[max(0, idx - 1)].low
    return min(c.low for c in candles[start:end])


# ──────────────────────────────────────────────
# Pyramiding engine
# ──────────────────────────────────────────────

@dataclass
class PyramidConfig:
    """All strategy parameters."""
    lot_size: int = 65               # NIFTY lot size (Jan 2026+)
    initial_capital: float = 100_000  # Capital for sizing
    initial_risk_pct: float = 2.0    # % of capital risked on first entry
    profit_risk_pct: float = 0.80    # Risk 80% of unrealized on pyramids
    max_pyramids: int = 5            # Max add-ons (total entries = max_pyramids + 1)
    min_profit_to_pyramid: float = 5.0  # Min unrealized pts/lot before first pyramid
    pyramid_cooldown: int = 5        # Min bars between pyramid adds
    trail_lookback: int = 3          # Completed candles for trailing SL floor
    hl_lookback: int = 7             # Candles for higher-low detection
    eod_hour: int = 15               # EOD exit hour
    eod_minute: int = 20             # EOD exit minute
    initial_sl_candles: int = 5      # Look back N candles for initial SL (swing low)
    trail_activation_r: float = 1.0  # Start trailing after +1R profit
    max_risk_pct_of_price: float = 0.50  # Max SL distance as % of entry (50% default, options can be wide)


def run_pyramid(
    candles: List[Candle],
    symbol: str = "NIFTY_CE",
    config: PyramidConfig = None,
) -> PyramidResult:
    """
    Run the pyramiding strategy on a list of candles.

    Returns PyramidResult with full trade log, entries, exit, and P&L.
    """
    config = config or PyramidConfig()
    result = PyramidResult(symbol=symbol, lot_size=config.lot_size)
    log = result.log

    # State
    in_position = False
    entries: List[PyramidEntry] = []
    total_lots = 0
    total_cost = 0.0
    trail_sl = 0.0
    pyramid_count = 0
    last_pyramid_bar = -999

    for idx in range(len(candles)):
        c = candles[idx]

        # ── Compute indicators ──
        ind = compute_indicators(candles, idx)
        if ind is None:
            continue

        ema5 = ind["ema5"]
        bb_mid = ind["bb_middle"]
        bb_upper = ind["bb_upper"]
        rsi_val = ind["rsi"]
        rsi_ema3 = ind["rsi_ema3"]
        rsi_wma21 = ind["rsi_wma21"]

        if ema5 is None or rsi_ema3 is None:
            continue

        # ── EOD check ──
        if in_position and _is_eod(c.timestamp, config.eod_hour, config.eod_minute):
            avg = total_cost / total_lots
            log.append(
                f"[{c.timestamp}] EOD EXIT @ {c.close:.2f} | "
                f"Lots: {total_lots} | Avg: {avg:.2f} | "
                f"P&L: {(c.close - avg) * total_lots:.2f} pts"
            )
            result.entries = entries
            result.total_lots = total_lots
            result.total_cost = total_cost
            result.exit_price = c.close
            result.exit_time = c.timestamp
            result.exit_reason = "EOD"
            return result

        if in_position:
            avg = total_cost / total_lots
            unrealized = (c.close - avg) * total_lots
            result.peak_unrealized = max(result.peak_unrealized, unrealized)
            result.peak_lots = max(result.peak_lots, total_lots)

            # ── Check trail SL hit ──
            if c.low <= trail_sl:
                log.append(
                    f"[{c.timestamp}] TRAIL SL HIT @ {trail_sl:.2f} | "
                    f"Low: {c.low:.2f} | Lots: {total_lots} | Avg: {avg:.2f} | "
                    f"P&L: {(trail_sl - avg) * total_lots:.2f} pts"
                )
                result.entries = entries
                result.total_lots = total_lots
                result.total_cost = total_cost
                result.exit_price = trail_sl
                result.exit_time = c.timestamp
                result.exit_reason = "Trail SL"
                return result

            # ── Update trailing SL (only after position is +1R) ──
            old_sl = trail_sl
            entry_price_0 = entries[0].price
            initial_risk = entries[0].price - entries[0].sl_at_entry
            current_r = (c.close - entry_price_0) / initial_risk if initial_risk > 0 else 0

            if current_r >= config.trail_activation_r:
                # Method 1: Floor of last N completed candle lows (ratchet up only)
                floor_sl = floor_of_recent_lows(candles, idx, config.trail_lookback)
                trail_sl = max(trail_sl, floor_sl)

                # Method 2: Higher-low detection — strongest signal
                hl = find_higher_low(candles, idx, config.hl_lookback)
                if hl is not None:
                    trail_sl = max(trail_sl, hl)

            if trail_sl > old_sl:
                log.append(
                    f"[{c.timestamp}]   SL raised: {old_sl:.2f} → {trail_sl:.2f} | "
                    f"Close: {c.close:.2f} | R: {current_r:.1f} | EMA5: {ema5:.2f}"
                )

            # ── Pyramid check ──
            # Pyramid conditions are simpler than entry: the higher-low
            # structure IS the momentum confirmation. We only need:
            #   1. Price above EMA5 (still trending)
            #   2. Higher low above avg entry (structure intact)
            #   3. Enough unrealized profit to risk
            #   4. Cooldown between adds
            if (pyramid_count < config.max_pyramids
                    and unrealized > config.min_profit_to_pyramid * total_lots
                    and c.close > ema5
                    and rsi_val > 45  # Not in bearish territory
                    and idx - last_pyramid_bar >= config.pyramid_cooldown):

                # Find the best SL level for the add-on:
                # 1st choice: a confirmed higher-low pivot
                # 2nd choice: floor of recent candle lows (ratcheting support)
                hl_for_sl = find_higher_low(candles, idx, config.hl_lookback)
                floor_sl = floor_of_recent_lows(candles, idx, config.trail_lookback)
                candidate_sl = max(
                    hl_for_sl or 0,
                    floor_sl,
                )
                if candidate_sl > avg:
                    new_sl = candidate_sl
                    # Compute max lots we can add while keeping 20% of gains
                    locked_pnl = (new_sl - avg) * total_lots
                    max_risk = config.profit_risk_pct * unrealized
                    risk_budget = locked_pnl - (1 - config.profit_risk_pct) * unrealized
                    risk_per_lot = c.close - new_sl

                    if risk_per_lot > 0 and risk_budget > 0:
                        add_lots = int(risk_budget / risk_per_lot)
                        add_lots = max(1, min(add_lots, total_lots * 2))  # Cap at 2x current

                        # Update position
                        total_lots += add_lots
                        total_cost += c.close * add_lots
                        trail_sl = max(trail_sl, new_sl)
                        pyramid_count += 1
                        last_pyramid_bar = idx

                        entry = PyramidEntry(
                            bar_index=idx,
                            timestamp=c.timestamp,
                            price=c.close,
                            lots=add_lots,
                            sl_at_entry=trail_sl,
                            reason=f"Pyramid #{pyramid_count}",
                        )
                        entries.append(entry)

                        new_avg = total_cost / total_lots
                        remaining_unrealized = (c.close - new_avg) * total_lots
                        pnl_if_stopped = (trail_sl - new_avg) * total_lots

                        log.append(
                            f"[{c.timestamp}] PYRAMID #{pyramid_count} @ {c.close:.2f} | "
                            f"+{add_lots} lots → {total_lots} total | "
                            f"Avg: {new_avg:.2f} | SL: {trail_sl:.2f} | "
                            f"Unrealized: {remaining_unrealized:.2f} pts | "
                            f"If stopped: {pnl_if_stopped:.2f} pts | "
                            f"RSI: {rsi_val:.1f} > EMA3({rsi_ema3:.1f})"
                        )

        else:
            # ── Entry check ──
            entry_signal = (
                c.close > ema5               # Price above fast EMA
                and c.close > bb_mid         # Above BB middle
                and rsi_val > rsi_ema3       # RSI momentum up
                and rsi_val > 50             # RSI in bullish zone
                and (rsi_wma21 is None or rsi_val > rsi_wma21)  # RSI above trend
            )

            if entry_signal:
                # Initial SL: lowest low of last N candles (swing low) or BB lower
                swing_low = min(
                    candles[j].low
                    for j in range(max(0, idx - config.initial_sl_candles), idx + 1)
                )
                # Use the lower of swing low and BB lower — give max room
                initial_sl = min(swing_low, ind["bb_lower"])

                # Don't enter if SL is too close (< 1 pt) or too far
                risk_per_lot = c.close - initial_sl
                if risk_per_lot < 1.0 or risk_per_lot > c.close * config.max_risk_pct_of_price:
                    continue

                # Size: risk initial_risk_pct of capital
                max_risk = config.initial_capital * config.initial_risk_pct / 100
                init_lots = max(1, int(max_risk / (risk_per_lot * config.lot_size)))

                in_position = True
                total_lots = init_lots
                total_cost = c.close * init_lots
                trail_sl = initial_sl
                pyramid_count = 0
                last_pyramid_bar = idx

                entry = PyramidEntry(
                    bar_index=idx,
                    timestamp=c.timestamp,
                    price=c.close,
                    lots=init_lots,
                    sl_at_entry=initial_sl,
                    reason="Initial entry",
                )
                entries.append(entry)

                log.append(
                    f"[{c.timestamp}] ENTRY @ {c.close:.2f} | "
                    f"{init_lots} lots | SL: {initial_sl:.2f} (risk {risk_per_lot:.2f}/lot) | "
                    f"EMA5: {ema5:.2f} | BB: {ind['bb_lower']:.2f}-{bb_mid:.2f}-{bb_upper:.2f} | "
                    f"RSI: {rsi_val:.1f} > EMA3({rsi_ema3:.1f})"
                    + (f" > WMA21({rsi_wma21:.1f})" if rsi_wma21 else "")
                )

    # If still in position at end of data
    if in_position:
        last = candles[-1]
        avg = total_cost / total_lots
        log.append(
            f"[{last.timestamp}] END OF DATA — still holding | "
            f"Lots: {total_lots} | Avg: {avg:.2f} | "
            f"Last: {last.close:.2f} | P&L: {(last.close - avg) * total_lots:.2f} pts"
        )
        result.entries = entries
        result.total_lots = total_lots
        result.total_cost = total_cost
        result.exit_price = last.close
        result.exit_time = last.timestamp
        result.exit_reason = "End of data"
    else:
        log.append("No entry signal triggered.")

    return result


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _is_eod(timestamp: str, hour: int, minute: int) -> bool:
    """Check if timestamp is at or past EOD cutoff."""
    try:
        ts = timestamp.replace("+05:30", "").replace("T", " ")
        parts = ts.split(" ")
        if len(parts) >= 2:
            time_parts = parts[1].split(":")
            h, m = int(time_parts[0]), int(time_parts[1])
            return h > hour or (h == hour and m >= minute)
    except (ValueError, IndexError):
        pass
    return False


def run_pyramid_with_chart_data(
    candles: List[Candle],
    symbol: str = "NIFTY_CE",
    config: PyramidConfig = None,
) -> dict:
    """
    Run the pyramid strategy and return chart-ready data for the React UI.

    Returns dict with:
      - kpis: summary numbers
      - candles: [{t, o, h, l, c, v, ema5, bb_upper, bb_mid, bb_lower, rsi, rsi_ema3, rsi_wma21}]
      - trail_sl: [{t, sl}] — SL level at each bar while in position
      - position: [{t, lots, avg_entry, unrealized}] — position state at each bar
      - entries: [{t, price, lots, reason, sl, cumulative_lots}]
      - exit: {t, price, reason, lots}
      - log: [str]
    """
    config = config or PyramidConfig()
    result = run_pyramid(candles, symbol, config)

    # Build per-candle indicator series
    candle_data = []
    for idx, c in enumerate(candles):
        ind = compute_indicators(candles, idx)
        row = {
            "t": c.timestamp,
            "o": round(c.open, 2),
            "h": round(c.high, 2),
            "l": round(c.low, 2),
            "c": round(c.close, 2),
            "v": c.volume,
        }
        if ind:
            row.update({
                "ema5": ind["ema5"],
                "bb_upper": ind["bb_upper"],
                "bb_mid": ind["bb_middle"],
                "bb_lower": ind["bb_lower"],
                "rsi": ind["rsi"],
                "rsi_ema3": ind["rsi_ema3"],
                "rsi_wma21": ind["rsi_wma21"],
            })
        candle_data.append(row)

    # Replay to capture trail SL and position state per bar
    trail_sl_series = []
    position_series = []
    _replay_for_chart(candles, result, config, trail_sl_series, position_series)

    # Build entries for UI
    entry_data = []
    cum_lots = 0
    for e in result.entries:
        cum_lots += e.lots
        entry_data.append({
            "t": e.timestamp,
            "price": round(e.price, 2),
            "lots": e.lots,
            "reason": e.reason,
            "sl": round(e.sl_at_entry, 2),
            "cumulative_lots": cum_lots,
        })

    # KPIs
    avg_entry = result.avg_entry
    # Capital deployed = premium paid for all lots (avg_entry × total_lots × lot_size)
    capital_deployed = avg_entry * result.total_lots * result.lot_size if result.total_lots > 0 else 0
    # Initial risk = risk on first entry (entry_price - SL) × lots × lot_size
    initial_risk = 0.0
    if result.entries:
        e0 = result.entries[0]
        initial_risk = (e0.price - e0.sl_at_entry) * e0.lots * result.lot_size
    # ROI = P&L / capital deployed
    roi_pct = (result.total_pnl_rupees / capital_deployed * 100) if capital_deployed > 0 else 0

    kpis = {
        "total_pnl_pts": round(result.total_pnl_points, 2),
        "total_pnl_inr": round(result.total_pnl_rupees, 0),
        "peak_unrealized_pts": round(result.peak_unrealized, 2),
        "peak_unrealized_inr": round(result.peak_unrealized * result.lot_size, 0),
        "total_lots": result.total_lots,
        "peak_lots": result.peak_lots,
        "pyramid_count": len(result.entries) - 1 if result.entries else 0,
        "avg_entry": round(avg_entry, 2),
        "exit_price": round(result.exit_price, 2),
        "exit_reason": result.exit_reason,
        "lot_size": result.lot_size,
        "won": result.total_pnl_points > 0,
        "capital_deployed": round(capital_deployed, 0),
        "initial_risk_inr": round(initial_risk, 0),
        "roi_pct": round(roi_pct, 2),
    }

    exit_data = None
    if result.exit_price > 0:
        exit_data = {
            "t": result.exit_time,
            "price": round(result.exit_price, 2),
            "reason": result.exit_reason,
            "lots": result.total_lots,
        }

    return {
        "symbol": symbol,
        "kpis": kpis,
        "candles": candle_data,
        "trail_sl": trail_sl_series,
        "position": position_series,
        "entries": entry_data,
        "exit": exit_data,
        "log": result.log,
    }


def _replay_for_chart(
    candles: List[Candle],
    result: PyramidResult,
    config: PyramidConfig,
    trail_sl_out: list,
    position_out: list,
):
    """Replay the trade to capture trail SL and position state per bar."""
    if not result.entries:
        return

    entry_map = {e.timestamp: e for e in result.entries}
    in_pos = False
    total_lots = 0
    total_cost = 0.0
    trail_sl = 0.0
    entry_price_0 = result.entries[0].price
    initial_risk = result.entries[0].price - result.entries[0].sl_at_entry

    for idx, c in enumerate(candles):
        # Check if this bar has an entry
        if c.timestamp in entry_map:
            e = entry_map[c.timestamp]
            if not in_pos:
                in_pos = True
                trail_sl = e.sl_at_entry
            total_lots += e.lots
            total_cost += e.price * e.lots
            trail_sl = max(trail_sl, e.sl_at_entry)

        if not in_pos:
            continue

        avg = total_cost / total_lots if total_lots > 0 else 0
        unrealized = (c.close - avg) * total_lots

        # Update trail SL (simplified replay — just ratchet up)
        current_r = (c.close - entry_price_0) / initial_risk if initial_risk > 0 else 0
        if current_r >= config.trail_activation_r:
            floor_sl = floor_of_recent_lows(candles, idx, config.trail_lookback)
            trail_sl = max(trail_sl, floor_sl)
            hl = find_higher_low(candles, idx, config.hl_lookback)
            if hl is not None:
                trail_sl = max(trail_sl, hl)

        trail_sl_out.append({"t": c.timestamp, "sl": round(trail_sl, 2)})
        position_out.append({
            "t": c.timestamp,
            "lots": total_lots,
            "avg_entry": round(avg, 2),
            "unrealized": round(unrealized, 2),
        })

        # Stop replaying after exit
        if result.exit_time and c.timestamp >= result.exit_time:
            break


def format_result(result: PyramidResult) -> str:
    """Pretty-print the pyramid result for CLI output."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  PYRAMID STRATEGY RESULT — {result.symbol}")
    lines.append(f"{'=' * 70}")

    if not result.entries:
        lines.append("  No trades taken.")
        lines.append(f"{'=' * 70}")
        return "\n".join(lines)

    # Entry summary
    lines.append(f"  Entries: {len(result.entries)}")
    for i, e in enumerate(result.entries):
        lines.append(
            f"    [{i+1}] {e.reason:>15} @ {e.price:>8.2f} | "
            f"{e.lots:>3} lots | SL: {e.sl_at_entry:.2f} | {e.timestamp}"
        )

    lines.append(f"{'─' * 70}")

    # Position summary
    avg = result.avg_entry
    lines.append(f"  Total lots:   {result.total_lots}")
    lines.append(f"  Avg entry:    {avg:.2f}")
    lines.append(f"  Peak lots:    {result.peak_lots}")
    lines.append(f"  Exit:         {result.exit_price:.2f} ({result.exit_reason})")
    lines.append(f"  Exit time:    {result.exit_time}")

    lines.append(f"{'─' * 70}")

    # P&L
    pnl_pts = result.total_pnl_points
    pnl_rs = result.total_pnl_rupees
    peak_rs = result.peak_unrealized * result.lot_size

    icon = "+" if pnl_pts >= 0 else ""
    lines.append(f"  P&L (points): {icon}{pnl_pts:.2f}")
    lines.append(f"  P&L (rupees): {icon}{pnl_rs:,.0f}")
    lines.append(f"  Peak unreal:  +{result.peak_unrealized:.2f} pts ({peak_rs:+,.0f} INR)")

    lines.append(f"{'=' * 70}")

    # Trade log
    lines.append(f"\n  TRADE LOG ({len(result.log)} events):")
    lines.append(f"{'─' * 70}")
    for entry in result.log:
        lines.append(f"  {entry}")

    lines.append(f"{'=' * 70}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Sample data generator
# ──────────────────────────────────────────────

def _generate_pyramid_sample() -> List[Candle]:
    """Synthetic 5m candles for backtest dry-runs.

    Six-phase trajectory designed to exercise initial entry, multiple
    pyramid adds, and an eventual trail-SL or EOD exit. Deterministic
    (seeded) so the UI always renders the same chart in dry-run mode.
    """
    import random
    from datetime import datetime

    candles: list[Candle] = []
    price = 180.0
    base_time = datetime(2026, 5, 5, 9, 15)
    random.seed(77)
    phases = {
        (0, 15): (0.1, 0.8),    # quiet open
        (15, 25): (0.8, 1.0),   # initial trend up
        (25, 45): (1.2, 0.7),   # extension
        (45, 55): (0.6, 0.5),   # consolidation
        (55, 65): (1.5, 0.9),   # second leg
        (65, 75): (-0.3, 1.2),  # late fade
    }
    for i in range(75):
        total_min = 15 + i * 5
        ts = base_time.replace(hour=9 + total_min // 60, minute=total_min % 60)
        if ts.hour >= 15 and ts.minute > 30:
            break
        drift, vol = 0.1, 0.8
        for (s, e), (d, v) in phases.items():
            if s <= i < e:
                drift, vol = d, v
                break
        open_p = price
        close_p = open_p + drift + random.gauss(0, vol)
        high_p = max(open_p, close_p) + abs(random.gauss(0, vol * 0.6))
        low_p = min(open_p, close_p) - abs(random.gauss(0, vol * 0.5))
        candles.append(Candle(
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            open=round(open_p, 2), high=round(high_p, 2),
            low=round(low_p, 2), close=round(close_p, 2),
            volume=random.randint(8000, 60000),
        ))
        price = close_p
    return candles
