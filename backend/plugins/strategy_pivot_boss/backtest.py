"""Small deterministic intraday backtester for the Pivot Boss setups.

Rules (all configurable):
- Entry at the OPEN of ``Signal.entry_idx`` (the bar after the trigger bar),
  adjusted by slippage against us.
- Stops/targets checked intrabar on 1-minute bars. If a bar touches both, the
  stop is assumed first (conservative). A bar opening through the stop/target
  fills at its open. Slippage is applied to every fill.
- Optional ATR trail (ratchets only) updated at completed 15-min bars.
- Flat by the close of the last bar supplied (callers pass bars ≤ 15:14 IST),
  or earlier at ``Signal.time_exit_idx``.
- Size = risk % of capital / |entry − stop|, capped at a notional % of capital.
  Never LLM-derived (CLAUDE.md invariant 3).
- NSE intraday-equity (MIS) costs per executed order.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .bars import Bar, hhmm, ist_minute


@dataclass(frozen=True)
class CostModel:
    brokerage_pct: float = 0.001  # 0.1 % of turnover …
    brokerage_cap: float = 20.0  # … capped at ₹20 per executed order
    stt_sell_pct: float = 0.00025  # 0.025 % on the sell side
    txn_pct: float = 0.0000297  # NSE transaction charge 0.00297 %
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.00003  # 0.003 % on the buy side
    gst_pct: float = 0.18  # on brokerage + transaction + SEBI

    def order_cost(self, price: float, qty: int, is_buy: bool) -> float:
        turnover = price * qty
        brokerage = min(self.brokerage_cap, self.brokerage_pct * turnover)
        txn = self.txn_pct * turnover
        sebi = self.sebi_per_crore * turnover / 1e7
        gst = self.gst_pct * (brokerage + txn + sebi)
        stt = 0.0 if is_buy else self.stt_sell_pct * turnover
        stamp = self.stamp_buy_pct * turnover if is_buy else 0.0
        return brokerage + txn + sebi + gst + stt + stamp

    def round_trip(self, entry: float, exit_: float, qty: int, side: int) -> float:
        """side +1 = long (buy then sell), −1 = short (sell then buy)."""
        return (self.order_cost(entry, qty, is_buy=side > 0)
                + self.order_cost(exit_, qty, is_buy=side < 0))


@dataclass(frozen=True)
class SizingConfig:
    capital: float = 500_000.0
    risk_pct: float = 0.5
    max_notional_pct: float = 20.0

    def qty(self, entry: float, stop: float) -> int:
        per_share = abs(entry - stop)
        if per_share <= 0 or entry <= 0:
            return 0
        q_risk = math.floor(self.capital * self.risk_pct / 100.0 / per_share)
        q_cap = math.floor(self.capital * self.max_notional_pct / 100.0 / entry)
        return max(0, min(q_risk, q_cap))


@dataclass(frozen=True)
class BacktestConfig:
    slippage_pct: float = 0.02  # per side, % of price
    costs: CostModel = field(default_factory=CostModel)
    sizing: SizingConfig = field(default_factory=SizingConfig)


@dataclass
class Signal:
    setup: str
    side: int  # +1 long, −1 short
    entry_idx: int  # index into the day's 1m bars; fill at that bar's open
    stop: float
    target: float | None = None
    trail_mult: float | None = None
    atr_at: Sequence[float | None] | None = None  # per 1m bar, set on 15m closes
    time_exit_idx: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class TradeResult:
    symbol: str
    day: str
    setup: str
    side: int
    entry_time: str
    entry_px: float
    exit_time: str
    exit_px: float
    qty: int
    stop: float
    target: float | None
    reason: str
    gross: float
    costs: float
    net: float
    risk_rs: float  # qty × |entry − initial stop|
    exit_t: int = 0

    @property
    def r_net(self) -> float:
        return self.net / self.risk_rs if self.risk_rs else 0.0


def simulate(symbol: str, day: str, bars: Sequence[Bar], sig: Signal,
             cfg: BacktestConfig) -> TradeResult | None:
    """Run one signal through the day's 1m bars. Returns None when the trade
    is not takeable (entry beyond stop/target at the open, or qty < 1)."""
    if sig.entry_idx >= len(bars) or sig.entry_idx < 0:
        return None
    side, slip = sig.side, cfg.slippage_pct / 100.0
    raw = bars[sig.entry_idx].o
    entry = raw * (1 + side * slip)
    stop, target = sig.stop, sig.target
    if side * (entry - stop) <= 0:
        return None
    if target is not None and side * (target - entry) <= 0:
        return None
    qty = cfg.sizing.qty(entry, stop)
    if qty < 1:
        return None
    init_stop = stop
    last = len(bars) - 1 if sig.time_exit_idx is None else min(sig.time_exit_idx, len(bars) - 1)
    extreme = raw
    exit_px, reason, exit_i = None, "", last
    for i in range(sig.entry_idx, last + 1):
        b = bars[i]
        first = i == sig.entry_idx
        if side > 0:
            if b.l <= stop:
                exit_px = stop if first or b.o > stop else b.o
                reason = "trail" if stop != init_stop else "stop"
            elif target is not None and b.h >= target:
                exit_px = target if first or b.o < target else b.o
                reason = "target"
        else:
            if b.h >= stop:
                exit_px = stop if first or b.o < stop else b.o
                reason = "trail" if stop != init_stop else "stop"
            elif target is not None and b.l <= target:
                exit_px = target if first or b.o > target else b.o
                reason = "target"
        if exit_px is not None:
            exit_i = i
            break
        extreme = max(extreme, b.h) if side > 0 else min(extreme, b.l)
        if sig.trail_mult and sig.atr_at is not None:
            atr = sig.atr_at[i]
            if atr:
                cand = extreme - side * sig.trail_mult * atr
                if side * (cand - stop) > 0:
                    stop = cand
    if exit_px is None:
        exit_px = bars[last].c
        reason = "time" if sig.time_exit_idx is not None and last == sig.time_exit_idx else "eod"
        exit_i = last
    exit_fill = exit_px * (1 - side * slip)
    gross = side * (exit_fill - entry) * qty
    costs = cfg.costs.round_trip(entry, exit_fill, qty, side)
    return TradeResult(
        symbol=symbol, day=day, setup=sig.setup, side=side,
        entry_time=hhmm(ist_minute(bars[sig.entry_idx].t)), entry_px=round(entry, 4),
        exit_time=hhmm(ist_minute(bars[exit_i].t)), exit_px=round(exit_fill, 4),
        qty=qty, stop=round(init_stop, 4), target=None if target is None else round(target, 4),
        reason=reason, gross=gross, costs=costs, net=gross - costs,
        risk_rs=qty * abs(entry - init_stop), exit_t=bars[exit_i].t,
    )


def summarize(trades: Sequence[TradeResult], sessions: int) -> dict:
    """Aggregate stats. Wins/losses are classified on NET P&L."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "sessions": sessions}
    wins = [t.net for t in trades if t.net > 0]
    losses = [t.net for t in trades if t.net <= 0]
    gross = sum(t.gross for t in trades)
    net = sum(t.net for t in trades)
    costs = sum(t.costs for t in trades)
    gp, gl = sum(wins), -sum(losses)
    eq = peak = mdd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_t):
        eq += t.net
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    gross_wins = sum(t.gross for t in trades if t.gross > 0)
    return {
        "trades": n,
        "sessions": sessions,
        "win_rate": len(wins) / n,
        "avg_win": gp / len(wins) if wins else 0.0,
        "avg_loss": -gl / len(losses) if losses else 0.0,
        "profit_factor": gp / gl if gl > 0 else math.inf,
        "gross": gross,
        "net": net,
        "costs": costs,
        "costs_pct_gross_wins": costs / gross_wins * 100 if gross_wins > 0 else None,
        "expectancy_rs": net / n,
        "expectancy_r": sum(t.r_net for t in trades) / n,
        "gross_expectancy_r": sum(t.gross / t.risk_rs for t in trades if t.risk_rs) / n,
        "max_drawdown": mdd,
        "trades_per_day": n / sessions if sessions else 0.0,
        "long_trades": sum(1 for t in trades if t.side > 0),
        "exit_reasons": _count(t.reason for t in trades),
    }


def _count(it) -> dict:
    d: dict = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return d
