"""StrategySpec → strategy_backtest engine pieces.

* ``PrecomputedDetector`` implements the engine's ``EntryDetector`` protocol
  from signals computed once per symbol (spec rules, screener replay, OK
  cycles all end up here), so every Studio backtest runs through the same
  ``BacktestEngine`` TIMELINE loop.
* ``spec_signals`` turns a compiled spec's masks into ``EntrySignal`` s with
  relative levels (stop/target distances re-anchored at the next-bar fill).
* ``RatchetStop`` is the owner trail (TRADING_RULES §2): at +``be_r``·R the
  stop moves to entry, then to best − ``atr_mult``·ATR; it only ratchets and
  only uses bars BEFORE the current one (no intrabar optimism).
* ``spec_exit_checks`` / ``spec_engine_config`` map the spec's exits, sizing
  and limits 1:1 onto ``ExitCheck`` s and ``EngineConfig``.
"""
from __future__ import annotations

from plugins.strategy_backtest.engine import EngineConfig
from plugins.strategy_backtest.exits import MaxHoldExit, StoplossExit, TargetExit, TrailingSLExit
from plugins.strategy_backtest.types import EntrySignal, PnLMode, TimeframeMode, TradeSide

from apps.strategies.services import costs as cost_models
from apps.strategies.services.rules import CompiledSpec, SeriesFrame, indicator_series
from apps.strategies.services.spec import TIMEFRAMES, _hhmm


class PrecomputedDetector:
    """``detect(symbol, bars, i, ctx)`` → the signal decided on bar i's close."""

    def __init__(self):
        self.signals: dict[str, dict[int, EntrySignal]] = {}

    def set(self, symbol: str, by_index: dict[int, EntrySignal]) -> None:
        self.signals[symbol] = by_index

    def detect(self, symbol, bars, bar_index, context):
        sig = self.signals.get(symbol, {}).get(bar_index)
        return [sig] if sig is not None else []


class RatchetStop(TrailingSLExit):
    """Breakeven at ``be_r``·R then ``best − atr_mult × ATR`` (ATR = the
    signal-time ATR carried on the trade). Subclasses TrailingSLExit so the
    ExitManager runs it before the stop check."""

    def __init__(self, be_r: float = 1.0, atr_mult: float | None = 1.5):
        self.be_r = be_r
        self.atr_mult = atr_mult

    def check(self, trade, bar):
        long = trade.side == TradeSide.BUY
        best = trade.metadata.get("_best", trade.entry_price)
        risk = trade.metadata.get("initial_risk") or trade.risk_points
        if risk and risk > 0:
            favour = (best - trade.entry_price) if long else (trade.entry_price - best)
            if favour >= self.be_r * risk:
                cand = trade.entry_price
                trade.breakeven_done = True
                if self.atr_mult and trade.atr and trade.atr > 0:
                    trail = best - self.atr_mult * trade.atr if long else best + self.atr_mult * trade.atr
                    if (long and trail > cand) or (not long and trail < cand):
                        cand = trail
                        trade.trail_active = True
                trade.stoploss = max(trade.stoploss, cand) if long else min(trade.stoploss, cand)
        trade.metadata["_best"] = max(best, bar.high) if long else min(best, bar.low)
        return None


def spec_signals(compiled: CompiledSpec, frame: SeriesFrame) -> dict[int, EntrySignal]:
    spec = compiled.spec
    stop = spec["exits"]["stop"]
    tgt = spec["exits"]["target"]
    atr_stop = indicator_series(frame, {"kind": "atr", "period": int(stop.get("atr_period", 14))})
    atr14 = indicator_series(frame, {"kind": "atr", "period": 14})
    long_m, short_m = compiled.entry_masks(frame)
    out: dict[int, EntrySignal] = {}
    for i in range(len(frame)):
        for side, mask in ((TradeSide.BUY, long_m), (TradeSide.SHORT, short_m)):
            if not mask[i]:
                continue
            px = frame.close[i]
            if stop["kind"] == "atr":
                if atr_stop[i] is None:
                    continue
                dist = float(stop["value"]) * atr_stop[i]
            else:
                dist = px * float(stop["value"]) / 100
            if dist <= 0:
                continue
            if tgt["kind"] == "r":
                tdist = float(tgt["value"]) * dist
            elif tgt["kind"] == "pct":
                tdist = px * float(tgt["value"]) / 100
            else:
                tdist = None
            sgn = 1 if side == TradeSide.BUY else -1
            out[i] = EntrySignal(
                symbol=frame.symbol, timestamp=frame.timestamps[i], side=side, entry_price=px,
                stoploss=px - sgn * dist, target=(px + sgn * tdist) if tdist else 0.0,
                risk_points=dist, atr=atr14[i] or atr_stop[i] or 0.0,
                metadata={"levels": "relative", "trade_type": side.value.lower(),
                          "signal_close": round(px, 2)},
            )
            break   # one signal per bar; long wins a tie (both masks rarely agree)
    return out


def spec_exit_checks(spec: dict) -> list:
    ex = spec["exits"]
    checks: list = []
    if ex.get("breakeven_r"):
        checks.append(RatchetStop(be_r=float(ex["breakeven_r"]), atr_mult=None))
    tr = ex["trail"]
    if tr["kind"] == "owner":
        checks.append(RatchetStop(be_r=1.0, atr_mult=1.5))
    elif tr["kind"] == "atr":
        checks.append(RatchetStop(be_r=float(tr["trigger_r"]), atr_mult=float(tr["atr_mult"])))
    checks.append(StoplossExit(gap_aware=True))
    if ex["target"]["kind"] != "none":
        checks.append(TargetExit())
    if ex.get("max_hold_bars"):
        checks.append(MaxHoldExit(int(ex["max_hold_bars"])))
    return checks


def engine_config(*, timeframe: str, capital: float, risk_pct: float, max_position_pct: float,
                  max_positions: int, daily_loss_pct: float, slippage_pct: float, cost_model: str,
                  entry_from: str, entry_to: str, eod_time: str | None = "15:14",
                  last_entry_time: str | None = None, max_trades_per_day: int = 0,
                  cooldown_bars: int = 0, day_gate=None) -> EngineConfig:
    daily = timeframe == "1d"
    return EngineConfig(
        pnl_mode=PnLMode.RUPEES, mode=TimeframeMode.TIMELINE, capital=capital,
        max_risk_pct=risk_pct, max_position_pct=max_position_pct, max_positions=max_positions,
        slippage_pct=slippage_pct, daily_loss_limit_pct=daily_loss_pct,
        cooldown_bars=cooldown_bars, max_concurrent_per_symbol=1,
        entry_from=entry_from, entry_to=entry_to,
        session_flat=not daily, eod_minute=(_hhmm(eod_time) if (eod_time and not daily) else None),
        bar_minutes=TIMEFRAMES[timeframe],
        last_entry_minute=_hhmm(last_entry_time) if (last_entry_time and not daily) else None,
        max_trades_per_day=max_trades_per_day,
        cost_fn=cost_models.trade_cost_fn(cost_models.resolve(cost_model, daily)),
        day_gate=day_gate,
    )


def spec_engine_config(spec: dict, *, capital: float, entry_from: str, entry_to: str,
                       day_gate=None) -> EngineConfig:
    daily = spec["timeframe"] == "1d"
    sz, lm, e = spec["sizing"], spec["limits"], spec["entry"]
    return engine_config(
        timeframe=spec["timeframe"], capital=capital, risk_pct=float(sz["risk_pct"]),
        max_position_pct=float(sz["max_position_pct"]), max_positions=int(sz["max_positions"]),
        daily_loss_pct=float(lm["daily_loss_pct"]), slippage_pct=float(spec["costs"]["slippage_pct"]),
        cost_model=spec["costs"]["model"], entry_from=entry_from, entry_to=entry_to,
        eod_time=spec["exits"].get("eod_time"),
        last_entry_time=None if daily else e.get("time_to"),
        max_trades_per_day=int(lm.get("max_trades_per_day") or 0),
        cooldown_bars=int(e.get("cooldown_bars") or 0), day_gate=day_gate,
    )
