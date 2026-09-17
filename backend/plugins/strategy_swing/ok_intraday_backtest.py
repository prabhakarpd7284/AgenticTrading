"""
Oliver Kell Intraday Backtest — multi-timeframe with RSI momentum.

Tests OK cycle + RSI EMA(3)/WMA(21) momentum on 3m, 5m, 15m bars.
Runs each TF independently and produces a comparison report.

Risk optimization: tests multiple SL ATR multipliers (1.0, 1.5, 2.0)
and R:R ratios (1.5, 2.0, 2.5) to find optimal config per TF.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from logzero import logger

from trading.config import config as trading_config
from plugins.strategy_swing.ok_intraday import IntradayCycleDetector, IntradayCycleResult


# ══════════════════════════════════════════════
# Trade + Result (reuse shape from ok_backtest)
# ══════════════════════════════════════════════

@dataclass
class IntradayTrade:
    symbol: str
    side: str
    phase: str
    entry_ts: str
    entry_price: float
    stoploss: float
    target: float
    quantity: int
    notional: float
    risk_points: float

    exit_ts: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rr_achieved: float = 0.0
    bars_held: int = 0

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class TFResult:
    """Result for a single timeframe."""
    timeframe: str
    sl_atr: float
    rr: float
    total_signals: int = 0
    trades: List[IntradayTrade] = field(default_factory=list)

    # Computed
    total_trades: int = 0
    winners: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    max_dd_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_bars: float = 0.0

    # Per phase
    phase_stats: Dict[str, dict] = field(default_factory=dict)

    def finalize(self, capital: float):
        self.total_trades = len(self.trades)
        if not self.trades:
            return

        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        self.winners = len(wins)
        self.win_rate = self.winners / self.total_trades

        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_pnl_pct = self.total_pnl / capital * 100

        self.avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        self.avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        gp = sum(t.pnl for t in wins)
        gl = abs(sum(t.pnl for t in losses))
        self.profit_factor = gp / gl if gl > 0 else float("inf")

        rr_vals = [t.rr_achieved for t in self.trades if t.risk_points > 0]
        self.avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0
        self.avg_bars = sum(t.bars_held for t in self.trades) / self.total_trades

        # Max drawdown
        eq = capital
        peak = eq
        dd = 0.0
        for t in self.trades:
            eq += t.pnl
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        self.max_drawdown = dd
        self.max_dd_pct = dd / capital * 100

        # Per-phase
        phases: Dict[str, list] = {}
        for t in self.trades:
            phases.setdefault(t.phase, []).append(t)
        for ph, ts in phases.items():
            pw = [t for t in ts if t.won]
            self.phase_stats[ph] = {
                "trades": len(ts),
                "win_rate": len(pw) / len(ts),
                "pnl": sum(t.pnl for t in ts),
            }


@dataclass
class MultiTFReport:
    """Comparison across timeframes and risk configs."""
    from_date: str
    to_date: str
    capital: float
    symbols: int
    results: List[TFResult] = field(default_factory=list)

    def best(self) -> Optional[TFResult]:
        """Best config by profit factor (min 5 trades)."""
        valid = [r for r in self.results if r.total_trades >= 5]
        return max(valid, key=lambda r: r.profit_factor) if valid else None

    def telegram_message(self) -> str:
        """HTML formatted Telegram report."""
        def r(n: float) -> str:
            sign = "+" if n >= 0 else "-"
            return f"{sign}₹{abs(n):,.0f}"

        lines = [
            "<b>📊 OK Intraday Multi-TF Backtest</b>",
            f"<i>{self.from_date} → {self.to_date}</i>",
            f"Capital: ₹{self.capital:,.0f} | Symbols: {self.symbols}\n",
        ]

        # Sort by profit factor descending
        sorted_results = sorted(
            [r for r in self.results if r.total_trades > 0],
            key=lambda x: x.profit_factor,
            reverse=True,
        )

        for res in sorted_results:
            pf_icon = "🟢" if res.profit_factor >= 1.5 else "🟡" if res.profit_factor >= 1.0 else "🔴"
            lines.append(
                f"{pf_icon} <b>{res.timeframe} | SL:{res.sl_atr}ATR | RR:{res.rr}</b>"
            )
            lines.append(
                f"  {res.total_trades}t | {res.win_rate:.0%}W | "
                f"PF:{res.profit_factor:.2f} | {r(res.total_pnl)} ({res.total_pnl_pct:+.1f}%)"
            )
            lines.append(
                f"  AvgW:{r(res.avg_win)} AvgL:{r(res.avg_loss)} "
                f"DD:{r(-res.max_drawdown)} Bars:{res.avg_bars:.0f}"
            )

            if res.phase_stats:
                parts = []
                for ph, s in sorted(res.phase_stats.items()):
                    parts.append(f"{ph}:{s['trades']}t/{s['win_rate']:.0%}")
                lines.append(f"  Phases: {' | '.join(parts)}")
            lines.append("")

        best = self.best()
        if best:
            lines.append(
                f"<b>🏆 Best: {best.timeframe} SL:{best.sl_atr}ATR RR:{best.rr} "
                f"→ PF {best.profit_factor:.2f}, {best.win_rate:.0%}W, "
                f"{r(best.total_pnl)}</b>"
            )

        return "\n".join(lines)

    def summary(self) -> str:
        """CLI summary."""
        lines = [
            f"{'═' * 80}",
            f"  OK Intraday Multi-TF Backtest — {self.from_date} → {self.to_date}",
            f"  Capital: ₹{self.capital:,.0f} | Symbols: {self.symbols}",
            f"{'═' * 80}",
            "",
            f"{'TF':>4} {'SL':>5} {'RR':>4} {'Trades':>7} {'Win%':>6} "
            f"{'PF':>6} {'P&L':>12} {'P&L%':>7} {'MaxDD':>10} {'AvgBars':>8}",
            f"{'─' * 80}",
        ]

        for res in sorted(self.results, key=lambda x: (-x.profit_factor if x.total_trades >= 5 else 0)):
            if res.total_trades == 0:
                continue
            lines.append(
                f"{res.timeframe:>4} {res.sl_atr:>5.1f} {res.rr:>4.1f} "
                f"{res.total_trades:>7} {res.win_rate:>5.0%} "
                f"{res.profit_factor:>6.2f} ₹{res.total_pnl:>+10,.0f} "
                f"{res.total_pnl_pct:>+6.1f}% ₹{res.max_drawdown:>8,.0f} "
                f"{res.avg_bars:>7.1f}"
            )

        best = self.best()
        if best:
            lines.append(f"{'─' * 80}")
            lines.append(
                f"  BEST: {best.timeframe} | SL {best.sl_atr} ATR | RR {best.rr} | "
                f"PF {best.profit_factor:.2f} | {best.win_rate:.0%} win | ₹{best.total_pnl:+,.0f}"
            )

        lines.append(f"{'═' * 80}")
        return "\n".join(lines)


# ══════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════

INTERVAL_MAP = {
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
}

# Exit before market close
EOD_CUTOFF_HOUR = 15
EOD_CUTOFF_MINUTE = 20


def run_intraday_backtest(
    symbols: List[str],
    from_date: str,
    to_date: str,
    timeframes: List[str] = None,
    sl_atr_mults: List[float] = None,
    rr_ratios: List[float] = None,
    capital: float = None,
    max_risk_pct: float = None,
    max_position_pct: float = None,
    cooldown_bars: int = 5,
    data_svc=None,
) -> MultiTFReport:
    """
    Run OK intraday backtest across multiple TFs and risk configs.

    Tests all combinations of (timeframe × sl_atr × rr) and produces
    a comparison report showing the optimal configuration.
    """
    timeframes = timeframes or ["3m", "5m", "15m"]
    sl_atr_mults = sl_atr_mults or [1.0, 1.5, 2.0]
    rr_ratios = rr_ratios or [1.5, 2.0, 2.5]

    risk_cfg = trading_config.risk
    capital = capital or risk_cfg.default_capital
    max_risk_pct = max_risk_pct or risk_cfg.max_risk_per_trade_pct
    max_position_pct = max_position_pct or risk_cfg.max_position_size_pct

    if data_svc is None:
        from trading.services.data_service import DataService
        data_svc = DataService()

    report = MultiTFReport(
        from_date=from_date,
        to_date=to_date,
        capital=capital,
        symbols=len(symbols),
    )

    # ── Fetch data for each TF (with disk cache) ──
    tf_data: Dict[str, Dict[str, List[dict]]] = {}
    import json
    from pathlib import Path
    cache_dir = Path("/tmp/ok_intraday_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        interval = INTERVAL_MAP.get(tf)
        if not interval:
            logger.warning(f"Unknown timeframe: {tf}")
            continue

        logger.info(f"Fetching {tf} data for {len(symbols)} symbols...")
        tf_data[tf] = {}
        fetch_start = time.time()
        fetched = 0
        cached_hits = 0

        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                logger.info(f"  {tf}: {i + 1}/{len(symbols)} ({cached_hits} cached)")

            # Disk cache: key = symbol_tf_from_to
            cache_file = cache_dir / f"{symbol}_{tf}_{from_date}_{to_date}.json"
            if cache_file.exists():
                try:
                    candles = json.loads(cache_file.read_text())
                    if candles and len(candles) >= 60:
                        tf_data[tf][symbol] = candles
                        cached_hits += 1
                        continue
                except (json.JSONDecodeError, IOError):
                    pass

            try:
                candles = data_svc.fetch_intraday_candles(
                    symbol, from_date, to_date, interval=interval
                )
                if candles and len(candles) >= 60:
                    tf_data[tf][symbol] = candles
                    fetched += 1
                    # Save to disk cache
                    try:
                        cache_file.write_text(json.dumps(candles))
                    except IOError:
                        pass
            except Exception as e:
                logger.warning(f"  Skip {symbol} {tf}: {e}")

        elapsed = time.time() - fetch_start
        logger.info(
            f"  {tf}: {len(tf_data[tf])}/{len(symbols)} symbols in {elapsed:.1f}s "
            f"({cached_hits} cached, {fetched} fetched)"
        )

    # ── Run backtests for each (TF × SL × RR) combo ──
    for tf in timeframes:
        if tf not in tf_data or not tf_data[tf]:
            continue

        for sl_atr in sl_atr_mults:
            for rr in rr_ratios:
                logger.info(f"Testing {tf} | SL:{sl_atr}ATR | RR:{rr}")
                detector = IntradayCycleDetector(sl_atr_mult=sl_atr)

                tf_result = TFResult(timeframe=tf, sl_atr=sl_atr, rr=rr)

                for symbol, candles in tf_data[tf].items():
                    signals = detector.scan(symbol, candles, min_rr=rr)
                    tf_result.total_signals += len(signals)

                    _simulate_trades(
                        signals, candles, tf_result,
                        capital=capital,
                        max_risk_pct=max_risk_pct,
                        max_position_pct=max_position_pct,
                        rr=rr,
                        cooldown_bars=cooldown_bars,
                    )

                tf_result.finalize(capital)
                report.results.append(tf_result)
                logger.info(
                    f"  → {tf_result.total_trades}t, "
                    f"{tf_result.win_rate:.0%}W, PF:{tf_result.profit_factor:.2f}, "
                    f"₹{tf_result.total_pnl:+,.0f}"
                )

    return report


def _simulate_trades(
    signals: List[IntradayCycleResult],
    candles: List[dict],
    result: TFResult,
    capital: float,
    max_risk_pct: float,
    max_position_pct: float,
    rr: float,
    cooldown_bars: int,
):
    """Walk forward from each signal with full trade management.

    Trade lifecycle:
      1. Entry with slippage
      2. Track max favorable/adverse excursion
      3. Breakeven: move SL to entry+buffer when +0.5R reached
      4. Partial exit: book 50% at +1.0R
      5. Trailing SL: after partial, trail at 0.3 ATR (tighten to 0.2 ATR at +1.5R)
      6. Full target: remaining qty exits at full target
      7. EOD hard close at 15:20
      8. Smart priority: check SL or target first based on open direction
    """
    from trading.config import config as tc
    mgr = tc.trade_manager

    # Build index lookup by timestamp for fast candle finding
    ts_to_idx: Dict[str, int] = {}
    for i, c in enumerate(candles):
        ts = c.get("timestamp", c.get("date", ""))
        ts_to_idx[str(ts)] = i

    last_exit_idx = -cooldown_bars

    for sig in signals:
        entry_idx = ts_to_idx.get(sig.timestamp, -1)
        if entry_idx < 0 or entry_idx + 1 >= len(candles):
            continue

        if entry_idx - last_exit_idx < cooldown_bars:
            continue

        if sig.risk_points <= 0:
            continue

        # Position sizing
        risk_rupees = capital * max_risk_pct / 100
        qty = int(risk_rupees / sig.risk_points)
        if qty <= 0:
            continue

        notional = qty * sig.close
        max_not = capital * max_position_pct / 100
        if notional > max_not:
            qty = int(max_not / sig.close)
        if qty <= 0:
            continue

        # Slippage on entry
        slip = 0.05 / 100
        entry = sig.close * (1 + slip) if sig.action == "BUY" else sig.close * (1 - slip)
        risk = sig.risk_points
        atr_val = sig.atr if sig.atr > 0 else risk

        trade = IntradayTrade(
            symbol=sig.symbol,
            side=sig.action,
            phase=sig.trade_type or sig.phase.value,
            entry_ts=sig.timestamp,
            entry_price=round(entry, 2),
            stoploss=sig.sl,
            target=sig.target,
            quantity=qty,
            notional=round(qty * entry, 2),
            risk_points=round(risk, 2),
        )

        # ── Trade management state ──
        # Intraday OK cycles: breakeven at +1R, then let target hit.
        # No partial, no trailing — backtests show fixed SL/target is optimal intraday.
        current_sl = sig.sl
        breakeven_done = False

        exited = False
        for j in range(entry_idx + 1, len(candles)):
            bar = candles[j]
            trade.bars_held += 1
            o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
            ts_str = str(bar.get("timestamp", ""))

            # Current R multiple (for breakeven trigger)
            if trade.side == "BUY":
                current_r = (h - entry) / risk if risk > 0 else 0
            else:
                current_r = (entry - l) / risk if risk > 0 else 0

            # ── Breakeven: lock in at entry when +1R reached ──
            if not breakeven_done and current_r >= 1.0:
                if trade.side == "BUY":
                    current_sl = max(current_sl, entry + 0.5)
                else:
                    current_sl = min(current_sl, entry - 0.5)
                breakeven_done = True

            # ── EOD hard close ──
            if _is_eod(ts_str):
                pnl = _pnl_per_share(trade.side, entry, c) * qty
                _close_full(trade, c, ts_str, "EOD close", pnl, entry)
                last_exit_idx = j
                exited = True
                break

            # ── Smart exit priority based on open direction ──
            if trade.side == "BUY":
                sl_dist = abs(o - current_sl)
                tgt_dist = abs(sig.target - o)
            else:
                sl_dist = abs(current_sl - o)
                tgt_dist = abs(o - sig.target)
            check_sl_first = sl_dist <= tgt_dist

            hit_sl = (trade.side == "BUY" and l <= current_sl) or \
                     (trade.side == "SHORT" and h >= current_sl)
            hit_tgt = (trade.side == "BUY" and h >= sig.target) or \
                      (trade.side == "SHORT" and l <= sig.target)

            if hit_sl and hit_tgt:
                if check_sl_first:
                    hit_tgt = False
                else:
                    hit_sl = False

            if hit_sl:
                pnl = _pnl_per_share(trade.side, entry, current_sl) * qty
                reason = "BE stop" if breakeven_done else "SL hit"
                _close_full(trade, current_sl, ts_str, reason, pnl, entry)
                last_exit_idx = j
                exited = True
                break

            if hit_tgt:
                pnl = _pnl_per_share(trade.side, entry, sig.target) * qty
                _close_full(trade, sig.target, ts_str, "Target hit", pnl, entry)
                last_exit_idx = j
                exited = True
                break

        if not exited:
            last_bar = candles[-1]
            pnl = _pnl_per_share(trade.side, entry, last_bar["close"]) * qty
            _close_full(trade, last_bar["close"], str(last_bar.get("timestamp", "")), "End of test", pnl, entry)
            last_exit_idx = len(candles) - 1

        result.trades.append(trade)


def _pnl_per_share(side: str, entry: float, exit_price: float) -> float:
    return exit_price - entry if side == "BUY" else entry - exit_price


def _close_full(
    trade: IntradayTrade, price: float, ts: str, reason: str,
    total_pnl: float, entry: float,
):
    """Close trade with pre-computed P&L (accounts for partial exits)."""
    trade.exit_price = round(price, 2)
    trade.exit_ts = ts
    trade.exit_reason = reason
    trade.pnl = round(total_pnl, 2)
    trade.pnl_pct = round(total_pnl / (entry * trade.quantity) * 100, 2) if trade.quantity > 0 else 0
    if trade.risk_points > 0:
        trade.rr_achieved = round(total_pnl / (trade.risk_points * trade.quantity), 2) if trade.quantity > 0 else 0


def _is_eod(ts: str) -> bool:
    """Check if timestamp is past EOD cutoff (15:20)."""
    try:
        # Timestamps like "2026-04-28T15:25:00+05:30"
        clean = ts.replace("+05:30", "").replace("T", " ")
        dt = datetime.strptime(clean[:16], "%Y-%m-%d %H:%M")
        return dt.hour >= EOD_CUTOFF_HOUR and dt.minute >= EOD_CUTOFF_MINUTE
    except (ValueError, IndexError):
        return False
