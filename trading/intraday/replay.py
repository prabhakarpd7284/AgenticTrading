"""
Intraday replay — derive paper trades for a *historical* trading day by driving
the live ``IntradayMonitor`` over that day's candles.

Why this exists
───────────────
The live intraday agent only creates ``Trade`` rows while the market is open
(it gates on the wall clock) and stamps ``trade_date = today``. So a day that
was never traded live can never be back-filled by it — which is exactly why the
Monthly report's trade-driven sections froze at the last live session.

This module reuses the live monitor *unchanged* — same structure detectors,
same risk engine, same ``process_signal`` trade-creation path — and only
changes two things:

  1. **Where candles come from** — a per-symbol full-day prefetch, sliced to a
     stepping replay cursor (so entries trigger at the time they'd have fired
     intraday, not all at once at the close).
  2. **What date trades are stamped with** — via ``state.trading_date``, which
     the monitor already honours after the surgical ``_trade_date()`` fix.

After the entry pass, :meth:`_simulate_exits` closes each filled position by
walking the remaining candles to the first SL/target touch (EOD close as the
fallback) and writes ``realized_pnl`` — reusing the same P&L math as the live
``_square_off_all`` — so the equity curve / analytics have real numbers.

Paper only: ``process_signal`` routes through ``BrokerService`` which paper-fills
when ``TRADING_MODE=paper`` (the default). This never places a live order.
"""
from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta
from typing import Dict, List, Optional

from logzero import logger

from trading.intraday.monitor import IntradayMonitor
from trading.intraday.state import IntradayState

# Step the replay cursor across the session. 09:45 gives the detectors enough
# 5-min candles to work with; 15:15 stops new entries before the close.
_REPLAY_START = dt_time(9, 45)
_REPLAY_END = dt_time(15, 15)
_STEP_MINUTES = 30


def _parse_ts(ts) -> datetime:
    """Broker candle timestamp → naive datetime (drop tz for cursor compares)."""
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    return datetime.fromisoformat(str(ts)).replace(tzinfo=None)


class IntradayReplay(IntradayMonitor):
    """Drives the live monitor over one past day's candles, in paper mode."""

    def __init__(self, state: IntradayState):
        super().__init__(state)
        self._full_candles: Dict[str, List[dict]] = {}   # symbol → full-day 5m
        self._cursor: Optional[datetime] = None           # current replay time
        self._entry_at: Dict[str, datetime] = {}          # symbol → entry time

    # ── candle source: prefetch once, slice to the cursor ──────────────
    def _fetch_intraday_candles(self, symbol: str, token: str, today: str) -> List[dict]:
        if symbol not in self._full_candles:
            self._full_candles[symbol] = self._prefetch(symbol, token, today)
        candles = self._full_candles[symbol]
        if self._cursor is None:
            return candles
        return [c for c in candles if _parse_ts(c["timestamp"]) <= self._cursor]

    def _prefetch(self, symbol: str, token: str, today: str) -> List[dict]:
        """Full-day 5-min candles for one symbol — bypasses the live-clock gate."""
        self.data_service._ensure_broker()
        if not token:
            from trading.services.ticker_service import ticker_service
            token = ticker_service.get_token(symbol) or ""
            if not token:
                return []
        raw = self.data_service._broker.fetch_candles(
            token, f"{today} 09:15", f"{today} 15:30", "FIVE_MINUTE",
        )
        out: List[dict] = []
        for row in raw or []:
            out.append({
                "timestamp": row[0],
                "open": float(row[1]), "high": float(row[2]),
                "low": float(row[3]), "close": float(row[4]),
                "volume": int(row[5]),
            })
        return out

    # ── one-day replay ─────────────────────────────────────────────────
    def replay_day(self, d: date) -> dict:
        """Build the watchlist for ``d``, step entries, then simulate exits."""
        from trading.intraday.agent import premarket_scan_node

        # Fresh per-day position state so caps/loss don't bleed across days.
        self.state.trading_date = d.isoformat()
        self.state.open_positions = 0
        self.state.daily_loss = 0.0
        self.state.trades_today = []
        self._full_candles = {}
        self._entry_at = {}
        self._cursor = None

        premarket_scan_node(self.state)
        if not self.state.watchlist:
            return {"date": d.isoformat(), "watchlist": 0, "entries": 0, "exits": 0}

        traded: set[str] = set()
        cursor = datetime.combine(d, _REPLAY_START)
        end = datetime.combine(d, _REPLAY_END)
        while cursor <= end:
            self._cursor = cursor
            try:
                signals = self.run_scan_cycle()
            except Exception as e:  # one bad step shouldn't abort the day
                logger.warning(f"replay {d} @ {cursor:%H:%M} scan error: {e}")
                signals = []
            for sig in signals:
                if sig.symbol in traded:
                    continue
                res = self.process_signal(sig)
                if res.get("action") == "TRADED":
                    traded.add(sig.symbol)
                    self._entry_at[sig.symbol] = cursor
            cursor += timedelta(minutes=_STEP_MINUTES)

        exits = self._simulate_exits(d)
        logger.info(f"replay {d}: watchlist={len(self.state.watchlist)} "
                     f"entries={len(traded)} exits={exits}")
        return {
            "date": d.isoformat(),
            "watchlist": len(self.state.watchlist),
            "entries": len(traded),
            "exits": exits,
        }

    # ── exit simulation ────────────────────────────────────────────────
    def _simulate_exits(self, d: date) -> int:
        """Close each FILLED trade at the first SL/target touch, else EOD close.

        Reuses the live square-off P&L math: (exit − entry)·qty for a long,
        (entry − exit)·qty for a short.
        """
        from django.utils import timezone
        from apps.trading.models import Trade

        closed = 0
        for t in Trade.objects.filter(trade_date=d, status=Trade.Status.FILLED):
            candles = self._full_candles.get(t.symbol)
            if not candles:
                continue
            entry = float(t.entry_price)
            stop = float(t.stop_loss) if t.stop_loss else None
            target = float(t.target) if t.target else None
            qty = t.quantity
            is_long = t.side == "BUY"

            entry_at = self._entry_at.get(t.symbol)
            forward = [
                c for c in candles
                if entry_at is None or _parse_ts(c["timestamp"]) > entry_at
            ] or candles[-1:]

            exit_px = float(forward[-1]["close"])  # EOD fallback
            reason = Trade.CloseReason.EOD
            exit_candle = forward[-1]
            for c in forward:
                hi, lo = float(c["high"]), float(c["low"])
                if is_long:
                    if stop is not None and lo <= stop:      # SL first (conservative)
                        exit_px, reason, exit_candle = stop, Trade.CloseReason.SL_HIT, c
                        break
                    if target is not None and hi >= target:
                        exit_px, reason, exit_candle = target, Trade.CloseReason.TARGET_HIT, c
                        break
                else:
                    if stop is not None and hi >= stop:
                        exit_px, reason, exit_candle = stop, Trade.CloseReason.SL_HIT, c
                        break
                    if target is not None and lo <= target:
                        exit_px, reason, exit_candle = target, Trade.CloseReason.TARGET_HIT, c
                        break

            pnl = (exit_px - entry) * qty if is_long else (entry - exit_px) * qty
            t.fill_price = round(entry, 2)
            t.fill_quantity = qty
            t.exit_price = round(exit_px, 2)
            t.exit_quantity = qty
            t.realized_pnl = round(pnl, 2)
            t.pnl_percent = round(pnl / (entry * qty) * 100, 2) if entry and qty else 0.0
            t.status = Trade.Status.CLOSED
            t.close_reason = reason
            # Real entry (cursor) + exit (the candle that triggered it) times,
            # so the chart can draw an accurate trade timeline.
            if entry_at is not None:
                t.filled_at = timezone.make_aware(entry_at) if timezone.is_naive(entry_at) else entry_at
            else:
                t.filled_at = timezone.now()
            try:
                t.closed_at = datetime.fromisoformat(str(exit_candle["timestamp"]))
            except (ValueError, TypeError):
                t.closed_at = timezone.now()
            t.save(update_fields=[
                "fill_price", "fill_quantity", "exit_price", "exit_quantity",
                "realized_pnl", "pnl_percent", "status", "close_reason",
                "filled_at", "closed_at",
            ])
            closed += 1
        return closed
