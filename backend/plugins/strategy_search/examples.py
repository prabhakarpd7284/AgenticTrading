"""Template signal source — copy this shape for new strategies.

Rules for a source:
* ``generate(data: SymbolData)`` sees ONE symbol; emit ``Intent`` only for
  ``session.in_period`` days (warm-up sessions are for indicators only).
* Use only information available at ``signal_t`` (bars that have CLOSED by
  then, ``prev_daily`` / ``daily_history``). The simulator fills at the open
  of the first 1m bar with ``t >= signal_t``.
* Stops/targets are absolute prices. Leave sizing to the simulator.
"""
from __future__ import annotations

from .data import SymbolData
from .sim import Intent


class OpeningRangeBreakout:
    """First close beyond the N-minute opening range (on ``tf``-minute bars),
    stop at the other side of the range, no fixed target (trail + EOD)."""

    def __init__(self, minutes: int = 15, tf: int = 5, target_r: float | None = None):
        self.minutes, self.tf, self.target_r = minutes, tf, target_r
        self.name = f"orb{minutes}"

    def generate(self, data: SymbolData):
        out = []
        n_or = self.minutes // self.tf
        for s in data.period_sessions:
            bars = s.resampled(self.tf)
            if len(bars) <= n_or:
                continue
            hi = max(b.h for b in bars[:n_or])
            lo = min(b.l for b in bars[:n_or])
            for b in bars[n_or:]:
                side = 1 if b.c > hi else -1 if b.c < lo else 0
                if side:
                    stop = lo if side > 0 else hi
                    tgt = None
                    if self.target_r:
                        tgt = b.c + side * self.target_r * abs(b.c - stop)
                    out.append(Intent(strategy=self.name, symbol=data.symbol, day=s.day,
                                      side=side, signal_t=b.t + self.tf * 60, stop=stop,
                                      target=tgt, trail_tf=self.tf))
                    break
        return out
