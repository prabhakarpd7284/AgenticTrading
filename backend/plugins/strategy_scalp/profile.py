"""Price-bin / value-area pressure profile.

The scalp strategy reads *buying-pressure vs selling-pressure buildup* from the
live LTP stream of a single option. We model it as a rolling **volume/time
profile**: ticks fall into fixed-width price bins (e.g. 20-pt bins → 280, 300,
320 …), each bin accumulates a weight, and from the shape + migration of that
histogram we derive a single ``pressure`` scalar in [-1, +1].

  pressure > 0  →  buying pressure  (value area migrating up + acceptance above)
  pressure < 0  →  selling pressure (value area migrating down + acceptance below)

Everything here is pure-Python and deterministic — same ticks in, same pressure
out — so it behaves identically in the historical simulator and on a live feed.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class PressureSnapshot:
    """One reading of the profile after a tick."""
    pressure: float          # composite buy/sell pressure ∈ [-1, +1]
    poc: float               # point of control — price (bin low) of heaviest bin
    vah: float               # value-area high
    val: float               # value-area low
    bins: list = field(default_factory=list)   # [{low, weight}] asc by price
    total_weight: float = 0.0
    poc_slope: float = 0.0   # POC migration in bins/step (signed)
    accept_up: float = 0.0   # fraction of recent ticks accepted above prior VAH
    accept_dn: float = 0.0   # fraction of recent ticks accepted below prior VAL
    skew: float = 0.0        # (weight above POC − below) / total ∈ [-1, +1]


class VolumeProfile:
    """Rolling price-bin histogram with incremental (O(1) amortised) updates.

    Driven by ``ScalpConfig`` for bin width, window length and the weighting of
    the three per-tick signals (tick count, dwell time, volume delta).
    """

    def __init__(self, config):
        self.cfg = config
        self._ticks: Deque[tuple[float, int, float]] = deque()  # (ts, bin_idx, weight)
        self._bin_weight: dict[int, float] = {}
        self._prev_ts: float | None = None
        self._prev_vol: float | None = None
        self._poc_ema: float | None = None
        self._accept: Deque[tuple[float, float]] = deque(maxlen=config.acceptance_ticks)
        self._prev_vah: float | None = None
        self._prev_val: float | None = None
        # session-cumulative profile (never evicted) — the "all visited bins" view
        self._sess_w: dict[int, float] = {}   # weight per bin
        self._sess_t: dict[int, int] = {}     # tick count per bin
        self._sess_v: dict[int, int] = {}     # revisit count per bin
        self._sess_prev: int | None = None

    # ── binning ────────────────────────────────────────────────────────
    def bin_index(self, ltp: float) -> int:
        return int(math.floor(ltp / self.cfg.bin_width))

    def bin_low(self, idx: int) -> float:
        return idx * self.cfg.bin_width

    # ── main update ────────────────────────────────────────────────────
    def update(self, ts: float, ltp: float, vol: float = 0.0) -> PressureSnapshot:
        cfg = self.cfg

        # 1. acceptance is judged against the value area as it stood BEFORE this
        #    tick reshapes the histogram.
        if self._prev_vah is not None:
            up = 1.0 if ltp >= self._prev_vah else 0.0
            dn = 1.0 if ltp <= self._prev_val else 0.0
        else:
            up = dn = 0.0
        self._accept.append((up, dn))

        # 2. ingest this tick's weight, then evict stale ticks
        dwell = 0.0 if self._prev_ts is None else max(0.0, ts - self._prev_ts)
        vol_delta = 0.0
        if self._prev_vol is not None and vol > 0:
            vol_delta = max(0.0, vol - self._prev_vol)
        weight = cfg.w_count + cfg.w_dwell * dwell + cfg.w_volume * vol_delta
        idx = self.bin_index(ltp)
        self._bin_weight[idx] = self._bin_weight.get(idx, 0.0) + weight
        self._ticks.append((ts, idx, weight))
        self._evict(ts - cfg.window_secs)
        # session-cumulative accumulation (count a revisit when price re-enters a bin)
        self._sess_w[idx] = self._sess_w.get(idx, 0.0) + weight
        self._sess_t[idx] = self._sess_t.get(idx, 0) + 1
        if idx != self._sess_prev:
            self._sess_v[idx] = self._sess_v.get(idx, 0) + 1
            self._sess_prev = idx
        self._prev_ts = ts
        if vol > 0:
            self._prev_vol = vol

        # 3. point of control + value area
        poc_idx, vah, val, total = self._value_area()
        poc = self.bin_low(poc_idx)

        # 4. POC migration (EMA-smoothed slope, in bins/step)
        if self._poc_ema is None:
            self._poc_ema = poc
        else:
            alpha = 2.0 / (cfg.poc_ema_span + 1.0)
            self._poc_ema = alpha * poc + (1.0 - alpha) * self._poc_ema
        poc_slope = (poc - self._poc_ema) / cfg.bin_width

        # 5. acceptance fractions over the recent tick window
        n = len(self._accept) or 1
        accept_up = sum(u for u, _ in self._accept) / n
        accept_dn = sum(d for _, d in self._accept) / n

        # 6. skew — is weight concentrated above or below the POC
        skew = self._skew(poc_idx, total)

        # 7. composite pressure
        pressure = _clamp(
            cfg.w_mig * math.tanh(poc_slope)
            + cfg.w_accept * (accept_up - accept_dn)
            + cfg.w_skew * skew,
            -1.0, 1.0,
        )

        snap = PressureSnapshot(
            pressure=pressure, poc=poc, vah=vah, val=val,
            bins=self._bins_list(), total_weight=total, poc_slope=poc_slope,
            accept_up=accept_up, accept_dn=accept_dn, skew=skew,
        )
        self._prev_vah, self._prev_val = vah, val
        return snap

    # ── internals ──────────────────────────────────────────────────────
    def _evict(self, cutoff: float) -> None:
        bw = self._bin_weight
        while self._ticks and self._ticks[0][0] < cutoff:
            _, idx, w = self._ticks.popleft()
            remaining = bw.get(idx, 0.0) - w
            if remaining <= 1e-9:
                bw.pop(idx, None)
            else:
                bw[idx] = remaining

    def _value_area(self) -> tuple[int, float, float, float]:
        bw = self._bin_weight
        if not bw:
            return 0, 0.0, 0.0, 0.0
        total = sum(bw.values())
        poc_idx = max(bw, key=lambda k: bw[k])
        lo = hi = poc_idx
        acc = bw[poc_idx]
        target = self.cfg.value_area_pct * total
        min_idx, max_idx = min(bw), max(bw)
        guard = (max_idx - min_idx) + 2
        while acc < target - 1e-9 and guard > 0:
            guard -= 1
            below = bw.get(lo - 1, 0.0) if lo - 1 >= min_idx else -1.0
            above = bw.get(hi + 1, 0.0) if hi + 1 <= max_idx else -1.0
            if below < 0 and above < 0:
                break
            if above >= below:
                hi += 1
                acc += max(0.0, above)
            else:
                lo -= 1
                acc += max(0.0, below)
        vah = self.bin_low(hi) + self.cfg.bin_width
        val = self.bin_low(lo)
        return poc_idx, vah, val, total

    def _skew(self, poc_idx: int, total: float) -> float:
        if total <= 0:
            return 0.0
        above = sum(w for k, w in self._bin_weight.items() if k > poc_idx)
        below = sum(w for k, w in self._bin_weight.items() if k < poc_idx)
        return _clamp((above - below) / total, -1.0, 1.0)

    def _bins_list(self) -> list[dict]:
        return [
            {"low": round(self.bin_low(k), 2), "weight": round(w, 4)}
            for k, w in sorted(self._bin_weight.items())
        ]

    def session_profile(self) -> dict:
        """Enriched stats for every bin price has visited this session (cumulative,
        not the rolling window): weight, % of total, ticks, and revisit count."""
        if not self._sess_w:
            return {"bins": [], "poc": 0.0, "total": 0.0}
        total = sum(self._sess_w.values()) or 1.0
        poc_idx = max(self._sess_w, key=lambda k: self._sess_w[k])
        bins = [
            {
                "low": round(self.bin_low(k), 2),
                "weight": round(self._sess_w[k], 2),
                "pct": round(self._sess_w[k] / total * 100, 1),
                "ticks": self._sess_t.get(k, 0),
                "visits": self._sess_v.get(k, 0),
                "poc": k == poc_idx,
            }
            for k in sorted(self._sess_w)
        ]
        return {"bins": bins, "poc": round(self.bin_low(poc_idx), 2), "total": round(total, 2)}
