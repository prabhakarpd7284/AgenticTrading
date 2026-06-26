"""Scalping engine — bin-pressure driven, both directions, incremental.

Sibling to the pyramid strategy, but scalping-oriented:

  - 10-minute candles set a directional **bias** at each candle close.
  - Within the candle, a live LTP tick stream is read for **buying vs selling
    pressure buildup** via the price-bin :class:`~.profile.VolumeProfile`.
  - Buying pressure → go **LONG** the option (+ pyramid adds); selling pressure
    → **SHORT** the premium. Both directions trade the *same* instrument.
  - The value-area boundaries drive stop-loss placement; sizing is the
    deterministic pyramid math (risk a fixed % of capital on entry, then risk a
    fraction of unrealized profit on each add).

The engine is **stateful but pure**: ``feed_tick`` / ``on_candle_close`` /
``apply_manual`` mutate internal state and return a list of :class:`ScalpEvent`.
Feed it synthesized ticks (simulator) or real ticks (live) — identical code path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .profile import PressureSnapshot, VolumeProfile
from .timeutil import is_eod, iso_from_epoch


# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────

@dataclass
class Tick:
    ts: float          # epoch seconds (UTC)
    ltp: float
    vol: float = 0.0   # cumulative day volume if available; 0 for LTP-only feeds
    oi: float = 0.0


@dataclass
class Candle:
    timestamp: str     # IST ISO
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    epoch: float = 0.0

    @staticmethod
    def from_fyers(raw: list) -> "Candle":
        """Fyers history row: ``[epoch, o, h, l, c, v]`` (epoch in seconds)."""
        ep = float(raw[0])
        return Candle(
            timestamp=iso_from_epoch(ep),
            open=float(raw[1]), high=float(raw[2]),
            low=float(raw[3]), close=float(raw[4]),
            volume=int(raw[5]) if len(raw) > 5 else 0,
            epoch=ep,
        )


@dataclass
class ScalpEntry:
    ts: float
    timestamp: str
    side: str          # LONG | SHORT
    price: float
    lots: int
    sl_at_entry: float
    reason: str


@dataclass
class ScalpEvent:
    ts: float
    type: str          # pressure | decision | position | exit | candle | log
    payload: dict


@dataclass
class ScalpConfig:
    """All scalp parameters (every threshold is tunable from the schema/CLI)."""
    # instrument / sizing
    lot_size: int = 65
    initial_capital: float = 100_000
    initial_risk_pct: float = 1.0       # % capital risked on first entry
    profit_risk_pct: float = 0.80       # risk this fraction of unrealized on adds
    max_pyramids: int = 4
    pyramid_cooldown_secs: float = 30.0
    # bins / profile
    bin_width: float = 20.0
    window_secs: float = 180.0
    value_area_pct: float = 0.70
    acceptance_ticks: int = 8
    poc_ema_span: float = 20.0
    w_count: float = 1.0                # per-tick weight: tick count
    w_dwell: float = 1.0                # per-tick weight: dwell seconds
    w_volume: float = 1.0               # per-tick weight: volume delta (auto-0 if no vol)
    w_mig: float = 0.35                 # pressure: POC migration term
    w_accept: float = 0.40              # pressure: acceptance term
    w_skew: float = 0.25                # pressure: skew term
    profile_every: int = 25             # emit the session "visited bins" profile every N ticks
    # decision thresholds — calibrated to the composite's natural range (≈ ±0.45);
    # the three terms rarely co-saturate, so |pressure| ~0.3 is already a strong read.
    entry_threshold: float = 0.30       # with-trend entries (the trend gate is the main filter)
    add_threshold: float = 0.22
    reverse_threshold: float = 0.55     # only flip on a STRONG opposite read (anti-churn)
    reentry_cooldown_secs: float = 60.0  # wait after any exit before re-entering
    sl_buffer: float = 10.0             # SL sits this far beyond the value-area edge
    min_sl_distance: float = 2.0        # reject entries with SL tighter than this
    max_risk_pct_of_price: float = 0.50 # reject entries with SL wider than this × price
    # trend filter (read the session direction from the bins/closes; trade WITH it)
    trend_only: bool = True              # only take with-trend entries; sit out chop
    trend_ema_span: int = 5              # EMA span (in 10-min candles) for the trend
    trend_flat_band: float = 0.012       # |EMA slope| below this ⇒ "flat" (no entries)
    # bin-reversal entry: after the bias is set, wait for price to pull back to a
    # bin and REVERSE (buy the dip off support / sell the bounce off resistance)
    # rather than chasing pressure. Thresholds are in units of bin_width.
    bin_reversal_entry: bool = True
    pullback_min_bins: float = 0.6       # require ≥ this pullback before a reversal counts
    reversal_bins: float = 0.35          # confirm the bounce/drop off the extreme by this much
    # bias gate (legacy path, used when trend_only=False)
    require_bias_alignment: bool = True
    allow_reverse: bool = True
    bias_ema_span: int = 3
    # session
    eod_hour: int = 15
    eod_minute: int = 20


@dataclass
class ScalpState:
    side: str = "FLAT"                  # FLAT | LONG | SHORT
    lots: int = 0
    cost: float = 0.0                   # Σ price×lots for the open position
    avg: float = 0.0
    trail_sl: float = 0.0
    pyramids: int = 0
    last_add_ts: float = -1e18
    bias: str = "neutral"               # buy | sell | neutral (last 10m candle)
    peak_unrealized: float = 0.0
    peak_lots: int = 0


@dataclass
class ScalpResult:
    symbol: str
    entries: List[ScalpEntry] = field(default_factory=list)
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    realized_pnl: float = 0.0           # points (Σ per-unit P&L × lots, both sides)
    peak_unrealized: float = 0.0
    peak_lots: int = 0
    trades: int = 0                     # closed round-trips
    lot_size: int = 65
    open_side: str = "FLAT"             # non-FLAT if still holding at end
    open_lots: int = 0
    log: List[str] = field(default_factory=list)

    @property
    def realized_inr(self) -> float:
        return self.realized_pnl * self.lot_size


def _dir(side: str) -> int:
    return 1 if side == "LONG" else -1


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

class ScalpEngine:
    def __init__(self, symbol: str = "NIFTY_CE", config: Optional[ScalpConfig] = None):
        self.symbol = symbol
        self.cfg = config or ScalpConfig()
        self.profile = VolumeProfile(self.cfg)
        self.st = ScalpState()
        self.log: List[str] = []
        self.entries: List[ScalpEntry] = []
        self._closes: List[float] = []
        self._pocs: List[float] = []     # bin point-of-control at each 10-min close
        self._trend = "flat"             # up | down | flat (session direction)
        self._trades = 0
        self._realized = 0.0
        self._peak_unrealized = 0.0          # session-wide peak across all positions
        self._last_exit_ts: float = -1e18    # for the re-entry cooldown
        self._rev_anchor: Optional[float] = None  # bin-reversal: trend-side extreme
        self._rev_ext: Optional[float] = None     # bin-reversal: pullback extreme
        self._last_snap: Optional[PressureSnapshot] = None
        self._last_ltp: float = 0.0
        self._last_ts: float = 0.0
        self._last_exit: Optional[dict] = None
        self._tick_n = 0

    # ── tick path ──────────────────────────────────────────────────────
    def feed_tick(self, tick: Tick) -> List[ScalpEvent]:
        snap = self.profile.update(tick.ts, tick.ltp, tick.vol)
        self._last_snap = snap
        self._last_ltp = tick.ltp
        self._last_ts = tick.ts

        events: List[ScalpEvent] = [ScalpEvent(tick.ts, "pressure", {
            "pressure": round(snap.pressure, 4),
            "poc": round(snap.poc, 2),
            "vah": round(snap.vah, 2),
            "val": round(snap.val, 2),
            "bins": snap.bins,
            "bias": self.st.bias,
        })]

        # periodically emit the session-cumulative "all visited bins" profile
        self._tick_n += 1
        if self._tick_n % max(1, self.cfg.profile_every) == 0:
            events.append(ScalpEvent(tick.ts, "profile", self.profile.session_profile()))

        if self.st.side != "FLAT" and is_eod(tick.ts, self.cfg.eod_hour, self.cfg.eod_minute):
            events += self._close(tick, "EOD")
            return events

        if self.st.side == "FLAT":
            events += self._maybe_enter(tick, snap)
        elif self.st.side == "LONG":
            events += self._manage(tick, snap, "LONG")
        else:
            events += self._manage(tick, snap, "SHORT")

        if self.st.side != "FLAT":
            events.append(self._position_event(tick))
        return events

    # ── candle path (decision timeframe, e.g. 10m) ─────────────────────
    def on_candle_close(self, candle: Candle) -> List[ScalpEvent]:
        self._closes.append(candle.close)
        # capture where the bins say price is being accepted (POC) at this close
        self._pocs.append(self._last_snap.poc if self._last_snap else candle.close)
        self._trend = self._compute_trend(candle)
        self.st.bias = {"up": "buy", "down": "sell", "flat": "neutral"}[self._trend]
        ep = candle.epoch or 0.0
        return [ScalpEvent(ep, "candle", {
            "t": candle.timestamp, "epoch": ep, "forming": False,
            "o": round(candle.open, 2), "h": round(candle.high, 2),
            "l": round(candle.low, 2), "c": round(candle.close, 2), "v": candle.volume,
            "bias": self.st.bias,
        })]

    # ── manual override ────────────────────────────────────────────────
    def apply_manual(self, action: str, ts: float, *, lots: Optional[int] = None,
                     price: Optional[float] = None, sl: Optional[float] = None) -> List[ScalpEvent]:
        """Inject a trader decision at the current tick. Manual actions are
        drained *before* the auto logic on a tick, so they always win."""
        p = price if price is not None else self._last_ltp
        tick = Tick(ts=ts, ltp=p)
        snap = self._last_snap
        events: List[ScalpEvent] = []
        act = (action or "").lower()

        if act in ("buy", "long"):
            if self.st.side == "SHORT":
                events += self._close(tick, "manual flip")
            sl_ = sl if sl is not None else (snap.val - self.cfg.sl_buffer if snap else p * 0.9)
            events += self._open(tick, "LONG", sl_, "manual buy", lots=lots)
        elif act in ("short", "sell"):
            if self.st.side == "LONG":
                events += self._close(tick, "manual flip")
            sl_ = sl if sl is not None else (snap.vah + self.cfg.sl_buffer if snap else p * 1.1)
            events += self._open(tick, "SHORT", sl_, "manual short", lots=lots)
        elif act == "add":
            if self.st.side != "FLAT" and snap is not None:
                events += self._add(tick, snap, force=True, lots=lots)
        elif act in ("exit", "flatten"):
            if self.st.side != "FLAT":
                events += self._close(tick, "manual exit")
        elif act == "adjust_sl" and sl is not None and self.st.side != "FLAT":
            self.st.trail_sl = sl
            line = f"[{iso_from_epoch(ts)}] MANUAL SL → {sl:.2f}"
            self.log.append(line)
            events += [ScalpEvent(ts, "log", {"line": line}), self._position_event(tick)]
        return events

    def _ema_dir(self, series: List[float]) -> int:
        """Sign of the EMA slope of `series` with a flat dead-band: +1 / -1 / 0."""
        n = len(series)
        if n < 2:
            return 0
        from trading.utils.indicators import _ema
        ema = _ema(series, min(self.cfg.trend_ema_span, n))
        if len(ema) < 2 or ema[-1] == 0:
            return 0
        rel = (ema[-1] - ema[-2]) / abs(ema[-1])
        return 1 if rel > self.cfg.trend_flat_band else -1 if rel < -self.cfg.trend_flat_band else 0

    def _compute_trend(self, candle: Candle) -> str:
        """Bias on the 10-min timeframe, confirmed by the bin price-pressure.

        The session is "up" only when the 10-min closes are trending up AND the
        bins (POC / value area) are not migrating down — i.e. price structure and
        where volume is being accepted agree. Either one disagreeing ⇒ flat (no
        new trade), which is what keeps the engine out of chop.
        """
        if len(self._closes) < 2:
            return "up" if candle.close >= candle.open else "down"
        close_dir = self._ema_dir(self._closes)
        poc_dir = self._ema_dir(self._pocs) if len(self._pocs) >= 2 else close_dir
        if close_dir > 0 and poc_dir >= 0:
            return "up"
        if close_dir < 0 and poc_dir <= 0:
            return "down"
        return "flat"

    # ── entry / management ─────────────────────────────────────────────
    def _bias_allows(self, side: str) -> bool:
        if not self.cfg.require_bias_alignment or self.st.bias == "neutral":
            return True
        return (self.st.bias == "buy") == (side == "LONG")

    def _bin_floor(self, price: float) -> float:
        return math.floor(price / self.cfg.bin_width) * self.cfg.bin_width

    def _bin_ceil(self, price: float) -> float:
        return math.ceil(price / self.cfg.bin_width) * self.cfg.bin_width

    def _reversal_entry(self, tick: Tick, snap: PressureSnapshot) -> List[ScalpEvent]:
        """Enter on a pullback-to-bin REVERSAL in the bias direction. Up bias:
        track the recent peak, the pullback low since the peak, and enter LONG
        once price bounces ``reversal_bins`` off that low after a ≥
        ``pullback_min_bins`` dip — i.e. price revisited a support bin and turned
        back up. SL sits just below the bin that held. Down bias mirrors it."""
        cfg, bw, ltp = self.cfg, self.cfg.bin_width, tick.ltp

        if self._trend == "up":
            if self._rev_anchor is None or ltp > self._rev_anchor:
                self._rev_anchor = self._rev_ext = ltp   # new peak resets the pullback
            elif ltp < self._rev_ext:
                self._rev_ext = ltp                       # pullback deepens
            pullback, bounce = self._rev_anchor - self._rev_ext, ltp - self._rev_ext
            if pullback >= cfg.pullback_min_bins * bw and bounce >= cfg.reversal_bins * bw and snap.pressure >= 0:
                support = self._bin_floor(self._rev_ext)
                return self._open(tick, "LONG", support - cfg.sl_buffer, f"reversal off {support:.0f} bin")
        elif self._trend == "down":
            if self._rev_anchor is None or ltp < self._rev_anchor:
                self._rev_anchor = self._rev_ext = ltp
            elif ltp > self._rev_ext:
                self._rev_ext = ltp
            pullback, drop = self._rev_ext - self._rev_anchor, self._rev_ext - ltp
            if pullback >= cfg.pullback_min_bins * bw and drop >= cfg.reversal_bins * bw and snap.pressure <= 0:
                resist = self._bin_ceil(self._rev_ext)
                return self._open(tick, "SHORT", resist + cfg.sl_buffer, f"reversal off {resist:.0f} bin")
        return []

    def _maybe_enter(self, tick: Tick, snap: PressureSnapshot) -> List[ScalpEvent]:
        cfg, p = self.cfg, snap.pressure
        # No fresh entries once the EOD cutoff has passed — otherwise the engine
        # opens and instantly EOD-exits on every remaining tick.
        if is_eod(tick.ts, cfg.eod_hour, cfg.eod_minute):
            return []
        # Cooldown after an exit — stops the rapid flip-flop churn.
        if tick.ts - self._last_exit_ts < cfg.reentry_cooldown_secs:
            return []

        if cfg.trend_only:
            if cfg.bin_reversal_entry:
                return self._reversal_entry(tick, snap)
            # Pressure-threshold path (when bin-reversal entry is off).
            if self._trend == "up" and p >= cfg.entry_threshold:
                return self._open(tick, "LONG", snap.val - cfg.sl_buffer, f"uptrend · buying pressure {p:+.2f}")
            if self._trend == "down" and p <= -cfg.entry_threshold:
                return self._open(tick, "SHORT", snap.vah + cfg.sl_buffer, f"downtrend · selling pressure {p:+.2f}")
            return []

        long_thr = cfg.entry_threshold if self._bias_allows("LONG") else cfg.reverse_threshold
        short_thr = cfg.entry_threshold if self._bias_allows("SHORT") else cfg.reverse_threshold
        if p >= long_thr:
            return self._open(tick, "LONG", snap.val - cfg.sl_buffer, f"buying pressure {p:+.2f}")
        if p <= -short_thr:
            return self._open(tick, "SHORT", snap.vah + cfg.sl_buffer, f"selling pressure {p:+.2f}")
        return []

    def _manage(self, tick: Tick, snap: PressureSnapshot, side: str) -> List[ScalpEvent]:
        cfg, st, d = self.cfg, self.st, _dir(side)
        ltp = tick.ltp
        unreal = d * (ltp - st.avg) * st.lots
        st.peak_unrealized = max(st.peak_unrealized, unreal)
        self._peak_unrealized = max(self._peak_unrealized, unreal)

        # trail the SL toward the value-area edge — ratchet only
        edge = (snap.val - cfg.sl_buffer) if d == 1 else (snap.vah + cfg.sl_buffer)
        if d == 1:
            st.trail_sl = max(st.trail_sl, edge)
        else:
            st.trail_sl = min(st.trail_sl, edge)

        # exits, in priority order
        if (d == 1 and ltp <= st.trail_sl) or (d == -1 and ltp >= st.trail_sl):
            return self._close(tick, "trail SL", price=st.trail_sl)
        opp = -d * snap.pressure  # how strong is pressure AGAINST us
        if opp >= cfg.reverse_threshold:
            events = self._close(tick, f"reverse (pressure {snap.pressure:+.2f})")
            new_side = "SHORT" if side == "LONG" else "LONG"
            with_trend = (new_side == "LONG" and self._trend == "up") or \
                         (new_side == "SHORT" and self._trend == "down")
            # In trend_only mode never flip into a counter-trend trade — just exit.
            if cfg.allow_reverse and (not cfg.trend_only or with_trend):
                new_sl = (snap.vah + cfg.sl_buffer) if new_side == "SHORT" else (snap.val - cfg.sl_buffer)
                events += self._open(tick, new_side, new_sl, f"reversed, pressure {snap.pressure:+.2f}")
            return events
        # NOTE: no soft "bias flipped" exit — the trail SL (which trails the value
        # area) is what protects the position. Exiting on every minor bias wobble
        # is what caused the churn; we let winners run with the trend instead.

        # pyramid add
        aligned = d * snap.pressure
        if (st.pyramids < cfg.max_pyramids and aligned >= cfg.add_threshold
                and tick.ts - st.last_add_ts >= cfg.pyramid_cooldown_secs):
            return self._add(tick, snap)
        return []

    def _open(self, tick: Tick, side: str, sl: float, reason: str,
              lots: Optional[int] = None) -> List[ScalpEvent]:
        cfg, st, d = self.cfg, self.st, _dir(side)
        entry = tick.ltp
        risk_per_unit = abs(entry - sl)
        if lots is None:
            if risk_per_unit < cfg.min_sl_distance or risk_per_unit > entry * cfg.max_risk_pct_of_price:
                return []
            max_risk = cfg.initial_capital * cfg.initial_risk_pct / 100.0
            lots = max(1, int(max_risk / (risk_per_unit * cfg.lot_size)))
        lots = max(1, int(lots))

        st.side, st.lots, st.cost, st.avg = side, lots, entry * lots, entry
        st.trail_sl, st.pyramids, st.last_add_ts = sl, 0, tick.ts
        st.peak_unrealized = 0.0
        st.peak_lots = max(st.peak_lots, lots)

        self.entries.append(ScalpEntry(tick.ts, iso_from_epoch(tick.ts), side, entry, lots, sl, f"initial · {reason}"))
        line = f"[{iso_from_epoch(tick.ts)}] ENTER {side} @ {entry:.2f} | {lots} lots | SL {sl:.2f} | {reason}"
        self.log.append(line)
        return [
            ScalpEvent(tick.ts, "decision", {
                "action": "enter_long" if side == "LONG" else "enter_short",
                "side": side, "price": round(entry, 2), "lots": lots,
                "sl": round(sl, 2), "reason": reason,
            }),
            ScalpEvent(tick.ts, "log", {"line": line}),
        ]

    def _add(self, tick: Tick, snap: PressureSnapshot, *, force: bool = False,
             lots: Optional[int] = None) -> List[ScalpEvent]:
        cfg, st, d = self.cfg, self.st, _dir(self.st.side)
        price = tick.ltp
        new_sl = (snap.val - cfg.sl_buffer) if d == 1 else (snap.vah + cfg.sl_buffer)
        unreal = d * (price - st.avg) * st.lots
        locked = d * (new_sl - st.avg) * st.lots
        risk_per_unit = d * (price - new_sl)

        if lots is None:
            if risk_per_unit <= 0 or unreal <= 0:
                return []
            # risk profit_risk_pct of unrealized, keeping the rest locked
            risk_budget = locked - (1 - cfg.profit_risk_pct) * unreal
            if risk_budget <= 0 and not force:
                return []
            add_lots = max(1, int(max(risk_budget, 0) / risk_per_unit)) if risk_budget > 0 else 1
            add_lots = min(add_lots, st.lots * 2)
        else:
            add_lots = max(1, int(lots))

        st.lots += add_lots
        st.cost += price * add_lots
        st.avg = st.cost / st.lots
        if d == 1:
            st.trail_sl = max(st.trail_sl, new_sl)
        else:
            st.trail_sl = min(st.trail_sl, new_sl) if st.trail_sl else new_sl
        st.pyramids += 1
        st.last_add_ts = tick.ts
        st.peak_lots = max(st.peak_lots, st.lots)

        tag = "manual add" if force else f"pyramid #{st.pyramids}"
        self.entries.append(ScalpEntry(tick.ts, iso_from_epoch(tick.ts), st.side, price, add_lots, st.trail_sl, tag))
        line = (f"[{iso_from_epoch(tick.ts)}] ADD {st.side} #{st.pyramids} @ {price:.2f} | "
                f"+{add_lots} → {st.lots} lots | avg {st.avg:.2f} | SL {st.trail_sl:.2f}")
        self.log.append(line)
        return [
            ScalpEvent(tick.ts, "decision", {
                "action": "add", "side": st.side, "price": round(price, 2),
                "lots": add_lots, "total_lots": st.lots, "sl": round(st.trail_sl, 2), "reason": tag,
            }),
            ScalpEvent(tick.ts, "log", {"line": line}),
        ]

    def _close(self, tick: Tick, reason: str, price: Optional[float] = None) -> List[ScalpEvent]:
        st = self.st
        d = _dir(st.side)
        exit_price = price if price is not None else tick.ltp
        pnl = d * (exit_price - st.avg) * st.lots
        self._realized += pnl
        self._trades += 1
        self._last_exit_ts = tick.ts
        side, lots, avg = st.side, st.lots, st.avg
        line = (f"[{iso_from_epoch(tick.ts)}] EXIT {side} @ {exit_price:.2f} | {lots} lots | "
                f"avg {avg:.2f} | P&L {pnl:+.2f} pts | {reason}")
        self.log.append(line)
        self._last_exit = {"t": iso_from_epoch(tick.ts), "price": round(exit_price, 2),
                           "reason": reason, "side": side, "lots": lots, "pnl_pts": round(pnl, 2)}
        # reset
        st.side, st.lots, st.cost, st.avg, st.trail_sl, st.pyramids = "FLAT", 0, 0.0, 0.0, 0.0, 0
        self._rev_anchor = self._rev_ext = None   # fresh pullback tracking next time
        return [
            ScalpEvent(tick.ts, "decision", {"action": "exit", "side": side, "price": round(exit_price, 2),
                                             "lots": lots, "pnl_pts": round(pnl, 2), "reason": reason}),
            ScalpEvent(tick.ts, "exit", dict(self._last_exit)),
            ScalpEvent(tick.ts, "log", {"line": line}),
        ]

    def _position_event(self, tick: Tick) -> ScalpEvent:
        st = self.st
        d = _dir(st.side)
        unreal = d * (tick.ltp - st.avg) * st.lots
        return ScalpEvent(tick.ts, "position", {
            "side": st.side, "lots": st.lots, "avg": round(st.avg, 2),
            "sl": round(st.trail_sl, 2), "ltp": round(tick.ltp, 2),
            "unrealized_pts": round(unreal, 2), "unrealized_inr": round(unreal * self.cfg.lot_size, 0),
            "pyramids": st.pyramids,
        })

    # ── result / KPIs ──────────────────────────────────────────────────
    def result(self) -> ScalpResult:
        st = self.st
        open_unreal = 0.0
        if st.side != "FLAT":
            open_unreal = _dir(st.side) * (self._last_ltp - st.avg) * st.lots
        res = ScalpResult(
            symbol=self.symbol, entries=list(self.entries),
            exit_price=(self._last_exit or {}).get("price", 0.0),
            exit_time=(self._last_exit or {}).get("t", ""),
            exit_reason=(self._last_exit or {}).get("reason", ""),
            realized_pnl=self._realized + open_unreal,
            peak_unrealized=self._peak_unrealized, peak_lots=st.peak_lots,
            trades=self._trades, lot_size=self.cfg.lot_size,
            open_side=st.side, open_lots=st.lots, log=list(self.log),
        )
        return res

    def kpis(self) -> dict:
        res = self.result()
        return {
            "realized_pnl_pts": round(res.realized_pnl, 2),
            "realized_pnl_inr": round(res.realized_inr, 0),
            "peak_unrealized_pts": round(res.peak_unrealized, 2),
            "peak_unrealized_inr": round(res.peak_unrealized * res.lot_size, 0),
            "trades": res.trades,
            "entries": len(res.entries),
            "peak_lots": res.peak_lots,
            "open_side": res.open_side,
            "open_lots": res.open_lots,
            "lot_size": res.lot_size,
            "won": res.realized_pnl > 0,
        }
