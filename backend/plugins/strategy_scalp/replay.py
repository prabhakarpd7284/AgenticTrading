"""Replay helpers — turn sub-minute candles into a tick stream and drive the engine.

Fyers serves seconds-resolution candles (5S/10S/…); the engine consumes a *tick*
stream. ``synthesize_ticks`` interpolates each seconds bar into a deterministic
intra-bar path, ``aggregate_decision_candles`` rolls the same bars up into the
10-minute decision timeframe that sets bias, and ``iter_session`` interleaves the
two in time order. The same iterator drives both the headless CLI/tests and the
live-simulation WebSocket controller — only the pacing differs.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

from .engine import Candle, ScalpConfig, ScalpEngine, ScalpResult, Tick
from .timeutil import decision_bucket, iso_from_epoch

# Fyers resolution string → seconds
RESOLUTION_SECS = {
    "5S": 5, "10S": 10, "15S": 15, "30S": 30, "45S": 45,
    "1": 60, "2": 120, "3": 180, "5": 300, "10": 600, "15": 900,
}


def resolution_to_secs(resolution: str) -> int:
    return RESOLUTION_SECS.get(str(resolution).upper(), 5)


# ──────────────────────────────────────────────
# Tick synthesis
# ──────────────────────────────────────────────

def synthesize_ticks(candles: List[Candle], resolution_secs: int, mode: str = "ohlc") -> List[Tick]:
    """Expand seconds candles into a deterministic tick stream.

    ``mode="close"`` → one tick per bar (at bar close).
    ``mode="ohlc"``  → four ticks per bar tracing O→L→H→C (up bars) /
                       O→H→L→C (down bars), with volume distributed across them.
    Volume is emitted as a running cumulative total so the profile can take a
    per-tick delta (and naturally degrades to 0 when bars carry no volume).
    """
    ticks: List[Tick] = []
    cum_vol = 0.0
    for c in candles:
        start = c.epoch
        if mode == "close":
            cum_vol += c.volume
            ticks.append(Tick(ts=start + resolution_secs, ltp=c.close, vol=cum_vol))
            continue
        seq = [c.open, c.low, c.high, c.close] if c.close >= c.open else [c.open, c.high, c.low, c.close]
        n = len(seq)
        per_v = c.volume / n
        for k, px in enumerate(seq):
            cum_vol += per_v
            ts = start + resolution_secs * (k + 1) / n
            ticks.append(Tick(ts=ts, ltp=float(px), vol=cum_vol))
    return ticks


def aggregate_decision_candles(candles: List[Candle], bucket_secs: int = 600) -> List[Tuple[float, Candle]]:
    """Roll seconds candles into ``bucket_secs`` (default 10m) OHLC candles.

    Returns ``[(end_epoch, Candle), …]`` so the iterator knows when each
    decision candle has *closed*.
    """
    buckets: dict[int, dict] = {}
    order: List[int] = []
    for c in candles:
        b = decision_bucket(c.epoch, bucket_secs)
        if b not in buckets:
            buckets[b] = {"o": c.open, "h": c.high, "l": c.low, "c": c.close, "v": c.volume}
            order.append(b)
        else:
            bk = buckets[b]
            bk["h"] = max(bk["h"], c.high)
            bk["l"] = min(bk["l"], c.low)
            bk["c"] = c.close
            bk["v"] += c.volume
    out: List[Tuple[float, Candle]] = []
    for b in order:
        bk = buckets[b]
        out.append((b + bucket_secs, Candle(
            timestamp=iso_from_epoch(b), open=bk["o"], high=bk["h"],
            low=bk["l"], close=bk["c"], volume=bk["v"], epoch=b,
        )))
    return out


def iter_session(candles: List[Candle], *, resolution_secs: int, decision_secs: int = 600,
                 mode: str = "ohlc") -> Iterator[Tuple[str, object]]:
    """Yield ``("candle", Candle)`` and ``("tick", Tick)`` in strict time order.

    A decision candle is emitted as soon as its close epoch is reached, so the
    engine's bias is set *before* the ticks of the next candle arrive.
    """
    decision = aggregate_decision_candles(candles, decision_secs)
    ticks = synthesize_ticks(candles, resolution_secs, mode)
    di = 0
    for tick in ticks:
        while di < len(decision) and decision[di][0] <= tick.ts:
            yield ("candle", decision[di][1])
            di += 1
        yield ("tick", tick)
    while di < len(decision):
        yield ("candle", decision[di][1])
        di += 1


# ──────────────────────────────────────────────
# Headless drivers
# ──────────────────────────────────────────────

def run_scalp(candles: List[Candle], symbol: str = "NIFTY_CE", config: ScalpConfig = None, *,
              resolution_secs: int = 5, decision_secs: int = 600, tick_synthesis: str = "ohlc") -> ScalpResult:
    """Feed a full session through the engine and return the result."""
    eng = ScalpEngine(symbol=symbol, config=config or ScalpConfig())
    for kind, item in iter_session(candles, resolution_secs=resolution_secs,
                                   decision_secs=decision_secs, mode=tick_synthesis):
        if kind == "candle":
            eng.on_candle_close(item)  # type: ignore[arg-type]
        else:
            eng.feed_tick(item)        # type: ignore[arg-type]
    return eng.result()


def run_scalp_with_chart_data(candles: List[Candle], symbol: str = "NIFTY_CE", config: ScalpConfig = None, *,
                              resolution_secs: int = 5, decision_secs: int = 600,
                              tick_synthesis: str = "ohlc") -> dict:
    """Run the session collecting a chart-ready payload (decision candles +
    entries + exits + KPIs + log). Used as a non-streaming fallback / for tests."""
    eng = ScalpEngine(symbol=symbol, config=config or ScalpConfig())
    decision_rows: List[dict] = []
    entries: List[dict] = []
    exits: List[dict] = []
    for kind, item in iter_session(candles, resolution_secs=resolution_secs,
                                   decision_secs=decision_secs, mode=tick_synthesis):
        if kind == "candle":
            for e in eng.on_candle_close(item):  # type: ignore[arg-type]
                if e.type == "candle":
                    row = dict(e.payload)
                    if eng._last_snap is not None:
                        row["pressure"] = round(eng._last_snap.pressure, 4)
                    decision_rows.append(row)
        else:
            for e in eng.feed_tick(item):        # type: ignore[arg-type]
                if e.type == "decision" and e.payload.get("action") in ("enter_long", "enter_short", "add"):
                    entries.append({"t": iso_from_epoch(e.ts), **e.payload})
                elif e.type == "exit":
                    exits.append({"t": iso_from_epoch(e.ts), **e.payload})
    res = eng.result()
    return {
        "symbol": symbol,
        "kpis": eng.kpis(),
        "candles": decision_rows,
        "entries": entries,
        "exits": exits,
        "exit": exits[-1] if exits else None,
        "log": res.log,
    }


def format_result(res: ScalpResult) -> str:
    """Pretty-print a scalp result for the CLI."""
    L = ["=" * 70, f"  SCALP STRATEGY RESULT — {res.symbol}", "=" * 70]
    if not res.entries:
        L += ["  No trades taken.", "=" * 70]
        return "\n".join(L)
    L.append(f"  Entries: {len(res.entries)} | Round-trips: {res.trades}")
    for i, e in enumerate(res.entries):
        L.append(f"    [{i+1}] {e.reason:>22} {e.side:<5} @ {e.price:>8.2f} | "
                 f"{e.lots:>3} lots | SL {e.sl_at_entry:.2f} | {e.timestamp}")
    L.append("-" * 70)
    if res.open_side != "FLAT":
        L.append(f"  Still holding: {res.open_side} {res.open_lots} lots")
    L.append(f"  Last exit:    {res.exit_price:.2f} ({res.exit_reason}) {res.exit_time}")
    L.append("-" * 70)
    icon = "+" if res.realized_pnl >= 0 else ""
    L.append(f"  P&L (points): {icon}{res.realized_pnl:.2f}")
    L.append(f"  P&L (rupees): {icon}{res.realized_inr:,.0f}")
    L.append(f"  Peak unreal:  +{res.peak_unrealized:.2f} pts ({res.peak_unrealized*res.lot_size:+,.0f} INR)")
    L.append(f"  Peak lots:    {res.peak_lots}")
    L += ["=" * 70, f"\n  TRADE LOG ({len(res.log)} events):", "-" * 70]
    L += [f"  {ln}" for ln in res.log]
    L.append("=" * 70)
    return "\n".join(L)


# ──────────────────────────────────────────────
# Deterministic sample (dry-run, no broker needed)
# ──────────────────────────────────────────────

def generate_scalp_sample(resolution_secs: int = 5, candles: int = 4500) -> List[Candle]:
    """Synthetic seconds candles for the FULL trading session (09:15→15:30 IST
    by default — 4500 × 5s), tracing the reference 23800 CE shape: a base near
    258, a momentum leg to ~425, chop, then a fade back toward ~385. Seeded, so
    dry-runs are reproducible. ``candles`` caps the length (tests use a short
    window)."""
    import random
    from datetime import datetime

    from .timeutil import IST

    random.seed(2380)
    # session open 09:15 IST so the sample aligns to real decision-candle grid
    base = datetime(2026, 6, 24, 9, 15, tzinfo=IST)
    epoch0 = base.timestamp()
    price = 258.0
    # (fraction_of_session, per-bar drift, volatility) — scaled to `candles`
    shape = [
        (0.07, 0.00, 0.8),   # base / quiet open
        (0.20, 0.18, 1.0),   # momentum leg up
        (0.13, 0.04, 0.9),   # pause / acceptance
        (0.13, 0.10, 1.0),   # second leg up
        (0.13, -0.03, 1.1),  # top consolidation
        (0.34, -0.02, 1.2),  # afternoon chop / fade
    ]
    out: List[Candle] = []
    i = 0
    for frac, drift, vol in shape:
        n = max(1, round(frac * candles))
        for _ in range(n):
            if i >= candles:
                break
            ep = epoch0 + i * resolution_secs
            o = price
            c = o + drift + random.gauss(0, vol)
            h = max(o, c) + abs(random.gauss(0, vol * 0.5))
            lo = min(o, c) - abs(random.gauss(0, vol * 0.5))
            out.append(Candle(
                timestamp=iso_from_epoch(ep), open=round(o, 2), high=round(h, 2),
                low=round(lo, 2), close=round(c, 2),
                volume=random.randint(2000, 18000), epoch=ep,
            ))
            price = c
            i += 1
    return out
