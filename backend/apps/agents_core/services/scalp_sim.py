"""Scalp simulation playback controller.

Owns a :class:`ScalpEngine` and a materialised session (interleaved decision
candles + ticks from :func:`iter_session`), and streams it to a single WebSocket
client at a controllable speed with pause / step / manual-override support.

Consumer-owned asyncio loop (the OpsConsumer pattern) — no Celery, no channel
layer, so it works on the dev default in-memory layer. The same engine runs on a
live Fyers feed later by swapping the tick source for the controller's session.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Awaitable, Callable

from plugins.strategy_scalp.timeutil import decision_bucket, iso_from_epoch


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _make_scalp_config(config: dict):
    from plugins.strategy_scalp.engine import ScalpConfig
    return ScalpConfig(
        lot_size=config.get("lot_size", 65), initial_capital=config.get("capital", 100_000),
        initial_risk_pct=config.get("risk_pct", 1.0), max_pyramids=config.get("max_pyramids", 4),
        bin_width=config.get("bin_width", 20.0), window_secs=config.get("window_secs", 180),
        value_area_pct=config.get("value_area_pct", 0.70),
        entry_threshold=config.get("entry_threshold", 0.30),
        add_threshold=config.get("add_threshold", 0.22),
        reverse_threshold=config.get("reverse_threshold", 0.55),
        reentry_cooldown_secs=config.get("reentry_cooldown_secs", 60.0),
        trend_only=config.get("trend_only", True),
        trend_ema_span=config.get("trend_ema_span", 5),
        trend_flat_band=config.get("trend_flat_band", 0.012),
        bin_reversal_entry=config.get("bin_reversal_entry", True),
        pullback_min_bins=config.get("pullback_min_bins", 0.6),
        reversal_bins=config.get("reversal_bins", 0.35),
        require_bias_alignment=config.get("require_bias_alignment", True),
        allow_reverse=config.get("allow_reverse", True),
    )


def _display_symbol(config: dict, suffix: str = "") -> str:
    s = f"{config.get('underlying', 'NIFTY')} {config.get('strike')} {config.get('type', 'CE')}"
    return s + suffix


def build_session(config: dict, tenant=None):
    """Resolve config → (engine, materialised session, ticks_per_bar, decision_secs, source).

    Synchronous (DB + broker I/O); call via ``sync_to_async`` from the consumer.
    Raises on a failed live fetch so the caller can fall back to the sample.
    """
    from plugins.strategy_scalp.engine import ScalpEngine
    from plugins.strategy_scalp.replay import generate_scalp_sample, iter_session, resolution_to_secs

    res = config.get("resolution", "5S")
    res_secs = resolution_to_secs(res)
    mode = config.get("tick_synthesis", "ohlc")
    ticks_per_bar = 4 if mode == "ohlc" else 1
    decision_secs = int(config.get("decision_secs", 600))

    source = "sample"
    candles = None
    meta = {"date": "", "expiry": ""}
    if not config.get("dry_run"):
        from plugins.strategy_scalp.data import fetch_scalp_candles, resolve_session
        candles = fetch_scalp_candles(
            underlying=config.get("underlying", "NIFTY"), strike=config["strike"],
            opt_type=config.get("type", "CE"), expiry=config.get("expiry", ""),
            date=config.get("date", ""), resolution=res, tenant=tenant,
        )
        day, exp = resolve_session(config.get("underlying", "NIFTY"), config.get("expiry", ""), config.get("date", ""))
        meta = {"date": day, "expiry": exp.strftime("%d-%b-%Y")}
        source = "fyers"
    if not candles:
        candles = generate_scalp_sample(res_secs, int(config.get("sample_candles", 4500)))
        source = "sample"
        meta = {"date": "2026-06-24", "expiry": "sample"}

    eng = ScalpEngine(symbol=_display_symbol(config, "" if source == "fyers" else " (sample)"),
                      config=_make_scalp_config(config))
    session = list(iter_session(candles, resolution_secs=res_secs,
                                decision_secs=decision_secs, mode=mode))
    return eng, session, ticks_per_bar, decision_secs, source, meta


def build_live_session(config: dict, tenant):
    """Resolve config → (engine, fyers_symbol, fyers_creds, decision_secs).

    For LIVE mode: resolves the option's Fyers ticker + the tenant's Fyers creds
    so the consumer can open a live tick socket. Raises if Fyers is unavailable.
    """
    from datetime import date as _date

    from plugins.strategy_scalp.engine import ScalpEngine
    from plugins.broker_fyers.symbols import resolve_option_symbol
    from plugins.strategy_scalp.data import _fyers_link, resolve_expiry
    from apps.market_data.adapters.factory import build_adapter

    # Live = today's session; empty expiry → nearest listed weekly/monthly.
    exp = resolve_expiry(config.get("underlying", "NIFTY"), config.get("expiry", ""), _date.today())
    fyers_symbol = resolve_option_symbol(
        config.get("underlying", "NIFTY"), exp, int(config["strike"]), config.get("type", "CE"))
    link = _fyers_link(tenant)
    adapter = build_adapter(link)
    if not adapter.authenticate():
        raise RuntimeError("Fyers daily re-login required — re-auth the Fyers broker link.")
    creds = adapter.credentials
    if not creds.get("app_id") or not creds.get("access_token"):
        raise RuntimeError("Fyers credentials incomplete.")

    decision_secs = int(config.get("decision_secs", 600))
    eng = ScalpEngine(symbol=_display_symbol(config, " · LIVE"), config=_make_scalp_config(config))
    return eng, fyers_symbol, creds, decision_secs


class ScalpPlaybackController:
    def __init__(self, *, engine, session, decision_secs: int, ticks_per_bar: int,
                 send: Callable[[dict], Awaitable[None]], base_interval: float = 0.1,
                 speed: float = 4.0):
        self.engine = engine
        self.session = session                 # list[(kind, item)]
        self.decision_secs = decision_secs
        self.ticks_per_bar = max(1, ticks_per_bar)
        self.send = send
        self.base_interval = base_interval
        self.speed = _clamp(speed, 0.25, 64.0)

        self.cursor = 0
        self._seq = 0
        self._run = asyncio.Event()
        self._run.set()                        # set = playing, clear = paused
        self._step_remaining = 0
        self._manual: deque[tuple[str, dict]] = deque()
        self._annotations: list[dict] = []
        self._stopped = False
        # forming 10m candle (tick-derived) for a smoothly animating chart
        self._forming_bucket: int | None = None
        self._forming: dict | None = None

    # ── control (called from the consumer's receive_json) ──────────────
    def control(self, op: str, payload: dict) -> None:
        if op == "pause":
            self._run.clear()
        elif op == "resume":
            self._run.set()
        elif op == "speed":
            self.speed = _clamp(float(payload.get("value", self.speed)), 0.25, 64.0)
        elif op == "step":
            self._step_remaining = self.ticks_per_bar
            self._run.set()
        elif op in ("manual_order", "adjust_sl"):
            self._manual.append((op, payload))
            self._run.set()                    # let the loop apply it promptly
        elif op == "annotate":
            self._annotations.append({"candle_ts": payload.get("candle_ts"), "note": payload.get("note", "")})
        elif op == "stop":
            self._stopped = True
            self._run.set()

    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    # ── playback loop ──────────────────────────────────────────────────
    async def run(self) -> dict:
        await self._emit("meta", {
            "symbol": self.engine.symbol, "decision_secs": self.decision_secs,
            "bin_width": self.engine.cfg.bin_width, "total": len(self.session),
            "speed": self.speed,
        })
        while self.cursor < len(self.session) and not self._stopped:
            if self._step_remaining <= 0:
                await self._run.wait()
            if self._stopped:
                break

            await self._drain_manual()

            kind, item = self.session[self.cursor]
            if kind == "candle":
                for ev in self.engine.on_candle_close(item):
                    await self._emit_event(ev)
            else:
                await self._update_forming(item)
                await self._emit("tick", {"ts": item.ts, "ltp": round(item.ltp, 2)})
                for ev in self.engine.feed_tick(item):
                    await self._emit_event(ev)
                if self._step_remaining > 0:
                    self._step_remaining -= 1
                    if self._step_remaining == 0:
                        self._run.clear()      # re-pause after a stepped bar
            self.cursor += 1
            await asyncio.sleep(self.base_interval / max(self.speed, 1e-6))

        return await self._finish()

    # ── internals ──────────────────────────────────────────────────────
    async def _drain_manual(self) -> None:
        if not self._manual:
            return
        ts = self.engine._last_ts or 0.0
        while self._manual:
            op, payload = self._manual.popleft()
            if op == "manual_order":
                evs = self.engine.apply_manual(
                    payload.get("action", "exit"), ts,
                    lots=payload.get("lots"), price=payload.get("price"), sl=payload.get("sl"),
                )
            else:  # adjust_sl
                evs = self.engine.apply_manual(
                    "adjust_sl", ts, sl=payload.get("value", payload.get("price")),
                )
            for ev in evs:
                await self._emit_event(ev)

    async def _emit_event(self, ev) -> None:
        # Engine events carry their own ts — markers/log on the client need it.
        await self._emit(ev.type, {**ev.payload, "ts": ev.ts})

    async def _update_forming(self, tick) -> None:
        b = decision_bucket(tick.ts, self.decision_secs)
        if b != self._forming_bucket:
            self._forming_bucket = b
            self._forming = {"o": tick.ltp, "h": tick.ltp, "l": tick.ltp, "c": tick.ltp}
        else:
            f = self._forming
            f["h"] = max(f["h"], tick.ltp)
            f["l"] = min(f["l"], tick.ltp)
            f["c"] = tick.ltp
        f = self._forming
        await self._emit("candle", {
            "t": iso_from_epoch(b), "epoch": b, "forming": True,
            "o": round(f["o"], 2), "h": round(f["h"], 2), "l": round(f["l"], 2), "c": round(f["c"], 2),
            "bias": self.engine.st.bias,
        })

    async def _emit(self, type_: str, payload: dict) -> None:
        self._seq += 1
        await self.send({"type": type_, "seq": self._seq, **payload})

    async def _finish(self) -> dict:
        # Note: the consumer emits "done" AFTER persisting, so a fast client
        # close can't cancel the persist and leave the run stuck "running".
        res = self.engine.result()
        return {
            "kpis": self.engine.kpis(),
            "entries": [vars(e) for e in res.entries],
            "exit": {"price": res.exit_price, "time": res.exit_time, "reason": res.exit_reason},
            "annotations": self._annotations,
            "log": res.log,
            "stopped": self._stopped,
        }


class LiveScalpController:
    """Real-time variant: consumes live Fyers ticks (pushed from the socket
    thread into an asyncio queue), feeds the SAME engine, aggregates 10-min
    decision candles on the fly, and optionally routes opening decisions to a
    paper-order callback. No speed/step — it's wall-clock real-time.
    """

    def __init__(self, *, engine, decision_secs: int, send: Callable[[dict], Awaitable[None]],
                 loop, on_decision: Callable[[dict], Awaitable[None]] | None = None):
        self.engine = engine
        self.decision_secs = decision_secs
        self.send = send
        self.loop = loop
        self.on_decision = on_decision
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=20_000)
        self._seq = 0
        self._stopped = False
        self._paused = False
        self._manual: deque[tuple[str, dict]] = deque()
        self._annotations: list[dict] = []
        self._forming_bucket: int | None = None
        self._forming: dict | None = None

    def control(self, op: str, payload: dict) -> None:
        if op == "pause":
            self._paused = True
        elif op == "resume":
            self._paused = False
        elif op in ("manual_order", "adjust_sl"):
            self._manual.append((op, payload))
        elif op == "annotate":
            self._annotations.append({"candle_ts": payload.get("candle_ts"), "note": payload.get("note", "")})
        elif op == "stop":
            self._stopped = True
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def push_tick(self, raw: dict) -> None:
        """Called from the Fyers socket thread — schedule onto the loop thread."""
        if self._stopped or self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._enqueue, raw)

    def _enqueue(self, raw) -> None:
        try:
            self.queue.put_nowait(raw)
        except asyncio.QueueFull:
            pass

    async def run(self) -> dict:
        import time as _time

        from plugins.strategy_scalp.engine import Candle, Tick

        await self._emit("meta", {"symbol": self.engine.symbol, "decision_secs": self.decision_secs,
                                  "bin_width": self.engine.cfg.bin_width, "live": True})
        while not self._stopped:
            raw = await self.queue.get()
            if raw is None:
                break
            await self._drain_manual()
            ts = float(raw.get("ts") or 0) or _time.time()
            tick = Tick(ts=ts, ltp=float(raw["ltp"]), vol=float(raw.get("vol") or 0))

            # close the previous 10-min candle when a new bucket starts
            b = decision_bucket(ts, self.decision_secs)
            if self._forming_bucket is not None and b > self._forming_bucket and self._forming:
                f = self._forming
                candle = Candle(timestamp=iso_from_epoch(self._forming_bucket), open=f["o"], high=f["h"],
                                low=f["l"], close=f["c"], epoch=self._forming_bucket)
                for ev in self.engine.on_candle_close(candle):
                    await self._emit_event(ev)
                self._forming = None

            await self._update_forming(tick)
            await self._emit("tick", {"ts": tick.ts, "ltp": round(tick.ltp, 2)})
            if not self._paused:
                for ev in self.engine.feed_tick(tick):
                    await self._emit_event(ev)
                    if ev.type == "decision" and self.on_decision:
                        await self.on_decision(ev.payload)
        return await self._finish()

    # shared-shape helpers (mirror ScalpPlaybackController)
    async def _drain_manual(self) -> None:
        if not self._manual:
            return
        ts = self.engine._last_ts or 0.0
        while self._manual:
            op, payload = self._manual.popleft()
            if op == "manual_order":
                evs = self.engine.apply_manual(payload.get("action", "exit"), ts,
                                               lots=payload.get("lots"), price=payload.get("price"),
                                               sl=payload.get("sl"))
            else:
                evs = self.engine.apply_manual("adjust_sl", ts, sl=payload.get("value", payload.get("price")))
            for ev in evs:
                await self._emit_event(ev)
                if ev.type == "decision" and self.on_decision:
                    await self.on_decision(ev.payload)

    async def _emit_event(self, ev) -> None:
        await self._emit(ev.type, {**ev.payload, "ts": ev.ts})

    async def _update_forming(self, tick) -> None:
        b = decision_bucket(tick.ts, self.decision_secs)
        if b != self._forming_bucket:
            self._forming_bucket = b
            self._forming = {"o": tick.ltp, "h": tick.ltp, "l": tick.ltp, "c": tick.ltp}
        else:
            f = self._forming
            f["h"] = max(f["h"], tick.ltp)
            f["l"] = min(f["l"], tick.ltp)
            f["c"] = tick.ltp
        f = self._forming
        await self._emit("candle", {"t": iso_from_epoch(b), "epoch": b, "forming": True,
                                    "o": round(f["o"], 2), "h": round(f["h"], 2),
                                    "l": round(f["l"], 2), "c": round(f["c"], 2),
                                    "bias": self.engine.st.bias})

    async def _emit(self, type_: str, payload: dict) -> None:
        self._seq += 1
        await self.send({"type": type_, "seq": self._seq, **payload})

    async def _finish(self) -> dict:
        res = self.engine.result()
        return {"kpis": self.engine.kpis(), "entries": [vars(e) for e in res.entries],
                "annotations": self._annotations, "log": res.log, "live": True,
                "stopped": self._stopped}
