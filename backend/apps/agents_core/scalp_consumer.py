"""WebSocket consumer driving an interactive scalp simulation.

One simulation per connection (the OpsConsumer model): the consumer owns an
asyncio playback task and forwards client control ops to a
:class:`ScalpPlaybackController`. No channel layer / Celery — works on the dev
in-memory layer.

  client → server   {"op": "start"}                              (config comes from the AgentRun)
                    {"op": "pause" | "resume" | "step" | "stop"}
                    {"op": "speed", "value": 4.0}
                    {"op": "manual_order", "action": "buy"|"short"|"add"|"exit"|"flatten",
                                            "lots"?, "price"?, "sl"?}
                    {"op": "adjust_sl", "value": 312.5}
                    {"op": "annotate", "candle_ts": "...", "note": "..."}
  server → client   {"type": "meta"|"tick"|"candle"|"pressure"|"decision"|
                              "position"|"exit"|"log"|"done"|"error"|"warn", "seq": n, …}

Auth: JWT subprotocol (same as the other consumers); tenant-scoped to the run.
"""
from __future__ import annotations

import asyncio

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from apps.agents_core.services.scalp_sim import (
    LiveScalpController, ScalpPlaybackController, build_live_session, build_session,
)


class ScalpSimConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self) -> None:
        user = self.scope.get("user")
        if user is None or getattr(user, "is_anonymous", True):
            await self.close(code=4401)
            return
        tenant = self.scope.get("tenant")
        if tenant is None:
            await self.close(code=4403)
            return
        self.tenant = tenant
        self.run_id = str(self.scope["url_route"]["kwargs"]["run_id"])
        subprotocol = "jwt" if "jwt" in (self.scope.get("subprotocols") or []) else None
        await self.accept(subprotocol=subprotocol)
        self.controller = None
        self._task: asyncio.Task | None = None
        self._stream = None
        self._completed = False

    async def disconnect(self, code: int) -> None:
        if self.controller:
            self.controller.control("stop", {})
        if self._stream:
            self._stream.stop()
        if self._task and not self._task.done():
            self._task.cancel()
        # An abandoned sim (navigated away mid-run) must not stay stuck "running".
        if not self._completed:
            await self._cancel_if_running()

    async def _cancel_if_running(self) -> None:
        if not getattr(self, "tenant", None) or not getattr(self, "run_id", None):
            return
        from apps.agents_core.models import AgentRun

        def _upd():
            AgentRun.objects.filter(tenant=self.tenant, pk=self.run_id, status="running").update(status="cancelled")
        await sync_to_async(_upd)()

    async def receive_json(self, content: dict, **kwargs) -> None:
        op = content.get("op")
        if op == "start":
            await self._start()
        elif self.controller is not None:
            # All transport / trade / annotate ops route to the controller.
            self.controller.control(op, content)
        # Unknown ops before start are ignored — keeps the protocol forgiving.

    # ── start handshake ────────────────────────────────────────────────
    async def _start(self) -> None:
        if self._task is not None and not self._task.done():
            await self.send_json({"type": "error", "detail": "simulation already running"})
            return

        run = await self._load_run()
        if run is None:
            await self.send_json({"type": "error", "detail": "run not found"})
            return
        config = run.config or {}
        self._run = run

        if config.get("mode") == "live":
            await self._start_live(config)
            return

        try:
            engine, session, tpb, dsec, source, meta = await sync_to_async(build_session)(config, self.tenant)
        except Exception as exc:  # noqa: BLE001 — data fetch failed
            # Never silently present synthetic data as real. Fall back to the
            # sample ONLY when the caller explicitly asked for dry_run; otherwise
            # surface the failure so misleading KPIs can't slip past a reviewer.
            if not config.get("dry_run"):
                await self.send_json({"type": "error", "detail": f"Fyers data unavailable: {exc}"})
                return
            await self.send_json({"type": "warn", "detail": f"Fyers data unavailable ({exc}); using sample."})
            try:
                engine, session, tpb, dsec, source, meta = await sync_to_async(build_session)(
                    {**config, "dry_run": True}, None)
                source = "sample (fallback)"
            except Exception as exc2:  # noqa: BLE001
                await self.send_json({"type": "error", "detail": f"could not build session: {exc2}"})
                return

        await self._set_status("running")
        await self.send_json({"type": "started", "source": source, "bars": len(session),
                              "date": meta.get("date", ""), "expiry": meta.get("expiry", "")})

        self.controller = ScalpPlaybackController(
            engine=engine, session=session, decision_secs=dsec, ticks_per_bar=tpb,
            send=self.send_json, speed=float(config.get("speed", 4.0)),
        )
        self._task = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        try:
            result_blob = await self.controller.run()
            # persist BEFORE emitting done so the client's close-on-done can't
            # race the DB write (which used to leave runs stuck "running").
            await self._persist(result_blob, "succeeded")
            self._completed = True
            await self.send_json({"type": "done", "kpis": result_blob.get("kpis", {}),
                                  "stopped": result_blob.get("stopped", False)})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            try:
                await self.send_json({"type": "error", "detail": str(exc)})
                await self._set_status("failed")
            except Exception:  # noqa: BLE001
                pass

    # ── live mode ──────────────────────────────────────────────────────
    async def _start_live(self, config: dict) -> None:
        from plugins.broker_fyers.stream import FyersTickStream

        try:
            engine, fyers_symbol, creds, dsec = await sync_to_async(build_live_session)(config, self.tenant)
        except Exception as exc:  # noqa: BLE001
            await self.send_json({"type": "error", "detail": f"live setup failed: {exc}"})
            return

        self._engine = engine
        self._fyers_symbol = fyers_symbol
        self._lot_size = int(config.get("lot_size", 65))
        self._place_orders = bool(config.get("place_orders", False))
        self._user = self.scope.get("user")
        self._portfolio = await sync_to_async(lambda: self._run.portfolio)()
        self._portfolio_mode = await sync_to_async(lambda: self._run.portfolio.mode)()
        self._order_n = 0

        await self._set_status("running")
        await self.send_json({"type": "started", "source": "fyers-live", "symbol": fyers_symbol,
                              "place_orders": self._place_orders, "portfolio_mode": self._portfolio_mode})

        loop = asyncio.get_running_loop()
        self.controller = LiveScalpController(
            engine=engine, decision_secs=dsec, send=self.send_json, loop=loop,
            on_decision=self._on_live_decision,
        )
        self._stream = FyersTickStream(creds["app_id"], creds["access_token"],
                                       on_tick=self.controller.push_tick)
        # the socket spawns its own thread; start it off the event loop
        await sync_to_async(self._stream.start)([fyers_symbol])
        self._task = asyncio.create_task(self._drive())

    async def _on_live_decision(self, payload: dict) -> None:
        """Opening decisions → advisory risk check, plus a real paper order when
        `place_orders` is enabled AND the portfolio is paper (never live-money)."""
        if payload.get("action") not in ("enter_long", "enter_short", "add"):
            return
        side = "BUY" if payload.get("side") == "LONG" else "SELL"
        entry = float(payload.get("price") or 0)
        sl = float(payload.get("sl") or 0)
        risk = abs(entry - sl) or max(1.0, entry * 0.05)
        tp = entry + 1.5 * risk if side == "BUY" else entry - 1.5 * risk
        qty = int(payload.get("lots", 1)) * self._lot_size

        placed_id = None
        try:
            if self._place_orders and self._portfolio_mode == "paper":
                placed_id, approved, reason = await sync_to_async(self._do_place)(side, entry, sl, tp, qty)
            else:
                approved, reason = await sync_to_async(self._risk_advisory)(side, entry, sl, tp, qty)
        except Exception as exc:  # noqa: BLE001
            approved, reason = False, str(exc)

        await self.send_json({
            "type": "order", "mode": "placed" if placed_id else "advisory",
            "action": payload.get("action"), "side": side, "qty": qty,
            "price": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "approved": approved, "reason": reason, "order_id": placed_id,
        })

    def _risk_advisory(self, side, entry, sl, tp, qty):
        from apps.trading.services.risk_engine import RiskEngine, TradeDraft
        snap = getattr(self._engine, "_last_snap", None)
        conf = max(0.55, 0.5 + 0.5 * abs(snap.pressure)) if snap else 0.55
        draft = TradeDraft(symbol=self._fyers_symbol, side=side, entry_price=entry,
                           stop_loss=sl, target=tp, quantity=qty, confidence=conf)
        d = RiskEngine().validate(draft, portfolio_id=self._portfolio.id)
        return d.approved, d.reason

    def _do_place(self, side, entry, sl, tp, qty):
        from apps.common.exceptions import RiskRejected
        from apps.trading.domain.entities import OrderDraft
        from apps.trading.services.place_order import PlaceOrder
        self._order_n += 1
        draft = OrderDraft(symbol=self._fyers_symbol, side=side, qty=qty, order_type="MARKET",
                           product="INTRADAY", price=entry, sl=sl, tp=tp, origin="strategy")
        try:
            res = PlaceOrder().execute(self.tenant, self._user, self._portfolio, draft,
                                       idempotency_key=f"scalp-{self.run_id}-{self._order_n}")
            return str(res.id), True, f"queued ({res.status})"
        except RiskRejected as e:
            return None, False, f"risk: {e}"

    # ── persistence ────────────────────────────────────────────────────
    async def _load_run(self):
        from apps.agents_core.models import AgentRun

        def _get():
            return AgentRun.objects.filter(tenant=self.tenant, pk=self.run_id,
                                           strategy_name="scalp").first()
        return await sync_to_async(_get)()

    async def _set_status(self, status: str) -> None:
        from apps.agents_core.models import AgentRun

        def _upd():
            AgentRun.objects.filter(tenant=self.tenant, pk=self.run_id).update(status=status)
        await sync_to_async(_upd)()

    async def _persist(self, result_blob: dict, status: str) -> None:
        from django.utils import timezone

        from apps.agents_core.models import AgentRun

        def _upd():
            AgentRun.objects.filter(tenant=self.tenant, pk=self.run_id).update(
                result=result_blob, status=status, completed_at=timezone.now())
        await sync_to_async(_upd)()
