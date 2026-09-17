"""End-to-end test for the interactive scalp simulation WebSocket.

Drives ScalpSimConsumer with a Channels WebsocketCommunicator (no running
server). Validates: the start handshake, the streaming frame protocol
(meta/tick/candle/pressure/decision/position), that the engine actually trades
on the dry-run sample over the wire, and that manual override flows through.
"""
from __future__ import annotations

import pytest
from asgiref.sync import sync_to_async
from channels.testing import WebsocketCommunicator

from apps.accounts.models import User
from apps.accounts.services.tenant_bootstrap import ensure_tenant
from apps.agents_core.models import AgentRun
from apps.agents_core.scalp_consumer import ScalpSimConsumer

pytestmark = pytest.mark.django_db


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@sync_to_async
def _setup(email: str, config: dict):
    user = User.objects.create_user(email=email, password="longpassword123")
    tenant = ensure_tenant(user)
    from apps.trading.models import Portfolio
    Portfolio.objects.create(tenant=tenant, name="Default", capital=500000, mode="paper")
    run = AgentRun.objects.create(
        tenant=tenant, triggered_by=user, strategy_name="scalp",
        strategy_version="1.0.0",
        portfolio=Portfolio.objects.filter(tenant=tenant).first(),
        config=config, status=AgentRun.Status.QUEUED,
    )
    return user, tenant, run


async def _open(user, tenant, run):
    comm = WebsocketCommunicator(ScalpSimConsumer.as_asgi(), f"/ws/scalp/{run.id}/")
    comm.scope["url_route"] = {"args": (), "kwargs": {"run_id": str(run.id)}}
    comm.scope["user"] = user
    comm.scope["tenant"] = tenant
    comm.scope["subprotocols"] = ["jwt"]
    connected, _ = await comm.connect()
    assert connected
    return comm


async def _collect(comm, *, until_type=None, cap=1200):
    """Read frames until we've seen `until_type` (or cap), returning all seen."""
    seen = []
    for _ in range(cap):
        msg = await comm.receive_json_from(timeout=5)
        seen.append(msg)
        if msg.get("type") == "done":
            break
        if until_type and msg.get("type") == until_type:
            break
    return seen


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sim_streams_and_trades_on_sample(db):
    user, tenant, run = await _setup("scalp1@x.io",
                                     {"strike": 23800, "type": "CE", "underlying": "NIFTY",
                                      "dry_run": True, "speed": 64, "sample_candles": 300, "trend_only": False})
    comm = await _open(user, tenant, run)
    await comm.send_json_to({"op": "start"})

    started = await comm.receive_json_from(timeout=5)
    assert started["type"] == "started" and started["source"] == "sample"

    frames = await _collect(comm, until_type="decision")
    types = {f["type"] for f in frames}
    assert "meta" in types
    assert "tick" in types
    assert "pressure" in types
    # the bullish sample must produce at least one entry over the wire
    decisions = [f for f in frames if f["type"] == "decision"]
    assert decisions, "no decision frame streamed"
    assert decisions[-1]["action"] in ("enter_long", "enter_short", "add")
    # every frame carries a monotonic seq for reconnect dedup
    seqs = [f["seq"] for f in frames]
    assert seqs == sorted(seqs)

    await comm.send_json_to({"op": "stop"})
    await comm.disconnect()


@pytest.mark.asyncio
async def test_manual_override_over_ws(db):
    user, tenant, run = await _setup("scalp2@x.io",
                                     {"strike": 23800, "type": "CE", "underlying": "NIFTY",
                                      "dry_run": True, "speed": 64, "sample_candles": 300,
                                      "require_bias_alignment": False, "trend_only": False})
    comm = await _open(user, tenant, run)
    await comm.send_json_to({"op": "start"})
    await comm.receive_json_from(timeout=5)  # started

    # let a few ticks flow so the engine has an LTP, then force a manual short
    await _collect(comm, until_type="tick", cap=20)
    await comm.send_json_to({"op": "manual_order", "action": "short", "lots": 3, "sl": 999})

    frames = await _collect(comm, until_type="decision", cap=400)
    manual = [f for f in frames if f["type"] == "decision" and f.get("reason", "").startswith("manual")]
    assert manual, "manual order did not produce a decision frame"
    assert manual[0]["side"] == "SHORT"
    assert manual[0]["lots"] == 3

    await comm.send_json_to({"op": "stop"})
    await comm.disconnect()


@pytest.mark.asyncio
async def test_live_controller_processes_ticks_and_decides():
    """LiveScalpController: queued ticks → frames + an opening decision routed to
    the order callback. No DB / no real socket — pure controller logic."""
    import asyncio

    from apps.agents_core.services.scalp_sim import LiveScalpController
    from plugins.strategy_scalp.engine import ScalpConfig, ScalpEngine

    frames: list = []
    decisions: list = []

    async def send(m):
        frames.append(m)

    async def on_dec(p):
        decisions.append(p)

    loop = asyncio.get_running_loop()
    eng = ScalpEngine("T", ScalpConfig(require_bias_alignment=False, trend_only=False, min_sl_distance=1.0))
    ctrl = LiveScalpController(engine=eng, decision_secs=600, send=send, loop=loop, on_decision=on_dec)
    task = asyncio.create_task(ctrl.run())

    px, ts = 280.0, 1_700_000_000.0
    for i in range(160):
        ctrl._enqueue({"ltp": px, "vol": 0, "ts": ts + i * 2})
        px += 1.0
    await asyncio.sleep(0.5)  # let the loop drain the queue
    ctrl.control("stop", {})
    await task

    types = {f["type"] for f in frames}
    assert {"meta", "tick", "candle", "pressure"} <= types
    assert decisions, "no opening decision routed to on_decision"
    assert decisions[0]["action"] in ("enter_long", "enter_short", "add")


@pytest.mark.asyncio
async def test_persists_result_on_completion(db):
    """Run to completion at max speed; AgentRun.result + status must be set."""
    user, tenant, run = await _setup("scalp3@x.io",
                                     {"strike": 23800, "type": "CE", "underlying": "NIFTY",
                                      "dry_run": True, "speed": 64, "sample_candles": 300, "trend_only": False})
    comm = await _open(user, tenant, run)
    await comm.send_json_to({"op": "start"})
    frames = await _collect(comm, cap=20000)  # drain to "done"
    assert frames[-1]["type"] == "done"
    await comm.disconnect()

    refreshed = await sync_to_async(lambda: AgentRun.objects.get(pk=run.id))()
    assert refreshed.status == "succeeded"
    assert refreshed.result and "kpis" in refreshed.result
