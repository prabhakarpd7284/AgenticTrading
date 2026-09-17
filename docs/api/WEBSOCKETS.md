# WebSocket channel reference

All WS endpoints are under `/ws/` on the same ASGI host as the REST API.

## Auth

Send this frame immediately after connection:

```json
{ "op": "auth", "token": "<JWT access token>" }
```

If the token is invalid or missing `tenant_id`, the server closes with **4401**.

## Channels

### `/ws/ticks/`
Server-push streaming market ticks. Per-socket subscribe/unsubscribe list.

**Client → server:**
```json
{ "op": "subscribe",   "tokens": ["NSE:HDFCBANK", "NSE:RELIANCE"] }
{ "op": "unsubscribe", "tokens": ["NSE:RELIANCE"] }
```

**Server → client (tick):**
```json
{ "ts": 1729000000.123, "exchange": "NSE", "token": "1333",
  "ltp": 1492.30, "oi": null, "vol": 1_245_000 }
```

### `/ws/pnl/`
Per-tenant, server-push 1s throttled P&L.

**Server → client:**
```json
{ "portfolio_id": "…", "mtm": 9420.00, "day_pnl": -1200.00,
  "unrealized": 1540.00 }
```

### `/ws/agents/{run_id}/`
Streams `AgentEvent`s from a running agent. Sequence is monotonically increasing;
reconnect-after-disconnect replays from last `seq` in client memory.

**Server → client:**
```json
{ "seq": 7, "node": "planner", "type": "result",
  "payload": { "symbol": "HDFCBANK", "side": "BUY", "qty": 10, "sl": 1460, "tp": 1520 } }
```

Possible `type` values: `token` (LLM streaming chunk), `state` (intermediate
state patch), `result` (node-terminal output), `info`, `error`.

### `/ws/alerts/`
Server-push notifications (order status, RiskGuard blocks, system alerts).

**Server → client:**
```json
{ "id": "…", "severity": "warn",
  "title": "Daily loss limit reached",
  "body": "New entries blocked until 00:00 IST." }
```

## Backpressure

Each socket has a 1000-message outbound buffer. When full, tick frames are dropped
(oldest first). Agent/alert frames are never dropped — if the buffer is full, the
socket is closed with **1013 (try again later)**.

## Heartbeats

Client MUST send `{"op":"ping"}` every 30 s; server replies `{"op":"pong"}`.
After 60 s of silence the server closes the socket.
