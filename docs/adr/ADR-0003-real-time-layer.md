# ADR-0003: Real-time layer (Channels + Redis)

- Status: Accepted
- Date: 2026-04-18

## Context
The UI needs three live streams: market ticks, portfolio P&L, and agent-run tokens.
Polling is unacceptable (latency + load). SSE works for one-way but we also want
client→server messages (subscribe/unsubscribe).

## Decision
**Django Channels (ASGI) with Redis as the channel layer.**

- Channels consumers run in a separate ECS service group from REST.
- Redis pub/sub powers fan-out; Redis Streams power the broker→worker tick pipeline.
- WS auth via JWT in the first message; reject after 5s if not authenticated.
- Per-tenant groups enforce isolation at the layer boundary.

## Key designs

### Tick fan-out
1. `broker.market_data.worker` holds the broker WS connection.
2. Normalizes ticks → publishes to Redis Stream `ticks.global`.
3. `dispatcher.worker` reads the stream, looks up subscribed tenants per token,
   writes to channel groups `tenant.{id}.ticks`.
4. Channels consumers fan out to user sockets.

This decouples broker WS health from user-facing WS health.

### Agent-run streaming
LangGraph nodes call `ctx.publisher.emit(node, payload)`. The publisher writes to
`tenant.{tid}.agent.{run_id}`. The frontend opens that channel when a run starts.
On disconnect, the run continues; reconnect replays from the last `seq` (stored in Redis).

### Backpressure
- Per-socket send queue capped at 1k messages → drop oldest tick frames (never alerts/agent).
- Rate-limit subscribe calls per socket.

## Alternatives considered
- **Server-Sent Events.** Simpler, but one-way. We want subscribe/unsubscribe and ping/pong.
- **Pure WebSocket without Channels.** Rolling our own group/auth is reinventing wheels.
- **MQTT.** Overkill for browser clients; great if we add a mobile/IoT layer later.

## Consequences
- Separate service-group means we can scale REST and WS independently.
- Redis becomes a critical dependency; multi-AZ ElastiCache with failover.
- WS disconnects on deploy → graceful drain hook waits for `<5s` of message idle.
