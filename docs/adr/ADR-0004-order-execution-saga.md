# ADR-0004: Order execution via Outbox + Saga

- Status: Accepted
- Date: 2026-04-18

## Context
Placing an order touches: our DB, RiskGuard, broker API, journal, notifications.
A naive `place()` inside a request handler has three failure modes:
1. We commit the DB row but the broker call fails → ghost order.
2. The broker accepts but our response times out → user retries → duplicate order.
3. RiskGuard pass races with portfolio mutation → over-leverage.

## Decision
**Transactional Outbox + Saga orchestrator.**

### Flow
```
POST /orders ──┐
               │ (1) RiskGuard.validate (sync, in transaction)
               │ (2) Insert Order(status=QUEUED) AND Insert OutboxEvent in same TX
               ▼
            HTTP 202 + order_id
                                  ┌─ Celery beat polls Outbox every 1s
                                  │
                                  ▼
                         OrderSaga.handle(event):
                            ├─ broker.place()
                            ├─ on success: Order.status=OPEN, write Journal
                            ├─ on retryable error: backoff + retry
                            └─ on terminal error: Order.status=FAILED, alert
```

### Properties
- **Atomicity** — DB write + outbox row are in the same TX; broker call is async.
- **Idempotency** — `Idempotency-Key` header dedup'd in Redis (24h window) before TX.
- **Retry** — exponential backoff with jitter; max 5 attempts; DLQ on overflow.
- **Compensation** — if portfolio capital was reserved, saga releases on terminal failure.
- **Observability** — every outbox event carries a trace-id; saga steps emit spans.

### Why not direct sync placement?
Brokers occasionally take 10s+. Holding a request handler open ties up a worker, hurts p95,
and risks gateway timeouts. The user gets immediate feedback ("Order accepted, queued for
broker"); FE polls or subscribes via WS for terminal status.

### Why not Step Functions?
Operational simplicity. We own the saga code, can debug it locally, and the retry policy
lives next to the business logic. Revisit if sagas grow > 5 steps with branching.

## Consequences
- All orders go through the outbox; CLI commands and automated strategies use the same path.
- We must have a "drain" job during deploys: complete in-flight saga steps before stopping.
- Adds 0–1s latency vs sync placement. Acceptable.

## Implementation pointers
- `apps/orders/models.py:OutboxEvent(payload jsonb, status, attempts, next_run_at)`.
- `apps/orders/tasks/outbox.py` — Celery task `process_outbox()` selects rows
  `WHERE next_run_at <= now() AND status='PENDING' FOR UPDATE SKIP LOCKED LIMIT 50`.
- One row = one saga; saga state machine in `apps/orders/services/order_saga.py`.
