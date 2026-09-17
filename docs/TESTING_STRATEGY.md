# AlphaDesk — Testing Strategy

> **Owner:** Platform Eng · **Last updated:** 2026-04-18 · **Status:** Living doc
>
> The testing strategy exists so we can ship fast **without breaking the invariants that keep customer capital safe**. Speed and safety are not in tension here — they compound, because a well-tested system is one we can change with confidence.

---

## 1. Philosophy

AlphaDesk moves real money through an agentic pipeline that is part-LLM, part-deterministic, part-external-broker. That blend shapes our tests in three ways:

1. **Guard the deterministic core at 100% coverage, unit-tested in isolation.** `@RiskGuard`, the Outbox saga, tenant scoping, and idempotency are not allowed to regress. These are the last lines of defence against a hallucinating LLM, a misbehaving broker, or a noisy-neighbour tenant.
2. **Bound the LLM boundary.** We never assert on exact LLM prose. We assert on the *structured output* (Pydantic schemas, tool-call shape, risk flag) and on the *contract* with downstream nodes.
3. **Integration tests should exercise real wiring.** In-memory sqlite, in-memory channel layer, and `CELERY_TASK_ALWAYS_EAGER` let us run the full graph — data → planner → risk → execute → journal — in milliseconds. Use them.

---

## 2. The Pyramid

```
         /   e2e (5%)    \      Playwright — login → plan → risk → paper fill
        /  integration    \     Django TestCase + channels + httpx + schemathesis
       /  (25%)            \
      /  unit tests         \   pytest / vitest — risk rules, analyzers, components
     /  (70%)                \
```

Absolute targets, measured in CI:

| Layer        | Target coverage | Budget (per run) |
|--------------|-----------------|------------------|
| Unit         | 80% lines       | < 30s            |
| Integration  | 65% lines       | < 3m             |
| Contract     | 100% of public endpoints | < 90s  |
| e2e          | 5 critical flows | < 8m            |

The frontend repo starts at 40/35/40/40 (statements/branches/functions/lines) and ratchets up one notch per quarter until it matches the backend.

---

## 3. Strategy by Component

### 3.1 `@RiskGuard` — `apps/orders/services/risk_guard.py`

**Criticality: P0.** Must never regress.

| Test type | What to cover |
|-----------|---------------|
| Unit (100% branch) | Every rule in isolation — qty ≤ 0, trading mode halted, cash below margin, position size > `MAX_POSITION_SIZE_PCT`, per-trade risk > `MAX_RISK_PER_TRADE_PCT`, daily realised + unrealised loss > `MAX_DAILY_LOSS_PCT`, kill-switch, short-straddle premium hard-stop, expiry-day time-of-day cut-off. |
| Property-based (hypothesis) | For any portfolio snapshot and any order, RiskGuard's decision is **deterministic** — running twice returns identical verdicts. |
| Integration | Planner → RiskGuard returns a structured `RiskResult`; a rejected plan never produces an order row. |

**Forbidden**: skipping RiskGuard in tests by mocking it out. If a test needs a passing risk outcome, construct a valid plan — never bypass.

### 3.2 Order saga — `apps/orders/services/order_saga.py`

| Test type | What to cover |
|-----------|---------------|
| Unit | `attempt < MAX_ATTEMPTS` → backoff schedule is `2^attempt` seconds (capped); `attempt == MAX_ATTEMPTS` → routes to DLQ. |
| Integration | Idempotency: submitting the same `client_order_id` twice produces exactly one broker call. Outbox row transitions `pending → sent → acked` under happy path; `pending → failed → retried → dlq` under broker 5xx. |
| Contract | Paper broker adapter satisfies the same interface as Angel One adapter (`zope.interface`-style conformance test). |

### 3.3 Tenancy — `apps/common/tenancy.py`

| Test type | What to cover |
|-----------|---------------|
| Unit | `TenantManager.for_tenant(t)` filters by `tenant_id`; omitting the call in a view raises `TenantScopeMissing` in debug. |
| Integration | User A cannot read User B's positions, orders, journal entries, or audit log rows — even with a crafted query param. A pytest fixture (`two_tenants`) sets up the scenario once. |
| Contract | DRF permission classes reject `X-Tenant-ID` mismatches with 403, not 404 (to avoid enumeration). |

### 3.4 LLM boundary — `@DirectionalTrader`, `@OptionsStrategist`

| Test type | What to cover |
|-----------|---------------|
| Unit | Prompt builders are pure functions — given fixed input, emit byte-identical prompts. Snapshot-tested. |
| Integration | With the Anthropic client **stubbed to return canned JSON**, the graph produces a valid `TradePlan` / `StraddleAction`. |
| Recorded fixtures | `tests/fixtures/llm/*.json` — a small gallery of real past responses we replay. Refresh quarterly. |
| Contract | Every LLM output is validated by Pydantic before leaving its node. Invalid output → graph routes to `fallback_node`, never to `execute`. |

### 3.5 Data pipelines — `trading/services/data_service.py`, `trading/options/data_service.py`

Input validation, schema stability, idempotency:

- Given a malformed Angel One response, `@DataAnalyst` returns `DataFetchError`, never corrupts state.
- Given the same window twice, the second call hits cache and is ≥ 10× faster (performance test, tagged `@pytest.mark.perf`).
- VIX / NIFTY candle enrichment is monotonic — timestamps strictly increasing.

### 3.6 Frontend — React components

| Test type | What to cover |
|-----------|---------------|
| Component | `Button`, `Input`, `KPI`, `DataTable`, `CommandPalette`, `Tabs`, `Tooltip` — variants, states, keyboard nav, `aria-*` correctness. |
| Interaction | `AgentConsolePage` — WS event with `payload.approved === false` renders a `role="alert"` banner. `NewRunDialog` — escape closes, focus returns to trigger. |
| Accessibility | Every page runs `@axe-core/react` in dev mode with zero serious violations. (CI gate: zero critical, zero serious.) |
| Visual regression | Chromatic (or Percy) on `DashboardPage`, `AgentConsolePage`, `PositionsPage` — three viewports: 1440 / 768 / 375. |

### 3.7 WebSockets — Channels consumers

- Unauthenticated connect → 4401 close code.
- Cross-tenant subscribe → 4403.
- `group_send` to `/ws/pnl/` reaches only subscribers of the correct tenant's group (`pnl.<tenant_id>`).
- Backpressure: consumer drops ticks when `send_buffer > 10_000` with a counter metric, never blocks the event loop.

### 3.8 Infrastructure — Smoke + chaos

- **Post-deploy smoke**: `/healthz/live`, `/healthz/ready`, `POST /api/orders/` with a paper order, `GET /ws/pnl/` handshake — under 10s, run on every deploy.
- **Chaos** (monthly in staging): kill a worker mid-saga → recovery within 2× backoff; drop Redis → WS clients reconnect with exponential jitter; broker returns 500 for 30s → circuit opens, DLQ stays empty, recovers automatically.

---

## 4. What to Skip

We do not test:

- Trivial getters/setters, dataclass `__repr__`, Django admin boilerplate.
- Framework code (DRF serializer internals, Channels routing internals).
- One-off management commands unless they touch money (`run_trading_agent` and `manage_straddle` **do** touch money — they are tested).
- Exact LLM prose. Assert on structured fields, schema validity, and downstream effect.
- Third-party SDK behaviour — we test our adapter over it, not the SDK.

---

## 5. Tooling

### Backend — Python

```
pytest
pytest-django
pytest-asyncio
pytest-cov
factory-boy
hypothesis
schemathesis      # contract tests against drf-spectacular schema
freezegun         # for expiry-day / EOD rules
responses         # HTTP stubbing (Angel One)
```

Layout:

```
backend/tests/
├── conftest.py           # shared fixtures — tenant, user, portfolio, broker stub
├── factories.py          # factory-boy factories per model
├── unit/
│   ├── test_risk_guard.py
│   ├── test_analyzer.py  # straddle P&L / delta / phase
│   └── test_prompts.py   # snapshot LLM prompts
├── integration/
│   ├── test_tenancy.py
│   ├── test_orders_saga.py
│   ├── test_trading_graph.py
│   └── test_straddle_graph.py
├── contract/
│   └── test_openapi.py   # schemathesis against /api/schema/
└── e2e/
    └── test_plan_to_fill.py
```

### Frontend — TS

```
vitest
@testing-library/react
@testing-library/user-event
@testing-library/jest-dom
@vitest/coverage-v8
jsdom
axe-core                  # via vitest-axe or @axe-core/react
```

Layout:

```
frontend/src/
├── test/setup.ts                         # jest-dom + window shims
├── components/ui/__tests__/
│   ├── Button.test.tsx
│   ├── Input.test.tsx
│   ├── Card.test.tsx
│   └── KPI.test.tsx
├── features/agents/__tests__/
│   └── RiskBreach.test.tsx
└── features/dashboard/__tests__/
    └── DashboardPage.test.tsx
```

---

## 6. CI gates

On every PR:

1. `ruff check` + `mypy --strict` — backend.
2. `eslint` + `tsc -b --noEmit` — frontend.
3. `pytest -m "not e2e and not perf"` — must pass + coverage ≥ target.
4. `vitest --run --coverage` — must pass + thresholds hit.
5. `schemathesis run http://localhost:8000/api/schema/` — zero contract breaks.
6. `npx @axe-core/cli http://localhost:5173` on the main flows — zero critical/serious.

On `main` merge:

7. Full `pytest` including `e2e` against ephemeral staging.
8. Chromatic visual diff approval gate.

---

## 7. Critical paths (never allowed to regress)

1. **Plan → risk → execute → journal** for equity buy — smoke + integration.
2. **Register → analyze → close** for straddle — integration.
3. **Daily loss breach** halts the desk and rejects new plans with a surfaced reason.
4. **Idempotent order submission** under duplicate client-id.
5. **Tenant A cannot see Tenant B's data** via any endpoint.
6. **Expiry-day auto-close** fires before 15:15 IST (freezegun).
7. **Kill-switch** toggled → all new plans rejected, no new broker calls.

Any PR touching these paths requires a linked test change and a CODEOWNERS review from Platform.

---

## 8. Gaps in current coverage (as of 2026-04-18)

Known gaps to close in the next two sprints, in priority order:

1. No property-based tests on RiskGuard — add hypothesis strategies for `PortfolioSnapshot` + `OrderIntent`.
2. No chaos tests around the Outbox saga — partial failures are only tested happy-path retry.
3. Straddle prompts have no snapshot tests — regressions in the prompt silently change decisions.
4. Frontend has zero accessibility tests in CI — add `@axe-core/react` dev overlay + CI lint.
5. WS consumers lack cross-tenant leakage tests — add `test_ws_tenancy.py`.
6. No contract tests — stand up schemathesis once drf-spectacular schema is stable.

---

## 9. Examples

### 9.1 Unit — RiskGuard rejects oversized position

```python
def test_position_size_cap_rejects_order_over_10pct(portfolio_500k):
    guard = DeterministicRiskGuard(portfolio_500k)
    intent = OrderIntent(symbol="HDFCBANK", qty=100, price=1_600, side="BUY")
    # 100 * 1600 = 160_000 → 32% of 500k capital → must reject
    result = guard.validate(intent)
    assert result.approved is False
    assert "position_size" in result.reason
```

### 9.2 Integration — cross-tenant isolation

```python
@pytest.mark.django_db
def test_tenant_a_cannot_read_tenant_b_positions(two_tenants, api_client):
    tenant_a, tenant_b = two_tenants
    Position.objects.create(tenant=tenant_b, symbol="INFY", qty=10)
    api_client.force_authenticate(tenant_a.owner)
    resp = api_client.get("/api/positions/")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
```

### 9.3 Component — Button loading state

```tsx
it("sets aria-busy when loading", () => {
  render(<Button loading>Save</Button>);
  const btn = screen.getByRole("button", { name: /save/i });
  expect(btn).toHaveAttribute("aria-busy", "true");
  expect(btn).toBeDisabled();
});
```

### 9.4 Interaction — RiskGuard breach renders alert

```tsx
it("renders an alert when risk rejects the plan", async () => {
  const ws = mockSocket();
  render(<AgentConsolePage runId="abc" />);
  ws.emit({ node: "risk", payload: { approved: false, reason: "daily_loss_cap" }});
  expect(await screen.findByRole("alert")).toHaveTextContent(/daily_loss_cap/);
});
```

---

## 10. Review cadence

This doc is reviewed every quarter, and after every post-incident postmortem that touches the testing surface. Owners: Platform Eng. Reviewers: one backend lead, one frontend lead, one SRE.
