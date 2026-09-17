# Backend Structure (Django 5 + DRF + Channels + Celery)

## 1. Principles

1. **Layered architecture per app** — `api/` (HTTP), `domain/` (entities + business rules),
   `services/` (use-cases), `repositories/` (persistence), `integrations/` (external).
   Dependencies point inward. Only `api/` imports Django REST pieces.
2. **Apps are feature-oriented, not layer-oriented.** Every app owns its full stack.
3. **No circular imports.** Shared kernel lives in `apps/common/`.
4. **Plugin boundary is explicit.** Strategies, retrievers, brokers are registered through
   entry-points, not hard imports.
5. **Multi-tenant by default.** Every model has `tenant_id`; every view applies the tenant
   filter through middleware + DRF permission.
6. **Async-first where it matters.** REST is sync (Django ORM is still sync in prod).
   WS uses Channels; workers use Celery. Async inside workers is fine (httpx, aiohttp).

## 2. Top-level tree

```
backend/
├── manage.py
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── config/
│   ├── __init__.py
│   ├── asgi.py
│   ├── wsgi.py
│   ├── urls.py
│   ├── celery.py
│   ├── channels_router.py
│   └── settings/
│       ├── base.py
│       ├── dev.py
│       ├── prod.py
│       └── test.py
├── apps/
│   ├── common/              # shared kernel: base models, pagination, exceptions, tenant mixins
│   ├── accounts/            # users, auth, MFA
│   ├── tenants/             # orgs, memberships, RBAC roles
│   ├── billing/             # plans, subscriptions, Razorpay/Stripe webhooks
│   ├── broker/              # broker links, credential vault, token refresh
│   ├── market_data/         # symbols, candles, ticks, VIX, options chain cache
│   ├── portfolio/           # holdings, capital, snapshots, P&L calc
│   ├── orders/              # order DSL, validation, outbox, execution
│   ├── strategies/          # strategy plugins + backtester
│   ├── agents_core/         # Agent ABC, registry, LangGraph builder
│   ├── rag/                 # retriever/embedder/vectorstore interfaces + registry
│   ├── journals/            # immutable decision log
│   ├── audit/               # security audit log
│   └── notifications/       # email, Telegram, in-app, webhooks
├── plugins/                 # first-party strategy & broker plugins (installed as packages)
│   ├── strategy_vwap_fade/
│   ├── strategy_short_straddle/
│   ├── broker_angel/
│   └── broker_zerodha/
└── tests/
    ├── integration/
    └── e2e/
```

### 2.1 Each app, uniformly

```
apps/orders/
├── __init__.py
├── apps.py
├── api/
│   ├── __init__.py
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   └── permissions.py
├── domain/
│   ├── __init__.py
│   ├── entities.py          # pydantic/dataclasses — no Django
│   ├── events.py            # domain events
│   └── exceptions.py
├── services/
│   ├── __init__.py
│   ├── place_order.py       # use-case classes
│   └── cancel_order.py
├── repositories/
│   ├── __init__.py
│   └── orders_repo.py       # thin wrappers over querysets
├── integrations/
│   ├── __init__.py
│   └── broker_gateway.py    # calls broker adapter
├── tasks/
│   ├── __init__.py
│   └── outbox.py            # celery tasks
├── models.py                # Django ORM models
├── admin.py
├── signals.py
├── consumers.py             # Channels consumer if app has WS
├── migrations/
└── tests/
```

## 3. Tenancy model

See `ADR-0001`. Single DB, shared schema, row-level isolation by `tenant_id`.

```python
# apps/common/tenancy.py
class TenantModel(models.Model):
    tenant = models.ForeignKey("tenants.Tenant", on_delete=models.CASCADE,
                               db_index=True, related_name="+")
    class Meta:
        abstract = True

# apps/common/middleware.py
class TenantMiddleware:
    def __call__(self, request):
        request.tenant = resolve_tenant_from_jwt(request)
        return self.get_response(request)

# apps/common/permissions.py
class TenantScoped(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return obj.tenant_id == request.tenant.id
```

Every ViewSet defines:

```python
def get_queryset(self):
    return super().get_queryset().filter(tenant=self.request.tenant)
```

Enforced by a check in CI (pytest-based linter that scans ViewSets).

## 4. API style

- **Base path:** `/api/v1/…`
- **Auth:** `Authorization: Bearer <JWT>` (djangorestframework-simplejwt).
- **Errors:** RFC 7807 problem+json.
- **Pagination:** cursor pagination (`?cursor=…`).
- **Idempotency:** `Idempotency-Key` header on mutating endpoints; dedup'd via Redis.
- **Versioning:** URL-path (`/v1/`). Breaking changes → `/v2/` and parallel run.
- **Docs:** drf-spectacular → `/api/schema/` (OpenAPI) + `/api/docs/` (Swagger UI).

## 5. WebSocket channels

| Channel | Direction | Payload |
|---------|-----------|---------|
| `ws/ticks/` | server→client | `{ts, exchange, token, ltp, oi, vol}` |
| `ws/pnl/` | server→client | `{portfolio_id, mtm, day_pnl, unrealized}` |
| `ws/agents/{run_id}/` | server→client | `{step, node, type: token|result|error, payload}` |
| `ws/alerts/` | server→client | `{id, severity, title, body}` |

All authenticated via JWT in the subprotocol or first message. Tenant scoped by group.

## 6. Domain→service→view shape (example: place order)

```python
# apps/orders/domain/entities.py
class OrderDraft(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    sl: float
    tp: float
    product: Literal["INTRADAY", "DELIVERY"]

# apps/orders/services/place_order.py
class PlaceOrder:
    def __init__(self, risk: RiskGuard, outbox: OrderOutbox):
        self.risk = risk
        self.outbox = outbox

    def execute(self, tenant: Tenant, user: User, draft: OrderDraft) -> OrderResult:
        decision = self.risk.validate(tenant, draft)
        if not decision.approved:
            raise RiskRejected(decision.reason)
        order = self.outbox.enqueue(tenant, user, draft)
        return OrderResult(id=order.id, status="QUEUED")

# apps/orders/api/views.py
class OrderViewSet(viewsets.ModelViewSet):
    def create(self, request):
        draft = OrderDraftSerializer(data=request.data).validated()
        place = container.resolve(PlaceOrder)  # DI
        result = place.execute(request.tenant, request.user, draft)
        return Response(result.dict(), status=202)
```

## 7. Dependency injection

We use a tiny container (`dependency_injector` or hand-rolled `apps/common/di.py`) to wire
services. No globals, no Django signals for business logic. Tests swap implementations.

## 8. Testing layers

| Layer | Tool | What it asserts |
|-------|------|-----------------|
| Unit | pytest | Pure domain functions (risk engine, analyzer, prompts) |
| Service | pytest + factory_boy | Use-cases with fake repos/gateways |
| API | pytest + DRF test client | HTTP contract, auth, tenant isolation |
| Channels | pytest-asyncio + channels testing | WS auth + group messaging |
| Integration | docker-compose test file | Real Postgres + Redis + mock broker |
| E2E | Playwright | Scripted FE flows against staging |
| Contract | schemathesis | Fuzzes against OpenAPI |

## 9. Observability hooks

- `structlog` bound to request-id, tenant-id, user-id.
- OpenTelemetry auto-instrumentation for Django, DRF, Celery, requests.
- Sentry for exception tracking.
- Prometheus `/metrics` endpoint scraped by ADOT.

## 10. Migration from current repo

| Old file | New location | Notes |
|----------|--------------|-------|
| `trading/services/risk_engine.py` | `backend/apps/orders/services/risk_guard.py` | Rename class to `RiskGuard` |
| `trading/services/data_service.py` | `backend/apps/market_data/integrations/angel_one.py` | |
| `trading/options/straddle/graph.py` | `backend/plugins/strategy_short_straddle/graph.py` | As plugin |
| `trading/options/straddle/prompts.py` | `backend/plugins/strategy_short_straddle/prompts.py` | |
| `trading/agents/planner.py` | `backend/plugins/strategy_directional/planner.py` | |
| `trading/graph/trading_graph.py` | `backend/plugins/strategy_directional/graph.py` | |
| `trading/rag/retriever.py` | `backend/apps/rag/retrievers/portfolio_retriever.py` | Implements `Retriever` |
| `trading/models.py` | Split across `accounts/`, `portfolio/`, `orders/`, `journals/`, `audit/` | |
| `trading/management/commands/*` | Kept; re-implemented to call new services | |
| `dashboard.py`, `dashboard_utils/` | **Deprecated** (replaced by React FE) | Keep for 30-day freeze |
