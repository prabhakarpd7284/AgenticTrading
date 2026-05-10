# ADR-0002: Pluggable Agentic RAG Architecture

- Status: Accepted
- Date: 2026-04-18

## Context
The product's value compounds when we (and partners) can ship new strategies and new
retrieval sources without touching core code. Today's repo hard-codes the directional
graph and the straddle graph. We need a plugin boundary.

## Decision
Three abstractions, registered through Python entry-points:

### 1. `Strategy` — produces a LangGraph runnable
```python
class Strategy(Protocol):
    name: str
    version: str
    asset_class: Literal["equity", "options", "futures"]
    def build_graph(self, ctx: AgentContext) -> CompiledGraph: ...
    def schema(self) -> StrategySchema: ...    # JSON-schema for params (powers FE form)
```

### 2. `Retriever` — fetches context for an agent run
```python
class Retriever(Protocol):
    name: str
    def retrieve(self, query: RetrievalQuery, k: int = 5) -> list[RetrievedDoc]: ...
```

### 3. `BrokerAdapter` — places/cancels/streams orders
```python
class BrokerAdapter(Protocol):
    name: str
    def place(self, order: OrderDraft) -> BrokerOrderId: ...
    def cancel(self, order_id: BrokerOrderId) -> None: ...
    def stream_ticks(self, tokens: list[str]) -> AsyncIterator[Tick]: ...
```

### Registry
`apps/agents_core/registry.py` exposes `get_strategy(name)`, `list_strategies()`, etc.
On boot, walks `importlib.metadata.entry_points(group="alphadesk.strategies")` and
registers each. A first-party plugin lives at `backend/plugins/strategy_xyz/` and is
declared in `pyproject.toml`:

```toml
[project.entry-points."alphadesk.strategies"]
short_straddle = "plugins.strategy_short_straddle:ShortStraddle"
directional = "plugins.strategy_directional:Directional"

[project.entry-points."alphadesk.retrievers"]
portfolio = "apps.rag.retrievers.portfolio:PortfolioRetriever"
news = "apps.rag.retrievers.news:NewsRetriever"

[project.entry-points."alphadesk.brokers"]
angel_one = "plugins.broker_angel:AngelOneAdapter"
zerodha = "plugins.broker_zerodha:ZerodhaAdapter"
```

### AgentContext (passed to every plugin)
```python
@dataclass
class AgentContext:
    tenant: Tenant
    user: User
    portfolio: Portfolio
    market_data: MarketDataPort
    rag: RAGRouter        # multi-retriever fan-out + rerank
    risk: RiskGuard
    journal: Journal
    publisher: EventPublisher  # for WS streaming
    config: dict          # per-tenant strategy config
```

The plugin sees only ports — never Django models directly.

## Alternatives considered
- **Hard imports of strategies.** Rejected: defeats the marketplace narrative; every new
  strategy is a backend deploy.
- **gRPC-isolated plugins.** Deferred: nicer security boundary, but adds latency and ops
  overhead. We trust first-party plugins; revisit when we accept third-party plugins.
- **A single God-graph parameterized by strategy.** Rejected: the conditional branching
  becomes unmaintainable; per-strategy graphs are clearer.

## Consequences
- Adding a new strategy = new package + entry-point. No core changes.
- We can publish a `cookiecutter-alphadesk-strategy` template.
- Versioning: each strategy ships a SemVer; tenant pins a version; rollouts are gradual.
- Marketplace billing hook: `Strategy.pricing` field surfaces to billing app.
