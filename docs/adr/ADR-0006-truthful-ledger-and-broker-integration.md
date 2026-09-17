# ADR-0006: Truthful Ledger & Broker Integration

- Status: Accepted
- Date: 2026-04-22
- Deciders: Platform eng, Trading eng
- Supersedes partial aspects of: ADR-0004 (Order saga remains; this ADR extends it with postback-driven fills and the MarketSnapshot surface)

## Context

AlphaDesk executes three distinct trade horizons across cash and derivatives
(intraday, swing, positional) through Angel One SmartAPI. The existing code wires
a single `broker_service` that pushes orders out; read paths poll `orderBook` /
`position` opportunistically. Two problems result:

1. **No canonical history.** SmartAPI exposes `tradeBook` / `orderBook` only
   intraday; after square-off the data is gone. The Monthly View, backtest
   validation, and audit workflows all need a durable, queryable ledger.
2. **No snapshot surface.** Post-trade analytics (MAE/MFE, entry-quality, IV
   context for options) need point-in-time market state stored against every
   plan / fill / exit. We don't store it today, so the Monthly View reverts to
   mock data.

The framework section of `TRADING_FRAMEWORK.md` names this the "slow loop"
feedback system. This ADR locks the data architecture that makes it real.

Constraints that shaped the design:

- **DB is the source of truth.** SmartAPI endpoints are sources; our tables are
  the system of record.
- **Paper mode is a peer**, not a placeholder. Flipping a tenant to live must be
  a single `broker_mode` update with zero code change.
- **Postback is preferred but unreliable.** Angel's webhook only fires for
  orders submitted via SmartAPI with the same API key; manual app / web orders
  never postback, and there is no documented HMAC signature. Design for
  at-most-once delivery with a polling reconciler as safety net.
- **SmartAPI has per-endpoint rate limits** (e.g. `getCandleData` at 3/sec,
  `ltpData` at 1/sec per request, order placement 20/sec with a 500/min ceiling).
  Every call path must share a rate budget.
- **Capital must be tracked by horizon.** Intraday rotates; positional locks.
  Monthly ROI on rotating vs locked capital is a first-class metric.

## Decision

Adopt a **postback-driven, polling-reconciled, snapshot-timestamped ledger**
with a single `BrokerAdapter` protocol over which `AngelBrokerAdapter` and
`PaperBrokerAdapter` are interchangeable. The DB schema is append-only at the
event layer and derived at the position layer.

The design has five pillars:

1. **Ledger (append-only):** `OrderRequest → OrderAck → BrokerPostback → TradeLeg`.
   Every mutation is a row; current state is derived.
2. **Position lifecycle (derived):** `TradeLeg`s FIFO-match into `PositionLeg`s;
   `PositionLeg`s roll up to `Underlying` for the Monthly view.
3. **Snapshots (polymorphic):** `MarketSnapshot` captures LTP, OHLC, OI, IV,
   greeks, and VIX at defined scopes (plan / entry / hold / exit) against the
   owning row.
4. **Broker abstraction:** Single `BrokerAdapter` protocol covering pricing,
   sizing, execution, and state; `broker_for(tenant)` is the only selector.
5. **Optimized plumbing:** Rate-limit-aware task scheduler, per-endpoint
   token buckets, Redis caching for margin/charges, Celery priority queues,
   partial indexes on hot paths, bulk upserts.

The rest of this document details each pillar.

---

## 1. Full trade lifecycle (end-to-end)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ Pre-trade                                                                       │
│   @DataAnalyst.snapshot(scope=PLAN) ──────────┐                                 │
│   @DirectionalTrader / @OptionsStrategist     │                                 │
│     produces TradePlan(basket)               ─┤                                 │
│   broker.estimateMargin(basket)               │                                 │
│   broker.estimateCharges(basket)              │                                 │
│   @RiskGuard.validate(basket, margin, rms)   ─┘                                 │
│     → reject (journal + notify) OR                                              │
│     → persist TradePlan(status=APPROVED) + OrderRequest(s) in single TX        │
│       + enqueue OutboxEvent(s)                                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│ Submission (OrderSaga, per OrderRequest)                                       │
│   dedup Idempotency-Key → broker.placeOrder(request)                           │
│   on success: OrderAck(broker_order_id, ack_time)                              │
│     MarketSnapshot(scope=ENTRY_REQUEST)                                        │
│   on retryable err: backoff + retry                                            │
│   on terminal err: OrderRequest.status=FAILED + release reserved margin        │
├────────────────────────────────────────────────────────────────────────────────┤
│ Execution (Postback + poll reconciler, both feed same handler)                 │
│   BrokerPostback(raw_payload) ────┐                                            │
│   OR poll_orderbook()      ───────┤→ ingest_event()                            │
│                                    ├→ upsert OrderState(latest)                │
│                                    ├→ if fill(partial/full): TradeLeg row      │
│                                    │    + MarketSnapshot(scope=FILL)           │
│                                    │    + apply charges from calculator        │
│                                    └→ if cancel/reject: terminal flags         │
│   TradeLeg FIFO-matches into PositionLeg                                       │
│     OPEN leg if none exists; else reduce / close existing leg                 │
├────────────────────────────────────────────────────────────────────────────────┤
│ Hold                                                                            │
│   @DataAnalyst cadence snapshots (scope=HOLD) every N minutes for open         │
│     PositionLegs — interval scales by horizon:                                 │
│       INTRADAY: 1-5 min | SWING: 15 min | POSITIONAL: 1 hr                     │
│   Compute MAE/MFE rolling on every snapshot                                    │
│   @OptionsStrategist / @RiskGuard may trigger exit OrderRequests               │
├────────────────────────────────────────────────────────────────────────────────┤
│ Exit                                                                            │
│   Exit OrderRequest → same submission + execution path                         │
│   TradeLeg(side opposite) → FIFO-match against open PositionLeg                │
│     close or partially close; set closed_at, realized_pnl                      │
│   MarketSnapshot(scope=EXIT)                                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│ Reconciliation (daily + on-demand)                                              │
│   3:31 PM IST: fetch tradeBook, position, holding, allholding, rmsLimit        │
│   Diff broker state vs DB → ReconFinding rows (kind, severity, payload)        │
│   Auto-heal soft drifts (e.g. missing snapshot); page on hard drifts           │
├────────────────────────────────────────────────────────────────────────────────┤
│ Aggregation                                                                     │
│   Delta-driven regenerate of MonthSnapshot on TradeLeg write                   │
│   YTD summary = materialized view refreshed every 5 min during market hours    │
│   WebSocket push to /ws/monthly/ when current-month rows change                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Order state machine

```
     ┌──────┐  submit   ┌────────────┐  ack    ┌─────────┐
     │ DRAFT├──────────►│ SUBMITTING ├────────►│ OPEN    │
     └──┬───┘           └──────┬─────┘         └────┬────┘
        │ reject               │ reject             │ partial fill ─┐
        │                      │                    │               ▼
        ▼                      ▼                    │         ┌────────────┐
     ┌──────┐             ┌─────────┐                │         │ PART_FILLED│
     │FAILED│             │ REJECTED│                │         └────┬───────┘
     └──────┘             └─────────┘                │              │ full fill
                                                    │              ▼
                                            ┌───────▼──┐     ┌──────────┐
                                            │ CANCELLED│◄────│  FILLED  │
                                            └──────────┘     └──────────┘
                                         user / broker-initiated
```

The state machine is derived, not stored as the primary key of truth — the
ledger is. `OrderState` is a materialized view of `(OrderRequest, latest
OrderAck, latest postback, latest poll)` computed via `ORDER BY update_time
DESC LIMIT 1` with a covering partial index.

### Position lifecycle

```
                                 FIFO-match
  TradeLeg(BUY 100) ───────────► PositionLeg(qty_open=100, side=LONG)
                                          │
  TradeLeg(BUY 50)  ───────────► qty_open=150 (weighted avg cost)
                                          │
  TradeLeg(SELL 80) ───────────► qty_open=70, realized_pnl += (exit-entry)*80
                                          │
  TradeLeg(SELL 70) ───────────► qty_open=0, closed_at=now, status=CLOSED
```

Matching key: `(tenant_id, source, symbol_token, strategy_instance_id,
product_type, horizon)`. Two open `PositionLeg`s cannot exist for the same key
— a new `TradeLeg` either extends or reduces an existing open leg, or opens a
fresh one if none is open. Cross-strategy positions in the same token (e.g.
two strategies both long RELIANCE) are intentionally separate rows.

---

## 2. Database schema (PostgreSQL)

Design principles:

- **Enums as native Postgres enums** (not CHECKs) for type safety in Django
  and direct readability in SQL.
- **Append-only at the ledger layer** (`OrderRequest`, `OrderAck`,
  `BrokerPostback`, `TradeLeg`, `MarketSnapshot`). Updates only on
  `PositionLeg`, `StrategyInstance`, derived materialized tables.
- **Idempotency through composite UNIQUE constraints**, not application-layer
  checks.
- **Partial indexes for hot reads** (open positions, unacknowledged requests).
- **JSONB + queryable generated columns** for broker raw payloads: keep the
  blob, surface the fields that matter for queries.

### Enums

```sql
CREATE TYPE broker_mode      AS ENUM ('paper', 'live');
CREATE TYPE source           AS ENUM ('angel_one');            -- future-proof
CREATE TYPE asset_class      AS ENUM ('cash', 'fno_options', 'fno_futures',
                                      'commodity', 'currency');
CREATE TYPE horizon          AS ENUM ('intraday', 'swing', 'positional');
CREATE TYPE side             AS ENUM ('buy', 'sell');
CREATE TYPE order_status     AS ENUM ('draft', 'submitting', 'open',
                                      'part_filled', 'filled', 'cancelled',
                                      'rejected', 'failed');
CREATE TYPE product_type     AS ENUM ('intraday', 'delivery', 'carryforward',
                                      'margin', 'bo', 'co', 'amo');
CREATE TYPE variety          AS ENUM ('normal', 'stoploss', 'amo', 'robo');
CREATE TYPE snapshot_scope   AS ENUM ('plan', 'entry_request', 'fill', 'hold',
                                      'exit', 'eod');
CREATE TYPE recon_kind       AS ENUM ('order_state', 'trade_leg', 'position',
                                      'holding', 'margin', 'charges');
CREATE TYPE recon_severity   AS ENUM ('info', 'warning', 'error');
CREATE TYPE ca_kind          AS ENUM ('dividend', 'bonus', 'split',
                                      'rights', 'merger', 'demerger');
CREATE TYPE gtt_status       AS ENUM ('new', 'active', 'triggered',
                                      'cancelled', 'expired');
```

### Core tables (trimmed, full SQL in migration)

```sql
-- Broker credentials & config, one row per tenant + broker
CREATE TABLE broker_account (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL REFERENCES tenant(id),
    source          source NOT NULL,
    client_code     TEXT NOT NULL,                  -- Angel client id
    api_key_ref     TEXT NOT NULL,                  -- KMS ref, never raw
    totp_secret_ref TEXT NOT NULL,                  -- KMS ref
    broker_mode     broker_mode NOT NULL DEFAULT 'paper',
    feed_token      TEXT,                            -- refreshed each session
    session_expires_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (tenant_id, source, client_code)
);

-- Instrument master, refreshed daily from ScripMaster
CREATE TABLE instrument (
    token           TEXT NOT NULL,                  -- SmartAPI symboltoken
    exchange        TEXT NOT NULL,                  -- NSE | BSE | NFO | MCX
    tradingsymbol   TEXT NOT NULL,
    name            TEXT,
    asset_class     asset_class NOT NULL,
    expiry          DATE,                           -- NULL for cash
    strike          NUMERIC(12,2),
    option_type     TEXT,                           -- CE | PE | NULL
    lot_size        INT NOT NULL DEFAULT 1,
    tick_size       NUMERIC(6,4),
    active          BOOLEAN DEFAULT TRUE,
    updated_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (exchange, token)
);
CREATE INDEX idx_instrument_ts ON instrument (tradingsymbol, exchange);
CREATE INDEX idx_instrument_expiry ON instrument (expiry) WHERE active;

-- Strategy instance groups multiple legs (e.g. a straddle is 2 legs, 1 strategy)
CREATE TABLE strategy_instance (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL REFERENCES tenant(id),
    kind            TEXT NOT NULL,                  -- 'short_straddle', 'directional_equity', ...
    horizon         horizon NOT NULL,
    underlying      TEXT NOT NULL,                  -- 'NIFTY', 'RELIANCE'
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at       TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_strat_open ON strategy_instance (tenant_id, horizon)
    WHERE closed_at IS NULL;

-- TradePlan = pre-trade intent (the basket the LLM produced)
CREATE TABLE trade_plan (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    strategy_instance_id BIGINT REFERENCES strategy_instance(id),
    agent           TEXT NOT NULL,                  -- '@DirectionalTrader' | ...
    rationale       TEXT,
    expected_margin NUMERIC(14,2),
    expected_charges NUMERIC(12,2),
    risk_result     JSONB,                           -- full @RiskGuard payload
    status          TEXT NOT NULL,                  -- 'approved' | 'rejected'
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- OrderRequest = one intended order (1 basket → N requests)
CREATE TABLE order_request (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    broker_account_id BIGINT NOT NULL REFERENCES broker_account(id),
    trade_plan_id   BIGINT REFERENCES trade_plan(id),
    strategy_instance_id BIGINT REFERENCES strategy_instance(id),
    client_order_id UUID NOT NULL DEFAULT gen_random_uuid(),
    exchange        TEXT NOT NULL,
    symbol_token    TEXT NOT NULL,
    tradingsymbol   TEXT NOT NULL,
    side            side NOT NULL,
    variety         variety NOT NULL DEFAULT 'normal',
    product_type    product_type NOT NULL,
    horizon         horizon NOT NULL,
    quantity        INT NOT NULL,
    price           NUMERIC(12,2),                   -- NULL for MARKET
    order_type      TEXT NOT NULL,                   -- LIMIT | MARKET | SL | SL-M
    trigger_price   NUMERIC(12,2),
    disclosed_qty   INT,
    validity        TEXT NOT NULL DEFAULT 'DAY',     -- DAY | IOC
    squareoff       NUMERIC(12,2),                   -- BO target
    stoploss        NUMERIC(12,2),                   -- BO/CO stop
    trailing_sl     NUMERIC(12,2),                   -- BO trail
    broker_order_id TEXT,                            -- filled on ack
    status          order_status NOT NULL DEFAULT 'draft',
    created_at      TIMESTAMPTZ DEFAULT now(),
    acknowledged_at TIMESTAMPTZ,
    terminal_at     TIMESTAMPTZ,
    UNIQUE (tenant_id, client_order_id)
);
CREATE INDEX idx_order_request_open
    ON order_request (tenant_id, broker_order_id)
    WHERE terminal_at IS NULL;

-- BrokerPostback = raw webhook payloads (and synthesized paper events). Never mutated.
CREATE TABLE broker_postback (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    source          source NOT NULL,
    broker_order_id TEXT NOT NULL,
    orderstatus     TEXT NOT NULL,                  -- Angel string as-is
    update_time     TIMESTAMPTZ NOT NULL,           -- from payload
    payload         JSONB NOT NULL,
    received_at     TIMESTAMPTZ DEFAULT now(),
    processed_at    TIMESTAMPTZ,
    UNIQUE (source, broker_order_id, orderstatus, update_time)
);
CREATE INDEX idx_postback_unprocessed ON broker_postback (received_at)
    WHERE processed_at IS NULL;

-- TradeLeg = one fill (partial or full). FIFO unit of P&L.
CREATE TABLE trade_leg (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    order_request_id BIGINT NOT NULL REFERENCES order_request(id),
    source          source NOT NULL,
    broker_trade_id TEXT NOT NULL,                   -- uniquely identifies fill
    exchange        TEXT NOT NULL,
    symbol_token    TEXT NOT NULL,
    tradingsymbol   TEXT NOT NULL,
    side            side NOT NULL,
    product_type    product_type NOT NULL,
    horizon         horizon NOT NULL,
    quantity        INT NOT NULL,
    price           NUMERIC(12,2) NOT NULL,
    filled_at       TIMESTAMPTZ NOT NULL,
    brokerage       NUMERIC(10,2) NOT NULL DEFAULT 0,
    stt             NUMERIC(10,2) NOT NULL DEFAULT 0,
    exch_txn_chg    NUMERIC(10,2) NOT NULL DEFAULT 0,
    sebi_chg        NUMERIC(10,4) NOT NULL DEFAULT 0,
    gst             NUMERIC(10,2) NOT NULL DEFAULT 0,
    stamp_duty      NUMERIC(10,2) NOT NULL DEFAULT 0,
    total_charges   NUMERIC(10,2) NOT NULL DEFAULT 0, -- sum of above
    strategy_instance_id BIGINT REFERENCES strategy_instance(id),
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (tenant_id, source, broker_trade_id)
);
CREATE INDEX idx_trade_leg_strategy ON trade_leg (strategy_instance_id);
CREATE INDEX idx_trade_leg_by_day ON trade_leg (tenant_id, filled_at DESC);

-- PositionLeg = derived open/closed position
CREATE TABLE position_leg (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    source          source NOT NULL,
    strategy_instance_id BIGINT REFERENCES strategy_instance(id),
    exchange        TEXT NOT NULL,
    symbol_token    TEXT NOT NULL,
    tradingsymbol   TEXT NOT NULL,
    side            side NOT NULL,                  -- LONG / SHORT
    product_type    product_type NOT NULL,
    horizon         horizon NOT NULL,
    qty_opened      INT NOT NULL,
    qty_open        INT NOT NULL,                   -- decreases on exits
    avg_entry       NUMERIC(12,2) NOT NULL,
    last_exit       NUMERIC(12,2),
    realized_pnl    NUMERIC(14,2) NOT NULL DEFAULT 0,
    charges_total   NUMERIC(10,2) NOT NULL DEFAULT 0,
    mae             NUMERIC(14,2),                  -- max adverse excursion
    mfe             NUMERIC(14,2),                  -- max favorable excursion
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX ux_position_open
    ON position_leg (tenant_id, source, exchange, symbol_token,
                     strategy_instance_id, product_type, horizon)
    WHERE closed_at IS NULL;
CREATE INDEX idx_position_open
    ON position_leg (tenant_id, horizon) WHERE closed_at IS NULL;

-- MarketSnapshot = point-in-time market state against owning row
CREATE TABLE market_snapshot (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    scope           snapshot_scope NOT NULL,
    -- owner pointers (exactly one is non-null)
    trade_plan_id   BIGINT REFERENCES trade_plan(id),
    order_request_id BIGINT REFERENCES order_request(id),
    trade_leg_id    BIGINT REFERENCES trade_leg(id),
    position_leg_id BIGINT REFERENCES position_leg(id),
    -- market data
    exchange        TEXT NOT NULL,
    symbol_token    TEXT NOT NULL,
    ltp             NUMERIC(12,2),
    open            NUMERIC(12,2),
    high            NUMERIC(12,2),
    low             NUMERIC(12,2),
    close           NUMERIC(12,2),
    volume          BIGINT,
    oi              BIGINT,
    bid             NUMERIC(12,2),
    ask             NUMERIC(12,2),
    bid_qty         INT,
    ask_qty         INT,
    -- options context
    iv              NUMERIC(8,4),
    delta           NUMERIC(8,4),
    gamma           NUMERIC(10,6),
    vega            NUMERIC(8,4),
    theta           NUMERIC(8,4),
    rho             NUMERIC(8,4),
    -- index context
    india_vix       NUMERIC(8,4),
    underlying_ltp  NUMERIC(12,2),
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    source          TEXT NOT NULL DEFAULT 'angel_ltp',  -- 'angel_ltp' | 'angel_full' | 'paper_sim'
    CHECK (
        (trade_plan_id IS NOT NULL)::int +
        (order_request_id IS NOT NULL)::int +
        (trade_leg_id IS NOT NULL)::int +
        (position_leg_id IS NOT NULL)::int = 1
    )
);
CREATE INDEX idx_snapshot_position_time
    ON market_snapshot (position_leg_id, captured_at DESC)
    WHERE position_leg_id IS NOT NULL;
CREATE INDEX idx_snapshot_hold
    ON market_snapshot (tenant_id, captured_at DESC)
    WHERE scope = 'hold';

-- OutboxEvent retained from ADR-0004 for saga handling
-- (no schema change; referenced here for continuity)

-- GTT rules (positional entry levels)
CREATE TABLE gtt_rule (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    broker_rule_id  TEXT NOT NULL,
    strategy_instance_id BIGINT REFERENCES strategy_instance(id),
    exchange        TEXT NOT NULL,
    symbol_token    TEXT NOT NULL,
    side            side NOT NULL,
    quantity        INT NOT NULL,
    trigger_price   NUMERIC(12,2) NOT NULL,
    limit_price     NUMERIC(12,2),
    product_type    product_type NOT NULL,
    status          gtt_status NOT NULL,
    payload         JSONB,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (tenant_id, broker_rule_id)
);

-- MarginCache for expensive basket margin calls
CREATE TABLE margin_cache (
    cache_key       TEXT PRIMARY KEY,                -- hash of basket
    tenant_id       BIGINT NOT NULL,
    basket_json     JSONB NOT NULL,
    result_json     JSONB NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL
);

-- Daily end-of-day capture for durable history (SmartAPI drops intraday books)
CREATE TABLE eod_snapshot (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    broker_account_id BIGINT NOT NULL,
    as_of_date      DATE NOT NULL,
    order_book      JSONB NOT NULL,
    trade_book      JSONB NOT NULL,
    positions       JSONB NOT NULL,
    holdings        JSONB NOT NULL,
    rms             JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (tenant_id, broker_account_id, as_of_date)
);

-- Corporate actions (V1 = flag; V2 = auto-rebase)
CREATE TABLE corporate_action (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    exchange        TEXT NOT NULL,
    tradingsymbol   TEXT NOT NULL,
    kind            ca_kind NOT NULL,
    ex_date         DATE NOT NULL,
    ratio           TEXT,                            -- '2:1', '1:10', '₹5/share'
    metadata        JSONB,
    applied         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Reconciliation findings
CREATE TABLE recon_finding (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       BIGINT NOT NULL,
    run_id          UUID NOT NULL,
    kind            recon_kind NOT NULL,
    severity        recon_severity NOT NULL,
    entity_ref      TEXT,                            -- 'order_request:123'
    broker_state    JSONB,
    db_state        JSONB,
    resolved_at     TIMESTAMPTZ,
    resolution_note TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Materialized monthly rollup (delta-driven refresh)
CREATE MATERIALIZED VIEW month_snapshot AS
SELECT
    tenant_id,
    date_trunc('month', filled_at)::date AS month,
    horizon,
    COUNT(DISTINCT position_leg.id) FILTER (WHERE closed_at IS NOT NULL) AS closed_legs,
    SUM(realized_pnl) FILTER (WHERE closed_at IS NOT NULL) AS realized_pnl,
    SUM(realized_pnl) FILTER (WHERE closed_at IS NOT NULL AND realized_pnl > 0) AS gross_wins,
    ABS(SUM(realized_pnl) FILTER (WHERE closed_at IS NOT NULL AND realized_pnl < 0)) AS gross_losses,
    -- capital_deployed = peak of (margin-used snapshots) within month
    MAX(margin_used) AS peak_capital_deployed,
    AVG(margin_used) AS avg_capital_deployed
FROM position_leg
JOIN trade_leg USING (tenant_id)
GROUP BY tenant_id, month, horizon;
CREATE UNIQUE INDEX ON month_snapshot (tenant_id, month, horizon);
```

### Why these constraints?

- `UNIQUE (source, broker_order_id, orderstatus, update_time)` on
  `broker_postback` makes replays (Angel retries, our reconciler finding a
  missed event) no-ops. Any duplicate is discarded at the DB layer.
- `UNIQUE (tenant_id, source, broker_trade_id)` on `trade_leg` makes fills
  idempotent whether they arrive via postback or poll.
- `UNIQUE (tenant_id, client_order_id)` on `order_request` makes retries from
  the outbox saga safe — the same `client_order_id` is sent every attempt.
- Partial `UNIQUE` on `position_leg` enforces the "only one open leg per key"
  invariant in the database, not application code.

---

## 3. Broker API usage — maximum coverage

Every SmartAPI endpoint we touch, with purpose and call pattern.

| Endpoint | Stage | Purpose | Cadence |
|---|---|---|---|
| `generateSession` / `renewAccessToken` | session | Login, refresh feed token | per session, + 5 min before expiry |
| `getProfile` | session | Verify client code post-login | on session start |
| `searchScrip` | bootstrap | Resolve `tradingsymbol` → `symboltoken` on strategy creation; refresh when an instrument is referenced without a cached token | ad-hoc + on plan build |
| `ScripMaster (daily dump)` | bootstrap | Full instrument master refresh 7:00 IST daily | daily |
| `ltpData` | snapshots (PLAN, HOLD-intraday) | Cheap single quote | rate-budget-shared |
| `getMarketData(LTP)` | snapshots | Batch LTP (up to 50 tokens per call) | preferred over `ltpData` whenever > 1 token |
| `getMarketData(OHLC)` | snapshots (PLAN) | Single call for day OHLC across a basket | batch by tenant |
| `getMarketData(FULL)` | snapshots (ENTRY, FILL, EXIT) | Depth + OI + circuit limits — used for quality audit on fills | every fill + periodic for positional |
| `getCandleData` | pre-trade, hold-positional, backfill | 1m/5m/15m/60m/1D candles; IV/greek historical replay | on-demand + scheduled backfill |
| `gainersLosers` | Cascade Stage 3 ("ROTATION") | Sector heat; monthly context tooltip | every 15 min market hours |
| `getMarginApi` (basket) | RiskGuard #11, pre-submit | Accurate pre-trade margin incl. hedge benefit | before every submit; cache 5 min |
| `estimateCharges` (brokerage calculator) | RiskGuard, trade leg | Exact STT/STX/GST/SEBI/stamp for realistic planning; stamp on every fill | pre-submit + on fill (no separate call if cached) |
| `placeOrder` / `modifyOrder` / `cancelOrder` | submit | OrderSaga — placement, modification, cancellation | per saga step |
| `individual_order_details` | verify | Postback round-trip verification when `orderstatus` looks inconsistent | on demand (recon) |
| `orderBook` | reconcile | EOD capture; also short-interval fallback when postback stream is stale | 3:31 PM IST + every 5 min during market hours when postback idle |
| `tradeBook` | reconcile | Source of truth diff vs our `trade_leg` table | 3:31 PM IST + on-demand |
| `position` | reconcile | Net position diff vs derived `position_leg` | 3:31 PM IST + intraday every 15 min |
| `holding` / `allholding` | reconcile (swing/positional) | Delivered holdings; map to positional carryforward legs | daily + on settlement (T+1) |
| `rmsLimit` | RiskGuard #12, capital tracking | Available margin, used margin, net cash | every 5 min during market hours |
| `convertPosition` | lifecycle | Convert INTRADAY → DELIVERY/CARRYFORWARD (swing upgrade), or DELIVERY → INTRADAY when desk decides to square off | on-demand, journaled |
| `gttCreateRule` / `gttModifyRule` / `gttCancelRule` / `gttRuleDetails` / `gttList` | positional entries | Store broker rule id in `gtt_rule`; daily list reconciliation | on plan + daily poll |
| `Postback webhook` | stream | Primary execution feed | event |
| `marketData (WS v2)` | live UI, hold-intraday snapshots | Tick stream for open positions; **subscribe only to tokens with open legs + shortlist tokens** | event |

### Key design rules around the API surface

1. **Prefer batched reads.** `getMarketData(LTP mode, 50 tokens)` is always
   preferred over 50 × `ltpData`. The snapshotter groups requests by scope and
   flushes.
2. **Separate rate buckets per endpoint.** A token-bucket per endpoint, owned
   by a central `RateLimiter` service. All adapters obtain a permit before
   calling SmartAPI. Buckets are sized from the documented limits with a 20%
   safety margin.
3. **Cache where safe.** `getMarginApi` and `estimateCharges` are cached in
   Redis keyed by a stable hash of the basket (tenant + symbol+side+qty+price
   tuples, sorted). 5-min TTL for margin; 1-hour TTL for charges (tariff is
   stable intra-day). `searchScrip` results cached 24h. `ScripMaster` loaded
   daily into a read-optimized table; tokens resolved from there first.
4. **WS subscriptions are dynamic.** Subscribe on position open; unsubscribe
   on close. Plus a persistent subscription for every shortlist token.
   Otherwise we'd hit the WS subscription cap.
5. **Postback is trusted but verified.** Every postback writes to
   `broker_postback` unchanged. The reconciler diffs order states against
   `orderBook` every 5 min during market hours; on a divergence we call
   `individual_order_details` for that order to get authoritative state.
6. **convertPosition is a first-class lifecycle action.** When the desk
   decides to hold an intraday leg overnight, the workflow is:
   `convertPosition(INTRADAY → CARRYFORWARD)` →
   `strategy_instance.horizon = 'positional'` → reconciler confirms on next
   `position` call. Audit log has a ConversionEvent row linking the
   intraday `PositionLeg` to the post-conversion lineage.

---

## 4. Edge case matrix

One row per scenario: trigger, detection, resolution.

### Order lifecycle

| # | Case | Detection | Resolution |
|---|---|---|---|
| 1 | **Partial fill** | `tradeBook`/postback fill qty < request qty | Insert `TradeLeg(quantity=fill_qty)`; `OrderRequest.status=part_filled`. Position_leg grows by fill_qty. |
| 2 | **Multiple partial fills** | Sequential postbacks with same `broker_order_id` | Each writes a distinct `TradeLeg(broker_trade_id=unique)`. FIFO matcher accumulates. |
| 3 | **Modify before fill** | User / strategy adjusts price or qty | `OrderModRequest` row (append-only); saga calls `modifyOrder`; postback with new `orderstatus` writes new `broker_postback` row — latest wins in OrderState materialization. |
| 4 | **Cancel before fill** | Manual cancel or RiskGuard trip mid-flight | `cancelOrder` saga step; terminal postback sets `OrderRequest.status=cancelled`, `terminal_at=now()`. |
| 5 | **Reject** | Broker returns `rejected`; postback carries `rejectionreason` | Map to `order_status=rejected`, journal with reason, alert on risk-relevant rejections (margin shortage). |
| 6 | **Duplicate postback** | Same `(broker_order_id, orderstatus, update_time)` | UNIQUE constraint rejects insert silently. Handler treats as no-op. |
| 7 | **Out-of-order postback** | Postback with earlier `update_time` after later one | OrderState materialization sorts by `update_time DESC LIMIT 1` — earlier is ignored for state but retained in ledger. |
| 8 | **Missed postback** | Webhook failure, retry disabled | Reconciler every 5 min calls `orderBook`; diffs against open `OrderRequest`s; writes synthesized `broker_postback` row with `source_hint='poll'`. |
| 9 | **Postback for unknown order** | Order placed by manual app using different API key → no DB record | Reject at webhook handler with 400 after writing to `broker_postback(orphan=true)`. Daily recon flags as info. |
| 10 | **Retry after broker timeout** | Saga timeout → unsure if order placed | Use `tag` on placement (our `client_order_id` converted to a prefix). Pre-placement reconciler fetches `orderBook` with that `client_order_id` pattern; if present, absorb; if absent, retry. |
| 11 | **Bracket order (BO) child legs** | Parent order fills, broker auto-creates target + stop | Each child is a distinct `broker_order_id` — postbacks arrive independently; handler ignores `variety=robo` child orders when joining back to `order_request` (they have no matching `client_order_id`) and treats them as `order_request.variety=robo, parent_broker_order_id=...`. Add a nullable `parent_order_request_id` link. |
| 12 | **Basket atomicity** | 4-leg iron condor — want all-or-nothing | `place_basket()` wraps 4 saga steps; if any `placeOrder` fails terminally, previously-placed legs are cancelled. Margin pre-check via `getMarginApi(basket)` minimizes the likelihood. |
| 13 | **Order modification with different margin** | Modify qty upward → broker may reject for margin | Pre-flight `getMarginApi` with the new basket; if shortfall, fail locally before `modifyOrder` call. |

### Market & instrument

| # | Case | Detection | Resolution |
|---|---|---|---|
| 14 | **Token churn across expiries** | Weekly option's token is reused after expiry for a different strike | `PositionLeg` keys include `opened_at` tiebreak plus `strategy_instance_id`; stale instrument rows flagged `active=false` after `ScripMaster` refresh. |
| 15 | **Corporate action (bonus/split)** | NSE `CorpAction` feed (future) / manual entry | V1: `corporate_action` row; Monthly view annotates affected underlying; hold_snapshots unaffected (LTP already post-adjustment). V2: auto-rebase `avg_entry` and qty for positional holdings. |
| 16 | **Dividend on holding** | CA table entry `kind=dividend` | Credit `PortfolioSnapshot` cash, annotate `PositionLeg` with dividend received; do NOT alter `realized_pnl`. |
| 17 | **Stale LTP** | `market_snapshot.captured_at` age > 60s during market hours | Snapshotter invalidates; RiskGuard criterion #7 (stale-price gate) rejects plan with reason. |
| 18 | **Market-wide circuit breaker** | Exchange notification / bid-ask spread explosion | Circuit-breaker flag in `rmsLimit` or detected via `getMarketData(FULL)` depth anomaly → RiskGuard blocks new plans; existing stops remain in place. |
| 19 | **NSE holiday / half-day** | Exchange calendar | Cached calendar table; `market_open()` gate on every planning / snapshotter run. |
| 20 | **AMO (after-market order)** | User plans before market open | `variety=amo`, `product_type=amo` on request; broker queues; postback arrives on open. Paper mode simulates open-price fill. |

### Session & connectivity

| # | Case | Detection | Resolution |
|---|---|---|---|
| 21 | **Session expires mid-trade** | SmartAPI 401 mid-call | Saga catches `SessionExpired`; calls `generateSession(totp)`; retries request. Max 3 re-auths per hour per account. |
| 22 | **TOTP clock drift** | Session generation fails with invalid OTP | NTP-sync check on server; alert if drift > 30s. |
| 23 | **Rate limit hit** | 429 / specific error code | Token bucket should prevent; if hit, exponential backoff + jitter; log as incident above threshold. |
| 24 | **Postback endpoint DDoS** | Spike in unknown-source postbacks | IP allowlist (Angel publishes egress range); path-secret; request-per-second cap at edge. |
| 25 | **WS disconnect** | Heartbeat timeout | Auto-reconnect with exponential backoff; re-subscribe from in-memory state; snapshotter gap-fills from `getMarketData`. |

### Position lifecycle

| # | Case | Detection | Resolution |
|---|---|---|---|
| 26 | **Intraday-to-delivery conversion** | Manual desk decision or @OptionsStrategist recommendation on positional upgrade | `convertPosition` call + strategy_instance.horizon update + audit row; reconciler verifies on next `position` call. |
| 27 | **Settlement lag on carryforward** | Holding only appears T+1 on `holding` | `position` call on day 0 carries `netqty`; `holding` matches on T+2; reconciler is lenient for T vs T+1 mismatches on recently-converted legs. |
| 28 | **Multiple strategies on same token** | Two strategies both long RELIANCE | Separate `PositionLeg`s keyed by `strategy_instance_id`; broker's aggregated `netqty` matched only on the `(tenant, token)` sum, not per leg. |
| 29 | **Manual trade from mobile app** | Postback never arrives (different API key); appears in `orderBook`/`tradeBook` | Daily recon writes an `orphan_trade_leg` with `strategy_instance_id=NULL` → flagged in Monthly view under "External trades" section. |
| 30 | **Expiry-day exit missed** | @OptionsStrategist failed to recommend close before 3:15 PM IST on expiry | Cron at 3:10 PM IST auto-submits closing orders for any open options PositionLeg whose expiry = today; RiskGuard override permitted only by the expiry cron. |
| 31 | **Strategy close without leg closes** | `strategy_instance.closed_at=now()` while legs still open | DB trigger raises; only allowed path: closing strategy closes all legs (or converts to orphan legs flagged in Monthly). |

---

## 5. Paper mode as peer

`PaperBrokerAdapter` must be behaviorally indistinguishable from
`AngelBrokerAdapter` at the interface layer. Flipping a tenant live is
`UPDATE broker_account SET broker_mode='live' WHERE tenant_id=X;` — nothing
else changes.

### Design

- **Pricing**: `PaperBrokerAdapter` reads real `getMarketData(LTP)` /
  `getCandleData` for pricing; paper-ness is purely about execution.
- **Fill simulator**: Market orders fill at LTP ± configurable slippage
  (bps tier by asset class). Limit orders wait until LTP crosses price;
  stop orders wait until trigger crosses. Simulation clock is real wall
  clock during market hours (replayable in backtests).
- **Synthesized postbacks**: Fill simulator emits payloads in Angel's
  postback schema, writes to `broker_postback` with `payload.source='paper'`.
  The same ingest handler runs — no branching.
- **Real margin, real charges**: `PaperBrokerAdapter.estimate_margin()`
  and `estimate_charges()` call the **real** Angel endpoints. This keeps
  capital sizing honest.
- **No real RMS**: `rmsLimit` is derived from `PortfolioSnapshot` for paper;
  live mode hits SmartAPI.
- **GTT in paper**: `gtt_rule` rows in `status=active`; a cron scans them
  against LTP each minute and triggers `placeOrder` in the same paper path.
- **`convertPosition` in paper**: metadata flip only; no broker call.

### Selector

```python
def broker_for(tenant_id: int) -> BrokerAdapter:
    acct = BrokerAccount.objects.get(tenant_id=tenant_id)
    if acct.broker_mode == 'paper':
        return PaperBrokerAdapter(acct)
    return AngelBrokerAdapter(acct)
```

The selector is the **only** place `broker_mode` is checked. Never in
planners, risk engine, saga, or reconciler.

### Paper-mode Monthly View correctness

All ledger tables are populated by paper mode exactly as they are for live:
`trade_leg` rows have realistic charges from the brokerage calculator,
`market_snapshot` rows have real LTP/greeks, `position_leg` rows have real
mark-to-market. The Monthly view cannot tell the difference — which is the
point: we've been training on synthetic P&L means we've been lying to the
strategist.

---

## 6. Efficiency & optimization

### Rate-limit-aware scheduler

```python
class RateLimiter:
    def __init__(self, buckets: dict[str, TokenBucket]): ...
    async def acquire(self, endpoint: str, cost: int = 1) -> None: ...

# Bucket sizing (20% safety margin below documented limits)
buckets = {
    'ltpData':        TokenBucket(rps=0.8,  burst=3),
    'getMarketData':  TokenBucket(rps=0.8,  burst=3),
    'getCandleData':  TokenBucket(rps=2.4,  burst=5),
    'placeOrder':     TokenBucket(rps=16,   burst=20,  minute_cap=400),
    'modifyOrder':    TokenBucket(rps=16,   burst=20,  minute_cap=400),
    'cancelOrder':    TokenBucket(rps=16,   burst=20,  minute_cap=400),
    'orderBook':      TokenBucket(rps=1,    burst=2),
    'tradeBook':      TokenBucket(rps=1,    burst=2),
    'position':       TokenBucket(rps=1,    burst=2),
    'rmsLimit':       TokenBucket(rps=0.5,  burst=1),
    'getMarginApi':   TokenBucket(rps=0.8,  burst=2),
    'estimateCharges':TokenBucket(rps=2,    burst=3),
    'gainersLosers':  TokenBucket(rps=0.2,  burst=1),   # we only need every 15 min
}
```

Every call path (snapshotter, saga, reconciler, planner) obtains a permit.
Redis-backed buckets (SETNX + TTL) so multi-worker deployments share quota.

### Caching

| Resource | Cache | TTL | Invalidation |
|---|---|---|---|
| `searchScrip` results | Redis + `instrument` table | 24h | Daily `ScripMaster` load |
| `ScripMaster` | Postgres `instrument` | 24h | 7:00 IST daily cron |
| `getMarginApi(basket)` | Redis, key = sha256(basket) | 5 min | Market hour flush on holiday |
| `estimateCharges(basket)` | Redis, key = sha256(basket-without-price) | 1h | On tariff change event (manual) |
| `rmsLimit` | Redis | 30s | On every fill (fill handler busts) |
| `gainersLosers` | Redis | 15 min | Market-close invalidation |
| Instrument LTP for UI | Redis | 1s during hours, 60s after | WS push bypass |

### Bulk upserts

- `MarketSnapshot` writes are buffered in a producer queue, flushed by a
  background writer in batches of 500 rows / 1 sec using
  `COPY ... FROM STDIN`. WebSocket fanout runs off the pre-flush queue for
  real-time UI without waiting on DB commit.
- `TradeLeg` writes are 1-at-a-time (rare), not batched — correctness matters
  more than throughput here.
- `EodSnapshot` is one row per tenant per day, written inside a daily cron.

### Partial & covering indexes

- `idx_order_request_open` — only open orders (majority of queries on a
  market day look at open orders)
- `idx_position_open` — only open positions (same reasoning)
- `idx_snapshot_position_time (position_leg_id, captured_at DESC)` —
  covers MAE/MFE queries
- `idx_postback_unprocessed` — small hot set for the processor

### Celery queue topology

```
broker-critical    → postback ingest (1 worker, 2 processes)  — never falls behind
broker-saga        → order placement/modify/cancel            — one per tenant shard
snapshot-hot       → hold snapshots for open positions        — fires 1m/5m/15m/1h by horizon
snapshot-plan      → plan-time snapshots                      — burst at signals
reconcile-daily    → EOD reconciliation                       — 3:31 PM IST
reconcile-intraday → 5-min order-state sanity                  — every 5 min market hours
aggregate          → monthly materialized view refresh        — delta-triggered
backfill           → historical candles                       — off-hours only
```

Priority: `broker-critical` > `broker-saga` > `snapshot-hot` > `reconcile-intraday`
> `reconcile-daily` > `snapshot-plan` > `aggregate` > `backfill`.

### WS topology

- `ws/monthly/{tenant_id}` — pushes month-card updates when a new
  `trade_leg` for the current month is written.
- `ws/positions/{tenant_id}` — pushes per-position MTM every N seconds or
  on significant move.
- `ws/alerts/{tenant_id}` — existing heartbeat/alert channel (ADR-0003).

Single Django Channels consumer group per tenant; consumers filter on
payload.

---

## 7. Phased delivery (revised, 7 phases, ~20 working days)

| Phase | Days | Outcome |
|---|---|---|
| **P1 — Schema + adapter protocol** | 3 | Migrations land; `BrokerAdapter` protocol defined; `PaperBrokerAdapter` stub that emits synthetic postbacks; feature flag gates ingestion. |
| **P2 — Ledger ingestion** | 3 | Postback endpoint + reconciler; `OrderRequest → OrderAck → BrokerPostback → TradeLeg → PositionLeg` FIFO matcher wired; recon runs nightly. |
| **P3 — Snapshot surface** | 3 | Plan / entry / hold / exit snapshot hooks in every agent; batched writer; local Black-Scholes greeks solver. |
| **P4 — Aggregation + Monthly API** | 2 | `MonthSnapshot` materialized view + refresh trigger; `/api/v1/monthly/` endpoint backed by real tables; flip `MONTHLY_USE_MOCK=false` behind feature flag. |
| **P5 — AngelBrokerAdapter** | 4 | Real SmartAPI calls for place/modify/cancel/snapshot/margin/charges/gtt/convert; rate limiter; session mgmt + TOTP re-auth. |
| **P6 — Edge case hardening** | 3 | BO/CO/AMO/GTT flows; basket atomicity; corporate actions V1 (flag-only); orphan trade handling; circuit-breaker gate. |
| **P7 — Observability + runbooks** | 2 | Metrics, traces, alert rules; runbook for postback outage; on-call flip-to-paper switch for incident response. |

P1 + P2 + P3 + P4 = 11 days to flip the Monthly View to real paper data.
P5 adds 4 days to enable live trading on the same plumbing.

---

## 8. Alternatives considered

- **Poll-only (no postback).** Rejected: adds 1-15s lag on state changes;
  noisy on the order-book endpoint; misses cancel/reject reasons.
- **Postback-only (no reconciler).** Rejected: Angel's webhook is
  best-effort; missing events would silently desync our ledger.
- **Mutable position table updated in place.** Rejected: loses the ability
  to replay a day, impedes audit, makes bugs unrecoverable. Append-only
  ledger + derived position is strictly better.
- **Single `Order` table conflating request, ack, and fill.** Rejected:
  one failure mode conflates with another; harder to reason about
  idempotency; worse for reporting.
- **Broker-agnostic adapter from day 1.** Partially rejected: the
  `BrokerAdapter` protocol is broker-agnostic, but every concrete adapter
  has quirks (Angel-specific enum strings, per-endpoint rate limits).
  We do not generalize until we add a second broker.
- **Column-level separation per broker.** Rejected: would create `angel_*`
  columns everywhere. Instead, keep broker-specific payloads in JSONB
  (`broker_postback.payload`) and normalized fields only when cross-broker
  needs justify it.

## Consequences

### Good

- Monthly view, backtests, and audits share one truth.
- Paper → live is a one-row UPDATE.
- Ingestion survives postback outages, duplicate events, out-of-order events.
- Rate-limit discipline is enforced at the infra layer, not ad-hoc in call sites.

### Costs

- Schema is wider than what the UI needs today — we're optimizing for
  where we're going, not where we are.
- `MarketSnapshot` table will grow fast (conservatively: 20 open positions ×
  ~240 snapshots/day = 4,800 rows/day/tenant). Plan to partition by month
  once we cross ~50M rows (likely Year 2).
- Postback handler must be rock-solid — a bug here desynchronizes the
  ledger. Mitigated by: ledger is append-only, so "bad ingest" is
  recoverable by re-running the matcher from the raw `broker_postback`
  table.

### Follow-ups

- **ADR-0007** — Backtesting on top of the ledger: the snapshot surface
  is the backtest input.
- **ADR-0008** — Corporate action automation (V2).
- **ADR-0009** — Second broker integration (Zerodha Kite?) — validates
  the adapter protocol.

## Implementation pointers

- `apps/broker/adapters/base.py` — `BrokerAdapter` protocol.
- `apps/broker/adapters/angel.py` — `AngelBrokerAdapter`.
- `apps/broker/adapters/paper.py` — `PaperBrokerAdapter` + fill simulator.
- `apps/broker/webhooks/postback.py` — POST handler with IP allowlist,
  path secret, raw insert.
- `apps/broker/services/ingest.py` — single ingest pipeline, postback and
  poll both feed it.
- `apps/broker/services/reconciler.py` — order/trade/position/holding diffs.
- `apps/broker/services/rate_limiter.py` — Redis token buckets.
- `apps/broker/services/snapshotter.py` — batched writer + scope dispatch.
- `apps/broker/services/greeks.py` — Black-Scholes IV solver + greeks.
- `apps/monthly/api/views.py` — `/api/v1/monthly/` backed by real tables.
- `apps/monthly/services/aggregator.py` — `MonthSnapshot` delta refresh.
- `trading/options/data_service.py` — refactor to call `BrokerAdapter`
  instead of `SmartApi` directly.
- `trading/services/data_service.py` — same refactor for equity path.
