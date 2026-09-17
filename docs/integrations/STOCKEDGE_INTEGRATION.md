# StockEdge Integration — Design Doc

> Status: **Draft / exploration complete** · Owner: AlphaDesk · Last updated: 2026-06-28
>
> Goal: pull StockEdge's ready-made market analytics (breadth, sector rotation, FII/DII
> flows, F&O OI/PCR, momentum scans) into AlphaDesk as a **research / confirmation
> overlay** — *not* as a live execution feed. Angel One stays the source of truth for
> prices and orders.

---

## 1. Why StockEdge

StockEdge (Club subscription) already computes a large amount of analytics that AlphaDesk
currently derives from raw candles, plus several it doesn't have at all. Using it as an
overlay lets the AI team cross-check its own signals against an independent, broad-universe
data source.

| StockEdge feature | AlphaDesk consumer | Value |
|---|---|---|
| **Market Breadth** (% constituents > SMA 20/50/100/200, RS>0, per index universe) | `/pulse` (Cascade Stage 1) | Independent breadth read across Nifty 50 → Microcap 250; today e.g. Nifty Bank 93% > SMA20, Nifty IT weak |
| **Sector Rotation** (per-sector RS / RSI>50 / SMA breadth / deliveries / VWAP) | `/rotation` (Stage 3) | Pre-ranked sector momentum with MCap context |
| **FII / DII Activity** + **FII Cash** flow (multi-day) | planner context, `/pulse` | Institutional directional bias |
| **F&O Zone → Futures** (OI buildup: New Long / New Short / Short Cover / Long Unwind) | directional planner, pyramid | Confirms/!confirms structural bias per F&O name |
| **F&O Zone → Options** (per-stock **PCR**, 1M PCR trend, PCR 1D change) | straddle workflow | Options sentiment input |
| **Scans** (Price / Volume&Delivery / Technical / Fundamental / Candlestick / **Volume&Options (OI)** / StockEdge Score; + Custom & Combination) | screener (8 strategies), `/shortlist` | Ready-made screeners with live counts (e.g. "Closing Above Previous High" = 70, "Increase in FII Shareholding" = 225) |
| **Trading Strategies** (named momentum/reversal/continuation setups + backtests) | screener, planner | Curated setups with "Perfect Match / Almost There" buckets per day |
| **Chart Patterns** (AI: Triple Top, Falling Wedge, Symmetrical Triangle + price target) | planner context | Pattern + target enrichment |
| Trending: Volume Shockers, 52W / All-time H/L, Most Visited; Deals (bulk/block) | screener, watchlist | Breadth of unusual activity |

---

## 2. Technical access analysis

### 2.1 API surface
The web app is an Ionic/Angular **PWA** that calls a clean REST API at
`https://api.stockedge.com/Api/...`. Endpoints confirmed during exploration:

```
# Market home dashboard
MarketHomeDashboardApi/GetLatestFIIAndMainIndices?lang=en
MarketHomeDashboardApi/GetIndexAdvanceDeclines?lang=en
MarketHomeDashboardApi/GetTopPriceMovers?gainerLosersTypeEnum=1&lang=en
MarketHomeDashboardApi/GetTrendingStockInsights?lang=en
MarketHomeDashboardApi/GetTopNewsItems?lang=en
MarketHomeDashboardApi/GetIntradayChartSecurities?lang=en
MarketHomeDashboardApi/GetLatestTVVideos?lang=en

# Trading strategy / ready-combination scans  (Cannon Momentum = id 60)
AlertsApi/GetRelevantListingCategories?lang=en
ReadyCombinationScanAlertTypesDashboardApi/GetAlertsForReadyCombinationScan/{id}/{yyyy}/{mm}/{dd}?relevantListings=10&page={n}&pageSize=10&lang=en
ReadyCombinationScanAlertTypesDashboardApi/GetAlertsForReadyCombinationScanForCSV/{id}/{yyyy}/{mm}/{dd}?relevantListings=10&lang=en   # <-- CSV export

# Page routes whose data endpoints follow the same *DashboardApi pattern
#   (capture exact URLs via the harness's own network interception — see §3):
/market-breadth         (Breadth · Scores · Periodic H/L · Adv/Dec)
/sector-rotation        (Sectors · Industries → Breadth · Scores · Deliveries · VWAP)
/derivative-analytics   (Futures OI buildup · Options PCR)   ?section=options
/scan-groups            (scan library + per-scan results)
```

**CSV export pattern:** every data table with a download button hits a sibling
`...ForCSV/...` endpoint that returns the full dataset (not just the rendered page).
This is the cleanest dataset to ingest — it's StockEdge's own supported export.

### 2.2 Auth
- JWT **Bearer** access token, `iss=accounts.stockedge.com` (IdentityServer / OIDC,
  Google IdP), `aud=api.stockedge.com`. **~24h expiry**, refreshable via `offline_access`.
- The token's `Features[]` claim is the subscription scope and confirms Club access:
  `marketbreadth, derivativeanalytics, sectorrotation, combinationscans, customscan,
  advancedfilter, score, fscore, sectoranalytics, indexanalytics, download, download2,
  readycombinationscans, chartpattern, industryrotation, technicalpeer, ...`

### 2.3 The blocker (important — drove the architecture choice)
Direct/out-of-band calls to these endpoints **fail with HTTP 504 even with a valid Bearer
token**, while the app's own identical requests return 200. Diagnosis:
- Requests issued from an **injected / isolated browser context** (and, by extension, a
  naive server-side HTTP client) are fingerprinted by the gateway and dropped (504). It is
  **not** an auth failure — a valid token does not help.
- The PWA service worker (`se-worker.js` / Angular `ngsw`) caches **app assets only**;
  zero `api.stockedge.com` data is in Cache Storage or IndexedDB, so there's no local
  cache to read.

**Conclusion:** forging these API calls server-side is brittle *and* a bypass of bot
protection (against StockEdge ToS). **We will not do that.** The only robust, defensible
mechanism is to let the **genuine app session** fetch, and capture the result.

---

## 3. Architecture — browser-automation harness

A headless **Playwright** (Chromium) worker that holds a logged-in StockEdge session and
extracts data through the app's own legitimate requests.

```
┌─────────────────────────────────────────────────────────────────┐
│  stockedge_harness  (Playwright, headless Chromium)               │
│                                                                   │
│  1. Restore session  (persisted storage_state.json; Google OIDC)  │
│  2. For each target page:                                         │
│       a. page.goto(route)                                         │
│       b. page.on('response') captures api.stockedge.com JSON      │
│          bodies  ── OR ──  click the page's Download button and   │
│          read the CSV the app writes                              │
│  3. Normalize → write to stockedge_* staging tables               │
│  4. Emit events.Event(type='stockedge.snapshot') (non-blocking)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        backend/apps/market_data/integrations/stockedge/
```

Why Playwright (not the exploration tool / not raw HTTP):
- `page.on('response')` reads bodies from the **main world** — the isolated-world and 504
  problems in §2.3 don't apply.
- Rides the real session, so no API forging; uses official export buttons where present.
- Same engine can solve OIDC login once and persist `storage_state` (refresh tokens give
  ~offline access; re-auth only when the refresh token lapses).

### 3.1 Where it lives
- New module: `backend/apps/market_data/integrations/stockedge/`
  - `client.py` — Playwright session manager (login, storage_state, response capture)
  - `pages.py` — `{page → route, capture strategy, parser}` registry
  - `parsers.py` — JSON/CSV → normalized rows
  - `tasks.py` — Celery tasks (one per dataset) + a `refresh_all` orchestrator
- Models: `backend/apps/market_data/models.py` (new `stockedge_*` tables, §4)
- Schedule: Celery beat (§5)

---

## 4. Staging data model (sketch)

Keep it **append-only snapshots** keyed by `(dataset, as_of_date, captured_at)` so we never
overwrite history and can diff our own signals against StockEdge over time.

```python
class StockEdgeSnapshot(models.Model):       # one row per (dataset, as_of, capture)
    dataset      = models.CharField()         # 'market_breadth' | 'sector_rotation' | 'fii_dii' | 'fno_oi' | 'fno_pcr' | 'scan' | 'strategy'
    as_of_date   = models.DateField()
    captured_at  = models.DateTimeField(auto_now_add=True)
    source_url   = models.URLField()
    raw          = models.JSONField()         # full normalized payload
    meta         = models.JSONField(default=dict)  # e.g. scan_id, strategy_id, listing filter

# Optional flattened tables for hot queries (derive from raw):
#   StockEdgeBreadthRow(index_name, rs_pos, pct_above_sma20/50/100/200, as_of)
#   StockEdgeSectorRow(sector, rs_pos, rsi_gt50, pct_above_sma*, mcap_cr, as_of)
#   StockEdgeFnoRow(symbol, oi_trend, spot_chg_pct, cum_oi, cum_oi_chg_pct, voldel, lot, as_of)
#   StockEdgePcrRow(symbol, pcr, pcr_chg_1d, spot, chg_pct, as_of)
#   StockEdgeScanHit(scan_id, scan_name, symbol, ltp, as_of)
```

Multi-tenant note: this is **shared reference data**, not tenant-scoped. Store it
un-tenanted (or under a system tenant) and expose read-only to all tenants — do **not**
put it behind `TenantModel`.

---

## 5. Refresh schedule (Celery beat)

| Dataset | Cadence | Notes |
|---|---|---|
| FII/DII + FII Cash | EOD ~18:30 IST | Settles after market close |
| Market Breadth | EOD + optional 15:35 IST | Daily breadth snapshot |
| Sector Rotation | EOD | |
| F&O OI buildup / PCR | EOD ~18:00 IST | After F&O data settles |
| Momentum Scans / Strategies | Premarket ~08:45 IST | Feeds the day's shortlist |
| Chart Patterns | Premarket | |

Token/session health-check task hourly; re-auth flow if the refresh token lapses.

---

## 6. Integration points into AlphaDesk

- **`/pulse`** — overlay StockEdge breadth + FII alongside our computed pulse; flag divergence.
- **`/rotation`** — blend StockEdge sector RS/RSI ranking with ours (or show side-by-side).
- **Screener** — use StockEdge scan/strategy hits as an *independent confirmation* set; a
  symbol firing both our screener and a StockEdge momentum scan = higher conviction.
- **Straddle (`@OptionsStrategist`)** — feed index/stock PCR + OI buildup into analyzer
  context (sentiment only; @RiskGuard unchanged).
- **Planner (`@DirectionalTrader`)** — attach chart-pattern + OI-buildup tags to the RAG
  context for a candidate.
- Always **advisory** — StockEdge never reaches @RiskGuard or position sizing. Invariant #3
  (deterministic sizing) and #1 (@RiskGuard last gate) stand.

---

## 7. Guardrails (ToS / legal / operational)

1. **Personal Club subscription.** Confirm StockEdge's terms before productionizing.
   Preferred long-term path: **ask StockEdge about an official API / data-licensing tier**
   for a commercial product. Track as an open item.
2. **No API forging / no bot-protection bypass.** Harness uses the real session + official
   export buttons only. If StockEdge rate-limits or blocks automation, we stop.
3. **Polite cadence.** EOD/premarket batch only; no intraday hammering; jittered, single
   concurrency. Cache aggressively (data is daily).
4. **Secrets.** StockEdge session/refresh token stored like any broker secret (env / secret
   manager), never committed. Short-lived access token never persisted in code or logs.
5. **Advisory only.** Overlay/confirmation; never an execution input.

---

## 8. Open questions / next steps

- [ ] Confirm StockEdge ToS position on automated personal-account export; open licensing
      conversation.
- [ ] Stand up the Playwright session manager + OIDC login persistence (the only non-trivial
      engineering piece).
- [ ] Capture the exact `*DashboardApi` endpoints for market-breadth / sector-rotation /
      derivative-analytics via `page.on('response')` (couldn't enumerate out-of-band due to §2.3).
- [ ] Decide JSON-capture vs CSV-download per dataset (CSV is cleaner where a download
      button exists).
- [ ] Build `stockedge_*` models + one end-to-end dataset (suggest **Market Breadth** first).
```
