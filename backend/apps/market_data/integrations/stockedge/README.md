# StockEdge — Market Breadth ingestion (vertical slice)

StockEdge analytics are an **advisory overlay** for AlphaDesk — an independent,
broad-universe read used to cross-check our own signals. They are **shared
reference data** (NOT tenant-scoped) and **never** reach @RiskGuard or position
sizing (Invariant #1 / #3 stand). Full design:
`docs/integrations/STOCKEDGE_INTEGRATION.md`.

This package ships the first end-to-end dataset: **Market Breadth** (% of each
index universe's constituents with RS > 0 and above SMA 20/50/100/200, across
Nifty 50 → Microcap 250).

## Layout

| File | Purpose |
|------|---------|
| `parser.py` | Pure normalization — `parse_breadth_payload()` / `summarize_breadth()`. No Django, no Playwright. |
| `harness.py` | Production capture via Playwright (lazy import). `capture_breadth()`. |
| `sample_breadth.json` | A captured payload for the MVP (2026-06-25 NSE). |
| `README.md` | This file. |

Models live in `apps/market_data/models.py`
(`StockEdgeSnapshot`, `StockEdgeBreadthRow`); the management command is
`apps/market_data/management/commands/pull_stockedge_breadth.py`.

## MVP — ingest a captured payload (no browser needed)

The MVP path ingests a captured JSON payload. With no flags it defaults to the
bundled sample:

```bash
# Ingest the bundled sample (upserts a StockEdgeSnapshot + 11 breadth rows)
.venv/bin/python manage.py pull_stockedge_breadth

# Ingest a payload you captured yourself
.venv/bin/python manage.py pull_stockedge_breadth --from-json /path/to/breadth.json

# Parse + print only, persist nothing
.venv/bin/python manage.py pull_stockedge_breadth --no-persist
```

The command upserts on `(dataset, as_of_date, exchange)` so re-pulls of the same
day replace that day's rows in place (history across days is preserved), prints
an aligned table sorted by breadth score, and prints the derived regime line.

## Production — Playwright harness + storage_state

Out-of-band API calls to `api.stockedge.com` are fingerprinted and return 504
even with a valid Bearer token (design doc §2.3). The robust, ToS-defensible
mechanism is to ride a **genuine logged-in session** and capture what the app
itself fetches. That requires Playwright (an optional dependency — it is lazily
imported, never a hard requirement) and a persisted logged-in session:

```bash
pip install playwright && playwright install chromium
```

1. **Create a `storage_state.json` once** by logging into
   <https://web.stockedge.com> with a persistent Playwright context (Google
   OIDC). Store it like any broker secret — never commit it.
2. **Capture live:**

```bash
.venv/bin/python manage.py pull_stockedge_breadth \
    --live --storage-state /secrets/stockedge_state.json
```

`capture_breadth()` launches headless Chromium, restores the session, navigates
to `/market-breadth`, waits for the grid, and extracts the breadth rows from the
DOM. It also registers a `page.on('response')` hook that snapshots any
`api.stockedge.com` breadth JSON as a fallback source. The result is the same
payload shape `parse_breadth_payload()` expects, so the persistence path is
identical to the MVP.

If Playwright isn't installed (or `--storage-state` is missing), the command
fails with a clear, actionable message and points you back at the `--from-json`
MVP path.

## Guardrails

- Personal Club subscription; no API forging / no bot-protection bypass — the
  harness uses the real session and official export surfaces only.
- Polite cadence: EOD/premarket batch, single concurrency, cache aggressively
  (breadth is daily).
- Advisory only: overlay/confirmation, never an execution input.
