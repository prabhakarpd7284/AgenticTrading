# AlphaDesk — Agentic Trading System

AI-driven Indian stock market trading desk. Built on Angel One SmartAPI,
with a single Django backend (REST + WebSocket + Celery) and a React UI.
Supports paper and live modes for equity intraday, options straddles, and
pyramid options strategies.

```
NIFTY · BANKNIFTY · SENSEX
     ↓ strategy plugins (7 ship today)
     ↓ canonical 10-criterion RiskEngine          ← last gate before broker
     ↓ Angel One (paper or live)
     ↓ events.Event firehose + Monthly Feedback   ← learn from every fired/skipped trade
```

---

## What's Inside

| Surface | Stack | Purpose |
|---|---|---|
| **React SPA** ([frontend/](frontend/)) | Vite · React 18 · TanStack Query · Tailwind | Trader UI — market pulse, positions, agents, backtester, monthly feedback report, pyramid live |
| **Django ASGI** ([backend/](backend/)) | Django 5 · DRF · Channels · Celery · Postgres · Redis | One process on :8000 — REST, WebSocket, broker WS holder. Multi-tenant. |
| **Strategy plugins** ([backend/plugins/](backend/plugins/)) | Plain Python packages, registered via entry-points | One package per engine — directional, short_straddle, pyramid, intraday_screener, swing_scanner, premarket_basket, backtest |

Phase notes: Phase 6 dropped the legacy SQLite DB and the second backend
port; Phase 8 dropped the Streamlit dashboard; Phase 4 collapsed 18 apps
into 13. See `docs/MIND_PALACE.md` for the current map.

---

## Quick Start (Local Dev)

> **TL;DR**: bring up Postgres + Redis in Docker, then one script starts
> the three native processes.
>
> ```bash
> docker compose -f docker-compose.dev.yml up -d
> bash scripts/dev_up.sh
> ```

### Prereqs

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12+ | `python --version` |
| Node | 20+ | `node --version` |
| Docker | any recent | Just for Postgres + Redis |
| [`uv`](https://github.com/astral-sh/uv) | recent | Python package manager — `brew install uv` |

### 1. Clone + secrets

```bash
git clone https://github.com/prabhakarpd7284/AgenticTrading.git
cd AgenticTrading

# Copy the env templates (these files are gitignored)
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

Defaults work for local dev. Live trading needs `SMARTAPI_*` in `.env`;
LLM-driven workflows need `ANTHROPIC_API_KEY` in `backend/.env`.

### 2. Infrastructure (Postgres :5436, Redis :6379)

```bash
docker compose -f docker-compose.dev.yml up -d
```

Only these two services run in Docker — everything else is native.

### 3. Backend (one-time setup)

```bash
cd backend
uv sync                          # installs deps + all 7 strategy plugins
uv pip install -e .              # registers entry-points (alphadesk.strategies/.brokers)
.venv/bin/python manage.py migrate
cd ..
```

### 3b. Move data to a new machine (optional — for a populated UI)

App data (trades, signals, portfolios, events) lives in Postgres, not git — and
the DB holds **broker credentials + PII**, so dumps are **gitignored and must be
transferred out-of-band** (scp / cloud), never committed.

On the source machine, dump it:

```bash
docker exec alphadesk-pg pg_dump -U alphadesk -d alphadesk -Fc \
  --no-owner --no-privileges > dumps/alphadesk.dump
```

Copy `dumps/alphadesk.dump` to the target machine, then (Postgres up, step 2):

```bash
docker exec -i alphadesk-pg pg_restore -U alphadesk -d alphadesk \
  --clean --if-exists --no-owner < dumps/alphadesk.dump
```

Skip this for an empty DB. To (re)generate trade data instead of restoring,
replay it from live candles (paper):

```bash
cd backend
.venv/bin/python manage.py derive_trades       --from 2026-05-01 --to 2026-05-31  # intraday
.venv/bin/python manage.py derive_swing_trades  --from 2026-05-01 --to 2026-05-31  # swing (Oliver Kell)
```

### 4. Frontend (one-time setup)

```bash
cd frontend && npm install && cd ..
```

### 5. Start everything

```bash
bash scripts/dev_up.sh
```

Brings up:

| Port | Process | Logs |
|---|---|---|
| `:8000` | Django ASGI (REST + WebSocket) | `logs/web.log` |
| —     | Celery worker (order outbox, agent runs) | `logs/celery.log` |
| `:5173` | Vite (React SPA) | `logs/vite.log` |

Stop them with `bash scripts/dev_down.sh` (leaves the docker containers running).

**Verify**:
- API docs at <http://localhost:8000/api/docs/>
- UI at <http://localhost:5173> — sign up, log in, land on Market Pulse

### Tests

```bash
cd backend && .venv/bin/python -m pytest -q
# expect: 99 passed
```

Pytest builds a fresh test DB from migrations every run — doubles as the
canonical fresh-install verification.

### Common first-run issues

| Symptom | Cause | Fix |
|---|---|---|
| `dev_up.sh` says "Postgres container 'alphadesk-pg' isn't running" | Step 2 skipped | `docker compose -f docker-compose.dev.yml up -d` |
| `dev_up.sh` says `.venv/bin/python not found` | Step 3 skipped | `cd backend && uv sync && uv pip install -e .` |
| `connection refused` on port 5432 | Stale `DATABASE_URL` | Edit `backend/.env` — port is **5436** (compose exposes it that way) |
| Frontend 401 errors | JWT expired or not logged in | Sign up at `/signup`, log in at `/login` |
| `/pyramid` says "No option candles found for SENSEX…" | SmartAPI not logged in | Set valid `SMARTAPI_*` in `.env` or use the page's mock-data toggle |

---

## CLI Workflows

The strategy engines are also exposed as Django management commands.
Run from `backend/` (or use `.venv/bin/python` as shown).

### Equity intraday
```bash
.venv/bin/python manage.py run_trading_agent "Plan a BUY trade for HDFCBANK"
.venv/bin/python manage.py run_trading_agent --show-journal
```

### Live screener (intraday opportunities)
```bash
.venv/bin/python manage.py run_screener
.venv/bin/python manage.py run_screener --telegram
.venv/bin/python manage.py run_screener --backtest --from 2026-04-01 --to 2026-04-30
```

### Oliver Kell swing scanner (daily/weekly cycles)
```bash
.venv/bin/python manage.py run_ok_scanner --actionable-only --telegram
```

### Pyramid options
```bash
.venv/bin/python manage.py run_pyramid --strike 24200 --type CE --underlying NIFTY
```

### Straddle lifecycle
```bash
.venv/bin/python manage.py manage_straddle --analyze --position 1
```

### EOD signal enrichment (feeds the Monthly Feedback report)
```bash
.venv/bin/python manage.py enrich_signals          # today
.venv/bin/python manage.py enrich_signals --all    # backfill
```

### Full trading day
```bash
.venv/bin/python manage.py run_trading_day
```

---

## Configuration

### Environment files

| File | Purpose |
|---|---|
| `.env` | Root — Angel One credentials, Telegram tokens |
| `backend/.env` | Django secret, DB URL, Redis URL, Anthropic key, risk caps |
| `frontend/.env.local` | API base URLs for the Vite dev server |

All `.env*` files are gitignored. Templates are checked in as `.env.example`.

### Key variables

| Variable | Default | Notes |
|---|---|---|
| `TRADING_MODE` | `paper` | `paper` or `live`. Set deliberately. |
| `PLANNER_MODE` | `cli` | `cli` (free, Max plan) or `api` (Anthropic credits) |
| `LLM_MODEL` | `claude-sonnet-4-6` | Used by both modes |
| `DEFAULT_CAPITAL` | `500000` | Starting capital in INR |
| `MAX_RISK_PER_TRADE_PCT` | `1.0` | RiskEngine criterion 5 |
| `MAX_DAILY_LOSS_PCT` | `3.0` | RiskEngine criterion 6 |
| `MAX_POSITION_SIZE_PCT` | `10.0` | RiskEngine criterion 7 |

### Index lot sizes (Jan 2026+)

| Index | Lot size | Weekly expiry | Exchange |
|---|---|---|---|
| NIFTY | 65 | Tuesday | NSE (NFO) |
| BANKNIFTY | 30 | Last Tuesday (monthly only) | NSE (NFO) |
| SENSEX | 20 | Thursday | BSE (BFO) |

Single source of truth: [`frontend/src/lib/market-config.ts`](frontend/src/lib/market-config.ts).

---

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  React SPA (Vite, :5173)                                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │ REST + WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  ONE Django ASGI process (:8000)                                 │
│    · DRF REST (/api/v1/...)                                      │
│    · Channels WebSocket (/ws/...)                                │
│    · Broker WS consumer (Channels worker, in-process)            │
│    · Multi-tenant (Tenant FK + JWT tenant_id)                    │
│    · Canonical 10-criterion RiskEngine                           │
│    · Workflow framework + plugin registry                        │
└──────┬──────────────────────────────────────────┬───────────────┘
       │                                          │
       ▼                                          ▼
  Celery worker                          Angel One SmartAPI
  (outbox, snapshots,                    (NSE NFO + BSE BFO)
   agent runs)
       │
       ▼
  Postgres (:5436)    Redis (:6379)         ← docker compose -f docker-compose.dev.yml
  pgvector for RAG    cache + queue
                      + pub/sub
```

See [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md)
for the full spec, and [`docs/MIND_PALACE.md`](docs/MIND_PALACE.md) for the
spatial mental model.

---

## Key Invariants (Never Break These)

1. **RiskEngine is the last gate** before every order — paper or live. No bypass.
2. **`TRADING_MODE=paper`** is the default. Live is opt-in and explicit.
3. **Position sizing is deterministic** (% of capital). LLM never touches it.
4. **Every decision lands in `events.Event`** — including rejected trades.
5. **Event writes are non-blocking** — logging failures never stop a trade.
6. **Expiry-day options must close before 15:15 IST** — gamma risk explodes after.
7. **Short straddle hard stop** — combined premium > 1.3× sold → close immediately.

---

## Documentation

| Document | What it covers |
|---|---|
| [`docs/MIND_PALACE.md`](docs/MIND_PALACE.md) | Spatial map of the whole system |
| [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) | Formal architecture spec |
| [`docs/architecture/BACKEND_STRUCTURE.md`](docs/architecture/BACKEND_STRUCTURE.md) | Django app layout |
| [`docs/api/WEBSOCKETS.md`](docs/api/WEBSOCKETS.md) | WebSocket channel spec |
| [`docs/api/openapi.yaml`](docs/api/openapi.yaml) | OpenAPI 3 spec |
| [`docs/strategies/`](docs/strategies/) | Strategy playbooks |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records |
| [`docs/SCREENER_SPEC.md`](docs/SCREENER_SPEC.md) | Live screener spec |
| [`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md) | Test pyramid + coverage targets |
| [`TRADING_FRAMEWORK.md`](TRADING_FRAMEWORK.md) | The Cascade — 6-stage pre-trade framework |
| [`CLAUDE.md`](CLAUDE.md) | Project instructions for Claude Code |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5 · DRF · Channels · Celery · Postgres · Redis (pgvector) |
| LLM | Claude (CLI via Max plan, or API) · LangGraph for workflow nodes |
| Broker | Angel One SmartAPI (`SmartApi` package) |
| Real-time | Django Channels · Redis pub/sub · WebSocket |
| Frontend | Vite · React 18 · TypeScript · TanStack Query · Tailwind |
| Charting | Recharts · TradingView lightweight-charts |
| Logging | structlog · Sentry |
| Plugin loading | Python entry-points (`importlib.metadata`) |

---

## License

Private. All rights reserved.
