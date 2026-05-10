# AlphaDesk — Agentic Trading System

AI-driven Indian stock market trading platform. Built on Angel One SmartAPI, with a Django backend, React frontend, and a Streamlit operator dashboard. Supports paper and live modes for equity intraday, options straddles, and pyramid options strategies.

```
NIFTY · BANKNIFTY · SENSEX  →  Screener / Scanner  →  @RiskGuard  →  Broker  →  Journal + Feedback Loop
```

---

## What's Inside

| Surface | Stack | Purpose |
|--------|-------|---------|
| **Streamlit dashboard** (`dashboard.py`) | Streamlit + Django ORM | Operator console — all original tooling, multi-timeframe charts, scanners, manual workflow. |
| **React SPA** (`frontend/`) | Vite + React 18 + TanStack Query + Tailwind | Trader-facing UI: market pulse, positions, agents, backtester, **monthly feedback report**, **pyramid live**. |
| **v2 Django stack** (`backend/`) | Django 5 + DRF + Channels + Celery | Multi-tenant API, WebSocket ticks, async order execution. |
| **Trading core** (`trading/`) | Pure-Python + Django models | Strategies, screener, scanner, backtester, broker integration, risk engine. |

---

## Quick Start (Local Dev)

### Prereqs
- Python 3.12+
- Node 20+
- Angel One SmartAPI credentials (for live data; paper mode works without)

### 1. Clone + secrets

```bash
git clone https://github.com/prabhakarpd7284/AgenticTrading.git
cd AgenticTrading

# Set up your .env files (NEVER commit these)
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local

# Edit each .env to fill in your credentials
```

### 2. Trading core (legacy Django + Streamlit)

```bash
# In project root
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Database
python manage.py migrate

# Streamlit dashboard
streamlit run dashboard.py
```

### 3. v2 Backend (Django REST + WebSocket)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000

# In another shell — Celery worker for async order execution
celery -A config worker -Q default,agents,orders -l info
```

API docs: http://localhost:8000/api/docs/

### 4. Frontend (React SPA)

```bash
cd frontend
npm install
npm run dev
```

Opens at http://localhost:5173. Vite proxies `/api/v1/legacy/*` → port 8001 (legacy bridge), everything else → port 8000 (v2 backend).

### 5. Optional — Docker Compose (full stack in one command)

```bash
docker compose -f docker-compose.local.yml up
```

Brings up Postgres, Redis, backend, worker, frontend, and the legacy bridge.

---

## CLI Workflows

### Equity intraday agent
```bash
python manage.py run_trading_agent --seed-strategies        # one-time
python manage.py run_trading_agent --init-portfolio 500000  # one-time
python manage.py run_trading_agent "Plan a BUY trade for HDFCBANK"
python manage.py run_trading_agent --show-journal
```

### Live screener (intraday opportunities)
```bash
python manage.py run_screener                # live
python manage.py run_screener --telegram     # with Telegram alerts
python manage.py run_screener --backtest --from 2026-04-01 --to 2026-04-30
```

### Oliver Kell swing scanner (daily/weekly cycles)
```bash
python manage.py run_ok_scanner --actionable-only --telegram
```

### Pyramid options strategy
```bash
python manage.py run_pyramid --strike 24200 --type CE --underlying NIFTY
```

### Straddle lifecycle
```bash
python manage.py manage_straddle --register --underlying NIFTY \
  --strike 24200 --expiry 2026-05-13 \
  --ce-symbol NIFTY13MAY2624200CE --ce-token 41762 --ce-sell 394.85 \
  --pe-symbol NIFTY13MAY2624200PE --pe-token 41763 --pe-sell 138.35 --lots 1

python manage.py manage_straddle --analyze --position 1
```

### End-of-day signal enrichment (feeds the monthly feedback report)
```bash
python manage.py enrich_signals             # today
python manage.py enrich_signals --all       # backfill
```

### Full trading day (premarket → screener → straddle monitor → review)
```bash
python manage.py run_trading_day
```

---

## Configuration

### Environment files

| File | Purpose |
|------|---------|
| `.env` | Root — Angel One credentials, Telegram, symbol master path |
| `backend/.env` | v2 backend — Django secret, DB URL, Redis URL, Anthropic key |
| `frontend/.env.local` | API base URLs for the React dev server |

All `.env*` files are gitignored. Use the `.env.example` files as a template.

### Key environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `TRADING_MODE` | `paper` | `paper` or `live`. Set deliberately. |
| `PLANNER_MODE` | `cli` | `cli` (free, Max plan) or `api` (Anthropic credits) |
| `LLM_MODEL` | `claude-sonnet-4-6` | Used by both modes |
| `DEFAULT_CAPITAL` | `500000` | Starting capital in INR |
| `MAX_RISK_PER_TRADE_PCT` | `1.0` | @RiskGuard gate |
| `MAX_DAILY_LOSS_PCT` | `3.0` | @RiskGuard gate |
| `MAX_POSITION_SIZE_PCT` | `10.0` | @RiskGuard gate |

### Index lot sizes (Jan 2026+)

| Index | Lot size | Weekly expiry | Exchange |
|-------|---------|---------------|----------|
| NIFTY | 65 | Tuesday | NSE (NFO) |
| BANKNIFTY | 30 | Last Tuesday (monthly only) | NSE (NFO) |
| SENSEX | 20 | Thursday | BSE (BFO) |

Single source of truth: `frontend/src/lib/market-config.ts`.

---

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React, port 5173)                                     │
│   ↓                              ↓                                │
│  v2 Backend (port 8000)         Legacy bridge (port 8001)        │
│   • Auth, Tenants, Portfolios    • Pyramid, Screener, Straddle   │
│   • Orders (outbox + Celery)     • TradeJournal, AuditLog,       │
│   • Market pulse, Monthly         SignalLog, StraddlePosition    │
│   • WebSocket /ws/ticks/                                          │
│   ↓                                                               │
│  Postgres + Redis (v2)          SQLite (legacy)                  │
└─────────────────────────────────────────────────────────────────┘
           │                                │
           └────────────┬───────────────────┘
                        ▼
              Angel One SmartAPI
              (NSE, NFO, BSE, BFO)
```

The legacy `trading/` package owns all strategy logic, broker integration, and the @RiskGuard implementation. The v2 backend wraps it with a multi-tenant API layer. The Streamlit dashboard talks directly to the legacy DB.

See `docs/architecture/ARCHITECTURE.md` for the full diagram, and `docs/MIND_PALACE.md` for the spatial mental model of the whole system.

---

## Key Invariants (Never Break These)

1. **@RiskGuard is the last gate** before every order — paper or live. No bypass.
2. **`TRADING_MODE=paper`** is the default. Live is opt-in and explicit.
3. **Position sizing is deterministic** (% of capital). LLM never touches it.
4. **Every decision is journaled** — including rejected trades.
5. **Audit log failures never block trading** — non-blocking.
6. **Expiry-day options must close before 15:15 IST** — gamma risk explodes after.
7. **Short straddle hard stop** — combined premium > 1.3× sold → close immediately.

---

## Documentation

| Document | What it covers |
|----------|---------------|
| `docs/MIND_PALACE.md` | Spatial mental map of the whole system — every floor, every desk, every data pipe |
| `docs/architecture/ARCHITECTURE.md` | Formal architecture spec — components, security, SLOs, capacity |
| `docs/architecture/BACKEND_STRUCTURE.md` | v2 Django app layout |
| `docs/api/WEBSOCKETS.md` | WebSocket channel spec |
| `docs/api/openapi.yaml` | OpenAPI 3 spec |
| `docs/strategies/` | Strategy playbooks (e.g. directional vertical spread) |
| `docs/adr/` | Architecture Decision Records (ADR-0001..0006) |
| `docs/SCREENER_SPEC.md` | Live screener strategy + filter spec |
| `docs/TESTING_STRATEGY.md` | Test pyramid + coverage targets |
| `TRADING_FRAMEWORK.md` | The Cascade — 6-stage pre-trade framework |
| `CLAUDE.md` | Project instructions for Claude Code (the AI assistant) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend (v2) | Django 5 · DRF · Channels · Celery · Postgres · Redis |
| Backend (legacy) | Django 5 · LangGraph · SQLite |
| Frontend | Vite · React 18 · TypeScript · TanStack Query · Tailwind |
| Charting | Recharts · TradingView lightweight-charts |
| LLM | Claude (CLI via Max plan, or API) |
| Broker | Angel One SmartAPI (`SmartApi` package) |
| Real-time | Django Channels · Redis pub/sub · WebSocket |
| Logging | structlog · Logzero · Sentry |
| Deploy | Docker · ECS Fargate · Amplify · CloudWatch · Terraform |

---

## License

Private. All rights reserved.
