# AlphaDesk Frontend (Vite + React 18 + TS)

## Dev
```bash
cp .env.example .env.local
npm install
npm run dev
```
Open http://localhost:5173. The Vite dev server proxies:
- `/api/v1/legacy/*` → port 8001 (legacy Django bridge — pyramid, screener, straddle)
- `/api/*` → port 8000 (v2 Django backend — auth, portfolios, monthly, orders)
- `/ws/*` → port 8000 (WebSocket — ticks, agent runs)

Set `VITE_MONTHLY_LIVE=1` in `.env.local` once your backend is running and you want the `/monthly` page to hit the live API instead of mock data. (There's also an in-page Mock/Live toggle in the Monthly page header for runtime switching.)

## Build
```bash
npm run build
```
Outputs static assets to `dist/` — deploy to Amplify / S3+CloudFront / Nginx.

## Structure
- `src/app/` — router (all routes), shell, guards
- `src/features/` — pages, one per feature:
  - **Cascade**: `pulse/`, `rotation/`, `shortlist/`, `setup/`
  - **Operational**: `dashboard/`, `positions/`, `agents/`, `strategies/`, `backtester/`, `broker/`
  - **Strategies**: `pyramid/`, `swing-scanner/`
  - **Feedback**: `monthly/`
  - **Auth**: `auth/`
- `src/components/{layout,ui}/` — shared components (Card, Button, Badge, KPI, DataTable, …)
- `src/lib/`
  - `api.ts` — axios + JWT refresh + paginated unwrap
  - `ws.ts` — WebSocket helper with subprotocol JWT auth
  - `market-config.ts` — ⭐ single source of truth for index metadata (NIFTY=65, BANKNIFTY=30, SENSEX=20, expiry days, tokens, exchange)
  - `market-pulse.ts` — pulse hook (used by `/pulse` and `/pyramid` ticker)
  - `monthly.ts` — monthly report types + hook + mock data
  - `legacy.ts` — typed wrappers around legacy bridge endpoints
- `src/stores/` — Zustand stores (auth, ws subscriptions)
- `src/types/` — shared TS types
- `src/styles/globals.css` — design tokens (CSS vars, `--fg-subtle: 138 143 152` style)
