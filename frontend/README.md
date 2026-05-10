# AlphaDesk Frontend (Vite + React 18 + TS)

## Dev
```bash
cp .env.example .env
npm install
npm run dev
```
Open http://localhost:5173. The Vite dev server proxies `/api` and `/ws` to the
Django backend on :8000.

## Build
```bash
npm run build
```
Outputs static assets to `dist/` — deploy to Amplify / S3+CloudFront / Nginx.

## Structure
- `src/app/` — router, shell, guards
- `src/features/{dashboard,positions,agents,strategies,backtester,broker,auth}/` — pages
- `src/components/{layout,ui,charts}/` — shared components
- `src/lib/` — API client, WS client, utils
- `src/stores/` — Zustand stores
- `src/types/` — shared TS types
