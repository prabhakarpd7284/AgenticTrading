# R0 inventory: the 21 sidebar pages

Date: 2026-10-21. There is no usage telemetry, so this inventory rests on three things:

- the code (each page's hooks and API calls);
- live `GET` requests to `:8000`, made one at a time on 2026-10-21 between 19:15 and 19:20 IST (after the close), as tenant `istpduckymomo`;
- the date of the last git commit per feature, as a weak usage proxy.

No POST requests were sent and no broker refresh was forced.

Spec: `docs/ALPHADESK_UX_REDESIGN_PLAN.md` §1–2. Destination = where the page now sits in the regrouped sidebar.

| Page (route) | What it shows | API calls | Live data on this machine | Last touched | Destination |
|---|---|---|---|---|---|
| Market Pulse `/pulse`, index `/` | Regime, Indian and global quotes, sector heatmap, guidance | GET `/market-data/pulse/` | 200, 29 s cold. **Real**: NIFTY 22811, VIX 15.7, 11 sectors. Quotes come from yfinance | 2026-05-10 | Today |
| Rotation `/rotation` | Sector rotation drill-in (Cascade stage 3) | GET `/market-data/rotation/`, pulse | Over 30 s cold; 200 once warm. **Real**: 11 sectors | 2026-05-10 | Today |
| Shortlist `/shortlist` | Tradeable names from hot sectors (stage 4) | GET `/market-data/shortlist/`, pulse | Over 30 s cold. **Real**: 4 hot sectors, 15 candidates | 2026-05-10 | Today |
| Setup `/setup/:symbol` (not in the sidebar) | One stock's setup plus the @RiskGuard breakdown | GET `/market-data/setup/`, `/events/…`, `/watchlists/…`, `/positions/`, `/trades/` | 200. **Real** candles and plan. Signals, saved setups and watchlists are all 0 | 2026-06-26 | Today (drill-down) |
| Swing Scanner `/swing-scanner` | Oliver Kell phase scan of 98 names | GET `/market-data/swing-scanner/`, pulse | 97 s cold. **Real**: 13 active, 0 buy-aligned | 2026-05-18 | Today |
| Basket `/basket` | Premarket mood plus equity/option legs | GET `/market-data/basket/`, pulse | Over 30 s cold. **Real**: BEARISH, 2 legs | 2026-05-18 | Today |
| Nightly Scan `/nightly-scan` | Next-day levels, news sentiment, allowed directions, cooldown | GET `/nightly-scan/latest/`, `/trading-rules/cooldown/` | **Real**: for 2026-10-21, success, 49 rows, 5 sources OK | 2026-10-21 | Today |
| Desk `/dashboard` | Capital, P&L, risk, AI pause, positions, agent runs, audit | GET `/portfolios/`, `/portfolios/summary/`, `/positions/`, `/risk/`, `/risk/alerts/`, `/system/`, `/events/audit/`, `/agents/runs/`; WS `/ws/pnl/` | 200, but **empty/zero**: capital 500k, 0 positions, 0 P&L, 0 alerts. 3 agent runs | 2026-05-18 | Portfolio |
| Positions `/positions` | Equity and options positions with live LTP, trades | GET `/positions/`, `/options-positions/`, `/trades/`; WS `/ws/ticks/` | **Empty** (0 / 0 / 0) | 2026-06-28 | Portfolio |
| Options Desk `/options` | Options cockpit: chain, expiries, risk, start an agent | GET `/market-data/options-chain/`, `/market-data/expiries/`, `/brokers/`, risk, positions; POST `/agents/runs/` | **Real Angel One chain** (`is_fallback=false`), 21 strikes, 18 expiries. Positions are empty | 2026-06-03 | Portfolio |
| Monthly `/monthly` | Monthly feedback: capture matrix, signal audit, equity curve, lessons | GET `/portfolios/monthly/`, `trades/<id>/chart/`, `trades/<id>/feedback/` | Backend returns **real but thin** data: 29 signals, 0 trades. **The UI shows mock data by default** (`lib/monthly.ts` `USE_MOCK` unless `VITE_MONTHLY_LIVE=1`) | 2026-06-03 | Portfolio (Performance) |
| Mutual Funds `/mutual-funds` | Holdings, allocation, advisor findings, NAV freshness, search, screener | GET `/mutual-funds/holdings/`, `status/`, `advisor/findings/`, `schemes/`, `screener/` | Holdings are **empty** (no CAS imported). **Real catalog**: 14,833 schemes, NAV date 2026-10-16 | 2026-10-21 | Portfolio |
| Agent Console `/agents` | Start and watch LangGraph agent runs | GET `/agents/catalog/`, `/agents/runs/…`; POST `/agents/runs/`; WS `/ws/agents/<id>/` | **Real**: 10 strategies, 3 runs. The last `scalp` run **failed** (2026-10-18) and nothing surfaced it | 2026-06-27 | Runs |
| Strategies `/strategies` | Strategy instances, "new strategy" (saves empty params) | GET `/strategies/instances/`, `/agents/catalog/`, `/rag/knowledge/` | **Empty**: 0 instances, 0 knowledge docs | 2026-05-18 | Strategies |
| Backtester `/backtester` | OK-cycle backtest in 3 hard-wired modes | GET `/market-data/ok-backtest/` (on Run) | Not probed (heavy, on demand) | 2026-05-18 | Strategies |
| Pyramid `/pyramid` | Pyramid options backtest, then "Go Live (Paper)" | GET `/strategies/pyramid/backtest/` (on Run); WS `/ws/ops/` | Not probed. It has a **synthetic sample candles** toggle | 2026-06-24 | Strategies |
| Scalping `/scalping` | Bin-pressure scalp simulator, tick replay | GET `/strategies/scalp/defaults/`; POST `/strategies/scalp/runs/`; WS `/ws/scalp/<id>/` | Defaults are **real** (12 expiries) | 2026-06-28 | Strategies |
| Broker Link `/brokers` | Link and manage brokers, OAuth, diagnose, combined positions | GET `/brokers/`, `/brokers/available/`, `/brokers/combined/positions/`, `/brokers/<id>/diagnose/` | **angel_one active**; zerodha and fyers **expired** (daily login). Combined positions empty | 2026-10-21 | Settings |
| Broker Monitor `/broker-monitor` | SmartAPI breaker, call rate, queues | GET `/brokers/monitor/` | **Real**: breaker closed, 0 trips | 2026-06-24 | Settings |
| TradingView `/tradingview` | Webhook links, Pine scripts, watchlists, signals | GET `/notifications/tradingview/…`, `/watchlists/…` | 0 links, 0 watchlists. The "signals" tab actually lists OK_SCANNER signals | 2026-06-26 | Settings |
| Ops Console `/ops` | Stream any `manage.py` command (owner-gated) | GET `/ops/commands/`; WS `/ws/ops/` | **Real**: 66 commands, admin tier | 2026-09-17 | Runs |
| Daily Pipeline `/pipeline` | Pipeline flow, run history, manual triggers | GET `/system/pipeline/`; POST `/system/pipeline/<task>/run/` | **Real but misleading**: defaults to 2026-10-18 rather than today. **2 PipelineRun rows have been stuck `running` since 2026-10-20 18:01 IST** (`swing_scan`, `premarket_basket`; no reaper) | 2026-06-03 | Runs |

## Findings

1. **Data trust.**
   - Angel One is the only active link. The option chain is real; Pulse quotes come from yfinance.
   - Zerodha and Fyers need the daily browser login.
   - **Stored candles end on 2026-10-19.** 2026-10-20 was a holiday in the NSE calendar, but 2026-10-21 was a trading day and produced no 1m or 1d candles. So the top bar reads **stale**, not live.
   - R3 (Action Board) and R5 (Lab) need fresh real candles. Check why today's screener session wrote nothing before starting them.
2. **Portfolio data is empty everywhere.** Desk, Positions, Options risk, broker combined positions, MF holdings, trades and strategy instances are all 0. See `REDESIGN_PORTFOLIO_SOURCE.md`. That note also covers a summary-view bug that mixes the two portfolios' snapshots.
3. **Mock data in Monthly by default.** This is the only page whose UI shows fabricated numbers without being asked.
4. **Cold-start latency.** Pulse (29 s), rotation, shortlist, basket, nightly-scan and swing-scanner (97 s) each take over 30 s on first load. Today's drill-downs feel broken on first open each day.
5. **Command palette drift (fixed in R0).**
   - `/journal`, `/broker` and `/settings` pointed at routes that do not exist.
   - 8 pages were missing from the palette: Options, Pyramid, Scalping, Brokers, Broker Monitor, TradingView, Ops and Pipeline.
   - The palette now lists every page and the new destinations. `/broker` and `/settings` now redirect.
6. **Three run UIs with no shared receipt.** Agent Console, Ops Console and Pipeline each show runs in their own way, and failures (the scalp run, the stuck pipeline rows) were not surfaced anywhere. R1's Runs page and receipts address this.
7. **Activity proxy.** 18 of 22 feature folders have no commit since 2026-09-01. Recently active: mutual-funds, nightly-scan and broker (2026-10-21), and ops (2026-09-17).

## Open question for the owner (plan §12)

Which of these pages do you actually open? Nothing has been deleted; every old URL still works. The inventory suggests three candidates to fold in during R6:

- **Rotation and Shortlist** become Today drill-downs.
- **Backtester, Pyramid and Scalping** become Strategy Studio templates.
- **Agent Console, Ops and Pipeline** become Runs tabs.
