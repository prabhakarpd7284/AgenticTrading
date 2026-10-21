# Plan: Strategy Studio - build, simulate, backtest, paper-trade, then (maybe) go live

Owner: Abhinav. Audience: Claude Code running natively in this repo.
Read first: `CLAUDE.md`, `backend/plugins/strategy_backtest/ARCHITECTURE.md`,
`docs/MF_SCREENER_PLAN.md` (same conventions: deterministic core, no LLM computing numbers).

## 0. Goal

From the UI, the owner defines a **new trading strategy** (no code), and AlphaDesk then:
1. **Backtests** it deterministically on historical data,
2. **Agent-simulates** it (a Claude agent interprets the strategy and trades a replay, with its
   reasoning shown) as a second, different kind of evidence,
3. **Paper-trades** it live on real market data through the existing order/risk path,
4. Only after explicit gates **and an explicit human confirmation**, allows a live switch.

Personal-use product for now (single owner/tenant). Keep tenancy correct anyway.

## 1. What exists today (verified by reading the code, 2026-10-21)

Frontend `features/strategies/StrategyBuilderPage.tsx` ("Your playbook"):
- "New strategy" dialog only asks for a **name + template** and POSTs
  `{strategy_name, strategy_version: "1.0.0", name, params: {}}`. **Params are always empty.**
  The dialog text says params "can be tuned after creation from the instance view" - this file
  has no such view (confirm no other route does; if none, that is the dead end to remove).
- The "catalog" cards just list plugin names and "N fields". The "Legacy playbook" list is text notes.
- Enable/pause is shown, but nothing evaluates a `StrategyInstance`.
Backend `apps/strategies`:
- `StrategyInstance(strategy_name, strategy_version, name, params JSON, is_enabled, schedule)`,
  `Backtest(instance, from_date, to_date, status, metrics, equity_curve, error)`.
- `tasks/backtest.py::run_backtest` is a **placeholder**: sets metrics to zeros and `DONE`.
  (`BacktestViewSet.perform_create` calls it, so any "backtest" through that API returns zeros.)
- Each plugin's `schema()` returns JSON-Schema params ("the frontend builder reads this to render
  the form") - **no frontend renders it**.
`features/backtester/BacktesterPage.tsx` is hardwired to three built-in modes (Daily Swing,
Intraday Multi-TF, Morning Basket) via `GET /market-data/ok-backtest/`; it cannot test a user-defined strategy.
`plugins/strategy_backtest` has a good generic engine: `EngineConfig`, `BacktestEngine`,
composable `ExitCheck`s (stoploss, target, breakeven, trailing, EOD, max-hold), `PositionSizer`,
`StatsAggregator`, and an `EntryDetector` protocol ("new strategy = new config + new
EntryDetector, zero engine changes"). Its docs say Trade/ExitManager are designed to run
identically in live mode.
Paper path: `apps/trading/services/paper_fill_driver.py` + `paper_fills.py` + `place_order.py` +
`RiskEngine` (10 criteria). Its docstring says paper previously produced no fills/P&L; the driver
now exists (recently modified files - confirm it is committed and tested).
Interactive sim precedent: `apps/agents_core/services/scalp_sim.py` + `scalp_consumer.py`
(WebSocket playback) and `run_agent` + `ChannelsPublisher` (streams agent events to the UI).
Pyramid page already has "backtest then Go Live (Paper)" - a pattern to generalise.

**Data caveat (critical):** `HANDOFF_MUTUAL_FUNDS.md` states no broker links exist on the
second machine and "equity is on synthesized prices". A backtest on synthetic prices is
meaningless. The product must record and show **data provenance** on every result and refuse
(or loudly label) synthetic data.

## 2. Product shape: "Strategy Studio"

One area replaces both the Strategies and Backtester pages' dead ends:
`/strategies` list -> `/strategies/:id` studio with tabs: **Define - Test - Agent Sim - Paper - Promote**.

Create flows (all end in the same saved `StrategySpec`):
- **From a built-in template** (OK swing, screener, pyramid...) with a real, schema-rendered params form.
- **Rule builder**: entry conditions, filters, exits, sizing, universe - forms, no code.
- **Describe in words**: a Claude call drafts a `StrategySpec`; the owner reviews/edits it.
  The LLM only drafts the spec; it never runs it and the spec is validated like any other.
No arbitrary user code execution in v1 (multi-tenant security risk). Revisit later, sandboxed, if ever.

### 2.1 StrategySpec v1 (pydantic, versioned JSON, the single source of truth)
- `meta`: name, description (free text, also fed to the agent simulator), tags.
- `universe`: explicit symbols | index constituents | saved watchlist | scanner output.
- `timeframe`: 1m/3m/5m/15m/60m/1d (what the data source can actually supply - verify).
- `indicators`: named series with params (EMA/SMA, RSI, ATR, Bollinger, VWAP, prior-day
  high/low/close, N-bar high/low, volume vs average, gap %). Reuse the existing screener
  `IndicatorEngine` (verify) rather than writing a second indicator library.
- `entry`: direction (long/short/both) + a condition **tree** (AND/OR/NOT over comparisons,
  crosses, "N bars ago", rising/falling) + filters (time window, weekdays, market regime such as
  VIX/NIFTY trend) + cooldown.
- `exits`: stop (ATR multiple | % | structural level), target (R multiple | fixed), breakeven,
  trailing, time/EOD exit, max-hold - mapped 1:1 to the existing `ExitCheck` classes.
- `sizing`: risk % per trade, max position %, max concurrent positions.
- `limits`: daily loss limit, max trades/day, strategy-level kill switch.
- `costs`: brokerage, STT/taxes and slippage models (config, with source and as-of date for rates).
- `instruments`: equity cash first. Options legs/spreads are a later phase - the existing
  options strategies stay plugins.
Specs are **immutable once tested**: editing creates `StrategyVersion` n+1, and every result
points at the exact version it tested.

## 3. Backend design

New/changed models in `apps/strategies` (tenant-scoped):
- `StrategyVersion(instance, version, spec JSON, spec_hash, created_by, note)`.
- `StrategyInstance` gains `stage` (draft | backtested | simulated | paper | live_eligible | live |
  retired), `current_version`. `is_enabled` is derived from stage, not a free toggle.
- `Backtest` gains `version`, `config JSON` (universe, dates, costs, split), `data_provenance JSON`
  (source, synthetic?, bar count, gaps), `trades` (separate `BacktestTrade` table), `oos_metrics`,
  `sensitivity JSON`, `progress`.
- `SimRun` (agent simulation): version, window, symbols, model id, temperature, `decisions` table,
  `cost_tokens`, status, leakage_mode (see 5).
- `PaperRun`: version, started_at, status, link to portfolio, rolling stats snapshot.
- `StageEvent`: every stage transition with evidence refs and who approved it.
All meaningful actions go through `events.services.event_writer.emit()`, non-blocking.

Services (pure where possible, each unit-tested):
- `spec.py` (schema + validator + hash), `rules.py` (compile condition tree -> vectorised
  evaluator), `adapter.py` (`SpecEntryDetector` implementing the existing `EntryDetector`
  protocol; `spec_to_exit_checks`), `costs.py`, `backtest_service.py`, `robustness.py`,
  `agent_sim.py`, `paper_runner.py`, `promotion.py`.
- **One evaluator for both backtest and live.** The same compiled rules must produce the same
  signals on the same bars whether fed historically or live. This is enforced by a **parity
  test**: replay a historical day through the live path and assert identical trades to the backtest.

API (`/api/v1/strategies/...`): CRUD instances and versions; `POST /spec/validate/`,
`POST /spec/draft-from-text/`, `POST /spec/preview-signals/` (signals on one symbol/range,
fast, synchronous), `POST /versions/{v}/backtests/`, `GET /backtests/{id}/` (+ trades, curve, sensitivity),
`POST /versions/{v}/sims/`, WebSocket `/ws/sim/{run}/`, `POST /versions/{v}/paper/start|stop/`,
`GET /instances/{id}/gates/`, `POST /instances/{id}/promote/`. Tenant-isolation tests for each.

## 4. Backtest (deterministic) - what "good" means here

- Runs in Celery queue `backtests`; progress reported; results immutable.
- Reuses `BacktestEngine`; fixes the placeholder task. Existing built-in modes (OK, intraday,
  basket) are exposed through the same Test tab so the old Backtester page can be retired.
- **Data**: a historical candle cache table (`CandleBar`: symbol, tf, ts, OHLCV, source,
  adjusted flag) filled from the real broker feed; backtests read only the cache, so results are
  reproducible. Verify the broker's per-request history limits before designing fetchers.
  Corporate-action adjustment and survivorship (delisted names) must be handled or stated.
  If the data is synthetic or has gaps, the result is stamped `provenance.synthetic=true` and the UI
  shows a red banner; promotion gates refuse it.
- **Realism**: slippage, costs, NSE/BSE session rules, the repo's own `TRADING_RULES`
  (auto square-off 15:14, risk caps) so a backtest cannot trade when the live system would not.
  Entries only on bars after the signal bar (no same-bar lookahead); indicators warm up before
  the entry window (engine already supports `entry_from/entry_to`).
- **Overfitting guards (do not skip)**: mandatory in-sample / out-of-sample split (default last
  30% held out and shown separately); walk-forward option; minimum trade count before any metric
  is called meaningful; parameter-sensitivity grid (small, bounded) showing whether the result
  survives small parameter changes; number-of-variants-tried counter per strategy, with a warning that
  more tries inflate best results; benchmark comparison (NIFTY buy-and-hold, and a random-entry
  baseline using the same exits).
- **Metrics**: return, CAGR, max drawdown + duration, win rate, profit factor, expectancy in R,
  Sharpe/Sortino (document the daily-return convention), exposure, turnover, costs paid, per-symbol,
  per-month, per-hour. Trade list with chart context for each trade (drill-down to the bars).

## 5. Agent simulation (the second way)

Purpose: test strategies that are described in words / are partly discretionary, catch spec
misreadings, and explain *why* it traded - not to prove an edge. It is a different instrument from
a backtest and must never be presented as equivalent evidence.

Design:
- A Claude agent (reuse `run_agent` / LangGraph / `ChannelsPublisher` event streaming) is given the
  strategy description + spec, then steps through a **replay** day-by-day (or bar-by-bar for
  intraday). At each step it sees only data up to that time through an `AsOfMarketData` port that
  **raises on any future access** (unit-tested).
- The agent outputs structured decisions (`enter/skip/exit/adjust`, side, stop, target, reasons,
  confidence). The **same `Trade`/`ExitManager`/sizer/`RiskEngine`** then account for P&L - the LLM
  decides, it never does arithmetic.
- UI: playback like the existing scalp simulator: chart + decision log with the agent's reasoning
  per step, running equity, and a final summary. Saved and replayable.
- **Leakage / contamination control**: the model may have seen real history. Two modes:
  (a) *clean window* - only dates after the model's knowledge cutoff (this deployment's cutoff is
  end-June 2026, so roughly July 2026 onward; keep this as a config with the model id) and
  (b) *masked replay* - symbols and dates anonymised for older windows. Anything else is labelled
  "possibly contaminated - indicative only".
- Cost/determinism: temperature 0, store prompts, model id and tool results; cap steps and tokens
  per run; sample days if needed and say so. Results with tens of decisions are anecdotes, not statistics.
- **Cross-check with backtest**: when a spec has rules, run both on the same window and show where
  they disagree (trades only one took). Disagreements usually mean the rules do not capture the
  owner's intent - a useful product feature by itself.
- Prompt-injection safety: news/RAG text is data; the agent has read-only tools and cannot place orders.

## 6. Paper trading

- `paper_runner`: scheduled worker (Celery beat or a long-running consumer) that, during market
  hours, evaluates every `paper`-stage instance on completed bars with the **same compiled rules**,
  emits `Signal` rows (existing model) and submits orders through `place_order` -> `RiskEngine` ->
  paper fills (`paper_fill_driver`). The paper portfolio is separate per strategy (or tagged) so P&L
  is attributable.
- Start/stop from the UI; status shows last evaluated bar, signals, open positions, P&L, and any
  skipped signals with the risk-engine reason.
- **Drift monitor**: compare live-paper stats with the backtest's expectations (trade frequency,
  win rate, average R, slippage) and alert on large divergence - this is what exposes
  look-ahead bugs and data problems.
- Paper realism limits to state in the UI: fills are simulated from LTP/candles, not an order book.
- Requires real market data (broker link); refuses to start on synthetic prices.

## 7. Promotion gates (state machine in `promotion.py`)

`draft -> backtested -> (simulated, optional) -> paper -> live_eligible -> live`
Defaults (all configurable per strategy, stored with each approval):
- Backtested: real data, >= N trades (default 100), out-of-sample not materially worse than in-sample,
  survives the sensitivity grid, beats the random-entry baseline.
- Paper: >= 20 trading days AND >= 30 trades (whichever is later), drift within bounds, no risk-engine
  breaches, daily loss limit never hit. The repo's `GOALS.md` targets (Sharpe > 1.5, DD < 10%,
  PF > 1.5, 60 days) are good defaults for headline numbers; note that 60 days cannot statistically
  prove a Sharpe of 1.5, so the UI shows confidence ranges rather than a green tick on one number.
- Live: **never automatic.** Requires `live_eligible`, an explicit owner confirmation dialog showing
  the evidence, `TRADING_MODE=live` already set deliberately (existing guardrail), reduced initial
  size (1 unit/lot), tighter loss limit, a one-click kill switch, and auto-demotion to paper on a
  breach. Every transition is a `StageEvent`.

## 8. Frontend work (Strategy Studio)

- Remove/replace the dead parts: parameter-less "New strategy" dialog, catalog cards with no action,
  the stub `Backtest` flow. Keep the legacy-playbook list only if it links to migration into specs.
- **Define tab**: condition-row builder, exits/sizing/limits forms, JSON-Schema form renderer for
  built-in templates, live "explain in plain English" summary, validation errors inline, version list/diff.
- **Instant preview**: pick a symbol/date range -> chart with entry/exit markers (calls
  `preview-signals`). Fast feedback is what makes a builder usable.
- **Test tab**: run form (range, universe, costs, split) -> progress -> report: equity curve, drawdown,
  OOS vs IS, trade table with drill-down, sensitivity heat-map, benchmark lines, provenance banner.
- **Agent Sim tab**: window/mode/budget picker, streaming playback, decision log, comparison with backtest.
- **Paper tab**: status, signals, positions, P&L vs expectation, drift flags, gate progress bars.
- **Promote tab**: checklist with evidence links, approval history, live confirmation, kill switch.
- Follow existing UI components (`Card`, `DataTable`, `Tabs`, react-query, recharts). Frontend tests for
  the builder, report and gate UI; backend tenant-isolation tests for every endpoint.

## 9. Build order (each step shippable; stop at any milestone)

P0 Audit + decisions: confirm dead UI paths, branch hygiene (commit pending work first), data
   provenance flag, and the data-source spike (what history can the broker supply; cache design).
P1 `StrategySpec` + validator + rule compiler + `SpecEntryDetector`/exit mapping + CLI
   `run_spec_backtest`. Unit tests incl. no-lookahead and parity-ready design.
P2 Real backtest service replacing the placeholder, `BacktestTrade`, candle cache, costs, API.
   **Milestone M1: a hand-written spec backtests end to end from the CLI/API with true numbers.**
P3 Studio UI v1: builder + instant preview + Test tab + JSON-Schema renderer for templates.
   **Milestone M2: create and backtest a strategy entirely from the UI.**
P4 Robustness: IS/OOS, walk-forward, sensitivity, baselines, trial counter.
P5 Natural-language drafting (spec only) + "explain strategy".
P6 Paper runner + drift monitor + Paper tab (+ parity test). **Milestone M3: paper-trade from the UI.**
P7 Agent simulator + playback + backtest comparison.
P8 Promotion gates, live confirmation flow, kill switch, auto-demotion. **Milestone M4.**
Later: options strategies in the builder, optimisation helpers, multi-strategy portfolio view,
sandboxed custom code (only if still wanted).

## 10. Risks and rules to keep visible
- Overfitting is the main way this product misleads its owner; the guards in section 4 are core
  scope, not extras.
- Synthetic or gappy data -> meaningless results; provenance is enforced, not advisory.
- Backtest/live divergence -> single evaluator + parity test + drift monitor.
- LLM output is never trusted for numbers or orders; contamination handled per section 5.
- Live trading is a deliberate, reversible, human-approved step with size caps. No path from the
  LLM or from a strategy to a live order bypasses `RiskEngine`.
- SEBI: personal strategies on own capital only; letting other people run strategies or receive
  signals raises registration questions (see GOALS.md Phase 4 / MF plan notes).

## 11. Decisions needed from the owner (defaults in brackets)
Instruments in v1 [equity cash only]; timeframes [5m, 15m, daily]; history window [2 years daily,
6 months intraday, subject to data source]; agent-sim budget per run [capped, e.g. 200 steps];
whether custom Python is ever wanted [no]; paper gate length [20 days / 30 trades];
who else may use the Studio [nobody].
