# Plan: Make AlphaDesk usable - Today, Portfolio, Strategies, Lab, Runs

Owner: Abhinav. Audience: Claude Code running natively in this repo.
Companion plans: `docs/STRATEGY_STUDIO_PLAN.md` (strategy CRUD, backtest, agent sim, paper),
`docs/MF_SCREENER_PLAN.md` (mutual funds). This plan is the product shell that ties them together.

## 0. The owner's requirements (verbatim intent)

When I open AlphaDesk I want to see:
1. **The stocks I need to invest in today.**
2. **My full portfolio picture.**
3. **My strategies: create / read / update / delete many of them, and backtest them.**
4. **A full analysis of a selected group of stocks across different strategies over a time period.**
5. **Confidence that anything I run actually ran correctly** - today I don't know which page to
   open, how to run things, or whether an output is right.

Success test (see section 9): within 10 seconds of opening the app he can answer "what do I do
today, what am I worth, and are my jobs healthy?" without choosing a page.

## 1. What is wrong today (verified in `router.tsx`, `AppShell.tsx`, a few pages)

- **21 flat sidebar items** (Today, Rotation, Shortlist, Swing Scanner, Basket, Nightly Scan, Desk,
  Positions, Options Desk, Monthly, Mutual Funds, Agent Console, Strategies, Backtester, Pyramid,
  Scalping, Broker Link, Broker Monitor, TradingView, Ops Console, Daily Pipeline). No hierarchy,
  no "start here".
- The index route is `MarketPulsePage` ("What's Happening Today"), but **portfolio** lives on `/dashboard`
  ("Desk"), **positions** on `/positions`, **options** on `/options`, **mutual funds** on
  `/mutual-funds`. There is no single portfolio view, and no "stocks to act on today" list.
  Idea generation is scattered over Rotation, Shortlist, Swing Scanner, Basket, Nightly Scan.
- `DashboardPage` mixes legacy and "v2" data sources; its own comment says the v2 queries "are
  currently empty for most users" (two sources of truth; `UI_GAP_ANALYSIS_AND_PLAN.md` flagged this in April).
- **Four different notions of "a run"** with four different UIs: `OpRunPanel` (streams stdout of a
  `manage.py` command), `PipelineRun` (daily pipeline status/summary/error), `AgentRun` (agent console),
  `Backtest` (status + metrics; the task is a placeholder returning zeros). None gives a uniform
  answer to "did it run correctly and what did it produce?". Raw stdout is not verification.
- Strategies page: "New strategy" saves an empty strategy; Backtester is hardwired to three built-in
  modes; neither can analyse a stock group (details in the Studio plan).
- Data trust: the equity feed may be synthetic when no broker link exists, and some scheduled
  jobs have never been observed firing (per `HANDOFF_MUTUAL_FUNDS.md`). The UI does not say so.

## 2. New information architecture (6 destinations, not 21)

Sidebar (everything else becomes a tab, a drawer, or lives under Settings):

| New | Purpose | Absorbs today's pages |
|---|---|---|
| **Today** (home) | What to do now: action board, portfolio snapshot, job health, alerts | Market Pulse, Rotation, Shortlist, Swing Scanner, Basket, Nightly Scan (as "where today's ideas came from", read-only drill-downs) |
| **Portfolio** | Full picture, one place | Desk/Dashboard, Positions, Options Desk, Monthly (as "Performance" tab), Mutual Funds (tab) |
| **Strategies** | CRUD + Strategy Studio (define, test, simulate, paper, promote) | Strategies, Backtester, Pyramid, Scalping simulator (as built-in templates/sims) |
| **Lab** | Stock-group x strategy x period analysis | new; reuses backtest service |
| **Runs** | Every job/backtest/scan/agent run with receipts and health | Daily Pipeline, Ops Console (advanced), Agent Console |
| **Settings** | Broker link, broker monitor, TradingView webhooks, risk caps, mode | Broker Link, Broker Monitor, TradingView |

Rules: old URLs keep working via redirects for one release; the command palette keeps every old
destination searchable; a persistent top bar shows **mode (PAPER/LIVE), data status (live/stale/
synthetic), and today's job health**; first-run shows a **setup checklist** (broker linked? data fresh?
mode paper? first strategy created?).

## 3. Today (home)

Layout, top to bottom:
1. **Status strip**: mode badge, data freshness, "N of M scheduled jobs ran OK today" (links to Runs).
2. **Action Board** - "Stocks to act on today" (section 4).
3. **Portfolio snapshot**: total value (all asset classes), day P&L, risk budget used (daily loss gauge),
   open positions count; click -> Portfolio.
4. **Needs attention**: unacknowledged alerts, jobs that failed or did not run, MF actions
   (SIP due, plan drift, weekly MF review when built).
5. **Market context** (collapsed by default): regime, sector heatmap - the current Market Pulse content.

## 4. Action Board - "stocks I need to invest in today"

New backend concept `TodayAction` (tenant-scoped, one row per proposed action per day):
`date, kind (BUY | ADD | EXIT | REDUCE | MF_SIP | REVIEW), symbol, strategy_version, horizon
(intraday | swing | positional), side, entry_zone, stoploss, target, suggested_qty, capital_required,
risk_check JSON (RiskEngine criteria with pass/fail, run in dry-run mode), confidence, reasons,
source_signal_ids, expires_at, status (proposed | approved | skipped | executed | expired), decided_at, note`.
- Generated by a daily job `build_action_board` after the nightly scan / watchlist / signal jobs,
  for **enabled strategies in paper/live stage only** (see Studio plan stages). Deduplicates across
  strategies, ranks, and allocates capital (position sizing from the existing sizer; cash left shown).
- Each card shows: symbol, action, why (top 3 reasons), entry/stop/target, size and rupees at risk,
  the **risk-engine pre-check** (would pass / would fail and which criterion), and the **track record of
  the strategy that produced it** (paper/backtest stats, with sample size) so every idea carries its
  own evidence. No idea without a strategy owner.
- Buttons: Approve (creates the order through `place_order` -> `RiskEngine` -> paper fill, or live if the
  mode and gates allow), Skip (with optional reason), Snooze. All recorded; skipped-then-worked-out ideas
  feed the "missed opportunities" report.
- Empty state is informative: "No actions today - last scan ran 08:02, scanned 100 symbols, 0 met
  criteria (typical: 3). [View scan receipt]".
- Honest scope: this is a *queue of ideas from your own strategies*, not advice and not a promise.
  Intraday vs swing ideas are labelled; the board defaults to swing/positional, intraday on a tab.
- Telegram: same cards pushed in the morning with approve/skip (existing notifications app).

## 5. Portfolio - the full picture

Decision first: **one source of truth.** Use the v2 `Portfolio` model (trading + `Mutual Funds`
portfolios already exist) and make the legacy summary a fallback to be deleted. Verify which tables
actually hold data on this machine before choosing, and write the decision in `docs/`.
`GET /api/v1/portfolio/overview/` returns, per tenant:
total value and invested; day / week / month / since-start P&L; cash; allocation by asset class
(equity, options, mutual funds), by strategy, by sector; open risk (sum of stop distances), margin used,
daily loss budget used; max drawdown and equity curve (from `PortfolioSnapshot`); data
freshness per component. Mutual-fund holdings valued at latest NAV with their own staleness flag.
Tabs: Overview - Positions - Options - Mutual Funds - Performance (monthly report, capture rate) -
Attribution (P&L by strategy / by symbol / by month). Missing data is shown as "not connected" and never as 0.

## 6. Strategies - full CRUD + backtest

Delivered by the Studio plan. Product requirements added here:
- List view: name, type, stage (draft / backtested / paper / live), last backtest result with
  provenance badge, paper P&L, last run status. Actions: New, Duplicate, Edit (creates new version),
  Archive, Delete (blocked if it has paper/live trades - archive instead), Run backtest, Open Studio.
- Every strategy page starts with "what stage am I in and what is the next step".
- Built-in strategies (OK swing, screener, pyramid, ...) appear as templates with a real parameter form.

## 7. Lab - analyse a group of stocks across strategies over a period

Purpose: "take this group of stocks, run these strategies over this period, tell me what works where".
New models: `StockGroup(name, symbols[], source: manual | index | watchlist | scan | tag, notes)` with
full CRUD (also usable as a backtest universe and a scan universe), and
`AnalysisRun(group, strategy_versions[], from_date, to_date, costs, split, status, progress, provenance)`
with child `AnalysisCell(symbol, strategy_version, metrics JSON, trade_count, status, error)`.
- Fan-out: one Celery task per (symbol, strategy) cell using the Studio backtest service; cached by
  (spec_hash, symbol, period, data fingerprint) so reruns and period tweaks are cheap; partial results
  appear as cells finish; failed cells show their error, never a silent zero.
- UI: group + strategy multi-select + period picker -> **matrix** (rows = stocks, columns = strategies)
  with a metric toggle (return, win rate, profit factor, max drawdown, trades, expectancy in R),
  colour scale, sort by any column, click a cell -> trades + chart. Summary panel: best strategy per
  stock, best stock per strategy, group-level result for each strategy (equal-weight and capital-
  constrained), buy-and-hold benchmark, regime split (trending vs ranging periods), export CSV.
- **Statistical honesty built in**: cells with fewer than N trades are greyed ("too few"); the best
  cell in a big matrix is partly luck, so the UI shows in-sample vs out-of-sample side by side and a
  "stable across sub-periods?" flag, and counts how many combinations were tried.
- Provenance banner on the whole run if any data was synthetic or has gaps.

## 8. Runs - "did it run correctly?" (the backbone; build this first)

### 8.1 Run receipts
One registry for everything that executes: `RunReceipt(id, kind, name, trigger (manual|beat|api),
inputs JSON, started_at, finished_at, status (queued|running|success|partial|failed|not_run),
outputs JSON (counts, ids, links), warnings[], error, checks[], data_provenance)`.
A decorator/context manager `tracked_run(kind, name, inputs)` wraps Celery tasks, `manage.py`
commands run from Ops, pipeline stages, backtests, analysis runs, agent runs, imports. Existing
`PipelineRun`, `AgentRun`, `Backtest` stay as detail tables and link to their receipt; do not rewrite them.
### 8.2 Checks (this is what makes "success" mean something)
Each receipt carries pass/warn/fail checks, e.g.: input data fresh (latest candle age vs expected);
broker linked and **not synthetic**; output row counts vs expected universe size; **zero-output
anomaly** (a scan returning 0 signals vs its 20-run median); duration anomaly; downstream rows
actually written; no unhandled exceptions swallowed (count of skipped rows is reported, not hidden).
A run is `success` only if no check failed; `partial` if some units failed.
### 8.3 Expected-schedule registry
Config listing every scheduled job and when it should run (e.g. nightly scan, daily pipeline, AMFI
NAV pull 23:45 and 07:30 IST). A watcher creates a `not_run` receipt and an alert if a job misses
its window - this directly catches "beat never fired".
### 8.4 UI contract for every button that starts something
Same component everywhere (`RunButton` + `RunReceiptPanel`): (1) confirm inputs, (2) live progress,
(3) on finish a **receipt**: status, duration, what it produced ("scanned 100, 4 signals, 3 added to
Action Board"), warnings, checks, and "open result" links. No page ends a run with raw log text only
(the log stays one click away). Runs page: filter by kind/status/date, today's schedule with
green/amber/red/grey states, rerun, compare to previous run.
Telegram: failed or missed jobs push an alert.

## 9. Acceptance walkthrough (the definition of done)
Open app, no clicks:
1. Status strip says mode, data freshness, "9 of 9 jobs OK".
2. Action Board lists today's ideas with strategy, size, risk pre-check, track record; approve one.
3. Portfolio snapshot shows total value across equity, options and MFs; click through to full breakdown.
4. Strategies: create a new strategy, backtest it, see non-placeholder results with a provenance badge.
5. Lab: choose a saved group of 10 stocks and 3 strategies, 1 year; matrix fills as cells complete;
   open the best cell's trades.
6. Runs: trigger the nightly scan manually; receipt shows counts, checks, and links to its output.
7. Kill the broker link: the status strip and receipts turn amber/red with the reason.

## 10. Build order (each phase shippable)
R0 Decisions + quick wins (days): inventory real usage of the 21 pages; regroup sidebar into the 6
   destinations without deleting anything (redirects); top bar with mode/data/job status; decide the
   portfolio source of truth; commit pending work.
R1 Run receipts backbone: model, `tracked_run`, checks library, schedule registry, `RunButton` +
   panel, Runs page; wrap daily pipeline + scans + backtest first. **Milestone: "did it run?" answered.**
R2 Portfolio overview endpoint + page (equity, options, MF), data-freshness flags.
R3 Action Board: `TodayAction`, `build_action_board`, risk pre-check, approve/skip flow, Today page.
   Requires at least one strategy producing real signals on real data.
R4 Strategy CRUD + Studio P1-P3 (Studio plan). Replace placeholder backtest.
R5 Stock groups + Analysis Lab (needs the real backtest service).
R6 Merge/retire old pages, Telegram push of Action Board and failures, mobile layouts for Today and
   Portfolio, empty/first-run states, keyboard shortcuts update.
Dependencies: R3 and R5 need real market data (broker link) - check this in R0, because synthetic
prices make Action Board and Lab results meaningless.

## 11. Rules and risks
- Nothing here bypasses `RiskEngine`; Approve always goes through the order path; live mode keeps the
  existing explicit-confirmation guardrail.
- No idea or result is shown without its provenance (data source, strategy version, sample size).
- Do not add signup/billing/marketplace work now (personal use until profitability is proven).
- Keep the old pages reachable until the replacement is verified by the owner.
- Test each phase: backend tenant-isolation tests, frontend tests for new pages, and a manual run of
  the section 9 script before closing the phase.

## 12. Open questions for the owner (defaults)
"Invest today" horizon [swing/positional by default, intraday as a second tab]; capital per idea
[risk-based sizing, 1% risk, existing caps]; execution [paper first, manual approve on every action];
home screen at open [Today]; which of the 21 pages he actually uses [R0 inventory will ask]; Telegram
for push [yes if already configured].
