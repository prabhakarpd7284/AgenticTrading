# UI Gap Analysis + Operational Plan

*AlphaDesk / AgenticTrading — April 2026*

## The core problem, in one line

The UI is a **monitoring dashboard** bolted onto a **CLI-driven trading system**. An operator can *see* what the AI team does, but cannot *intervene*, *tune*, or *debug* from the UI — all of that still lives in `python manage.py run_trading_agent …` and `manage_straddle …`. That's fine for a solo trader with a terminal open, but it breaks the promise of "an AI virtual team you supervise" that the README and CLAUDE.md lay out.

Every missing piece below traces back to one of three root causes:

1. **Read/write asymmetry.** Reads (list trades, show P&L, show audit) are well-covered; writes (close a position, force an action, pause the AI, adjust risk caps) are near-absent.
2. **Agent opacity.** `@DirectionalTrader`, `@OptionsStrategist`, `@RiskGuard` each log rich state (prompt, LLM response, 9-criterion risk breakdown, Greek calcs), but the UI renders them as flat JSON cards or not at all.
3. **Two sources of truth.** Legacy sqlite (`db.sqlite3`, seeded) vs v2 Postgres (`apps.*`) are both queried from the dashboard with no indication of which is authoritative for which concept. Operators don't know whether "Portfolio Value" came from `PortfolioSnapshot` or `apps.portfolio.Portfolio`.

## Vision alignment — what the UI MUST respect

Any UI change has to preserve these invariants (they're from `CLAUDE.md`, non-negotiable):

- `@RiskGuard` is the **last gate before every execution**. If the UI adds a "force execute" button, it still goes through `validate_action`/`validate_trade` — never bypasses it. If an operator overrides (e.g. "execute anyway"), the override itself is the new data point the system records.
- `TRADING_MODE=paper` by default. Every mutating UI must display the current mode prominently; "live" should require a second-factor confirmation.
- **Daily loss hard stop** at 3% — UI must show the remaining budget as a gauge, not buried in a KPI strip.
- **Every operator action is journalled** to `AuditLog` with the operator's user_id, the intent, and the outcome (success/error). No silent state changes.
- **Paper vs live parity** — the UI must look identical in paper mode and live mode except for the mode badge, so muscle memory carries over.

## Gap analysis — ten concrete holes

Each row: **what's missing → why it matters → where it lives in the backend today.**

| # | Gap | Impact | Backend surface exists? |
|---|---|---|---|
| 1 | Pause/Resume AI (global + per-strategy) | Can't kill a runaway loop from the UI. Today requires editing `SystemControl` in sqlite or killing the process. | Partial — `legacy:ai-pause`/`ai-resume` exists. No UI button. |
| 2 | Force-close / force-action on positions | Operator watches a straddle bleed past the hard stop, can't close without SSH. | Yes — `manage_straddle --execute CLOSE_BOTH` works; no API or UI equivalent. |
| 3 | Risk-cap editor + kill switch | Can't tighten risk caps mid-day or halt all new trades. Hardcoded in `risk_engine.py`. | No API yet — caps come from env, not DB. |
| 4 | Agent reasoning inspector | `@DirectionalTrader`'s prompt + LLM response + scratchpad are logged as `AuditLog.raw`, never rendered. Blind-spot for debugging rejected trades. | Yes — `AuditLog.raw` has the full payload. |
| 5 | 9-criterion risk breakdown card | `validate_trade` returns a structured result with actual vs threshold per criterion. UI shows a single "REJECTED" pill. | Yes — `RiskResult` has `criteria: list[CriterionResult]`. |
| 6 | Straddle detail page (Greeks, P&L vs entry, action history, scenarios) | Straddle is the marquee product, and it has the thinnest UI — just a table row. No way to see *why* the strategist recommended HOLD. | Yes — `StraddleAnalysis` + `StraddleAction` records carry everything. |
| 7 | Portfolio capital curve | `PortfolioSnapshot` rows accumulate forever, never charted. Operators can't see "am I up or down on the month?" at a glance. | Yes — `PortfolioSnapshot` has daily rows. |
| 8 | Alerts acknowledgment + escalation | `legacy:alerts` returns critical alerts; UI shows them but can't ACK them. They never clear. | No — `alerts` is read-only today. |
| 9 | Broker reconciliation surface | `reconcile_positions` node logs mismatches silently. Operators don't know broker ↔ DB drift happened until a trade fails. | Partial — `AuditLog.type=RECONCILE` rows exist; no dedicated page. |
| 10 | Backtest parameter tuning + progress | Form only accepts date range. Async runs show "running" for minutes with no ETA or partial equity curve. | Yes — `Backtest` model has `status`, `progress`, `equity_curve`; no WS stream for progress. |

Secondary gaps that follow from the above:

- No **capital / portfolio init UI** (operators run `--init-portfolio 500000` once; no way to re-seed in a dev tenant)
- No **strategy seed UI** (`--seed-strategies` flag)
- **Agent catalog vs legacy StrategyDoc vs strategies/instances** — three overlapping concepts shown on the Strategies page with no hierarchy
- **Order fill status** — TradeJournal shows `status=FILLED/REJECTED` but no live streaming, no broker confirmation ID, no partial-fill handling
- **Mobile layout** — DataTables overflow horizontally; critical on a phone during market hours
- **Keyboard shortcuts** — an operational UI needs `g p` for positions, `k` for kill-switch, `?` for cheatsheet. None today.

## The plan — four milestones, ~4–5 weeks

Each milestone ships end-to-end (backend endpoint + UI page + integration test + runbook entry). No "backend now, UI later" — the vision is the UI, so every backend change must land a user-visible surface in the same PR.

### M1 — Operational Control (week 1–2)

**Goal: an operator can intervene in anything the AI does, from the UI, with every action journalled.**

- **Kill switch + pause/resume** — header bar toggle (`Trading: ENABLED / PAUSED / HALTED`); backend endpoint `POST /api/v1/control/pause/`, `POST /api/v1/control/resume/`, `POST /api/v1/control/halt/`. Writes `SystemControl` and journals `CONTROL_*` audit entries. `@RiskGuard` reads `SystemControl.is_halted` as the first check — all new trades reject when halted.
- **Force-close equity position** — "Close" button on `/positions/` rows. Confirmation dialog: type the symbol to confirm. Backend `POST /api/v1/positions/{id}/close/` → market order via broker → updates `TradeJournal.status=CLOSED_MANUAL`, audit `FORCE_CLOSE`.
- **Force-action on straddle** — detail-page buttons for `CLOSE_BOTH | CLOSE_CE | CLOSE_PE | HEDGE | ROLL | SHIFT`. Each still goes through `validate_action` (the 5-criterion straddle risk gate). If validate fails, dialog shows the breach and an "override" toggle that requires a typed reason.
- **Risk-cap editor** — `/settings/risk` page with sliders for `MAX_RISK_PER_TRADE_PCT` (0.25–2.0), `MAX_DAILY_LOSS_PCT` (1.0–5.0), `MAX_POSITION_SIZE_PCT` (5–20), `VIX_CEILING` (15–40). Persisted to new `RiskConfig` model (tenant-scoped); hot-reloaded by `@RiskGuard` on every request.
- **Mode badge + live confirmation** — red "LIVE" badge in header when `TRADING_MODE=live`. Any mutating action in live mode requires typing `LIVE` to confirm.

**Acceptance:** a new operator can join, kill a rogue run, and tighten the daily-loss cap, without ever opening a terminal.

### M2 — Agent Observability (week 2–3)

**Goal: debug an AI decision in under 60 seconds without grep-ing the database.**

- **Agent run timeline** — `/agents/:runId` gets a vertical timeline: `fetch_data → retrieve_context → planner → risk → execute → journal`. Each node expands to show its inputs, outputs, duration, and LLM prompt/response if applicable.
- **Risk-check breakdown card** — replaces the monolithic "REJECTED" pill. Renders all 9 criteria as rows: criterion, threshold, actual, pass/fail, remediation hint. For rejections, the failing rows are pinned to top.
- **LLM inspector** — two-pane viewer: system prompt (left), user prompt + response (right). Copy-to-clipboard per pane. Wire to `AuditLog.raw.prompt`/`AuditLog.raw.response`.
- **Event replay** — filter `/agents/` list by date/strategy/status/risk-outcome. Click any historical run → replays the timeline exactly as it happened. No re-execution, just visualization.
- **WebSocket reliability** — runs that stream over `/ws/agents/{runId}/` get an "offline" badge + auto-reconnect after 2s backoff. Drop-recovery fetches missed events from `/runs/{id}/events/`.

**Acceptance:** given a rejected trade from yesterday, an operator can find it, see the rejecting criterion, read the LLM's plan, and decide whether to loosen a cap — in one screen.

### M3 — Position & Portfolio Lifecycle (week 3–4)

**Goal: the marquee products (straddles, capital curve) get first-class detail pages.**

- **Straddle detail page** at `/positions/straddles/:id`. Four panels:
  1. **Greeks + P&L** — live Delta/Gamma/Vega/Theta, P&L vs entry-day premium (chart), VIX overlay.
  2. **Scenario table** — payoff at spot ±2%/±5%/±10%, at T and T+1d.
  3. **Action history** — every `StraddleAction` record, with LLM reasoning and risk-check summary.
  4. **Controls** — force-action buttons from M1, wired to this position.
- **Portfolio capital curve** — new `/portfolio` route (rename v2 tab); plots `PortfolioSnapshot.capital` + `.daily_pnl` + `.invested` over time (toggle 1d/1w/1m/all). Annotations: peak, drawdown, deploy events.
- **Alert acknowledgment** — inline "ACK" button on each alert; backend `POST /api/v1/alerts/{id}/ack/` sets `acked_by` + `acked_at`. Acked alerts collapse into a history drawer. Re-raise if the underlying condition fires again.
- **Broker reconciliation page** — `/brokers/reconciliation` shows the last N `RECONCILE` audit rows: broker position vs DB position, diff, resolution. "Re-reconcile" button triggers the node on demand.
- **Order fill tracker** — new "Orders" tab on positions page. Streams `OrderUpdate` over WS (new model, backs `orders_saga` app). Shows: submitted → placed → partial → filled (or rejected/cancelled) with broker order-id.

**Acceptance:** operator can pick a single straddle, see why the strategist said HOLD at 2:15 PM, verify the reconciliation passed at 9:16 AM, and know the portfolio is -0.8% on the day.

### M4 — Strategy Configuration + Backtest (week 4–5)

**Goal: tune strategies and run comparative backtests without editing code.**

- **Strategy instance editor** — `/strategies/:id/edit` exposes tunable params per agent catalog entry (entry_threshold, sl_pct, target_pct, size_pct, time_window). Params are validated against the catalog's pydantic schema. Save writes to `StrategyInstance.config` (jsonb).
- **Strategy enablement schedule** — time-of-day gates (e.g. "intraday-momentum runs 09:30–14:00 only"); expiry-day auto-disable for straddles; holiday calendar.
- **Backtest parameter overrides** — the backtest form accepts per-run param overrides layered on top of the instance config. Diff view shows "what changed from the default".
- **Backtest progress streaming** — WS channel `/ws/backtests/{id}/` streams `progress`, `partial_equity`, `current_date`, `current_trade`. UI shows a live equity curve that grows during the run.
- **Backtest compare** — pick 2-4 runs, side-by-side KPIs + overlaid equity curves. "Promote" button marks a run's config as the new default for its instance.

**Acceptance:** operator tunes `entry_threshold` from 30 → 35, backtests April–March with the override, compares to the baseline, and promotes the winner. No shell.

### Cross-cutting (runs through every milestone)

- **Every mutating endpoint journals** to `AuditLog` with `operator_id`, `intent`, `before_state`, `after_state`.
- **Every mutation shows a confirmation dialog** sized to the stakes: "Close position" = click, "Halt trading" = type "HALT", "Live mode override" = type `LIVE` + SMS/email OTP.
- **Keyboard shortcuts** — `g d` dashboard, `g p` positions, `g a` agents, `g s` strategies, `k` kill-switch (with confirm), `?` cheatsheet overlay.
- **Mobile-first for two pages only** — dashboard and positions. A trader on a phone at 3 PM needs to see P&L, risk gauge, and hit "close" on a bleeder. Everything else can stay desktop.
- **Single source of truth per concept** — we pick one: portfolio summary comes from v2 `/portfolios/` (not legacy); trade journal comes from legacy until the orders_saga app takes over; audit always legacy. Display the source on hover for transparency.

## What *not* to build (explicitly)

To stay focused on the vision ("AI virtual team you supervise"), we are explicitly **not** building:

- Free-form charting / custom indicators — TradingView is better at that, and we don't want to compete.
- Discretionary order entry without an agent — AlphaDesk is AI-first. Manual trading goes through the broker's own app.
- Social / sharing / leaderboard features — out of scope for a compliance-sensitive product.
- Custom strategy code editor — strategies come from the agent catalog; ops tune params, developers add new catalog entries in code + PRs.

## Execution sequencing — tickets

Eight workstream tasks queued (see task list). Estimated effort shown is realistic, not optimistic. Each ticket carries its own acceptance criteria; nothing ships without a demo against the acceptance line.

---

*Reviewed against CLAUDE.md invariants, `trading/services/risk_engine.py`, `trading/options/straddle/graph.py`, and the existing frontend at `frontend/src/features/`.*
