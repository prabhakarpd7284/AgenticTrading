# AlphaDesk AI Team — Project Briefing

This document is the **persistent context** every team agent reads on every
run so none of them re-derives how the system works from scratch. Treat it
as the team's shared CLAUDE.md.

The team has four roles, all sharing the same mind palace:

```
   trader_user  →  planner  →  executor  →  tester
   (asks for     (turns asks   (proposes    (verifies
    features +   + bugs into   code         everything
    reports)     tasks)        changes)     still green)
                       ↑                          │
                       └──────────────────────────┘
                          (bugs feed back as
                           high-priority tasks)
```

All four are Django management commands:

```bash
python manage.py run_ai_trader_user   # 1. feature_requests from a trader POV
python manage.py run_ai_planner       # 2. tasks from bugs + requests
python manage.py run_ai_executor      # 3. propose ONE code change for the next task
python manage.py run_ai_tester        # 4. deterministic test suite + findings
python manage.py run_ai_team          # all four in sequence
```

The **trader_user** agent has multiple **profiles** — each is a persona
with its own system prompt and `requested_by` tag on every request:

| Profile | Persona |
|---|---|
| `default` | Generalist quant (current behavior) |
| `options` | NIFTY/BANKNIFTY index options trader — skew, IV, gamma, pin risk |
| `futures` | Index / single-stock futures — basis, OI, rolls, SPAN margin |
| `equity` | Swing / positional equities — RS, sector rotation, breakouts |
| `intraday` | Equity intraday scalper — ORB, VWAP, tape speed |

```bash
python manage.py run_ai_trader_user --profile options    # one persona
python manage.py run_ai_trader_user --all                # every persona
python manage.py run_ai_team --profile futures           # team with one persona
python manage.py run_ai_team --all-profiles              # team across every persona
```

Adding a profile is one entry in `apps/agents_core/tester/trader_profiles.py`.

Important: the executor **never applies** code — it writes structured
proposals into the mind palace for human review (or for Claude Code in
another conversation to apply).

## What the AI Tester is

A deterministic Python test runner plus an optional LLM analysis layer.
The tester is one of the four team agents.

```bash
python manage.py run_ai_tester                   # quiet, exit code 0/1
python manage.py run_ai_tester --verbose         # per-test detail
python manage.py run_ai_tester --llm             # add Claude summary + fix proposals
python manage.py run_ai_tester --only plan_stock # filter to one suite
python manage.py run_ai_tester --reset           # wipe mind palace, start fresh
```

State lives at `docs/AI_TESTER_MIND_PALACE.json` — read first, written last,
shared by ALL four agents.

## Mind palace contract (shared by all four agents)

The mind palace is the team's common memory. It stores:

1. **`fingerprint`** — sha256 of (registered strategies + plugin file
   mtimes). When the fingerprint matches, the agents know the surface
   hasn't shifted.
2. **`open_bugs`** — findings the tester produces. Deduped by `slug(title)`.
3. **`fixed_bugs`** — closed findings.
4. **`runs`** — last 20 tester runs.
5. **`feature_requests`** — items the **trader_user** agent produces
   (what reports/charts/data the operator wants next).
6. **`tasks`** — items the **planner** agent produces (one per bug or
   feature request, ordered by priority, with files-to-touch + acceptance).
7. **`proposals`** — items the **executor** agent produces (structured
   code-change proposals, never auto-applied).
8. **`agent_runs`** — meta log of every run by every agent.
9. **`notes`** — short summaries each agent leaves between cycles.

All collections dedupe by stable `slug(title)` ids so re-running any agent
idempotently updates instead of duplicating.

## Test plan policy

Default plan: **NIFTY · nearest weekly expiry**. The runner queries
`/api/v1/legacy/expiries/?underlying=NIFTY&kind=weekly` and picks the
soonest expiry with DTE >= 0. **Never hard-code an expiry string** — pick it
fresh every run so the suite survives weekly rollovers.

Test scope (per run):

| Suite | What it covers |
|---|---|
| `auth` | login → JWT obtain; refresh works; bad creds rejected. |
| `meta` | `/api/v1/agents/catalog/` lists all 4 plugins with correct schemas; `/api/v1/legacy/expiries/` returns weekly + monthly for NIFTY. |
| `directional` | Fire on a stock (HDFCBANK), verify plan structure + indicator panel populated. NIFTY is index-only so directional is intentionally skipped on it. |
| `vertical_spread` | NIFTY weekly BULL CE spread — verify strikes resolved from scrip master (not step-rounded), LTPs > 0 when market is open, net_debit + max_profit + breakeven computed, structured error returned when LTPs are zero. |
| `pyramid` | NIFTY ATM CE weekly — verify candles fetched (>= 50), entries computed when present, plan.error surfaces when no candles. |
| `short_straddle` | Use the existing seeded position (id from mind palace, else first ACTIVE position) — verify snapshot legs all populated, scenarios list present, LLM action returned. |
| `plan_stock` | POST `/api/v1/agents/plan-stock/` for NIFTY monthly — all applicable strategies return data, allocations sum to 1.0, runtime < 30s. |
| `stock_summary` | `/api/v1/legacy/stock-summary/?symbol=NIFTY` — KPIs + buckets + rollups present, open positions have leverage > 0. |
| `step_replay` | Pick the most recent succeeded run, GET its detail, assert `steps[]` length matches the AgentStep count in DB. |
| `ui_plumbing` | Static cross-check: grep the React source for each field present in latest run responses. Flag fields that are in the payload but referenced nowhere in the UI (catches "data fetched but not rendered" bugs like the missing pyramid chart). |

## Known invariants (do not break)

These are the rules @RiskGuard + the legacy stack enforce. The tester treats
violations as **high-severity** findings:

- Position value > 10% of capital → RiskGuard rejects.
- Daily loss > 3% of capital → RiskGuard rejects.
- Risk per trade > 1% of capital → RiskGuard rejects.
- Confidence < 0.6 on a straddle action → validator overrides to MONITOR.
- Combined straddle premium > 1.3 × sold → validator forces CLOSE_BOTH.
- Expiry-day options with > 15:15 IST → CLOSE_BOTH.
- LegacyRouter must pin `trading.*` models to the legacy alias.
- AgentStep persistence must not raise SynchronousOnlyOperation.

## Where to look when something's broken

| Symptom | First place to check |
|---|---|
| Agent run stuck `queued` | Celery worker — `logs/celery.log` |
| `Listen failure: address already in use` | Orphan daphne — `lsof -nP -iTCP:8000 -sTCP:LISTEN -t \| xargs kill -9` |
| Empty option LTPs | `get_nfo_options` must accept both OPTIDX + OPTSTK; check `trading/services/ticker_service.py:309` |
| Empty NIFTY candles in plugin | `_fetch_option_candles` should auto-widen lookback; check pyramid plugin |
| Hooks error in React | Hook called after a conditional return — move all `useQuery`/`useMemo` above guards |
| `legacyApi` 404 | Path must include `/legacy/` prefix; baseURL is `/api/v1` not `/api/v1/legacy` |

## Credentials for tests

The runner authenticates as the seeded smoke user:

- email: `smoke@alphadesk.local`
- password: `smoke-1234`
- portfolio id: read from `/api/v1/portfolios/` (first row)

If those don't exist, the tester self-seeds via `User.objects.create_user(...)`
in `bootstrap()`. See `runner.py:_ensure_smoke_user()`.

## What "fixing" looks like for the AI agent

When the LLM layer is on (`--llm`), it gets the open-bugs list + the offending
file's contents and is asked to produce a unified-diff style fix proposal.
The agent **never edits files itself** — it writes the proposal into the
mind palace under each finding's `suggested_fix`, and a human (or Claude
Code) applies it. This keeps the test-and-fix loop auditable.

## Versioning

Bump `CONTEXT_VERSION` at the top of `state.py` whenever the test plan,
mind-palace shape, or this briefing changes meaningfully. The mind palace
records the version; mismatched versions cause a soft reset (open_bugs are
preserved, runs roll forward, fingerprint cleared).
