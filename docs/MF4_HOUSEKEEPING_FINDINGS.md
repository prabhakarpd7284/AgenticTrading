# MF-4 Housekeeping Findings

Read-only investigation, 2026-10-15, branch `abhinav/redesign-v2`. Nothing was
edited, deleted, committed, or run against the DB/broker (one `pytest
--collect-only` and `manage.py help` were run; neither writes). Every "decision"
below is a **recommendation for the owner to confirm**.

Line numbers refer to files as read today; `CLAUDE.md` changed during the
session, so prefer the quoted text over its line number.

## Summary table

| # | Item | Recommendation | Risk of acting |
|---|------|----------------|----------------|
| 1 | `audit`, `journals`, `broker`, `orders`, `portfolio`, `legacy` dirs | **Remove** (empty `__pycache__` husks) | None |
| 2 | `swing_scanner` vs `_v2`, `enrich_signals` vs `_v2`, `run_ok_scanner` vs `_v2` | **Keep both** (both live); fix a 16:00 enrichment clash | Medium if one is deleted |
| 3 | `db.sqlite3` + root legacy Django project | **Archive out of repo, then remove**; owner confirms the 2 straddle positions are not needed | Low |
| 4a | `fyers` plugin | **Keep** (live) and document | High if removed |
| 4b | `vertical_spread` plugin | **Keep** and document | Medium |
| 4c | StockEdge pipeline | **Keep as personal-use only; defer** any commercial reliance until ToS is read | Legal, not technical |
| 5 | Doc drift | **Fix** (list in section 5) | None |
| + | Legacy `trading/` package is missing from the Docker images | **Decide** (deployability) | High for prod |

---

## 1. The six "old" apps under `backend/apps/`

**Finding: they are not code. They are leftover `__pycache__` directories.**

Evidence:

```
audit:     total files=5   pyc=5   (no .py, no __init__.py)
journals:  total files=10  pyc=10
broker:    total files=10  pyc=10
orders:    total files=16  pyc=16
portfolio: total files=33  pyc=33
legacy:    total files=6   pyc=6
git ls-files backend/apps/<each> -> 0 tracked files
git status --ignored       -> "!! backend/apps/audit/" (ignored by *.py[oc] / __pycache__/)
```

- Unregistered in `LOCAL_APPS` (`backend/config/settings/base.py:60-73`): confirmed. Only
  `common, accounts, tenants, billing, market_data, mutual_funds, trading, strategies,
  agents_core, rag, events, system, notifications` are registered.
- Plain-Python imports reaching them (searched `backend/`, `trading/`, `scripts/`,
  `pyproject.toml`, `docker/`, `infra/`, `.github/`, `docker-compose*.yml`; patterns
  `apps.<name>`, `from apps import <name>`, `apps/<name>`): **zero live imports.** The only
  hits are comments/docstrings:
  - `backend/apps/events/services/journal_port.py:5`, `events/api/journal_views.py:3`
    (docstrings: "old `apps.journals.JournalEntry` was deleted")
  - `backend/apps/trading/models.py:26,92` and `trading/migrations/0001_initial.py:3`
    (comments: "was apps.portfolio / apps.orders")
  - `backend/apps/trading/services/risk_engine.py:6` (docstring mentions
    `apps/orders/services/risk_guard.py`)
  - `scripts/diagnose_legacy.sh:4` (comment referencing `apps.legacy.api.views._with_legacy`)
- Migration references: no live app's migration depends on an old label. The only string
  matches for `portfolio` are FKs to `trading.portfolio`; `broker` appears only as an
  `actor_kind` choice value in `events/migrations/0001_initial.py:29`.
- URL includes: `config/urls.py` already points `brokers/`, `portfolios/`, `orders/`,
  `journals/` at `market_data`, `trading`, `events`.
- Git history confirms the real removal: `2e4def4` (4a), `e0c1db5` (4b), `3a66fdd` (4c).

**Recommendation: REMOVE** all six directories (`rm -rf backend/apps/{audit,journals,broker,orders,portfolio,legacy}`).
Nothing tracked, nothing imported. They only look alive because a stale `.pyc` tree survives
branch switches. Same applies to root `dashboard_utils/` (contains only `__pycache__`).

Related, optional: `scripts/diagnose_legacy.sh` curls `/api/v1/legacy/*`; those routes are
now served by `apps/trading/api/legacy_compat_views.py` (reads v2 tables), so the script
still has a purpose, but its comment about `apps.legacy.api.views` is stale. **Update the
comment, keep the script.** Not verified that every path in it still resolves.

---

## 2. swing v1 vs v2

**Finding: both generations are live. Neither is dead. v2 is not a replacement.**

Correction to `HANDOFF_MUTUAL_FUNDS.md` item 3: the duplicate commands are **not** in
`backend/apps/trading/management/commands/` (that directory holds only `__init__.py`). They
are in the **root legacy package** `trading/management/commands/`: `enrich_signals.py`,
`enrich_signals_v2.py`, `run_ok_scanner.py`, `run_ok_scanner_v2.py`, plus `derive_swing_trades.py`,
`run_ok_backtest.py`, `run_ok_intraday_backtest.py`, `run_backtest.py`, `run_backtest_intraday.py`.
They are discovered only because `config/settings/dev.py:49-57` appends
`trading.apps.TradingLegacyConfig` (see section 6 for why that matters).

| Piece | v1 | v2 |
|-------|----|----|
| Plugin entry point | `swing_scanner` -> `plugins.strategy_swing:SwingScannerStrategy` (`pyproject.toml:96`) | `swing_scanner_v2` -> `plugins.strategy_swing.v2.plugin:SwingScannerV2Strategy` (`pyproject.toml:97`); registered in the live plugin log |
| Scan command | `run_ok_scanner` (daily/weekly cycles, NIFTY 100) | `run_ok_scanner_v2` (tiers: small 15m, medium 1h, long daily) |
| Scheduled by | `CELERY_BEAT_SCHEDULE["daily-swing-scan"]` 08:20 Mon-Fri -> `daily_pipeline.run_swing_scan` -> `call_command("run_ok_scanner")` (`apps/strategies/tasks/daily_pipeline.py:109`) | `config/celery.py:104-121` `on_after_finalize` adds `swing_v2_small` (`*/30` 9-15h), `swing_v2_medium` (10,12,14h :05), `swing_v2_long` (15:45), `enrich_swing_v2` (16:00) |
| Enrichment | `enrich_signals` via `daily-eod-enrichment` 16:00 (`daily_pipeline.py:457-461`) | `enrich_signals_v2` via `enrich_swing_v2` 16:00 |
| Persists | `Signal(source=OK_SCANNER)` | `Signal(source=OK_SCANNER, strategy="swing_v2_<tier>_<phase>")` |
| Frontend | `SwingScannerPage` -> `run_ok_scanner` (`features/swing-scanner/SwingScannerPage.tsx:77`, `MonthlyPage.tsx:1538`, `agentConsole.utils.ts:152`); API `GET swing-scanner/` -> `swing_scanner_service.py` | No frontend references to `_v2` |
| Ops allowlist | `run_ok_scanner`, `enrich_signals` in `PRODUCT_COMMANDS` (`apps/system/services/ops_access.py:53-63`) | `_v2` variants also listed |
| Tests | `test_swing_enrichment.py` | `test_swing_v2.py`, `test_enrich_v2.py` (import `trading.management.commands.*_v2`) |

Problems found (these matter more than the duplication):

1. **Likely enrichment clash at 16:00.** v1 `enrich_signals` only excludes
   `SWING_SOURCES = ["STOCKEDGE"]` (`apps/strategies/services/swing_enrichment.py:29`;
   used at `trading/management/commands/enrich_signals.py:92-97`). It does **not** exclude
   `source=OK_SCANNER, strategy startswith "swing_v2_"`. For today's v2 signals with
   `eod_price IS NULL`, v1 sets `eod_price`, same-day MFE/MAE, and flips
   `PENDING -> EXPIRED` (`enrich_signals.py:193-201`). v2's own docstring
   (`enrich_signals_v2.py:3-12`) says that is exactly the wrong behaviour for multi-day
   swings, and its default query only walks `outcome=PENDING` (`:66-72`), so signals v1
   already expired are skipped unless `--all` is passed. Both jobs are scheduled for
   16:00. **Not reproduced by running it**; this is from reading the code. Candidate fix:
   `.exclude(strategy__startswith="swing_v2_")` in v1, or add the v2 marker to the same
   exclusion helper.
2. v2 beat entries are registered through a signal hook in `config/celery.py`, not in
   `CELERY_BEAT_SCHEDULE`, so reading `base.py` alone suggests v2 is unscheduled.
3. `swing_scanner_v2` has no entry in `STRATEGY_CATALOG`
   (`apps/system/api/pipeline_views.py:77-`), so the /pipeline page does not show it.
4. `CLAUDE.md` mentions neither v2 nor these commands' real location.

**Recommendation: KEEP BOTH.** Ask the owner one question: "is v1 still wanted now that v2
runs intraday tiers, or should v1 be retired after a freeze period?" Do not delete either
until the 16:00 clash (item 1) is resolved, because deleting v1 enrichment changes what the
monthly report sees.

---

## 3. `db.sqlite3` and the old root Django project

Facts:

- `/db.sqlite3`: 426 KB, last modified 18 May 2026, **gitignored and untracked**
  (`.gitignore:27`).
- Opened read-only (`mode=ro`). 18 tables. Rows with data: `trading_watchlistentry` 58,
  `trading_auditlog` 16, `trading_tradejournal` 8, `trading_straddleposition` 2,
  `trading_portfoliosnapshot` 1. Empty: `auth_user`, `trading_signallog`,
  `trading_systemcontrol`, `trading_tradernote`, `trading_strategydoc`.
  I did not read row contents.
- Who reads it:
  - **Root** `config/settings.py:34-37` defaults `DB_ENGINE` to sqlite3 and `DB_NAME` to
    `BASE_DIR/db.sqlite3`. Root `manage.py` sets `DJANGO_SETTINGS_MODULE=config.settings`.
    So `python manage.py ...` from the repo root still talks to this file. Root
    `config/`, `manage.py`, `app.py`, `main.py` are all **tracked in git**.
  - **Backend** settings never point at it, except a fallback:
    `backend/config/settings/base.py:117` `env.db_url("DATABASE_URL", default="sqlite:///db.sqlite3")`.
    If `DATABASE_URL` is unset this silently creates/uses a SQLite file relative to the cwd.
    `config/settings/dev.py` has a comment saying there is "no SQLite fallback" (dev
    pins Postgres), but `base.py` and therefore `prod.py` still has the default.
  - `backend/config/settings/test.py:27` uses SQLite for tests (fine).
- Legacy `trading/` package: `trading/models.py` and `trading/migrations/` are SQLite-era;
  `trading/apps.py:12` overrides `import_models()` to a no-op and `dev.py` sets
  `MIGRATION_MODULES = {"trading_legacy": None}`, so it owns no tables. Several commands
  still print "legacy SQLite era" tombstones (`run_straddle_monitor.py:194`,
  `run_trading_day.py:3,36`).
- Root `app.py` is a **Streamlit** app (`import streamlit as st`) and `main.py` is the
  original notebook-era SmartAPI/LangGraph/Ollama script. Both are tracked.

**Recommendation:**
- Move `db.sqlite3` out of the repo (e.g. `~/archive/`) rather than deleting, after the
  owner confirms the 2 `StraddlePosition`, 8 `TradeJournal` and 58 `WatchlistEntry` rows are
  either already in Postgres or not needed. **Defer deletion to the owner.**
- Change `base.py:117` to have no default (fail closed like `prod.py` does for
  `DJANGO_SECRET_KEY`). **Defer** (touches settings, owner's call).
- Root `config/`, `manage.py`, `app.py`, `main.py`: **remove candidates**, but they are
  tracked and `CLAUDE.md` claims Phase 8 already removed Streamlit, so confirm first.
  `app.py` is the only remaining Streamlit entry point.

---

## 4. `fyers`, `vertical_spread`, `scalp`, StockEdge

### fyers broker plugin: KEEP, it is live

- Registered: `pyproject.toml:109` `fyers = "plugins.broker_fyers.adapter:FyersAdapter"`;
  dependency `fyers-apiv3>=3.1` (`pyproject.toml:59`).
- Real code: `plugins/broker_fyers/adapter.py` 310 lines, `symbols.py` 226, `stream.py` 71.
- Wired end to end: OAuth login/callback in `apps/market_data/api/oauth_views.py:79-117,204`;
  `DAILY_TOKEN_BROKERS = {"zerodha","fyers"}` in `broker_views.py:25`; frontend catalog
  `frontend/src/features/broker/brokerCatalog.ts:97-112`, `BrokerLinkPage.tsx`;
  `OptionsDashboardPage.tsx:472` treats `source === "fyers"` as live data.
- Depended on by the scalp strategy: `plugins/strategy_scalp/data.py:1-5,66` streams Fyers
  seconds candles; `ScalpingPage.tsx:235` has a "Go Live (Fyers)" button.
- Documented? Only in the dev-stack skill and the frontend; **absent from `CLAUDE.md`**.

### vertical_spread strategy plugin: KEEP

- Registered: `pyproject.toml:92`. 674-line `strategy.py` plus `bias.py`, `pricing.py`,
  `selector.py`, `lifecycle.py` and four test files inside the plugin.
- In `STRATEGY_CATALOG` (`pipeline_views.py:111`); has a playbook
  `docs/strategies/DIRECTIONAL_VERTICAL_SPREAD.md`. `agents_core/apps.py:13` references it
  in a comment about plugin registration. No frontend page of its own.
- Absent from `CLAUDE.md`.

### scalp strategy plugin (not asked, found while checking): KEEP

Registered (`pyproject.toml:93`), ~1,400 lines in `plugins/strategy_scalp/`, with
`agents_core/scalp_consumer.py`, `scalp_sim.py`, `strategies/api/scalp_views.py`,
`run_scalp` command, a WebSocket route in `config/channels_router.py`, and
`frontend/src/features/scalping/`. Tests: `tests/unit/test_scalp_engine.py`,
`tests/integration/test_scalp_sim_ws.py`. Absent from `CLAUDE.md`.

### StockEdge pipeline

What it does (`apps/market_data/integrations/stockedge/`, ~1,800 lines across
`parser.py`, `harness.py`, `csv_ingest.py`, `momentum.py`, `momentum_signals.py` and five
commands `pull_stockedge_breadth`, `pull_stockedge_csv`, `stockedge_login`,
`stockedge_momentum`, `stockedge_momentum_signals`):
- Ingests market-breadth snapshots, stock-wise momentum scores and CSV exports into
  `StockEdgeSnapshot`/`StockEdgeBreadthRow`/`StockEdgeScanRow` (shared, not tenant-scoped).
- `stockedge_momentum_signals` turns ranked momentum into `Signal(source=STOCKEDGE)` rows,
  which feed the monthly report; `SWING_SOURCES` makes the intraday enricher skip them.
- README states it is an **advisory overlay that never reaches @RiskGuard or sizing**.

How it authenticates (from `stockedge_login.py`, `harness.py`,
`docs/integrations/STOCKEDGE_INTEGRATION.md`):
- Google OIDC. The operator logs into `https://web.stockedge.com` in a **real browser**
  (fresh headed Chromium, or `--cdp` attach to the user's own Chrome), and the Playwright
  `storage_state` (cookies + localStorage) is saved to a JSON file.
- `--live` capture replays that saved session headless and reads either the DOM grid or the
  responses the page itself fetches. The CSV path ingests StockEdge's own "Download" export
  by hand.
- The design doc records that direct API calls with a valid Bearer token return HTTP 504
  (gateway fingerprinting) and says "forging these API calls ... is a bypass of bot
  protection (against StockEdge ToS). We will not do that." The harness therefore drives a
  genuine logged-in session with automation. That is a different thing from forging
  requests, but whether automating the logged-in web app is itself permitted is exactly the
  open question below.
- The handoff mentions env vars `SL_API_KEY` / `SL_USERNAME`. **I could not find them
  anywhere in the repo** (`grep -rnE "\bSL_[A-Z_]+"` only hit unrelated `SL_HIT` strings).
  Treat that handoff claim as unverified.

**I did not fetch StockEdge's terms of service.** Nothing below states what the ToS says.
What the repo's own guardrails (`STOCKEDGE_INTEGRATION.md` section 7, still unchecked at
section 8) say to verify:
1. Does the plan the account is on (a "Personal Club subscription" per the doc) permit
   automated/scripted access to the web app or exports, even for personal use?
2. Are downloaded exports, and values derived from them, allowed to be stored, and in any
   product shown to anyone other than the subscriber? Note exports carry a "Generated by and
   for: <email>" line, i.e. they are tied to the subscriber.
3. Is there an official API or data-licensing tier? The doc's preferred long-term path.
4. Any rate or concurrency limits (doc's own rule: EOD batch only, single concurrency, jittered).

**Recommendation: KEEP for personal use, DEFER** any plan to ship it to other users until
someone reads the actual terms (or asks StockEdge) and records the answer.

---

## 5. Doc drift (claims vs reality)

Reality sources: `.claude/skills/alphadesk-dev-stack/SKILL.md`, `docker-compose.dev.yml`,
`backend/config/settings/dev.py`, the code and the running DB.

Note on the skill: it says CLAUDE.md/README give Postgres 5436, Redis 6379 and Zerodha
"stub". **That is now partly stale.** `CLAUDE.md` and `README.md` already say Postgres
`:5444` and Redis `:6380` (`CLAUDE.md` lines ~57,109,391-392; `README.md:66,155,267`). Only
the Zerodha "stub" claim and the dev login are still wrong. Update the skill's "Common
Mistakes" row too.

### CLAUDE.md
| Claim | Reality |
|-------|---------|
| Layout: `broker_zerodha/ Zerodha Kite adapter (stub)` (~line 40) | `plugins/broker_zerodha/adapter.py` is 154 lines with `authenticate`, `health_check`, `fetch_positions`, `fetch_holdings`, `fetch_margin`, plus OAuth in `oauth_views.py`. Not a stub (read-only, no order placement verified) |
| Plugin list omits `broker_fyers`, `strategy_scalp`, `strategy_vertical_spread` | All three registered in `pyproject.toml:92-93,109` and live (section 4) |
| App list omits `mutual_funds` | Registered in `LOCAL_APPS` today |
| "pytest suite (99/99 passing)", "expect: 99 passed" (~lines 42,125) | `pytest --collect-only` collects **559** tests (includes the new untracked `tests/mutual_funds/`). Pass/fail not run |
| "UI: default dev login `admin@local.dev` / `devadmin`" (~line 120) | Skill: wrong tenant. In the current DB neither `admin@local.dev` nor `prabhakarpd7284@gmail.com` exists (users are `istpduckymomo@gmail.com`, `smoke@alphadesk.local`; tenants "Smoke Tenant" and "Abhinav dixit's workspace"). The roadmap text also points at a tenant that is not in this DB |
| "Phase 8 removed the Streamlit dashboard (`dashboard.py` / `dashboard_utils/`)" (line 15) | Root `app.py` is still a Streamlit app (tracked); `dashboard_utils/` still exists as a `__pycache__`-only dir |
| "How to Run": `manage.py run_trading_agent`, `run_ok_scanner`, `enrich_signals` from `backend/` | Works only under `config.settings.dev`, which alone installs `trading.apps.TradingLegacyConfig` (section 6) |
| Duplicate v2 commands not mentioned | `run_ok_scanner_v2`, `enrich_signals_v2`, `enrich_swing_signals`, `swing_scanner_v2` are live (section 2) |
| Docs index lists MIND_PALACE/ARCHITECTURE as the spatial/formal spec | Both describe the pre-Phase-6 design (below) |

### README.md
- `:143` "expect: 99 passed": 559 tests collected.
- Ports (`:66,155,267`) and the Phase 6/8 note (`:26-27`) are correct.
- Does not mention fyers/scalp/vertical_spread/mutual_funds.

### docs/MIND_PALACE.md
- `:291` "Station A: Streamlit Dashboard (The Original Bridge)" presents Streamlit as a
  current component; removed per `CLAUDE.md`, though `app.py` lingers.
- `:633` says `apps.common.db_router.LegacyRouter` pins `trading.*` to the `legacy` SQLite
  alias. **`LegacyRouter` no longer exists**: no `db_router.py` in `apps/common`, no
  `DATABASE_ROUTERS` in settings; `dev.py:9,20` says "LegacyRouter ... retired".
- `:373`, `:552`, `:715` describe `enrich_signals` walking `SignalLog`/`TradeJournal`.
  It now enriches `apps.strategies.Signal` (`enrich_signals.py:2`).

### docs/architecture/ARCHITECTURE.md
- `:31-33,47,91,129,184` describe "two cooperating Django processes", a Legacy bridge on
  SQLite, `LegacyRouter`, TradeJournal/SignalLog/AuditLog read from SQLite.
- `:246-254` migration table marks phases 5-7 as "In progress / Planned" and states "The
  current shape is **dual-Django** ... intentional and stable". `CLAUDE.md` says Phase 6
  retired `:8001` and SQLite; `dev.py` docstring agrees. There is a single ASGI process.
- `:279-280` "Dual-Django prod deployment", "Fine for SQLite" are obsolete.

**Recommendation: FIX.** Cheapest sequence: (1) correct `CLAUDE.md` facts above; (2) add a
"superseded by Phase 6" banner at the top of MIND_PALACE and ARCHITECTURE and rewrite
the stale sections later; (3) update the dev-stack skill's Common Mistakes row.

---

## 6. Extra findings (outside the five items, worth a decision)

1. **The root `trading/` package is not in the Docker images.** `docker/backend.Dockerfile:25`
   and `docker/worker.Dockerfile:18` copy only `backend/` into `/app`, and `pyproject.toml`
   packages only `apps*`, `config*`, `plugins*`. Separately, `trading.apps.TradingLegacyConfig`
   is installed **only in `config/settings/dev.py`**, not `base.py` or `prod.py`, while the
   Dockerfiles and Terraform set `DJANGO_SETTINGS_MODULE=config.settings.prod`. Consequences
   (by reading, not tested in a container): under prod settings `call_command("run_ok_scanner")`,
   `enrich_signals`, `run_screener` etc. in `daily_pipeline.py` and `config/celery.py` would
   fail with "Unknown command", and `from trading.services.data_service import ...` would
   raise `ImportError`. Dev works because `dev.py:41-45` puts the repo root on `sys.path`.
   The daily beat pipeline therefore only works locally today.
2. `base.py:180` still defaults `REDIS_URL` to `redis://localhost:6379/0`, which `dev.py`
   says is a **different project's Redis** on this machine. `dev.py` overrides it; any
   process started without `config.settings.dev` and without `REDIS_URL` set would use it.
3. `backend/apps/agents_core/tester/` exists on disk as a gitignored `__pycache__`-only
   directory (same husk pattern as section 1).
4. `tests/mutual_funds/` is untracked and was not created by this investigation.

## Questions for the owner

1. OK to delete the six empty husk dirs and `dashboard_utils/`?
2. v1 swing vs v2: keep both, or schedule v1 for retirement? And fix the 16:00 clash now?
3. `db.sqlite3`: archive it, and are the 2 straddle positions / 58 watchlist rows needed?
4. Remove root `app.py`, `main.py`, `manage.py`, `config/` (old Streamlit/SQLite project)?
5. Is production deployment of the daily pipeline planned soon? If so, the missing legacy
   `trading/` package (section 6.1) needs a decision first.
6. Will someone read StockEdge's actual terms before anything beyond personal use?
