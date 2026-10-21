# Handoff: Mutual Fund module + AlphaDesk operational state

Context for a Claude Code session running natively on this machine (with real
Docker/network access) to continue work started in a cloud session that only
had bridge access to this repo. Read this fully before doing anything —
several items below are things the cloud session got wrong on first guess and
corrected after checking real code/logs; don't re-assume the wrong version.

## Where this fits in the bigger picture (so priorities make sense)

The owner (Abhinav) and Prabhakar built AlphaDesk/AgenticTrading as an
AI-driven Indian stock-trading system (Django + LangGraph + Angel One).
Longer-term vision, explicitly sequenced:

1. **Now**: AlphaDesk trading engine stays personal-use only. Prove it's
   profitable before selling it to anyone. No SEBI advisory registration
   needed yet for this reason.
2. **Current focus (this handoff)**: add a **mutual fund data collector +
   advisor** as a second asset class alongside equity/options trading.
   Deliberately **not** building FDs, bonds, commodities, or a full
   cross-asset financial planner yet — those are explicitly deferred.
   Tax filing is also explicitly deferred (and conceptually belongs to a
   separate project, FinnBull, not AlphaDesk).
3. **Later**: a light conversational persona ("chatbot") sitting on top of
   AlphaDesk's existing data (positions, signals, risk status) — motivated by
   the owner missing trading opportunities from being busy. Not started yet.
4. **Later still, before selling to anyone**: SEBI RA/IA registration for
   personalized stock advice, and separately AMFI ARN or SEBI RIA
   registration if the MF advisor ever gives personalized fund
   recommendations to other people (not just the owner). Not urgent for
   personal use, but don't let the product drift past this unregistered.

## What's already built in this session (on disk, branch `abhinav/redesign-v2`)

New Django app `backend/apps/mutual_funds/`:

- `models.py`:
  - `MFScheme` — shared AMFI scheme catalog (scheme_code PK, ISINs, AMC name,
    scheme name, plan, option, category, is_direct). Not tenant-scoped —
    same data for everyone.
  - `MFNav` — daily NAV time series per scheme, unique per (scheme, date).
  - `MFHolding` — tenant-scoped, FK to `apps.trading.Portfolio` (reuses the
    existing Portfolio model, same role as equity `Position`). Has
    folio_number, units, avg_cost_per_unit, source (manual/cas).
  - `MFTransaction` — tenant-scoped, FK to `MFHolding`. For CAS-imported
    transaction history (buy/sell/sip/switch/dividend).
- `services/amfi_nav_collector.py` — fetches and parses AMFI's daily NAV
  file (`https://portal.amfiindia.com/spages/NAVAll.txt` — the `www.` host
  302-redirects here as of Oct 2026). **The parser was tested against real
  fetched AMFI data and confirmed correct** — the format is NOT clean CSV:
  category header lines and AMC-name header lines are interleaved among
  semicolon-delimited data rows with no explicit tag saying which is which
  (category lines contain a `(`, AMC lines don't). See the module docstring
  for the exact format.
- `management/commands/pull_amfi_nav.py` — daily pull command, `--dry-run`
  flag, upserts via `update_or_create`/`get_or_create`, structlog + non-crash
  on fetch failure. Matches the style of existing commands like
  `enrich_signals.py`.
- Registered in `backend/config/settings/base.py` `LOCAL_APPS`.
- Migration generated: `apps/mutual_funds/migrations/0001_initial.py`
  (ran successfully once the DB was reachable — see bug fixed below).

## Status update — 2026-10-17 (supersedes the "not yet migrated" notes above)

Everything above is now **built, migrated, and committed** on branch
`abhinav/redesign-v2` (commits `35985cc` code, `4164ad9` docs). On the
original machine: migrations applied, `pull_amfi_nav` loaded **14,376
schemes / 14,376 NAV rows** (latest NAV 2026-10-14). 89 tests pass
(`cd backend && .venv/bin/python -m pytest tests/mutual_funds -q`).

Added since this handoff was first written:
- `services/amfi_history.py` + `backfill_amfi_history` — per-AMC AMFI
  history (AMFI cannot filter by scheme; ~15 MB per AMC-year, so pass only
  held schemes). Verified: 494 rows / 2 years for scheme `148921`.
- `services/staleness.py` + `check_stale_navs` — flags 5,628 schemes today,
  mostly matured funds; noisy, restrict to holdings in MF-3.
- `tasks.py` + two beat entries in `settings/base.py` (23:45 and 07:30 IST).
  Wired and loading, but not observed firing (needs worker + beat).
- `services/cas_parser.py`, `cas_import.py`, `import_cas_statement`
  (MF-2) — via `casparser`; **synthetic fixtures only, never run on a real
  CAS PDF**.
- `services/advisor.py` + `run_mf_advisor` (MF-3) — direct-vs-regular
  (estimate), allocation vs target, category/AMC overlap, SIP continuity.
  **Never run on real holdings.**
- `docs/MF4_HOUSEKEEPING_FINDINGS.md` — read-only MF-4 evidence; no
  decisions made, nothing deleted.

### Bugs found and fixed loading real AMFI data
- `MFScheme.option` was `varchar(30)`; AMFI values reach 84 chars → widened
  to 100 (migration `0002_widen_scheme_option`).
- The NAV file was decoded via `resp.text` (latin-1 fallback) and mangled
  non-ASCII category names; now decoded as UTF-8 from `resp.content`.

### Continuing on a new machine
1. `git fetch && git checkout abhinav/redesign-v2`
2. Recreate `backend/.env` (gitignored — it did not travel) from
   `backend/.env.example`. `DATABASE_URL` must use port **5444**, not the
   stale 5436 (that range belongs to unrelated Hopzy projects).
3. `cd backend && uv sync` — `casparser` was added to `pyproject.toml` but
   `uv.lock` was not regenerated; commit the lock change.
4. `docker compose -f docker-compose.dev.yml up -d`, confirm Postgres/Redis
   healthy, then `manage.py migrate` and `manage.py pull_amfi_nav`. The NAV
   data is in the old machine's Postgres volume, not in git — a fresh
   machine starts empty.
5. Tests need `pytest pytest-django pytest-asyncio factory-boy` in the venv
   (they were missing from the original `.venv` and were installed ad hoc).

### Verified later on 2026-10-17 (second machine, uncommitted working tree)
- Working tenant here is `istpduckymomo` (`istpduckymomo@gmail.com`, id
  `000d8a50-ebbe-4a8c-9d1e-cd4bf7a4d026`). Portfolios: `Default` (paper,
  500,000 — the trading one) and `Mutual Funds` (paper, capital 0, id
  `ca053d3f-27b4-4717-8013-5944b72c48c9`) for MF holdings. Default-portfolio
  lookups (`legacy_compat_views`, monthly report view, RAG portfolio
  retriever) now `order_by("created_at")` so the MF portfolio never gets
  picked as the trading one.
- MF-1: `pull_amfi_nav` task ran on the live Celery worker (succeeded in
  68 s; 14,390 schemes, 87 new NAV rows, stale sweep 5,628). Both beat
  entries load (Asia/Kolkata, next 23:45). A scheduled fire is still **not
  observed** — check `logs/beat.log` after 23:45 IST.
- MF-0b step 3: `manage.py zerodha_login` written + 10 tests (kiteconnect
  mocked). Shares `apps/market_data/services/broker_connect.py`
  `connect_broker()` with `POST /brokers/{name}/connect/`. Never run
  against real Kite. `KITE_API_KEY` / `KITE_API_SECRET` added to settings
  and `backend/.env.example`.
- MF-3 smoke: `run_mf_advisor --tenant istpduckymomo --portfolio "Mutual
  Funds"` runs cleanly on Postgres → "No MF holdings found … 0 finding(s)."

### What is genuinely still open
1. **A real CAS PDF** (CAMS/KFintech) — the only thing that can verify MF-2
   and MF-3. First run: `import_cas_statement --file <pdf> --tenant
   istpduckymomo --portfolio "Mutual Funds" --dry-run`.
2. ~~A `trading.Portfolio` for MF holdings~~ — done (see above).
3. **Real Zerodha link** — owner registers a Kite Connect app (Personal
   plan), puts `KITE_API_KEY` / `KITE_API_SECRET` in `backend/.env`, then
   runs `manage.py zerodha_login --tenant istpduckymomo` every trading
   morning. No broker links exist on this machine yet, so equity is on
   synthesized prices.
3a. **MF-UI — Mutual Fund page** (new phase, after MF-3, not gated on MF-5):
   read-only `/api/v1/mutual-funds/` API (holdings + latest NAV/value/gain,
   scheme search + NAV history, latest advisor findings, NAV freshness),
   `frontend/src/features/mutual-funds/` + `/mutual-funds` route + nav entry
   (holdings table, allocation by category, findings, freshness badge, CAS
   empty state), backend tenant-isolation tests + a frontend test.
   **Built 2026-10-17 (uncommitted):** API in `apps/mutual_funds/api/`
   (holdings, schemes search/detail, advisor findings + owner-only POST run,
   status), page in `frontend/src/features/mutual-funds/`. 18 API tests + 2
   frontend tests; full backend suite 588 passed. Curled live on
   `istpduckymomo`: every endpoint 200 with empty holdings; a temporary
   holding (folio `TEMP-MFUI-VERIFY-DELETE-ME`) was created, curled and
   deleted again — zero MF holdings remain. Not visually checked in a
   browser. Acceptance with real holdings still waits on a CAS import.
4. **MF-4 owner decisions** — see the findings doc. Notably: remove the six
   empty old app dirs?; archive `db.sqlite3`?; investigate the possible
   v1/v2 `enrich_signals` clash (v1 may mark `swing_v2_*` signals
   `EXPIRED`); prod Docker images copy only `backend/`, so the legacy
   `trading/` package is likely missing in prod.
5. **Advisor event type** — findings currently use `workflow.step.completed`
   because `Event.Type` has no advisor member; a dedicated type needs an
   `events` enum change + migration.
6. Replace the advisor's assumed 0.75% regular-plan drag with real expense
   ratios when available.

## Original roadmap notes (kept for design rationale)

1. **CAS import** — parse a CAMS/KFin
   Consolidated Account Statement (the password-protected PDF a user gets,
   PAN+DOB as password) into `MFHolding` + `MFTransaction` rows. This is
   how real holdings get into the system — there's no broker API for MF
   folios in India; CAS parsing or the RBI Account Aggregator framework
   (Setu/Finvu/OneMoney as consent managers) are the two real options. CAS
   parsing is the faster personal-use path; AA integration is the "do this
   properly for other users later" path.
2. **MF advisor service** — keep this in its own module
   (`apps/mutual_funds/services/advisor.py`), deliberately separate from the
   data-collection code, so collection never silently grows advice-giving
   side effects. Rule-based + LLM-explained, same pattern as the existing
   Analyst/RiskGuard split elsewhere in the codebase:
   - category allocation vs. a target
   - direct-vs-regular plan expense drag (an easy, objectively correct win
     — regular plans cost more via distributor commission, direct plans
     don't)
   - overlap detection between held funds
   - rebalancing suggestions *against the owner's own stated targets*, not
     generic market calls
   - Log every suggestion through the existing `apps.events.services.
     event_writer.emit()` non-blocking pattern, same as every other
     decision in this codebase.
   - **This is personal-use advice on the owner's own money right now.**
     The moment this starts giving personalized recommendations to a
     second person, the AMFI ARN / SEBI RIA question above becomes live.

## Separate, lower-priority housekeeping surfaced during this session

These are real findings, not guesses, but intentionally **not acted on yet**
— each needs a decision from the owner before touching it:

1. **Half-finished app migration.** `backend/apps/` on disk still has
   `audit`, `journals`, `broker`, `orders`, `portfolio`, and `legacy`
   sitting alongside `events` and `trading`, which CLAUDE.md claims already
   replaced them (Phase 4a/4c). Confirmed via `grep` that `LOCAL_APPS` in
   `settings/base.py` does **not** include the old ones — Django isn't
   loading them. Likely safe to delete, but **confirm no plain Python
   import still reaches into them** (not just the Django app registry)
   before deleting anything.
2. **`db.sqlite3` still exists at repo root** even though README says Phase
   6 dropped the legacy SQLite DB entirely. Check whether anything still
   reads it before removing.
3. **Duplicate-looking v1/v2 management commands** in
   `backend/apps/trading/management/commands/`: `enrich_signals.py` /
   `enrich_signals_v2.py`, `run_ok_scanner.py` / `run_ok_scanner_v2.py`, and
   four backtest entry points. **Do not assume v1 is dead** — the plugin
   registry log shows `swing_scanner` AND `swing_scanner_v2` both actively
   registered simultaneously right now. Ask the owner which is still
   supposed to be live before removing either.
4. **Undocumented-in-CLAUDE.md pieces that are actually real/live, not
   dead**: a `fyers` broker plugin (fully registered alongside angel_one/
   zerodha/paper — not a stub), a `vertical_spread` strategy plugin, and a
   StockEdge data pipeline (`pull_stockedge_breadth.py`,
   `pull_stockedge_csv.py`, `stockedge_login.py`,
   `stockedge_momentum*.py`, env vars `SL_API_KEY`/`SL_USERNAME`) with a
   scripted login suggesting scraping — worth checking StockEdge's terms of
   service before relying on it in anything that gets sold.
5. **Doc drift, two layers deep.** `docs/MIND_PALACE.md` and
   `docs/architecture/ARCHITECTURE.md` describe an *older* design (dual
   Django process, legacy SQLite bridge, Streamlit dashboard) that
   README/CLAUDE.md say was already removed (Phases 6 and 8). But
   CLAUDE.md/README themselves are *also* wrong about ports and the dev
   login per the `alphadesk-dev-stack` skill. **Trust the skill over both
   doc layers** for anything operational.

## Conventions to follow when continuing this work

- Tenant-scoped models inherit `apps.common.tenancy.TenantModel`.
- Every meaningful decision/action gets logged via
  `apps.events.services.event_writer.emit()` — never a raw
  `Event.objects.create()` — and that logging must never block the main
  flow (wrap in try/except if needed, same as elsewhere in the codebase).
- Management commands follow the existing style: structlog logging,
  `--dry-run` where it makes sense, clear stdout success/error messages,
  don't crash the whole run on one bad row — skip and log instead.
- Keep advice-generating logic physically separate from data-collection
  code (see the advisor module note above) — this isn't just style, it's
  so a future audit of "what gives advice" doesn't have to read through
  plumbing code too.
