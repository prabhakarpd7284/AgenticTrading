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

**None of this has been migrated into a live DB yet** — that's the immediate
next step, blocked only on Docker being up (see below).

## Bug already found and fixed this session

`backend/.env` had `DATABASE_URL` pointing at port **5436** (stale — that
port now belongs to a different, unrelated "Hopzy" project per the repo's
own `alphadesk-dev-stack` skill). Fixed to **5444**, which matches
`docker-compose.dev.yml`'s actual port mapping (`5444:5432`). This fix is
already applied and saved in `backend/.env`.

## Immediate next steps, in order

1. `cd ~/abhinav-codes/AgenticTrading && docker compose -f docker-compose.dev.yml up -d`
   then `docker compose -f docker-compose.dev.yml ps` — confirm Postgres
   and Redis both show running/healthy. (This was blocked in the cloud
   session because its device-bridge shell has no Docker — needs to run
   from an actual native terminal, which this Claude Code session has.)
2. `cd backend && .venv/bin/python manage.py migrate`
3. `.venv/bin/python manage.py pull_amfi_nav` — first real NAV pull, expect
   on the order of ~16,000 scheme/NAV rows. Report scheme/NAV counts back.
4. **Read `.claude/skills/alphadesk-dev-stack/SKILL.md` before trusting
   anything about broker status or market data** — it documents two
   non-obvious failure modes that look fine but aren't real:
   - The advertised dev login `admin@local.dev` in README/CLAUDE.md is the
     **wrong tenant**. The working one is `prabhakarpd7284@gmail.com`.
   - If no broker link is `status: active` on the *right* tenant, the
     system silently serves a **synthesized options chain** with nothing
     louder than a log WARNING — strategies look normal while trading on
     fake prices.

## After the NAV pull works: the actual MF module roadmap

1. **CAS import** (next big piece, not started) — parse a CAMS/KFin
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
