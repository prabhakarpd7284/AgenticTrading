# AlphaDesk — Backlog

Durable record of work that's been **explicitly deferred**, not just untouched. Items are filed here when a session decides "yes, do this later, but not today." Each entry includes the trigger (why it matters), scope, and what would unblock it.

Open items are checkboxes; tick them when done and leave the entry as a history note. Don't delete completed items — the *why* is the most valuable part for the next operator.

---

## Open

### `[ ]` Build C — full cockpit historical orchestrator

**Filed:** Cycle 10 (2026-05-18) · **Estimate:** 3-4 days

A Celery batch job that pre-computes every cockpit panel for every trading day in a window, persists snapshots, and a UI timeline scrubber plays them back like a film reel.

**Why it's deferred:** Option A (cockpit time-travel via date picker) was shipped first because it gives the same functional reach (any panel, any historical day) with ~10× less infra. C only beats A if scrubbing through time frame-by-frame is genuinely useful — we won't know that until A has been used for a week.

**Scope:**
- Define snapshot schema for the 42 heterogeneous panels (each has its own shape)
- Celery batch task: fan 42 × N days = potentially thousands of broker calls
- Snapshot store + per-day invalidation
- Replay UI: timeline + play/pause + frame rate

**Unblocker:** Concrete operator request like "I want to watch sector rotation flip frame-by-frame across April 15-22 to confirm the regime shift." If date-jumping (Option A) keeps satisfying that need, C stays cold.

---

### `[ ]` Persistent candle store for closed sessions

**Filed:** Cycle 10 (2026-05-18) · **Estimate:** ~1 day · **Companion to:** cockpit time-travel

Historical session bars **never change** once market closes. Persist last-trading-day 1m/5m OHLCV in Redis (or DB) with multi-day TTL, pre-warm via a Celery task at 15:35 IST after market close.

**Why it matters:** The 400ms broker rate-limiter (`trading/services/data_service.py:121`, `self._min_interval = 0.4`) puts a hard floor on universe-scoped panels: 30 symbols × N broker calls × 0.4s = 12-36s cold-cache, regardless of threading. Parallelisation already extracted what it could (47-52% on the panels that aren't rate-bound) — the *next* perf cliff requires not fetching the same bars twice.

**Why it pairs with time-travel:** Once operators start jumping to historical dates, they hit cold cache repeatedly. Persistent store amortises broker pain into one nightly warm-up — every historical-date panel render then returns <50ms.

**Scope:**
- Storage layer: pick Redis with no TTL vs Postgres table; favour Redis for query simplicity
- Key shape: `bars:{interval}:{symbol}:{YYYY-MM-DD}`
- New Celery task: `warm_historical_bars` at 15:35 IST — iterates watchlist, fetches today's full session, stores
- Plumbing: `_fetch_1m_today` / `_fetch_5m_today` check persistent store before broker
- Invalidation: closed sessions never invalidate; only today's in-progress bars use the 30-60s TTL cache
- Backfill script: optionally pull N past days on demand to populate history

**Unblocker:** None — purely a perf project, ready to start whenever there's an afternoon for it.

---

## Conventions

- **One H3 per item.** Keep the title actionable.
- Always include **filed date** + **estimate** so triage knows the staleness.
- Every entry must answer **why deferred** — that protects the *next* operator from re-debating decisions already made.
- Tick the checkbox when shipped, but **leave the entry**. Append a `**Shipped:** YYYY-MM-DD · commit <sha>` line so the history is searchable.
- For tiny "follow up nice-to-haves," prefer a TODO comment in the code with the file/line — only file here for items that are >½ day or cross-cutting.

## What does NOT belong here

- Bugs (file a GitHub issue or fix in-session)
- Mid-session in-progress work (use the conversation task tracker)
- Vague aspirations ("improve UX") — only filed work with a definable doneness
- Anything already covered in `docs/MIND_PALACE.md` Cycle entries (link instead)
