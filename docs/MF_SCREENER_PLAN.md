# Plan: Mutual Fund Screener + Investment Planner (MF-S)

Owner: Abhinav. Audience: Claude Code running natively in this repo.
Branch: continue on `abhinav/redesign-v2`. Read `HANDOFF_MUTUAL_FUNDS.md` first.

## 0. Goal

Screen **every** Indian mutual fund scheme, measure its real performance and risk
from NAV history, rank funds within their category, and turn the owner's stated
profile (capital, horizon, risk) into a concrete, explainable allocation plan
that he can execute and then track.

PERSONAL USE ONLY (owner's own money). Same rule as `advisor.py`: the moment this
gives personalised output to a second person, AMFI ARN / SEBI RIA applies. Do not
build multi-user recommendation flows.

Hard design rules (carry over from the existing MF code):
- Deterministic, rule-based numbers. **No LLM picks funds or computes anything.**
  An LLM may only *explain* an already-computed result (phase S6).
- Collection code never gives advice; advice code never writes to collection tables.
  New modules: `services/metrics.py` (pure maths), `services/screener.py`
  (filter + rank), `services/planner.py` (profile -> allocation). Keep them separate.
- Shared catalog data (schemes, NAV, metrics) is NOT tenant-scoped. Profile, plans,
  holdings are `TenantModel`.
- Every plan / ranking run is logged via `event_writer.emit()` (non-blocking).
- Management commands: structlog, `--dry-run` where sensible, skip-and-log bad rows.
- Tests for every service; tenant-isolation tests for every new API endpoint.

## 1. What exists vs. what is missing (verified in code)

Exists: `MFScheme` (14,376 rows, category = AMFI header line, `is_direct`, `option`),
`MFNav` (**currently only the latest day per scheme** - 14,376 rows), `amfi_history.py`
(per-AMC download, ~15 MB per AMC-year, cannot filter by scheme), `advisor.py`
(holdings-level checks), read-only API + `/mutual-funds` page.

Missing - and these are the real work:
1. **NAV history for the universe.** Without it nothing can be screened. Biggest task.
2. Expense ratio (TER), AUM, benchmark, fund manager, launch date. AMFI's NAV file has none.
3. Benchmark / index return series (for alpha, capture ratios).
4. Metrics, scoring, planner, profile model, UI, monitoring.

## 2. Screening universe (define once, in `screener_config.py`)

Default universe: schemes that are **Direct plan + Growth option + open-ended +
actively reporting NAV in last ~7 days**. Exclude IDCW/dividend options, FMPs /
closed-ended, segregated portfolios, matured schemes (stale-NAV list already exists
in `staleness.py`). That should be roughly 1,500-2,500 schemes. Print the actual count
at the end of S1 and put it in the PR description.
Regular plans are only used to quantify the direct-vs-regular gap (already in advisor).

Categories are the AMFI/SEBI category (`category_label()` in advisor.py - reuse it).
Rank **within category only**; never compare a liquid fund to a small-cap fund.

## 3. Phases (each independently shippable, each ends with tests + a short doc note)

### S0 - Housekeeping (small, do first)
- Commit the uncommitted MF-UI work from 2026-10-17 as its own commit (check
  `git status`; do not mix with new work).
- Decide the NAV history source (see S1) with a 30-minute spike, record the decision in
  `docs/MF_SCREENER_DATA_SOURCES.md`.

### S1 - NAV history backfill for the whole universe
Option A (primary): AMFI NAV-history report per AMC (already coded). For the universe
you want ALL direct-growth schemes of an AMC, so pass the full scheme-code set for that
AMC, not "held schemes". Resolve AMC ids once and persist them in a small table/JSON
(the current probe loop of up to 150 HTTP calls should not run per backfill).
Fetch in 1-year chunks, back to 5 years (configurable, 10 later). Stream-parse, do not
hold a 15 MB body per chunk in several copies. Bulk insert with
`bulk_create(ignore_conflicts=True)` in batches.
Option B (spike and compare): a public per-scheme JSON NAV API such as mfapi.in.
Much smaller downloads and one call per scheme. Check availability from this machine,
its reliability and terms before depending on it. If it works, use it for backfill and
keep AMFI as the daily source of truth; cross-check a sample of 20 schemes between the
two sources and fail loudly on mismatch.
Requirements:
- Resumable and idempotent: track per (amc, year) completion in a `MFBackfillJob` table;
  re-running skips done chunks.
- Throttle + retries + User-Agent; never crash the whole run on one AMC.
- **Survivorship bias:** matured/merged schemes vanish from the current NAV file, so a
  ranking built only from today's survivors flatters the past. Keep any dead schemes the
  history files return (flag `is_active=False`) and include them in the point-in-time
  backtest in S5.
- Data quality pass: gaps > 10 trading days, NAV jumps > 25% in a day (splits/bonus/
  data errors), zero or negative NAV. Write findings to a `MFDataIssue` table; metrics
  must skip or flag affected windows, never silently interpolate.
- Daily `pull_amfi_nav` keeps appending; confirm the 23:45 and 07:30 IST beat entries
  actually fired (`logs/beat.log`) - still unobserved per the handoff.
Acceptance: row counts per scheme, % of universe with >= 3y / 5y history, runtime and
download size, all printed by a `mf_data_coverage` command.

### S2 - Reference data (cost, size, benchmark, manager)
New shared model `MFSchemeInfo` (one row per scheme or per scheme "family" since Direct
and Regular share a portfolio): `expense_ratio_pct`, `expense_ratio_as_of`, `aum_cr`,
`benchmark_name`, `launch_date`, `fund_manager`, `source`, `as_of`.
Sources to evaluate (verify availability, formats and terms before building; do not
assume): AMFI's published TER data and scheme-wise AUM disclosures; AMC factsheets;
SEBI-mandated monthly portfolio disclosures. Each field is nullable and carries its
source + as_of date. If a field cannot be sourced reliably, leave it null and have the
screener degrade gracefully (show "n/a", do not impute silently).
Benchmarks: add `MFBenchmarkNav` (index name, date, TRI value) for the handful of
indices that matter (Nifty 50 / Next 50 / Midcap 150 / Smallcap 250 / 500, plus
a debt index if available). Source from NSE Indices historical data; prefer TRI over
price index. If TRI is not obtainable, say so and use price index with an explicit warning.
Replace the advisor's assumed 0.75% regular-plan drag with real TER differences once
available (HANDOFF open item 6).

### S3 - Metrics engine (`services/metrics.py`, pure functions + a snapshot table)
Pure functions over a NAV series (pandas/numpy are fine; add deps via `uv`), unit-tested
against hand-computed fixtures. Compute and store in `MFMetricSnapshot`
(scheme, as_of, window columns) so the screener never recomputes on request:
- Trailing CAGR: 1y, 3y, 5y (and since-inception where available).
- Rolling returns: 3y rolling CAGR sampled monthly -> median, min, % of windows above the
  category median, % of windows positive. This is far more honest than point-to-point.
- Risk: annualised volatility, downside deviation, max drawdown, drawdown duration/recovery
  days, worst 1y rolling return.
- Risk-adjusted: Sharpe, Sortino, Calmar. Risk-free rate is a config value (not hard-coded
  in logic); document its source and date.
- Relative: excess return vs benchmark, beta, up/down capture, tracking error and
  information ratio (once S2 benchmarks exist); until then compare to category median.
- Quality flags: history length, staleness, data-issue overlap, AUM/TER availability.
Use NAV-based returns only, and state that dividends in IDCW options are excluded by the
universe rule. Windows with data-quality issues are marked, not smoothed.
Command: `compute_mf_metrics [--as-of DATE] [--category ...] [--dry-run]`, scheduled
weekly via celery beat.

### S4 - Screener (`services/screener.py`)
Input: a `ScreenCriteria` dataclass (category list, min history years, min AUM, max TER,
min Sharpe, etc.) - hard filters first, then rank.
Scoring: transparent composite **within category**, percentile-rank based, weights in
config and shown in the output. Suggested default (tune only via S5 evidence):
rolling-return consistency 30, risk-adjusted return (Sortino/Sharpe) 25, drawdown
behaviour 20, cost (TER) 15, size/age sanity 10. Every ranked fund must return its
per-component scores so the UI/CLI can show *why* it ranked where it did.
Index-vs-active check: for each equity category, report whether the best active funds
beat the cheapest index fund of that category after costs over rolling windows. If they do
not, the screener says so and the planner may prefer the index fund. This is the single
most useful honesty check - do not skip it.
Command: `screen_mf --category "Equity Scheme - Flexi Cap Fund" --top 10` prints a table;
`--json` for the API.

### S5 - Validate the screen before trusting it (do not skip)
Point-in-time backtest: at each past date t (say every quarter), rank using only data up
to t, then measure the next 12 / 36 months' return of top-quartile vs median vs bottom
quartile vs the category index fund. Include dead/merged schemes. Report honestly whether
the score has predictive value; fund-performance persistence is historically weak, so the
result may well be "weak/none". If so: reduce weight on trailing returns, raise weight on
cost and consistency, and make the planner lean to low-cost index funds. Write the result
in `docs/MF_SCREENER_VALIDATION.md` with the tables. This decides the default weights.

### S6 - Investor profile + planner (`services/planner.py`)
New tenant-scoped models: `MFInvestorProfile` (investable capital, monthly SIP capacity,
horizon in years per goal bucket, risk tolerance 1-5, max tolerable drawdown %,
emergency-fund-already-covered flag, tax regime) and `MFPlan` / `MFPlanLine`
(snapshot of a generated plan: scheme, target %, amount, rationale JSON, as_of, plan
version). Plans are immutable snapshots; regenerating creates a new version.
Planner logic (deterministic):
1. Map profile -> target asset-class mix (equity / hybrid / debt / liquid) via an explicit,
   documented rule table in config (horizon < 3y leans debt/liquid; the owner can
   override any weight). Show the rule that fired.
2. Split equity into categories (large/flexi/mid/small/index...), with caps (e.g. small cap
   <= X% of equity, configurable).
3. Pick per category: top-ranked funds from S4, max 1 per AMC per category, max N funds
   total (default 6-8; fewer funds is a feature), prefer index fund where S4's check says
   active does not earn its fee.
4. Lump sum: suggest staged entry (STP/tranches over N months) for equity portions, as a
   configurable policy, with the reasoning shown. SIP: monthly amounts per fund.
5. Output per fund: amount, % of plan, why chosen (component scores), what could go wrong
   (max drawdown seen, worst 1y), cost, and the exit/review trigger.
6. Taxes: keep rates and holding-period thresholds in a config table with an `as_of`
   field; look up the current rules when implementing and write the source URL in the
   config - do not hard-code from memory.
No guarantees or return forecasts in any output. Show historical ranges only, labelled as
history.
Commands: `mf_profile set ...`, `mf_plan --tenant X [--dry-run]`.

### S7 - Execution + tracking
Execution stays manual (owner places orders on his MF platform). Do not build order
placement. Instead: after he buys, he records holdings via CAS import (already built, not
yet verified on a real PDF) or manual entry. Add `MFPlan` <-> `MFHolding` matching so the
app shows plan vs actual drift, and a plan-vs-benchmark performance line (paper-track the
plan from its creation date even before he invests, so the screen's real-world record
builds up).
Verify `import_cas_statement` on a real CAMS/KFintech CAS as soon as the owner supplies
one; this is still the top unverified item.

### S8 - Monitoring (`run_mf_review`, monthly)
For each held fund: rank change vs previous snapshot, drawdown vs its history, TER change,
NAV staleness, category drift, plan drift. Emit findings through the existing advisor
`Finding` shape and event writer. Alert via the existing Telegram path if one exists.
Rule: a fund is flagged for review after N consecutive months in the bottom quartile -
**not** sold automatically, and one bad quarter is never a trigger.

### S9 - API + UI
Extend `apps/mutual_funds/api/`: `GET /screener/?category=&min_years=&...`,
`GET /schemes/{code}/metrics/`, `GET /schemes/{code}/nav-history/`, `GET /compare/?codes=`,
`GET/PUT /profile/`, `POST /plans/` (owner-only), `GET /plans/{id}/`.
Frontend `features/mutual-funds/`: Screener tab (category picker, filter chips, sortable
table with score breakdown tooltip), Fund detail (NAV chart vs benchmark, rolling-return
band, drawdown chart, cost/AUM), Compare (2-4 funds), Plan wizard (profile -> plan with
per-fund rationale), Plan-vs-actual. Show data freshness and "data missing" badges
prominently. Follow the existing page's patterns; add backend tenant-isolation tests and
frontend tests.

### S10 - Conversational layer (later, only after S5 shows the numbers are sound)
The planned chatbot persona may call these services as tools and **explain** results in
plain language, quoting the computed numbers. It must refuse to invent returns, must say
when data is missing, and must state it is the owner's personal tool.

## 4. Order of work and rough size
S0 -> S1 (largest; data) -> S3 -> S4 -> S5 (decision gate) -> S2 can run in parallel with
S3/S4 -> S6 -> S9 (screener part first, then plan wizard) -> S7 -> S8 -> S10.
Ship a usable CLI screener (S1+S3+S4) before any UI.

## 5. Risks to keep visible
- Past returns do not predict future returns; S5 exists to measure how much they do.
- Survivorship bias, category-reclassification, and NAV jumps from corporate actions.
- Missing TER/AUM/benchmark data: degrade visibly, never impute silently.
- AMFI download size/availability; scraping terms for any non-AMFI source.
- Tax/regulatory rules change: config with source + as_of, re-verified at implementation.
- Scope creep into advice for others (registration question) - keep personal-use banners.

## 6. Inputs needed from the owner (use these defaults, ask once if blocking)
Defaults to start with: lump sum + monthly SIP both possible; horizon >= 5y for equity
money; risk 3/5; max tolerable drawdown 25%; direct plans only; max 8 funds.
Needs from owner: actual capital amount and SIP amount, horizon per goal, whether an
emergency fund already exists, tax regime, a real CAS PDF, and a Kite Connect app (only
if/when linking Zerodha).

## 7. Definition of done for the first milestone (CLI screener)
- `mf_data_coverage` shows >= 3y NAV history for the large majority of the universe,
  with the exact numbers reported.
- `compute_mf_metrics` and `screen_mf` run end to end on real data; per-component scores
  displayed; unit tests for every metric against hand-computed values.
- `docs/MF_SCREENER_VALIDATION.md` exists with the S5 backtest results, or states
  clearly what blocked it.
- Full backend suite green; nothing in collection code imports from screener/planner.

## 8. Source verification (checked 2026-10-21 from a cloud session)

Method note: the cloud workspace's shell could not reach any of these hosts (network
allowlist), so checks used a web-fetch tool, which returns summaries, not raw bytes.
Treat every row as "looks viable" and **re-verify with a real HTTP call from this
machine before building** (curl + one-row parse). Do not take a row below as proof.

| Source | Finding | Status |
|---|---|---|
| AMFI daily NAV file `portal.amfiindia.com/spages/NAVAll.txt` | Content seen: semicolon rows, category/AMC header lines interleaved, latest date 20-Oct-2026. Matches the existing parser. Already ran for real on this machine (14,376 schemes). | Verified (by earlier runs) |
| AMFI NAV-history report (per AMC) | Already used for scheme 148921 (494 rows / 2y). Not re-checked today. | Verified earlier |
| mfapi.in `GET https://api.mfapi.in/mf/{scheme_code}` | Returned full JSON for 148921: `meta` (fund_house, scheme_category, scheme_code, scheme_name, isin_growth) + `data` [{date DD-MM-YYYY, nav string}], newest first, 19-10-2026 down to 10-05-2021 (~1,340 rows, i.e. the fund's whole life). Optional `startDate`/`endDate`; also `/mf`, `/mf/search?q=`, `/mf/{code}/latest`. Site says free, no key, "rate limiting" unspecified, refreshes ~6x daily. | Works. Unknowns: rate limit, ToS text, whether matured schemes are listed, upstream source |
| mfapi.in depth for older funds | HDFC Mid Cap Direct (118989) showed data back to at least 20-09-2019 but the tool's output was truncated, so the true start date is **unconfirmed**. | Unconfirmed - check |
| AMFI TER page `amfiindia.com/ter-of-mf-schemes` | Exists. Filters: financial year, month, fund type, category, mutual fund. Described as **monthly** (a 2018 article said daily). Table columns and any export/API were **not visible** to the tool. | Needs a real look (browser devtools: find the XHR/JSON or Excel it loads) |
| Community TER dataset on Hugging Face (`Na-Rajan/IndianMutualFundsTER`) | Exists, Apache-2.0, 1K-10K rows, columns mirror AMFI's TER page, date range and update cadence unstated. Not authoritative. | Use only as a cross-check |
| NSE TRI history `niftyindices.com/Backpage/getTotalReturnIndexString` | Endpoint named in the open-source `BharatFinTrack` library (field `TotalReturnsIndex`, JSON POST with `content-type: application/json`, date format DD-Mon-YYYY). Method/payload/limits not seen. | Plausible - read that library's `Helper` source and test. Fallback: manual CSV from niftyindices.com |
| AMFI "scheme performance" page | Fetch failed. | Unknown |

Consequences for the plan:
- S1: make mfapi.in the **primary backfill candidate** (one small call per scheme versus
  ~15 MB per AMC-year) but run the 20-scheme AMFI cross-check from S1 before trusting it,
  and test whether it holds history for matured/merged schemes (survivorship). Persist raw
  responses for reproducibility.
- S2: TER is monthly, not daily; store `as_of` month. Budget a spike to find the real
  data feed behind AMFI's TER page. If none, fall back to AMC factsheets or leave null.
- S2 benchmarks: TRI via niftyindices is the likely route; verify, else price index with
  an explicit warning.

### Tax rules (single secondary source - verify against the Income-tax Act / CBDT before coding)
Source: DSP Investment Managers page "STCG and LTCG rules in India", page-dated 16 Jun 2026.
- Equity-oriented funds: STCG (< 12 months) 20% under s.111A; LTCG (>= 12 months) 12.5%
  under s.112A, no indexation; LTCG exemption Rs 1.25 lakh per FY; effective 23 Jul 2024;
  the page says Budget 2026 made no change.
- Debt / "specified mutual funds" (< 35% equity), units bought on or after 1 Apr 2023: taxed
  at slab rate, no holding-period split, no indexation.
- Hybrid funds: no separate rule given. Equity exposure vs the 35% line decides treatment.
  Arbitrage and hybrid-equity cases need the statute text.
- Not covered by the source: pre-April-2023 debt units, surcharge/cess detail, STT.
Put these in `mf_tax_config` with `source_url` and `as_of`; the planner must print
"verify with a tax professional" next to any after-tax figure.
