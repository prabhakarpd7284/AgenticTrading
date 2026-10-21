# MF Screener — data sources (S0 spike + S1/S2 results)

Checked 2026-10-21 from the dev machine with real HTTP calls (curl + Python),
not from summaries. Companion to `docs/MF_SCREENER_PLAN.md` §8.

## Decision

| Need | Source | Status |
|---|---|---|
| Daily NAV (source of truth) | AMFI `portal.amfiindia.com/spages/NAVAll.txt` (`pull_amfi_nav`, unchanged) | In production since MF-1 |
| NAV history backfill | **mfapi.in** `GET https://api.mfapi.in/mf/{code}` (`backfill_mf_nav_history`) | **Adopted.** Cross-checked against AMFI: exact match |
| Cross-check / fallback history | AMFI NAV-history report per AMC (`amfi_history.py`) | Kept; AMC ids now persisted |
| Benchmarks | niftyindices.com TRI endpoint (`pull_mf_benchmarks`) | **Adopted**, 7 indices, TRI (not price) |
| TER (expense ratio) | AMFI TER page | **Not sourced.** Screener shows "n/a" |
| AUM | AMFI AUM disclosures | **Not sourced.** Screener shows "n/a" |
| Risk-free rate | 91-day T-bill cut-off, 5.52% (`screener_config.RISK_FREE_RATE`) | Secondary source, see below |

## mfapi.in (primary backfill source)

What was checked:

- **Availability / format.** `GET /mf/118989` returns 200, 134 KB of JSON: `meta`
  (fund_house, scheme_type, scheme_category, scheme_code, scheme_name, ISINs) and
  `data` `[{date: "DD-MM-YYYY", nav: "219.78400"}]`, newest first. Unknown codes
  return 200 with `scheme_code: 0` and empty `data` (handled as "empty").
  `startDate`/`endDate` (ISO) filters work. `GET /mf` lists **37,982** scheme
  codes (5.8 MB), versus 14,397 in today's AMFI file.
- **Depth.** HDFC Mid Cap Direct Growth (118989) goes back to **2013-01-01**
  (3,396 rows), i.e. the start of Direct plans. The plan's "unconfirmed" note is
  resolved. Axis Midcap Direct (120505) also goes back to 2013-01-02.
- **Dead / matured / merged schemes are present.** 100027 (Grindlays Super Saver
  Income, Standard Chartered MF) returns 531 NAVs from 2006-04-03 to 2008-05-29.
  23,585 listed codes are not in AMFI's current file. So survivorship-free
  history is available.
- **Rate limiting.** The site says "no authentication, no API keys, no rate
  limiting". 30 back-to-back requests (about 2.7 req/s) returned 30 × 200. No
  rate-limit headers are sent. We still throttle to 0.35 s between calls, retry
  5xx/timeouts with exponential backoff, back off on 429 (never seen), and send
  `User-Agent: AlphaDesk-MF/1.0 (personal research; single-user, throttled)`.
  The full backfill (2,201 calls) saw 0 failures.
- **Terms.** The footer links "Terms of Service" / "Privacy Policy", but
  `/terms`, `/terms-of-service` and `/tos` return 404, so no ToS text could be
  read. It is a free community mirror of AMFI data (upstream presumed AMFI;
  refreshed 6x daily per the homepage). Because there is no ToS, AMFI stays the
  daily source of truth and mfapi is only the history backfill. Raw responses are
  archived gzipped in `$MF_RAW_DIR` (default `~/.local/share/alphadesk/mf_raw`,
  outside the repo) for reproducibility.

### 20-scheme cross-check against AMFI (`manage.py mf_crosscheck_sources`)

The command picks 20 universe schemes over 5 AMCs (seeded), downloads one old
month (March 2019) and the last 30 days from **both** sources, and compares every
common date. It exits non-zero if any NAV differs by more than 1e-4 (relative), or
if more than 5% of weekday dates are missing on one side.

| Run | Schemes / AMCs | Common dates compared | Max relative NAV difference | Result |
|---|---|---|---|---|
| 1 (before the backfill; young schemes, so most old-window rows were empty) | 20 / 5 (LIC, Edelweiss, Navi, quant, Bajaj Finserv) | ~430 | **0** | One flag: Bajaj Finserv Overnight had 4 **weekend** NAVs in AMFI that mfapi omits. Weekend-only gaps are now reported separately and do not fail the check |
| 2 (after the backfill; schemes with history in March 2019) | 20 / 5 (HDFC, Baroda BNP, ICICI Pru, Nippon, Axis) | 778 | **0** | **Passed** (AMFI 13.2 MB, mfapi 0.04 MB) |

During the backfill, every date where an AMFI-daily row already existed was also
compared (`source_mismatch` issues): **0 mismatches**. Where the sources do
overlap, AMFI's row is always kept (`ignore_conflicts`).

## AMFI NAV-history report (fallback / cross-check)

Still works. AMC ids for 45 AMCs (ids 1..80 probed once on 2026-10-21) are
persisted in `backend/apps/mutual_funds/data/amfi_amc_ids.json`.
`resolve_amc_id()` reads that file first, so a normal run makes no probe calls
(the plan asked for this). A one-month report is about 0.5-3 MB per AMC.

## Benchmarks: niftyindices.com TRI (S2, verified)

`POST https://www.niftyindices.com/BackPage/getTotalReturnIndexString`,
`Content-Type: application/json`, body
`{"cinfo": "{'name':'NIFTY 50','startDate':'01-Jan-2024','endDate':'31-Jan-2024','indexName':'NIFTY 50'}"}`.

- The `.aspx` path named in the plan (`/Backpage.aspx/...`) returns 302 plus an
  error. The working path (`/BackPage/...`, no `.aspx`) was found in the site's
  own `IISLComponet.js`. No cookies or tokens are needed, and nothing here
  bypasses any protection.
- The site's script caps one request at 365 days, so we fetch 1-year chunks with a
  1.5 s pause.
- Loaded **TRI** (not price index) for NIFTY 50, NEXT 50, 100, 500, LARGEMIDCAP
  250, MIDCAP 150 and SMALLCAP 250: **3,464 rows each, 2012-10-22 .. 2026-10-19**.
  `NIFTY 500 MULTICAP 50:25:25` returned empty for 2014 and is not used.
- No debt index was pulled. Debt and hybrid categories are compared with their
  category median only.

## Not sourced (S2), and what blocked it

- **TER.** Not attempted beyond the plan's note: AMFI's TER page renders through
  a script whose data feed has not been identified (it needs a browser-devtools
  look). Without a verified feed the cost component is **dropped and the other
  weights are renormalised**. Every screener output says so, the UI shows a "TER
  / AUM data missing" badge, and nothing is imputed. The advisor's 0.75% assumed
  regular-plan drag is unchanged (HANDOFF item 6 is still open).
- **AUM.** Same. The size/age component uses fund age only.
- **Fund manager / launch date.** Not sourced. "History years" is measured from
  the first NAV the source has. For funds launched after 2013 that is
  effectively the launch date.

## Risk-free rate

5.52%, the 91-day T-bill cut-off YTM from the 1-Oct-2026 RBI auction, as
reported by Aditya Birla Sun Life AMC's "Daily Fixed Income Tracker" for that
date. It is a secondary source and has **not** been checked against RBI's own
press release (`verified_against_rbi: False` in config). One constant is used for
every window, including the backtest. That shifts Sharpe levels but barely
changes within-category ranks.

## Backfill run (S1) — actual numbers

`nice -n 10 manage.py backfill_mf_nav_history --years 14`, run detached and
logged to `logs/mf_backfill.log`:

- **Targets:** 2,202. That is 1,673 catalog schemes passing the universe rules
  (live or stale), plus 529 dead "Direct … Growth" candidates from mfapi's list
  (matched by name).
- **Result:** 2,107 fetched, 0 failed, 1 empty, 92 rejected as out-of-universe
  after reading their metadata (close-ended / interval). **436 dead schemes**
  were added to `MFScheme` with `is_active=False, catalog_source="mfapi"`.
- **Rows:** 3,323,675 NAV rows inserted. `MFNav` now holds 3,354,191 rows.
- **Download:** 140.4 MB in 2,201 HTTP calls (134.9 MB of scheme histories plus
  the 5.8 MB scheme list).
- **Runtime:** 1,121 s (about 19 min) wall clock. RSS stayed around 240 MB.
- **Depth.** `--years 14` was used, not the 5-year default, so the S5 backtest
  has enough history. mfapi returns a fund's whole life in one small response
  either way. The command's default stays 5 years (`MF_HISTORY_YEARS`).

`mf_data_coverage` after the backfill:

| | Schemes | NAV rows | ≥1y | ≥3y | ≥5y | ≥10y |
|---|---|---|---|---|---|---|
| Live universe (Direct + Growth + open-ended + NAV ≤7 days old) | **1,650** | 2,810,367 | 90.3% | **68.7%** | **50.7%** | 33.5% |
| Dead schemes kept for survivorship | 459 | 523,395 | 93.7% | 64.3% | 39.2% | 2.8% |

The ≥3y share is not 90%+ because 31% of the live universe launched less than 3
years ago. History starts at the fund's first NAV, so this is not missing data.
That is especially true of index funds (36.6% ≥3y) and multi-asset funds (40.5%).
Core equity categories are 67-97% ≥3y (`mf_data_coverage --by-category`).

### Data quality (`MFDataIssue`)

| Kind | Issues | Schemes | What it is |
|---|---|---|---|
| nonpositive | 1,375 | 51 | NAV = 0 rows in the source (e.g. PGIM India Money Market 737 rows, several dead Sundaram debt funds). Dropped from every series |
| jump (>25% in a day) | 55 | 45 | Mostly **×100 face-value changes** in liquid / money-market funds (HDFC Liquid / Money Market / Overnight 2015-08-28, Kotak Liquid 2013-01-18), plus a spike-and-revert (ABSL International Equity 2019-04-09/10) and real events (SBI Credit Risk 2020-03-19) |
| gap (>10 weekdays) | 26 | 13 | Missing stretches |
| source_mismatch | 0 | 0 | |

Metrics never compute across a jump: the series is cut at the most recent jump
known at `as_of` and flagged `history_cut_at_nav_jump`. The backtest also skips
any fund-date whose forward window contains a jump. Gaps and other issues flag
the snapshot (`data_issue_in_3y_window` / `data_issue_in_history`), and the
screener can exclude flagged funds.

## AMFI category renaming (found during S1)

AMFI's daily file currently mixes two naming schemes on the same day, and
different AMCs use different ones. Examples: "Equity Scheme - Flexi Cap Fund"
and "Equity Schemes - Flexi Cap Fund"; "Debt Scheme - Short Duration Fund" and
"Income/Debt Oriented Schemes - Short Term Fund". `services/universe.py
canonical_category()` folds these so funds are ranked against all their peers.
Two debt mappings are judgement calls and are marked APPROX in the code: "Dynamic
Term" → Dynamic Bond, and "Ultra Short to Short Term" → Low Duration.

Dead schemes keep the category mfapi reports. 231 have the pre-2018 label
"Income" and 11 have "Growth", so they cannot be placed in a SEBI category. They
are kept but sit in their own buckets, outside every ranked category. Each
category's current label is used for all of its history. Category
reclassification is a known bias.

## Scheduling

- Daily: unchanged `mf-nav-pull-night` (23:45) / `mf-nav-pull-morning` (07:30)
  append AMFI NAVs. `logs/beat.log` could not be checked on 2026-10-21: the dev
  stack (beat included) was not running, and the Docker containers had exited.
- **New:** `mf-metrics-weekly`, Saturday 08:15 IST. It tops up the benchmark TRI
  incrementally, then runs `compute_mf_metrics`. Beat must be restarted to pick it
  up.
