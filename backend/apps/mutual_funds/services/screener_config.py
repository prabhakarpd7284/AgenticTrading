"""Screener configuration — every tunable number in one place (S3/S4/S5).

The universe rule and canonical categories live in ``universe.py`` (shared
with collection code, which must not import anything screener-related).

PERSONAL USE ONLY: rankings are arithmetic on public NAV history for the
owner's own decisions, not advice to anyone else.
"""
from __future__ import annotations

# ── risk-free rate ───────────────────────────────────────────────────────
# Used by Sharpe/Sortino. A single constant is applied to every window
# (also in the S5 backtest), which shifts Sharpe levels but barely changes
# within-category ranks. Re-verify against RBI's T-bill auction results.
RISK_FREE_RATE = {
    "value": 0.0552,
    "instrument": "91-day Government of India Treasury Bill, auction cut-off YTM",
    "source": "Aditya Birla Sun Life AMC 'Daily Fixed Income Tracker' 01-Oct-2026 "
              "(https://mutualfund.adityabirlacapital.com/-/media/knowledgecenter/pdf/all-about-market/"
              "daily-fixed-income-tracker-01-oct-2026.pdf), reporting the RBI auction: 5.5199%",
    "as_of": "2026-10-01",
    "verified_against_rbi": False,
}

# ── metric windows ───────────────────────────────────────────────────────
RISK_WINDOW_YEARS = 3          # vol / downside / Sharpe / Sortino / beta / capture
DRAWDOWN_WINDOW_YEARS = 5      # max drawdown + worst 1y use up to this much history
ROLLING_YEARS = 3              # rolling-return consistency window
MIN_CATEGORY_FUNDS_FOR_MEDIAN = 3
STALE_NAV_DAYS = 7

# ── scoring (S4). Default weights were set from the S5 validation; see
# docs/MF_SCREENER_VALIDATION.md. Each component is a 0-100 within-category
# percentile; a component with no data for the whole category (TER / AUM
# today) is dropped and the remaining weights are renormalised — the output
# says so, nothing is imputed.
PLAN_DEFAULT_WEIGHTS = {        # the plan's suggested starting point (kept for reference)
    "consistency": 30, "risk_adjusted": 25, "drawdown": 20, "cost": 15, "size_age": 10,
}
# S5 (docs/MF_SCREENER_VALIDATION.md, 2026-10-21): no component predicted the
# next 12/36 months within category (|IC| < 0.07, nothing significant except
# drawdown-only being mildly NEGATIVE at 12m). Per the plan's rule for a weak
# result: drawdown and age weights cut, cost becomes the largest weight (the
# one factor with strong prior evidence; inactive until TER is sourced), and
# the screen is presented as a description of past behaviour, not a forecast.
DEFAULT_WEIGHTS = {
    "consistency": 30,
    "risk_adjusted": 25,
    "drawdown": 10,
    "cost": 30,
    "size_age": 5,
}
# "% of rolling windows above the category median" from a young fund with a
# handful of windows is thin evidence: shrink it toward 50 % with a prior
# worth this many windows: adj = (pct * n + 50 * K) / (n + K).
CONSISTENCY_PRIOR_WINDOWS = 24
COMPONENT_METRICS = {
    # component: [(metric field, higher_is_better), ...] — averaged percentiles
    "consistency": [("consistency_pct_above_shrunk", True), ("rolling_3y_min", True)],
    "risk_adjusted": [("sortino", True), ("sharpe", True)],
    "drawdown": [("max_drawdown", True), ("worst_1y", True)],   # both negative: less negative is better
    "cost": [("expense_ratio_pct", False)],                     # not sourced yet -> n/a
    "size_age": [("history_years", True)],                      # AUM not sourced yet -> age only
}
COMPONENT_NOTES = {
    "cost": "TER not sourced yet (AMFI's TER page has no verified feed) - component dropped, weights renormalised.",
    "size_age": "AUM not sourced yet - fund age (years of NAV history) only.",
}

# ── index-vs-active honesty check ────────────────────────────────────────
# category -> (benchmark TRI index name, regex for the index funds an investor
# would actually buy instead). Index-fund NAVs are after their own costs, as
# are active funds' NAVs, so that comparison is like-for-like after costs.
# TRI is gross of any fund cost: when used as a fallback it flatters the
# index side by roughly an index fund's TER (~0.1-0.3 %/yr) — conservative
# for active funds, and labelled in the output.
CATEGORY_BENCHMARKS = {
    "Equity Scheme - Large Cap Fund": ("NIFTY 100", r"\bnifty\s*50\b(?!.*next)"),
    "Equity Scheme - Large & Mid Cap Fund": ("NIFTY LARGEMIDCAP 250", r"large\s*mid\s*cap\s*250"),
    "Equity Scheme - Mid Cap Fund": ("NIFTY MIDCAP 150", r"midcap\s*150"),
    "Equity Scheme - Small Cap Fund": ("NIFTY SMALLCAP 250", r"small\s*cap\s*250"),
    "Equity Scheme - Flexi Cap Fund": ("NIFTY 500", r"\bnifty\s*500\b"),
    "Equity Scheme - Multi Cap Fund": ("NIFTY 500", r"\bnifty\s*500\b"),
    "Equity Scheme - Focused Fund": ("NIFTY 500", r"\bnifty\s*500\b"),
    "Equity Scheme - Value Fund": ("NIFTY 500", r"\bnifty\s*500\b"),
    "Equity Scheme - Contra Fund": ("NIFTY 500", r"\bnifty\s*500\b"),
    "Equity Scheme - Dividend Yield Fund": ("NIFTY 500", r"\bnifty\s*500\b"),
    "Equity Scheme - ELSS": ("NIFTY 500", r"\bnifty\s*500\b"),
}
# Factor / smart-beta / thematic index funds are not "the index fund".
INDEX_FUND_EXCLUDE = (
    r"momentum|quality|value|equal|alpha|low\s*vol|multi\s*cap|multicap|enhanced|dividend|esg|shariah|"
    r"smart|50:25:25|beta|elss|tax|fof|fund of fund|etf|arbitrage|sensex|bank|it\b|pharma|healthcare|"
    r"consumption|infra|psu|manufacturing|defence|auto|realty|metal|energy|financial"
)
MIN_INDEX_FUND_WINDOWS = 12    # rolling windows needed before an index-fund proxy is used
INDEX_FUND_MIN_COVERAGE = 0.6  # ... and it must cover this share of the TRI-comparable windows
ACTIVE_EARNS_FEE_PCT = 60.0    # median active fund beat the index in >= this % of windows -> "yes"
ACTIVE_NO_FEE_PCT = 40.0       # <= this % -> "no"; in between -> "mixed"
