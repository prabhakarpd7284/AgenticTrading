"""Screening universe + canonical categories — the single definition.

Pure helpers (no advice, no ranking), so both the collection side (NAV
backfill) and the screener can import this without the collection code ever
importing screener code.

Universe rule (MF_SCREENER_PLAN §2): Direct plan + Growth option +
open-ended + NAV seen in the last ``ACTIVE_NAV_DAYS`` days. Excluded: IDCW /
dividend / bonus options, closed-ended / interval / FMP, segregated
portfolios, "unclaimed redemption" plans, ETFs (no direct plan; exchange
traded) and IL&FS infra debt funds. Dead schemes (matured/merged) that a
history source still returns are kept with ``is_active=False`` for the
point-in-time backtest (survivorship).

Categories: AMFI is mid-way through renaming categories ("Equity Scheme -
Flexi Cap Fund" vs "Equity Schemes - Flexi Cap Fund", "Debt Scheme - Short
Duration Fund" vs "Income/Debt Oriented Schemes - Short Term Fund"), and
different AMCs use different spellings on the same day. ``canonical_category``
folds those into one label so funds are ranked against all their real peers.
Mappings marked APPROX below are judgement calls, not AMFI equivalences.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

ACTIVE_NAV_DAYS = 7

_IDCW_RE = re.compile(r"idcw|dividend|\bdcw\b|payout|reinvest|bonus|income distribution", re.I)
_EXCLUDE_NAME_RE = re.compile(
    r"segregat|unclaimed|\bfmp\b|fixed maturity|fixed term|interval|capital protection|"
    r"close[d -]?ended|\bseries\b|\betf\b|exchange traded",
    re.I,
)
_DEBT_INDEX_RE = re.compile(r"\b(sdl|g-?sec|gilt|bond|crisil|psu|debt|t-?bill|corporate|aaa|target maturity)\b", re.I)

# Exact (case-insensitive) label -> canonical label.
_CATEGORY_ALIASES: dict[str, str] = {
    "equity schemes - elss- tax saver fund": "Equity Scheme - ELSS",
    "elss": "Equity Scheme - ELSS",
    "equity scheme - sectoral/ thematic": "Equity Scheme - Sectoral/Thematic",
    "equity schemes - sectoral fund": "Equity Scheme - Sectoral/Thematic",
    "equity schemes - thematic fund": "Equity Scheme - Sectoral/Thematic",
    "hybrid schemes - balanced advantage fund/ dynamic asset allocation":
        "Hybrid Scheme - Dynamic Asset Allocation or Balanced Advantage",
    "hybrid schemes - multi asset allocation fund": "Hybrid Scheme - Multi Asset Allocation",
    "hybrid schemes - equity savings fund": "Hybrid Scheme - Equity Savings",
    # Debt: new AMFI names -> SEBI 2017 names.
    "income/debt oriented schemes - banking and psu debt fund": "Debt Scheme - Banking and PSU Fund",
    "income/debt oriented schemes - floating interest rates fund": "Debt Scheme - Floater Fund",
    "income/debt oriented schemes - long term fund": "Debt Scheme - Long Duration Fund",
    "income/debt oriented schemes - medium term fund": "Debt Scheme - Medium Duration Fund",
    "income/debt oriented schemes - medium to long term fund": "Debt Scheme - Medium to Long Duration Fund",
    "income/debt oriented schemes - short term fund": "Debt Scheme - Short Duration Fund",
    "income/debt oriented schemes - ultra short term fund": "Debt Scheme - Ultra Short Duration Fund",
    "income/debt oriented schemes - dynamic term fund": "Debt Scheme - Dynamic Bond",            # APPROX
    "income/debt oriented schemes - ultra short to short term fund": "Debt Scheme - Low Duration Fund",  # APPROX
    "income/debt oriented schemes - 10-year constant maturity gilt fund":
        "Debt Scheme - Gilt Fund with 10 year constant duration",
    "liquid": "Debt Scheme - Liquid Fund",
    "money market": "Debt Scheme - Money Market Fund",
    "gilt": "Debt Scheme - Gilt Fund",
    "other scheme - fof domestic": "Other Scheme - FoF Domestic",
    "fund of funds scheme (domestic) - fund of funds scheme (domestic)": "Other Scheme - FoF Domestic",
    "overseas fund of funds - fund of funds investing overseas": "Other Scheme - FoF Overseas",
    "solution oriented schemes ** - retirement fund": "Solution Oriented Scheme - Retirement Fund",
    "children’s fund - childrens' fund": "Solution Oriented Scheme - Children's Fund",
    "solution oriented scheme - children’s fund": "Solution Oriented Scheme - Children's Fund",
    "index funds - equity funds": "Index Funds - Equity",
    "index funds - debt funds": "Index Funds - Debt",
    "index funds - hybrid fund": "Index Funds - Hybrid",
}

# Categories the screener ranks with equity rules (index-vs-active check etc.).
EQUITY_CATEGORIES = (
    "Equity Scheme - Large Cap Fund",
    "Equity Scheme - Large & Mid Cap Fund",
    "Equity Scheme - Mid Cap Fund",
    "Equity Scheme - Small Cap Fund",
    "Equity Scheme - Flexi Cap Fund",
    "Equity Scheme - Multi Cap Fund",
    "Equity Scheme - Focused Fund",
    "Equity Scheme - Value Fund",
    "Equity Scheme - Contra Fund",
    "Equity Scheme - Dividend Yield Fund",
    "Equity Scheme - ELSS",
    "Equity Scheme - Sectoral/Thematic",
)


def raw_category_label(category: str) -> str:
    """'Open Ended Schemes(Equity Scheme - Large Cap Fund)' -> inner label.
    Same rule as advisor.category_label (kept here so collection code does
    not import advice code)."""
    m = re.search(r"\((.*)\)\s*$", category or "")
    label = m.group(1) if m else (category or "")
    return re.sub(r"\s+", " ", label).strip()


def canonical_category(category: str, scheme_name: str = "") -> str:
    label = raw_category_label(category)
    if not label:
        return "Uncategorised"
    low = label.lower()
    if low in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[low]
    if low == "other scheme - index funds":
        return "Index Funds - Debt" if _DEBT_INDEX_RE.search(scheme_name or "") else "Index Funds - Equity"
    # Generic plural/prefix folding.
    label = re.sub(r"^Equity Schemes\b", "Equity Scheme", label)
    label = re.sub(r"^Hybrid Schemes\b", "Hybrid Scheme", label)
    label = re.sub(r"^Income/Debt Oriented Schemes\b", "Debt Scheme", label)
    return label


def is_growth_option(option: str, scheme_name: str) -> bool:
    """Growth option and nothing that pays out / reinvests distributions."""
    text = f"{option} {scheme_name}"
    if _IDCW_RE.search(option or ""):
        return False
    if not re.search(r"growth", text, re.I):
        return False
    # Some AMCs leave option blank and put everything in the name.
    if not option and _IDCW_RE.search(scheme_name or ""):
        return False
    return True


def is_direct_plan(plan: str, scheme_name: str, is_direct_flag: bool = False) -> bool:
    if is_direct_flag:
        return True
    text = f"{plan} {scheme_name}".lower()
    return "direct" in text and "regular" not in text


def is_open_ended(category: str, scheme_type: str = "") -> bool:
    head = (scheme_type or (category or "").split("(")[0]).strip().lower()
    return head.startswith("open ended")


def passes_static_rules(*, plan: str, option: str, scheme_name: str, category: str,
                        is_direct: bool = False, scheme_type: str = "") -> bool:
    """Everything in the universe rule except the "NAV seen recently" part."""
    if not is_open_ended(category, scheme_type):
        return False
    if not is_direct_plan(plan, scheme_name, is_direct):
        return False
    if not is_growth_option(option, scheme_name):
        return False
    if _EXCLUDE_NAME_RE.search(scheme_name or ""):
        return False
    label = raw_category_label(category).lower()
    if "etf" in label or label.startswith("exchange traded") or "idf" == label:
        return False
    return True


def looks_like_direct_growth_name(scheme_name: str) -> bool:
    """Name-only prefilter for schemes we only know by name (mfapi's list of
    dead schemes) before spending an HTTP call on them."""
    return (
        is_direct_plan("", scheme_name)
        and is_growth_option("", scheme_name)
        and not _EXCLUDE_NAME_RE.search(scheme_name)
    )


def active_cutoff(catalog_date: date) -> date:
    return catalog_date - timedelta(days=ACTIVE_NAV_DAYS)


def universe_schemes(include_inactive: bool = False):
    """MFScheme rows passing the static rules. ``include_inactive`` keeps
    dead schemes (survivorship); otherwise only ``is_active`` ones."""
    from apps.mutual_funds.models import MFScheme

    qs = MFScheme.objects.filter(category__istartswith="Open Ended")
    if not include_inactive:
        qs = qs.filter(is_active=True)
    return [
        s for s in qs.only("scheme_code", "plan", "option", "scheme_name", "category", "is_direct", "is_active",
                           "amc_name")
        if passes_static_rules(plan=s.plan, option=s.option, scheme_name=s.scheme_name,
                               category=s.category, is_direct=s.is_direct)
    ]
