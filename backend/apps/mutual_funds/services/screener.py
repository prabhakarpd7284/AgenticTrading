"""MF screener (S4) — hard filters, then a transparent within-category score.

PERSONAL USE ONLY. Deterministic arithmetic over MFMetricSnapshot rows; no
LLM, no forecasts. Funds are only ever compared with their own canonical
category. Every ranked row carries its per-component scores (0-100
within-category percentiles) and the weights actually used, so the output
shows *why* a fund ranks where it does.

Reads snapshots / NAV / schemes; never writes collection tables.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd
import structlog
from django.core.cache import cache
from django.db.models import Max

from apps.mutual_funds.models import MFMetricSnapshot
from apps.mutual_funds.services import metrics as m
from apps.mutual_funds.services import screener_config as cfg
from apps.mutual_funds.services.metrics_runner import load_benchmark, load_nav, schemes_by_category

log = structlog.get_logger(__name__)

PERSONAL_USE_NOTE = ("Personal-use screen on public NAV history. Past returns do not predict future "
                     "returns; see docs/MF_SCREENER_VALIDATION.md for how weak the evidence is.")


@dataclass
class ScreenCriteria:
    categories: list[str] = field(default_factory=list)   # canonical labels; empty = all
    min_history_years: float = 3.0
    min_aum_cr: float | None = None
    max_ter_pct: float | None = None
    min_sharpe: float | None = None
    max_drawdown_floor: float | None = None     # e.g. -0.35 -> exclude funds that fell more than 35 %
    exclude_flags: list[str] = field(default_factory=list)   # e.g. ["data_issue_in_3y_window"]
    weights: dict[str, float] = field(default_factory=lambda: dict(cfg.DEFAULT_WEIGHTS))
    top: int | None = None
    as_of: date | None = None


def percentile_scores(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0-100 percentile within the given values (ties averaged); NaN stays NaN.
    A lone value scores 50."""
    v = values.dropna()
    out = pd.Series(index=values.index, dtype=float)
    if v.empty:
        return out
    if len(v) == 1:
        out.loc[v.index] = 50.0
        return out
    ranks = v.rank(method="average", ascending=higher_is_better)
    out.loc[v.index] = (ranks - 1) / (len(v) - 1) * 100
    return out


def score_frame(df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """Adds ``score_<component>`` columns and ``score``; returns
    (df, weights actually used, notes). Components with no data for the whole
    frame are dropped and the rest renormalised."""
    notes: list[str] = []
    used: dict[str, float] = {}
    if "rolling_3y_pct_above_cat_median" in df:
        k = cfg.CONSISTENCY_PRIOR_WINDOWS
        n = df.get("rolling_3y_windows", pd.Series(0, index=df.index)).fillna(0).astype(float)
        p = df["rolling_3y_pct_above_cat_median"].astype(float)
        df["consistency_pct_above_shrunk"] = (p * n + 50.0 * k) / (n + k)
    for comp, metrics_ in cfg.COMPONENT_METRICS.items():
        w = float(weights.get(comp, 0) or 0)
        parts = []
        for col, hib in metrics_:
            if col in df and df[col].notna().any():
                parts.append(percentile_scores(df[col].astype(float), hib))
        if parts:
            df[f"score_{comp}"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
            if w > 0:
                used[comp] = w
        else:
            df[f"score_{comp}"] = float("nan")
            if w > 0:
                notes.append(cfg.COMPONENT_NOTES.get(comp, f"{comp}: no data - dropped, weights renormalised."))
        if comp in cfg.COMPONENT_NOTES and parts and comp == "size_age":
            notes.append(cfg.COMPONENT_NOTES[comp])
    total = sum(used.values())
    used = {k: round(v / total * 100, 2) for k, v in used.items()} if total else {}

    def composite(row):
        num = den = 0.0
        for comp, w in used.items():
            s = row.get(f"score_{comp}")
            if s is not None and not pd.isna(s):
                num += w * s
                den += w
        return num / den if den else float("nan")

    df["score"] = df.apply(composite, axis=1) if len(df) else pd.Series(dtype=float)
    df["partial_score"] = df.apply(
        lambda r: any(pd.isna(r.get(f"score_{c}")) for c in used), axis=1) if len(df) else []
    return df, used, notes


def latest_as_of(category: str | None = None) -> date | None:
    qs = MFMetricSnapshot.objects.all()
    if category:
        qs = qs.filter(category__iexact=category)
    return qs.aggregate(m=Max("as_of"))["m"]


_ROW_FIELDS = [
    "cagr_1y", "cagr_3y", "cagr_5y", "cagr_since_inception", "rolling_3y_median", "rolling_3y_min",
    "rolling_3y_pct_positive", "rolling_3y_pct_above_cat_median", "rolling_3y_windows", "volatility",
    "downside_deviation", "max_drawdown", "max_drawdown_days", "recovery_days", "worst_1y", "sharpe",
    "sortino", "calmar", "benchmark_name", "excess_cagr_3y_vs_benchmark", "beta", "up_capture",
    "down_capture", "tracking_error", "information_ratio", "excess_cagr_3y_vs_category", "history_years",
    "first_nav_date", "last_nav_date", "quality_flags",
]


def _frame(category: str, as_of: date) -> pd.DataFrame:
    snaps = (MFMetricSnapshot.objects.filter(category__iexact=category, as_of=as_of, is_active=True)
             .select_related("scheme"))
    recs = []
    for s in snaps:
        r = {f: getattr(s, f) for f in _ROW_FIELDS}
        r.update(scheme_code=s.scheme_id, scheme_name=s.scheme.scheme_name, amc_name=s.scheme.amc_name,
                 category=s.category, as_of=s.as_of,
                 # S2 reference data is not sourced yet: explicit nulls, never imputed.
                 expense_ratio_pct=None, aum_cr=None)
        recs.append(r)
    return pd.DataFrame.from_records(recs)


def _apply_filters(df: pd.DataFrame, c: ScreenCriteria) -> tuple[pd.DataFrame, dict[str, int]]:
    excluded: dict[str, int] = {}

    def drop(mask, reason):
        n = int(mask.sum())
        if n:
            excluded[reason] = excluded.get(reason, 0) + n
        return df[~mask]

    if df.empty:
        return df, excluded
    df = drop(df["history_years"].fillna(0) < c.min_history_years, f"history < {c.min_history_years}y")
    if c.min_aum_cr is not None:
        df = drop(df["aum_cr"].isna() | (df["aum_cr"] < c.min_aum_cr), "AUM below min or n/a")
    if c.max_ter_pct is not None:
        df = drop(df["expense_ratio_pct"].isna() | (df["expense_ratio_pct"] > c.max_ter_pct), "TER above max or n/a")
    if c.min_sharpe is not None:
        df = drop(df["sharpe"].isna() | (df["sharpe"] < c.min_sharpe), "Sharpe below min or n/a")
    if c.max_drawdown_floor is not None:
        df = drop(df["max_drawdown"].isna() | (df["max_drawdown"] < c.max_drawdown_floor), "drawdown deeper than floor")
    for flag in c.exclude_flags:
        df = drop(df["quality_flags"].apply(lambda fl, f=flag: f in (fl or [])), f"flag {flag}")
    return df, excluded


def _clean(v):
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def screen_category(category: str, criteria: ScreenCriteria, with_index_check: bool = True) -> dict:
    as_of = criteria.as_of or latest_as_of(category)
    out = {"category": category, "as_of": as_of.isoformat() if as_of else None, "rows": [],
           "universe_count": 0, "eligible_count": 0, "excluded": {}, "weights_used": {}, "notes": [],
           "index_check": None}
    if as_of is None:
        out["notes"].append("No metric snapshots for this category - run compute_mf_metrics.")
        return out
    df = _frame(category, as_of)
    out["universe_count"] = len(df)
    df, out["excluded"] = _apply_filters(df, criteria)
    out["eligible_count"] = len(df)
    if df.empty:
        out["notes"].append("No fund passes the hard filters.")
    else:
        df, used, notes = score_frame(df.reset_index(drop=True), criteria.weights)
        out["weights_used"], out["notes"] = used, notes
        df = df.sort_values(["score", "scheme_code"], ascending=[False, True], na_position="last")
        df["rank"] = range(1, len(df) + 1)
        if criteria.top:
            df = df.head(criteria.top)
        comps = list(cfg.COMPONENT_METRICS)
        for _, r in df.iterrows():
            row = {k: _clean(r.get(k)) for k in ["rank", "scheme_code", "scheme_name", "amc_name", "score",
                                                    "partial_score", *_ROW_FIELDS, "expense_ratio_pct", "aum_cr"]}
            row["score"] = None if row["score"] is None else round(row["score"], 2)
            row["components"] = {c: (None if pd.isna(r.get(f"score_{c}")) else round(float(r[f"score_{c}"]), 2))
                                 for c in comps}
            out["rows"].append(row)
    if with_index_check:
        out["index_check"] = index_vs_active(category, as_of)
    return out


def screen(criteria: ScreenCriteria, with_index_check: bool = True, tenant=None) -> dict:
    """Runs the screen. With a ``tenant`` the run is logged through
    event_writer.emit() (non-blocking); Event rows are tenant-scoped, so a
    tenant-less CLI run is only logged to structlog."""
    cats = criteria.categories or sorted(set(
        MFMetricSnapshot.objects.values_list("category", flat=True).distinct()))
    result = {"criteria": {k: _clean(v) for k, v in asdict(criteria).items()},
              "personal_use_note": PERSONAL_USE_NOTE,
              "categories": [screen_category(c, criteria, with_index_check) for c in cats]}
    log.info("mf_screener.run", categories=len(cats), criteria=result["criteria"])
    if tenant is None:
        return result
    try:   # non-blocking audit trail of every ranking run
        from apps.events.models import Event
        from apps.events.services.event_writer import emit
        emit(tenant=tenant, type=Event.Type.WORKFLOW_STEP_COMPLETED, step_name="mf_screener.run",
             text=f"MF screen over {len(cats)} categor{'y' if len(cats) == 1 else 'ies'}",
             payload={"criteria": result["criteria"],
                      "counts": {c["category"]: c["eligible_count"] for c in result["categories"]}})
    except Exception as exc:  # noqa: BLE001
        log.warning("mf_screener.emit_failed", error=str(exc))
    return result


# ── index vs active ──────────────────────────────────────────────────────

def _median_rolling(codes: list[str], as_of: date, min_funds: int = 1) -> tuple[pd.Series, int]:
    rolls = []
    for code in codes:
        r = m.rolling_cagr(load_nav(code, as_of), cfg.ROLLING_YEARS)
        if not r.empty:
            rolls.append(r)
    if not rolls:
        return pd.Series(dtype=float), 0
    df = pd.concat(rolls, axis=1, sort=True)
    med = df.median(axis=1, skipna=True)
    return med[df.notna().sum(axis=1) >= min_funds], len(rolls)


def index_funds_for(category: str) -> list[str]:
    spec = cfg.CATEGORY_BENCHMARKS.get(category)
    if not spec:
        return []
    want, bad = re.compile(spec[1], re.I), re.compile(cfg.INDEX_FUND_EXCLUDE, re.I)
    return [s.scheme_code for s in schemes_by_category(include_inactive=True).get("Index Funds - Equity", [])
            if want.search(s.scheme_name) and not bad.search(s.scheme_name)]


def index_vs_active(category: str, as_of: date) -> dict | None:
    """Did the median active fund of the category (dead funds included) beat
    what an investor could have bought passively, over rolling 3y windows?
    Index-fund NAVs (after their costs) are preferred; with too little
    index-fund history the gross TRI is used and labelled."""
    spec = cfg.CATEGORY_BENCHMARKS.get(category)
    if not spec:
        return None
    key = "mf_index_check_v2_" + hashlib.sha1(f"{category}|{as_of}".encode()).hexdigest()
    try:
        hit = cache.get(key)
    except Exception:  # noqa: BLE001
        hit = None
    if hit is not None:
        return hit
    bench_name, _ = spec
    active_codes = [s.scheme_code for s in schemes_by_category(include_inactive=True).get(category, [])]
    active_med, n_active = _median_rolling(active_codes, as_of, cfg.MIN_CATEGORY_FUNDS_FOR_MEDIAN)
    idx_codes = index_funds_for(category)
    idx_med, n_idx = _median_rolling(idx_codes, as_of)
    tri = m.rolling_cagr(m.upto(load_benchmark(bench_name), as_of), cfg.ROLLING_YEARS)

    both_idx = pd.concat([active_med, idx_med], axis=1, join="inner").dropna()
    both_tri = pd.concat([active_med, tri], axis=1, join="inner").dropna()
    idx_ok = (len(both_idx) >= cfg.MIN_INDEX_FUND_WINDOWS
              and len(both_idx) >= cfg.INDEX_FUND_MIN_COVERAGE * max(len(both_tri), 1))
    if idx_ok:
        ref, both, basis = idx_med, both_idx, "index_fund"
        ref_label = f"median of {n_idx} index fund(s) matching /{spec[1]}/ (NAV, after their costs)"
    else:
        ref, both, basis = tri, both_tri, "tri"
        ref_label = (f"{bench_name} TRI (gross - no fund costs, flatters passive by ~an index fund's TER); "
                     f"index-fund history covers {len(both_idx)} of {len(both_tri)} windows")
    alt = None
    other = both_tri if idx_ok else both_idx
    if len(other):
        alt = {"basis": "tri" if idx_ok else "index_fund", "windows": int(len(other)),
               "pct_windows_median_active_beat": round(float((other.iloc[:, 0] > other.iloc[:, 1]).mean() * 100), 1)}
    res = {"category": category, "as_of": as_of.isoformat(), "basis": basis, "reference": ref_label,
           "active_funds_used": n_active, "index_funds_found": n_idx, "windows": int(len(both)),
           "alternative": alt}
    if both.empty:
        res.update(verdict="unknown", text="Not enough overlapping rolling windows to compare.")
        return res
    beat = (both.iloc[:, 0] > both.iloc[:, 1])
    gap = (both.iloc[:, 0] - both.iloc[:, 1])
    pct = float(beat.mean() * 100)
    recent = both[both.index >= both.index[-1] - pd.DateOffset(years=3)]
    pct_recent = float((recent.iloc[:, 0] > recent.iloc[:, 1]).mean() * 100) if len(recent) else None
    # share of individual active funds (alive or dead) whose own windows beat the reference most of the time
    winners = total = 0
    for code in active_codes:
        r = m.rolling_cagr(load_nav(code, as_of), cfg.ROLLING_YEARS)
        p = m.pct_windows_above(r, ref) if not r.empty else None
        if p is not None:
            total += 1
            winners += p > 50
    verdict = ("yes" if pct >= cfg.ACTIVE_EARNS_FEE_PCT else "no" if pct <= cfg.ACTIVE_NO_FEE_PCT else "mixed")
    res.update(
        pct_windows_median_active_beat=round(pct, 1),
        pct_windows_recent_3y=None if pct_recent is None else round(pct_recent, 1),
        median_gap_pp=round(float(gap.median()) * 100, 2),
        first_window_end=both.index[0].date().isoformat(), last_window_end=both.index[-1].date().isoformat(),
        funds_beating_reference_majority=f"{winners}/{total}",
        verdict=verdict,
        text={
            "yes": "The median active fund beat the passive alternative in most rolling 3y windows.",
            "no": "The median active fund did NOT beat the passive alternative in most rolling 3y windows - "
                  "a low-cost index fund is the honest default here.",
            "mixed": "Active vs passive is a coin-flip in this category over rolling 3y windows.",
        }[verdict],
    )
    try:
        cache.set(key, res, 7 * 24 * 3600)
    except Exception:  # noqa: BLE001
        pass
    return res
