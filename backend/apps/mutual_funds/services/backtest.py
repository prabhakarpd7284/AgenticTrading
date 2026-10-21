"""Point-in-time validation of the screener score (S5).

At each quarter-end t:
  1. Eligible funds = universe schemes (live AND dead) of the category that
     were priced at t (NAV within 7 days) and had >= 3y of history at t.
  2. Metrics are computed from NAV <= t only (``metrics.upto``), peers'
     rolling medians likewise, then scored with the same ``score_frame`` the
     live screener uses. Alternative single-signal scores are scored too.
  3. Forward return over h months from t. A fund that dies inside the
     horizon (merged / wound up) is NOT dropped: its return up to its last
     NAV is chained with the category's median forward return over the rest
     of the horizon (the scheme-merger mapping is not available, so the
     category median is the neutral assumption). This keeps losers in.
  4. Groups by score: top quartile, middle half, bottom quartile; plus the
     passive alternative (median index fund NAV, or TRI if none existed).

Memory: one category's NAV series held at a time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import structlog

from apps.mutual_funds.services import metrics as m
from apps.mutual_funds.services import screener_config as cfg
from apps.mutual_funds.models import MFDataIssue
from apps.mutual_funds.services.metrics_runner import (
    after_last_jump,
    category_metrics,
    issue_ranges_by_scheme,
    load_benchmark,
    load_nav,
    scheme_metrics,
    schemes_by_category,
)
from apps.mutual_funds.services.screener import index_funds_for, score_frame

log = structlog.get_logger(__name__)

PRICED_TOLERANCE_DAYS = 7

# Alternative scores to test against the composite (weights over components,
# or the special "trailing_3y" = rank by past 3y CAGR alone).
VARIANTS: dict[str, dict | str] = {
    "composite_default": dict(cfg.DEFAULT_WEIGHTS),     # S5-chosen defaults
    "plan_weights": dict(cfg.PLAN_DEFAULT_WEIGHTS),
    "consistency_only": {"consistency": 1},
    "risk_adjusted_only": {"risk_adjusted": 1},
    "drawdown_only": {"drawdown": 1},
    "age_only": {"size_age": 1},
    "risk_plus_drawdown": {"risk_adjusted": 1, "drawdown": 1},
    "trailing_3y_cagr": "trailing_3y",
}


def _price_at(nav: pd.Series, ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    pos = nav.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    d = nav.index[pos]
    if (ts - d).days > PRICED_TOLERANCE_DAYS:
        return None
    return d, float(nav.iloc[pos])


def forward_return(nav: pd.Series, t: pd.Timestamp, months: int) -> tuple[float | None, pd.Timestamp | None]:
    """(simple return t -> t+months, None) if alive at the horizon; otherwise
    (return up to its last NAV, last NAV date) so the caller can chain it."""
    start = _price_at(nav, t)
    if start is None:
        return None, None
    end_ts = t + pd.DateOffset(months=months)
    end = _price_at(nav, end_ts)
    if end is not None:
        return end[1] / start[1] - 1, None
    after = nav.loc[t:]
    if after.empty or after.index[-1] >= end_ts:
        return None, None
    return float(after.iloc[-1]) / start[1] - 1, after.index[-1]


def annualise(simple: float, months: int) -> float:
    return (1 + simple) ** (12 / months) - 1


@dataclass
class DateResult:
    t: date
    n: int
    groups: dict[str, dict[str, float | None]] = field(default_factory=dict)   # variant -> {top, mid, bottom, ic}
    passive: float | None = None
    passive_basis: str = ""
    dead_chained: int = 0


def _ic(score: pd.Series, fwd: pd.Series) -> float | None:
    df = pd.concat([score, fwd], axis=1).dropna()
    if len(df) < 5:
        return None
    return float(df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank()))


def _groups(score: pd.Series, fwd: pd.Series) -> dict[str, float | None]:
    df = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(df) < 8:
        return {"top": None, "mid": None, "bottom": None, "ic": None}
    pct = df["s"].rank(pct=True, method="average")
    return {
        "top": float(df.loc[pct > 0.75, "f"].mean()),
        "mid": float(df.loc[(pct > 0.25) & (pct <= 0.75), "f"].mean()),
        "bottom": float(df.loc[pct <= 0.25, "f"].mean()),
        "ic": _ic(df["s"], df["f"]),
    }


def quarter_ends(start: date, end: date) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="QE"))


def backtest_category(category: str, horizon_months: int, start: date, end: date,
                      rf: float | None = None) -> list[DateResult]:
    rf = cfg.RISK_FREE_RATE["value"] if rf is None else rf
    schemes = schemes_by_category(include_inactive=True).get(category, [])
    navs = {s.scheme_code: load_nav(s.scheme_code) for s in schemes}
    navs = {k: v for k, v in navs.items() if not v.empty}
    jumps = issue_ranges_by_scheme(list(navs), kinds=[MFDataIssue.KIND_JUMP])
    idx_navs = {c: load_nav(c) for c in index_funds_for(category)}
    bench_name = cfg.CATEGORY_BENCHMARKS.get(category, (None,))[0]
    tri = load_benchmark(bench_name) if bench_name else pd.Series(dtype=float)
    results: list[DateResult] = []
    last_data = max((v.index[-1] for v in navs.values()), default=None)
    for t in quarter_ends(start, end):
        if last_data is None or t + pd.DateOffset(months=horizon_months) > last_data:
            break
        rows, peer_rolls, fwd, chained = {}, [], {}, {}
        for code, nav in navs.items():
            hist, _ = after_last_jump(m.upto(nav, t), jumps.get(code, ()), t)
            if hist.empty or _price_at(hist, t) is None:
                continue
            res = scheme_metrics(hist, t.date(), rf)
            peer_rolls.append(res["_rolling"])
            if (res.get("history_years") or 0) < 3:
                continue
            rows[code] = res
            # A jump inside the forward window would fake a return: skip that fund-date.
            if any(pd.Timestamp(b) > t and pd.Timestamp(a) <= t + pd.DateOffset(months=horizon_months)
                   for a, b in jumps.get(code, ())):
                continue
            r, died = forward_return(nav, t, horizon_months)
            if r is None:
                continue
            if died is None:
                fwd[code] = r
            else:
                chained[code] = (r, died)
        if len(rows) < 8:
            continue
        category_metrics(rows, peer_rolls)
        # Chain dead funds with the category median forward return after their death.
        for code, (r, died) in chained.items():
            rest = []
            for oc, onav in navs.items():
                a = _price_at(onav, died)
                b = _price_at(onav, t + pd.DateOffset(months=horizon_months))
                if a and b:
                    rest.append(b[1] / a[1] - 1)
            fwd[code] = (1 + r) * (1 + float(np.median(rest))) - 1 if rest else r
        fwd_ann = pd.Series({k: annualise(v, horizon_months) for k, v in fwd.items()})

        df = pd.DataFrame.from_dict(rows, orient="index")
        df["scheme_code"] = df.index
        dr = DateResult(t=t.date(), n=len(fwd_ann), dead_chained=len(chained))
        for name, w in VARIANTS.items():
            if w == "trailing_3y":
                score = df["cagr_3y"].astype(float)
            else:
                scored, _, _ = score_frame(df.copy(), w)
                score = scored["score"]
            dr.groups[name] = _groups(score, fwd_ann)

        # Passive alternative: median index fund priced at t and at horizon, else TRI.
        idx_rets = []
        for nav in idx_navs.values():
            r, died = forward_return(nav, t, horizon_months)
            if r is not None and died is None:
                idx_rets.append(r)
        if idx_rets:
            dr.passive, dr.passive_basis = annualise(float(np.median(idx_rets)), horizon_months), "index_fund"
        elif not tri.empty:
            r, died = forward_return(tri, t, horizon_months)
            if r is not None and died is None:
                dr.passive, dr.passive_basis = annualise(r, horizon_months), "tri_gross"
        results.append(dr)
    return results


def summarise(results: list[DateResult], step: int = 1) -> dict:
    """Averages over dates. ``step`` thins dates (4 = annual) so overlapping
    forward windows do not inflate the t-stat."""
    res = results[::step]
    out: dict = {"dates": len(res), "variants": {}}
    if not res:
        return out
    out["first"], out["last"] = res[0].t.isoformat(), res[-1].t.isoformat()
    out["avg_funds"] = round(float(np.mean([r.n for r in res])), 1)
    out["dead_chained_total"] = int(sum(r.dead_chained for r in res))
    pas = [r.passive for r in res if r.passive is not None]
    out["passive_mean"] = float(np.mean(pas)) if pas else None
    out["passive_basis"] = sorted({r.passive_basis for r in res if r.passive_basis})
    for name in VARIANTS:
        g = [r.groups.get(name) for r in res if r.groups.get(name) and r.groups[name]["top"] is not None]
        if not g:
            continue
        top = np.array([x["top"] for x in g]); mid = np.array([x["mid"] for x in g]); bot = np.array([x["bottom"] for x in g])
        ics = np.array([x["ic"] for x in g if x["ic"] is not None])
        spread = top - bot
        paired_pas = [(x["top"], r.passive) for r, x in ((r, r.groups.get(name)) for r in res)
                      if x and x["top"] is not None and r.passive is not None]
        out["variants"][name] = {
            "top_mean": float(top.mean()), "mid_mean": float(mid.mean()), "bottom_mean": float(bot.mean()),
            "top_minus_bottom": float(spread.mean()),
            "top_beats_bottom_pct": float((spread > 0).mean() * 100),
            "ic_mean": float(ics.mean()) if len(ics) else None,
            "ic_t": (float(ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics))))
                     if len(ics) > 2 and ics.std(ddof=1) > 0 else None),
            "top_beats_passive_pct": (float(np.mean([a > b for a, b in paired_pas]) * 100) if paired_pas else None),
            "top_minus_passive": (float(np.mean([a - b for a, b in paired_pas])) if paired_pas else None),
        }
    return out
