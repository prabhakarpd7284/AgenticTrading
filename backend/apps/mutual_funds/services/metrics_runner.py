"""Computes MFMetricSnapshot rows category by category (S3).

Memory: one category's NAV series at a time (largest ~270 schemes x ~3,400
points), never the whole NAV table. ``scheme_metrics`` and
``category_metrics`` are pure (no ORM) so the S5 backtest reuses them at past
``as_of`` dates on truncated series — strictly point-in-time.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
import structlog

from apps.mutual_funds.models import MFBenchmarkNav, MFDataIssue, MFMetricSnapshot, MFNav, MFScheme
from apps.mutual_funds.services import metrics as m
from apps.mutual_funds.services import screener_config as cfg
from apps.mutual_funds.services.universe import canonical_category, universe_schemes

log = structlog.get_logger(__name__)


def _r(x, nd=6):
    return None if x is None or pd.isna(x) else round(float(x), nd)


# ── pure ─────────────────────────────────────────────────────────────────

def scheme_metrics(nav: pd.Series, as_of: date, rf: float, bench: pd.Series | None = None,
                   issue_ranges=(), benchmark_name: str = "") -> dict:
    """Every per-scheme metric that does not need category peers.
    ``nav`` must already be truncated to ``as_of``."""
    out: dict = {"quality_flags": []}
    flags = out["quality_flags"]
    if nav.empty:
        flags.append("no_nav")
        return out
    yrs = m.history_years(nav)
    out.update(first_nav_date=nav.index[0].date(), last_nav_date=nav.index[-1].date(),
               history_years=round(yrs, 3), nav_points=int(len(nav)))
    if (pd.Timestamp(as_of) - nav.index[-1]).days > cfg.STALE_NAV_DAYS:
        flags.append("stale_nav")
    if yrs < 3:
        flags.append("history_lt_3y")
    elif yrs < 5:
        flags.append("history_lt_5y")

    out["cagr_1y"] = _r(m.cagr(nav, 1))
    out["cagr_3y"] = _r(m.cagr(nav, 3))
    out["cagr_5y"] = _r(m.cagr(nav, 5))
    out["cagr_since_inception"] = _r(m.cagr_since_inception(nav))

    roll = m.rolling_cagr(nav, cfg.ROLLING_YEARS)
    rs = m.rolling_stats(roll)
    out.update(rolling_3y_median=_r(rs.median), rolling_3y_min=_r(rs.minimum),
               rolling_3y_pct_positive=_r(rs.pct_positive, 2), rolling_3y_windows=rs.windows)

    risk = m.trailing(nav, cfg.RISK_WINDOW_YEARS)
    if m.history_years(risk) >= cfg.RISK_WINDOW_YEARS - 0.05:
        rets = m.daily_returns(risk)
        ppy = m.periods_per_year(risk)
        out["volatility"] = _r(m.annualised_volatility(rets, ppy))
        out["downside_deviation"] = _r(m.downside_deviation(rets, ppy, rf))
        out["sharpe"] = _r(m.sharpe(rets, ppy, rf), 4)
        out["sortino"] = _r(m.sortino(rets, ppy, rf), 4)

    ddw = m.trailing(nav, cfg.DRAWDOWN_WINDOW_YEARS)
    if m.history_years(ddw) >= 1:
        dd = m.max_drawdown(ddw)
        if dd is not None:
            out["max_drawdown"] = _r(dd.max_drawdown)
            out["max_drawdown_days"] = dd.duration_days
            out["recovery_days"] = dd.recovery_days
            if dd.max_drawdown < 0 and dd.recovery is None:
                flags.append("drawdown_not_recovered")
        r1 = m.rolling_return(ddw, 12)
        out["worst_1y"] = _r(r1.min()) if not r1.empty else None
        out["calmar"] = _r(m.calmar(out.get("cagr_3y"), out.get("max_drawdown")), 4)

    if bench is not None and not bench.empty:
        b = m.upto(bench, nav.index[-1])
        out["benchmark_name"] = benchmark_name
        fb, bb = m.trailing(nav, cfg.RISK_WINDOW_YEARS), m.trailing(b, cfg.RISK_WINDOW_YEARS)
        bc = m.cagr(b, 3)
        if out.get("cagr_3y") is not None and bc is not None:
            out["excess_cagr_3y_vs_benchmark"] = _r(out["cagr_3y"] - bc)
        out["beta"] = _r(m.beta(fb, bb), 4)
        up, dn = m.capture_ratios(fb, bb)
        out["up_capture"], out["down_capture"] = _r(up, 2), _r(dn, 2)
        te, ir = m.tracking_error_and_ir(fb, bb)
        out["tracking_error"], out["information_ratio"] = _r(te), _r(ir, 4)
    else:
        flags.append("no_benchmark")

    if issue_ranges:
        start3 = nav.index[-1] - pd.DateOffset(years=3)
        if m.overlaps(issue_ranges, start3, nav.index[-1]):
            flags.append("data_issue_in_3y_window")
        elif m.overlaps(issue_ranges, nav.index[0], nav.index[-1]):
            flags.append("data_issue_in_history")
    out["risk_free_rate"] = rf
    out["_rolling"] = roll       # consumed by category_metrics, not stored
    return out


def after_last_jump(nav: pd.Series, jumps, as_of=None) -> tuple[pd.Series, bool]:
    """Drop everything up to the most recent NAV jump (face-value change,
    bonus, data error) known at ``as_of``: a return measured across one is
    meaningless, and we never splice or interpolate. Returns (series, cut?)."""
    cut_at = None
    limit = pd.Timestamp(as_of) if as_of is not None else None
    for _a, b in jumps:
        tb = pd.Timestamp(b)
        if limit is not None and tb > limit:
            continue
        cut_at = tb if cut_at is None or tb > cut_at else cut_at
    if cut_at is None:
        return nav, False
    return nav.loc[cut_at:], True


def category_median_rolling(rollings: list[pd.Series], min_funds: int = cfg.MIN_CATEGORY_FUNDS_FOR_MEDIAN) -> pd.Series:
    if not rollings:
        return pd.Series(dtype=float)
    df = pd.concat(rollings, axis=1, sort=True)
    med = df.median(axis=1, skipna=True)
    return med[df.notna().sum(axis=1) >= min_funds]


def category_metrics(rows: dict[str, dict], peer_rollings: list[pd.Series] | None = None) -> None:
    """Adds peer-relative fields in place: % of rolling windows above the
    category median and 3y CAGR excess over the category median."""
    rolls = peer_rollings if peer_rollings is not None else [r["_rolling"] for r in rows.values() if "_rolling" in r]
    med = category_median_rolling(rolls)
    c3 = [r["cagr_3y"] for r in rows.values() if r.get("cagr_3y") is not None]
    cat_c3 = float(pd.Series(c3).median()) if len(c3) >= cfg.MIN_CATEGORY_FUNDS_FOR_MEDIAN else None
    for r in rows.values():
        roll = r.get("_rolling")
        r["rolling_3y_pct_above_cat_median"] = _r(m.pct_windows_above(roll, med), 2) if roll is not None and not roll.empty else None
        if cat_c3 is not None and r.get("cagr_3y") is not None:
            r["excess_cagr_3y_vs_category"] = _r(r["cagr_3y"] - cat_c3)


# ── ORM-facing ───────────────────────────────────────────────────────────

def load_nav(scheme_code: str, as_of: date | None = None) -> pd.Series:
    qs = MFNav.objects.filter(scheme_id=scheme_code)
    if as_of:
        qs = qs.filter(nav_date__lte=as_of)
    return m.to_series(list(qs.order_by("nav_date").values_list("nav_date", "nav")))


_BENCH_CACHE: dict[str, pd.Series] = {}


def load_benchmark(index_name: str) -> pd.Series:
    if index_name not in _BENCH_CACHE:
        _BENCH_CACHE[index_name] = m.to_series(list(
            MFBenchmarkNav.objects.filter(index_name=index_name).order_by("nav_date").values_list("nav_date", "value")))
    return _BENCH_CACHE[index_name]


def issue_ranges_by_scheme(codes, kinds=None) -> dict[str, list[tuple[date, date]]]:
    qs = MFDataIssue.objects.filter(scheme_id__in=codes)
    if kinds:
        qs = qs.filter(kind__in=kinds)
    out = defaultdict(list)
    for code, a, b in qs.values_list("scheme_id", "start_date", "end_date"):
        out[code].append((a, b))
    return out


def schemes_by_category(include_inactive: bool = True) -> dict[str, list[MFScheme]]:
    cats = defaultdict(list)
    for s in universe_schemes(include_inactive=include_inactive):
        cats[canonical_category(s.category, s.scheme_name)].append(s)
    return cats


SNAPSHOT_FIELDS = [f.name for f in MFMetricSnapshot._meta.get_fields()
                   if f.concrete and f.name not in ("id", "scheme", "as_of", "category", "is_active", "computed_at")]


def compute_snapshots(as_of: date, categories: list[str] | None = None, dry_run: bool = False,
                      rf: float | None = None, progress=None) -> dict:
    rf = cfg.RISK_FREE_RATE["value"] if rf is None else rf
    cats = schemes_by_category(include_inactive=True)
    if categories:
        wanted = {c.lower() for c in categories}
        cats = {c: v for c, v in cats.items() if c.lower() in wanted}
    summary = {"categories": 0, "snapshots": 0, "schemes_live": 0, "dead_in_medians": 0}
    for cat, schemes in sorted(cats.items()):
        bench_name = cfg.CATEGORY_BENCHMARKS.get(cat, (None,))[0]
        bench = load_benchmark(bench_name) if bench_name else None
        codes = [s.scheme_code for s in schemes]
        issues = issue_ranges_by_scheme(codes)
        jumps = issue_ranges_by_scheme(codes, kinds=[MFDataIssue.KIND_JUMP])
        rows: dict[str, dict] = {}
        peer_rolls: list[pd.Series] = []
        for s in schemes:
            nav, cut = after_last_jump(load_nav(s.scheme_code, as_of), jumps.get(s.scheme_code, ()), as_of)
            if nav.empty:
                continue
            res = scheme_metrics(nav, as_of, rf, bench, issues.get(s.scheme_code, ()), bench_name or "")
            if cut:
                res["quality_flags"].append("history_cut_at_nav_jump")
            peer_rolls.append(res["_rolling"])        # dead funds count in the peer median
            if s.is_active:
                rows[s.scheme_code] = res
            else:
                summary["dead_in_medians"] += 1
        category_metrics(rows, peer_rolls)
        summary["categories"] += 1
        summary["schemes_live"] += len(rows)
        if not dry_run and rows:
            MFMetricSnapshot.objects.filter(as_of=as_of, scheme_id__in=list(rows)).delete()
            MFMetricSnapshot.objects.bulk_create([
                MFMetricSnapshot(scheme_id=code, as_of=as_of, category=cat, is_active=True,
                                 **{k: v for k, v in r.items() if k in SNAPSHOT_FIELDS})
                for code, r in rows.items()
            ], batch_size=500)
            summary["snapshots"] += len(rows)
        if progress:
            progress(cat, len(rows))
    return summary
