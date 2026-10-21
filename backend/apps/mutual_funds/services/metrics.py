"""MF metrics engine (S3) — pure functions over a NAV series.

Conventions (all deterministic, no I/O, no ORM):
- A NAV series is a ``pandas.Series`` of floats indexed by a sorted
  ``DatetimeIndex`` (one value per NAV date; gaps are left as gaps).
- Returns are fractions (0.12 = 12 %). Drawdowns are negative fractions.
- Everything is NAV-based. The universe is Growth options only, so no
  distributions leave the NAV; IDCW options are excluded by that rule.
- Point-in-time: callers truncate the series to ``as_of`` first
  (``upto``), so a backtest at date t never sees data after t.
- A function returns ``None`` when the input is too short, never a guess.
- Periods per year for daily returns are inferred from the data
  (``periods_per_year``): equity funds post ~248 NAVs/yr, liquid funds ~365.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
# Accept a start NAV up to this many days before the exact window start
# (weekends/holidays/missed days); beyond that the window is "not available".
START_TOLERANCE_DAYS = 7


def to_series(rows) -> pd.Series:
    """[(date, nav), ...] -> float Series on a DatetimeIndex, sorted, deduped."""
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows])
    s = pd.Series([float(v) for _, v in rows], index=idx, dtype=float)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s[s > 0]


def upto(nav: pd.Series, as_of) -> pd.Series:
    return nav.loc[: pd.Timestamp(as_of)]


def history_years(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    return (nav.index[-1] - nav.index[0]).days / DAYS_PER_YEAR


def _value_at_or_before(nav: pd.Series, ts: pd.Timestamp, tolerance_days: int = START_TOLERANCE_DAYS):
    """(date, value) of the last NAV on/before ``ts`` if within tolerance."""
    pos = nav.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    d = nav.index[pos]
    if (ts - d).days > tolerance_days:
        return None
    return d, float(nav.iloc[pos])


def cagr(nav: pd.Series, years: float) -> float | None:
    """Trailing CAGR over ``years`` ending at the last NAV. Uses the actual
    day count between the two NAVs used. None if history is too short."""
    if len(nav) < 2:
        return None
    end_d, end_v = nav.index[-1], float(nav.iloc[-1])
    start = _value_at_or_before(nav, end_d - pd.DateOffset(days=round(years * DAYS_PER_YEAR)))
    if start is None:
        return None
    start_d, start_v = start
    yrs = (end_d - start_d).days / DAYS_PER_YEAR
    if yrs <= 0 or start_v <= 0:
        return None
    return (end_v / start_v) ** (1 / yrs) - 1


def cagr_since_inception(nav: pd.Series, min_years: float = 1.0) -> float | None:
    yrs = history_years(nav)
    if yrs < min_years:
        return None
    return (float(nav.iloc[-1]) / float(nav.iloc[0])) ** (1 / yrs) - 1


def month_end(nav: pd.Series) -> pd.Series:
    """Last available NAV in each calendar month (index = month-end stamp)."""
    if nav.empty:
        return nav
    return nav.resample("ME").last().dropna()


def rolling_cagr(nav: pd.Series, years: int = 3) -> pd.Series:
    """Rolling ``years``-year CAGR sampled monthly (month-end to month-end).
    A window needs both endpoint months present; missing months drop out."""
    me = month_end(nav)
    if me.empty:
        return me
    months = years * 12
    shifted = me.copy()
    shifted.index = shifted.index + pd.offsets.MonthEnd(months)
    start = shifted.reindex(me.index)
    out = (me / start) ** (1 / years) - 1
    return out.dropna()


def rolling_return(nav: pd.Series, months: int = 12) -> pd.Series:
    """Simple (not annualised) rolling return over ``months``, monthly sampled."""
    me = month_end(nav)
    if me.empty:
        return me
    shifted = me.copy()
    shifted.index = shifted.index + pd.offsets.MonthEnd(months)
    return (me / shifted.reindex(me.index) - 1).dropna()


@dataclass(frozen=True)
class RollingStats:
    median: float | None
    minimum: float | None
    pct_positive: float | None
    windows: int


def rolling_stats(rolling: pd.Series) -> RollingStats:
    if rolling.empty:
        return RollingStats(None, None, None, 0)
    return RollingStats(float(rolling.median()), float(rolling.min()),
                        float((rolling > 0).mean() * 100), int(len(rolling)))


def pct_windows_above(rolling: pd.Series, reference: pd.Series) -> float | None:
    """% of the fund's rolling windows (dates present in both) where it beat ``reference``."""
    both = pd.concat([rolling, reference], axis=1, join="inner").dropna()
    if both.empty:
        return None
    return float((both.iloc[:, 0] > both.iloc[:, 1]).mean() * 100)


def daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def periods_per_year(nav: pd.Series) -> float:
    yrs = history_years(nav)
    if yrs <= 0:
        return 252.0
    return max((len(nav) - 1) / yrs, 1.0)


def trailing(nav: pd.Series, years: float) -> pd.Series:
    """The last ``years`` of the series (inclusive of the start NAV)."""
    if nav.empty:
        return nav
    start = nav.index[-1] - pd.DateOffset(days=round(years * DAYS_PER_YEAR))
    pos = max(nav.index.searchsorted(start, side="right") - 1, 0)
    return nav.iloc[pos:]


def annualised_volatility(returns: pd.Series, ppy: float) -> float | None:
    if len(returns) < 20:
        return None
    return float(returns.std(ddof=1) * math.sqrt(ppy))


def _per_period(rate_annual: float, ppy: float) -> float:
    return (1 + rate_annual) ** (1 / ppy) - 1


def downside_deviation(returns: pd.Series, ppy: float, mar_annual: float = 0.0) -> float | None:
    """sqrt(mean(min(r - MAR, 0)^2)) annualised — all periods in the
    denominator (Sortino & Price convention)."""
    if len(returns) < 20:
        return None
    mar = _per_period(mar_annual, ppy)
    shortfall = np.minimum(returns.to_numpy() - mar, 0.0)
    return float(math.sqrt(np.mean(shortfall ** 2)) * math.sqrt(ppy))


def sharpe(returns: pd.Series, ppy: float, rf_annual: float) -> float | None:
    if len(returns) < 20:
        return None
    excess = returns - _per_period(rf_annual, ppy)
    sd = returns.std(ddof=1)
    if not sd or not np.isfinite(sd):
        return None
    return float(excess.mean() / sd * math.sqrt(ppy))


def sortino(returns: pd.Series, ppy: float, rf_annual: float) -> float | None:
    dd = downside_deviation(returns, ppy, mar_annual=rf_annual)
    if not dd:
        return None
    excess_annual = (returns - _per_period(rf_annual, ppy)).mean() * ppy
    return float(excess_annual / dd)


@dataclass(frozen=True)
class Drawdown:
    max_drawdown: float          # <= 0
    peak: pd.Timestamp | None
    trough: pd.Timestamp | None
    recovery: pd.Timestamp | None   # first date back at/above the peak NAV; None = not recovered

    @property
    def duration_days(self) -> int | None:
        return None if self.peak is None else (self.trough - self.peak).days

    @property
    def recovery_days(self) -> int | None:
        return None if self.recovery is None else (self.recovery - self.trough).days


def max_drawdown(nav: pd.Series) -> Drawdown | None:
    if len(nav) < 2:
        return None
    running_max = nav.cummax()
    dd = nav / running_max - 1
    trough = dd.idxmin()
    mdd = float(dd.loc[trough])
    if mdd >= 0:
        return Drawdown(0.0, None, None, None)
    peak = nav.loc[:trough].idxmax()
    peak_v = float(nav.loc[peak])
    after = nav.loc[trough:]
    rec = after[after >= peak_v]
    return Drawdown(mdd, peak, trough, rec.index[0] if not rec.empty else None)


def calmar(cagr_value: float | None, mdd: float | None) -> float | None:
    if cagr_value is None or mdd is None or mdd >= 0:
        return None
    return cagr_value / abs(mdd)


# ── relative to a benchmark (monthly returns) ────────────────────────────

def monthly_returns(nav: pd.Series) -> pd.Series:
    return month_end(nav).pct_change().dropna()


def _aligned(fund: pd.Series, bench: pd.Series) -> pd.DataFrame:
    df = pd.concat([monthly_returns(fund), monthly_returns(bench)], axis=1, join="inner").dropna()
    df.columns = ["f", "b"]
    return df


def beta(fund: pd.Series, bench: pd.Series, min_months: int = 24) -> float | None:
    df = _aligned(fund, bench)
    if len(df) < min_months:
        return None
    var = df["b"].var(ddof=1)
    return None if not var else float(df["f"].cov(df["b"]) / var)


def _geo_mean(r: pd.Series) -> float:
    return float(np.prod(1 + r.to_numpy()) ** (1 / len(r)) - 1)


def capture_ratios(fund: pd.Series, bench: pd.Series, min_months: int = 24) -> tuple[float | None, float | None]:
    """(up capture, down capture) in %, geometric (Morningstar-style)."""
    df = _aligned(fund, bench)
    if len(df) < min_months:
        return None, None
    up, down = df[df["b"] > 0], df[df["b"] < 0]
    upc = (_geo_mean(up["f"]) / _geo_mean(up["b"]) * 100) if len(up) >= 3 and _geo_mean(up["b"]) else None
    dnc = (_geo_mean(down["f"]) / _geo_mean(down["b"]) * 100) if len(down) >= 3 and _geo_mean(down["b"]) else None
    return upc, dnc


def tracking_error_and_ir(fund: pd.Series, bench: pd.Series, min_months: int = 24) -> tuple[float | None, float | None]:
    df = _aligned(fund, bench)
    if len(df) < min_months:
        return None, None
    active = df["f"] - df["b"]
    te = float(active.std(ddof=1) * math.sqrt(12))
    ir = float(active.mean() * 12 / te) if te else None
    return te, ir


# ── quality flags ────────────────────────────────────────────────────────

def overlaps(issue_ranges, start, end) -> bool:
    """Does any (start_date, end_date) issue range intersect [start, end]?"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return any(pd.Timestamp(a) <= e and pd.Timestamp(b) >= s for a, b in issue_ranges)
