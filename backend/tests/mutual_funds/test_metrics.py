"""S3 metrics — pure functions against hand-computed fixtures (no DB)."""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from apps.mutual_funds.services import data_quality as dq
from apps.mutual_funds.services import metrics as m
from apps.mutual_funds.services.metrics_runner import category_median_rolling, category_metrics, scheme_metrics


def S(pairs):
    return m.to_series([(d if isinstance(d, date) else date.fromisoformat(d), v) for d, v in pairs])


def monthly(values, start="2018-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series([float(v) for v in values], index=idx)


# ── returns ──────────────────────────────────────────────────────────────

def test_cagr_two_points_hand_computed():
    nav = S([("2020-01-01", 100), ("2021-03-01", 110), ("2022-01-01", 121)])
    # 731 days = 2.001369 years; 1.21 ** (1/2.001369) - 1 = 0.099928
    assert m.cagr(nav, 2) == pytest.approx(0.099928, abs=1e-6)
    assert m.cagr(nav, 3) is None                      # not enough history -> None, never a guess
    assert m.cagr_since_inception(nav) == pytest.approx(0.099928, abs=1e-6)


def test_cagr_start_tolerance():
    # Start NAV 5 days before the window start is accepted; 10 days is not.
    end = date(2022, 1, 1)
    target = end - timedelta(days=365)
    ok = S([(target - timedelta(days=5), 100), (end, 110)])
    assert m.cagr(ok, 1) == pytest.approx(1.10 ** (365.25 / 370) - 1, abs=1e-9)
    bad = S([(target - timedelta(days=10), 100), (end, 110)])
    assert m.cagr(bad, 1) is None


def test_to_series_drops_nonpositive_and_duplicates():
    s = m.to_series([(date(2020, 1, 2), 10), (date(2020, 1, 1), 9), (date(2020, 1, 2), 11), (date(2020, 1, 3), 0)])
    assert list(s.values) == [9.0, 10.0]


def test_rolling_cagr_constant_monthly_growth():
    nav = monthly([100 * 1.01 ** i for i in range(38)])   # 38 month-ends -> 2 three-year windows
    roll = m.rolling_cagr(nav, 3)
    assert len(roll) == 2
    # (1.01^36)^(1/3) - 1 = 1.01^12 - 1 = 0.126825
    assert roll.iloc[0] == pytest.approx(1.01 ** 12 - 1, abs=1e-9)
    st = m.rolling_stats(roll)
    assert st.windows == 2 and st.pct_positive == 100.0
    assert st.minimum == pytest.approx(0.126825, abs=1e-6)


def test_rolling_cagr_skips_missing_endpoint_month():
    nav = monthly([100 * 1.01 ** i for i in range(38)]).drop(pd.Timestamp("2018-01-31"))
    assert len(m.rolling_cagr(nav, 3)) == 1


def test_rolling_return_12m():
    nav = monthly([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 90])
    r = m.rolling_return(nav, 12)
    assert len(r) == 1 and r.iloc[0] == pytest.approx(-0.10)


def test_pct_windows_above_and_category_median():
    idx = pd.date_range("2021-01-31", periods=4, freq="ME")
    a = pd.Series([0.10, 0.20, 0.30, 0.40], index=idx)
    b = pd.Series([0.20, 0.20, 0.20, 0.20], index=idx)
    c = pd.Series([0.00, 0.30, 0.10, 0.50], index=idx)
    med = category_median_rolling([a, b, c])
    assert list(med.round(6)) == [0.10, 0.20, 0.20, 0.40]
    # a vs median: 0.1>0.1 no, 0.2>0.2 no, 0.3>0.2 yes, 0.4>0.4 no -> 25 %
    assert m.pct_windows_above(a, med) == pytest.approx(25.0)
    # fewer than 3 funds on a date -> no median that date
    assert category_median_rolling([a, b]).empty


# ── risk ─────────────────────────────────────────────────────────────────

def test_volatility_sharpe_sortino_hand_computed():
    r = pd.Series([0.02, -0.01] * 15)
    # mean 0.005; sample sd = 0.015 * sqrt(30/29) = 0.0152564
    sd = 0.015 * math.sqrt(30 / 29)
    assert m.annualised_volatility(r, 252) == pytest.approx(sd * math.sqrt(252), rel=1e-9)
    assert m.sharpe(r, 252, 0.0) == pytest.approx(0.005 / sd * math.sqrt(252), rel=1e-9)
    # downside dev (MAR 0): sqrt(mean([0, 0.0001]*15)) = sqrt(0.00005) = 0.0070711, annualised x sqrt(252)
    dd = math.sqrt(0.00005) * math.sqrt(252)
    assert m.downside_deviation(r, 252, 0.0) == pytest.approx(dd, rel=1e-9)
    assert m.sortino(r, 252, 0.0) == pytest.approx(0.005 * 252 / dd, rel=1e-9)   # = 11.2249
    assert m.sortino(r, 252, 0.0) == pytest.approx(11.2249, abs=1e-3)


def test_sharpe_risk_free_lowers_ratio():
    r = pd.Series([0.001, 0.002] * 20)
    assert m.sharpe(r, 252, 0.05) < m.sharpe(r, 252, 0.0)


def test_risk_functions_need_enough_data():
    r = pd.Series([0.01] * 5)
    assert m.annualised_volatility(r, 252) is None
    assert m.sharpe(r, 252, 0.0) is None
    assert m.sortino(r, 252, 0.0) is None


def test_max_drawdown_peak_trough_recovery():
    nav = S([("2020-01-01", 100), ("2020-01-02", 120), ("2020-01-03", 90), ("2020-01-04", 100), ("2020-01-05", 130)])
    dd = m.max_drawdown(nav)
    assert dd.max_drawdown == pytest.approx(-0.25)          # 90/120 - 1
    assert dd.peak == pd.Timestamp("2020-01-02") and dd.trough == pd.Timestamp("2020-01-03")
    assert dd.duration_days == 1 and dd.recovery_days == 2
    assert m.calmar(0.10, dd.max_drawdown) == pytest.approx(0.4)


def test_max_drawdown_not_recovered_and_monotonic():
    dd = m.max_drawdown(S([("2020-01-01", 100), ("2020-01-02", 80), ("2020-01-03", 90)]))
    assert dd.max_drawdown == pytest.approx(-0.2) and dd.recovery is None and dd.recovery_days is None
    up = m.max_drawdown(S([("2020-01-01", 100), ("2020-01-02", 110)]))
    assert up.max_drawdown == 0.0 and m.calmar(0.1, up.max_drawdown) is None


def test_periods_per_year_inferred():
    idx = pd.date_range("2020-01-01", periods=366, freq="D")   # 365 days, 365 returns
    nav = pd.Series(range(1, 367), index=idx, dtype=float)
    assert m.periods_per_year(nav) == pytest.approx(365.25)


# ── relative ─────────────────────────────────────────────────────────────

def _bench_and_fund(mult):
    rets = [0.03, -0.02, 0.01, 0.04, -0.03, 0.02] * 5      # 30 months
    b, f = [100.0], [100.0]
    for r in rets:
        b.append(b[-1] * (1 + r))
        f.append(f[-1] * (1 + mult * r))
    return monthly(f), monthly(b)


def test_beta_capture_te_identical_series():
    f, b = _bench_and_fund(1.0)
    assert m.beta(f, b) == pytest.approx(1.0)
    up, dn = m.capture_ratios(f, b)
    assert up == pytest.approx(100.0) and dn == pytest.approx(100.0)
    te, ir = m.tracking_error_and_ir(f, b)
    assert te == pytest.approx(0.0, abs=1e-12) and ir is None


def test_beta_two_x_leverage():
    f, b = _bench_and_fund(2.0)
    assert m.beta(f, b) == pytest.approx(2.0)
    up, dn = m.capture_ratios(f, b)
    assert up > 190 and dn > 190


def test_relative_needs_24_months():
    f, b = _bench_and_fund(1.0)
    assert m.beta(f.iloc[:20], b.iloc[:20]) is None


# ── data quality ─────────────────────────────────────────────────────────

def test_find_issues_gap_jump_nonpositive():
    navs = [(date(2024, 1, 1), 10.0), (date(2024, 1, 2), 10.1),
            (date(2024, 1, 25), 10.2),            # Jan 3..24 = 16 weekdays missing -> gap
            (date(2024, 1, 26), 20.0),            # +96 % -> jump
            (date(2024, 1, 29), 0.0)]             # nonpositive
    kinds = [i.kind for i in dq.find_issues(navs)]
    assert kinds == ["gap", "jump", "nonpositive"]
    assert dq.weekdays_between(date(2024, 1, 2), date(2024, 1, 25)) == 16
    assert dq.weekdays_between(date(2024, 1, 5), date(2024, 1, 8)) == 0     # Fri -> Mon is no gap


def test_find_issues_clean_series():
    navs = [(date(2024, 1, 1) + timedelta(days=i), 10 + i * 0.01) for i in range(30)]
    assert dq.find_issues(navs) == []


def test_overlaps():
    assert m.overlaps([(date(2024, 1, 1), date(2024, 1, 5))], date(2024, 1, 4), date(2024, 2, 1))
    assert not m.overlaps([(date(2023, 1, 1), date(2023, 1, 5))], date(2024, 1, 1), date(2024, 2, 1))


# ── scheme_metrics / category_metrics ─────────────────────────────────────

def _daily(years, growth=0.12, start="2018-01-01"):
    idx = pd.bdate_range(start, periods=int(252 * years))
    t = (idx - idx[0]).days / 365.25
    wiggle = [1 + 0.004 * ((i % 7) - 3) for i in range(len(idx))]
    return pd.Series([(1 + growth) ** x * w * 10 for x, w in zip(t, wiggle)], index=idx)


def test_scheme_metrics_short_history_flags_and_nulls():
    nav = _daily(2)
    out = scheme_metrics(nav, nav.index[-1].date(), 0.05)
    assert "history_lt_3y" in out["quality_flags"] and "no_benchmark" in out["quality_flags"]
    assert out["cagr_3y"] is None and out.get("sharpe") is None and out["rolling_3y_windows"] == 0
    assert out["cagr_1y"] is not None


def test_scheme_metrics_full_history_and_point_in_time():
    nav = _daily(6)
    as_of = nav.index[-1].date()
    out = scheme_metrics(nav, as_of, 0.05)
    assert out["cagr_5y"] == pytest.approx(0.12, abs=0.01)
    assert out["sharpe"] is not None and out["max_drawdown"] < 0 and out["rolling_3y_windows"] > 30
    # truncating to an earlier as_of must ignore later data entirely
    early = m.upto(nav, nav.index[800])
    out_early = scheme_metrics(early, nav.index[800].date(), 0.05)
    assert out_early["last_nav_date"] == nav.index[800].date()


def test_scheme_metrics_flags_issue_in_window_and_stale():
    nav = _daily(4)
    last = nav.index[-1].date()
    out = scheme_metrics(nav, last + timedelta(days=30), 0.05,
                         issue_ranges=[(last - timedelta(days=100), last - timedelta(days=90))])
    assert "data_issue_in_3y_window" in out["quality_flags"] and "stale_nav" in out["quality_flags"]


def test_category_metrics_excess_vs_median():
    rows = {}
    for code, g in (("a", 0.08), ("b", 0.12), ("c", 0.16)):
        rows[code] = scheme_metrics(_daily(5, g), _daily(5, g).index[-1].date(), 0.05)
    category_metrics(rows)
    assert rows["b"]["excess_cagr_3y_vs_category"] == pytest.approx(0.0, abs=1e-9)
    assert rows["c"]["excess_cagr_3y_vs_category"] > 0 > rows["a"]["excess_cagr_3y_vs_category"]
    assert rows["c"]["rolling_3y_pct_above_cat_median"] == 100.0
    assert rows["a"]["rolling_3y_pct_above_cat_median"] == 0.0


def test_after_last_jump_cuts_history_point_in_time():
    from apps.mutual_funds.services.metrics_runner import after_last_jump

    nav = S([("2015-08-26", 28.5), ("2015-08-27", 28.56), ("2015-08-28", 2857.2), ("2015-08-31", 2857.9)])
    jumps = [(date(2015, 8, 27), date(2015, 8, 28))]
    cut, did = after_last_jump(nav, jumps)
    assert did and list(cut.values) == [2857.2, 2857.9]
    # A jump after as_of is not known yet at as_of.
    same, did2 = after_last_jump(m.upto(nav, "2015-08-27"), jumps, as_of=date(2015, 8, 27))
    assert not did2 and len(same) == 2
