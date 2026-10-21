"""MF-S pipeline: universe rules, mfapi backfill (network faked), metric
snapshots, screener scoring, index-vs-active, backtest, coverage command."""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from decimal import Decimal
from io import StringIO

import pandas as pd
import pytest
from django.core.cache import cache
from django.core.management import call_command

from apps.mutual_funds.models import (
    MFBackfillJob,
    MFBenchmarkNav,
    MFDataIssue,
    MFMetricSnapshot,
    MFNav,
    MFScheme,
)
from apps.mutual_funds.services import universe as U
from apps.mutual_funds.services.backtest import backtest_category, forward_return, summarise
from apps.mutual_funds.services.benchmarks import chunks, parse_tri_json
from apps.mutual_funds.services.metrics_runner import compute_snapshots
from apps.mutual_funds.services.mfapi_client import SchemeHistory, parse_scheme_json
from apps.mutual_funds.services.nav_backfill import BackfillStats, BackfillTarget, backfill_one, refresh_active_flags
from apps.mutual_funds.services.screener import ScreenCriteria, percentile_scores, score_frame, screen

FLEXI = "Open Ended Schemes(Equity Scheme - Flexi Cap Fund)"
FLEXI_NEW = "Open Ended Schemes(Equity Schemes - Flexi Cap Fund)"
INDEX = "Open Ended Schemes(Index Funds - Equity Funds)"


# ── universe / categories (pure) ─────────────────────────────────────────

def test_canonical_category_folds_amfi_spelling_variants():
    assert U.canonical_category(FLEXI) == U.canonical_category(FLEXI_NEW) == "Equity Scheme - Flexi Cap Fund"
    assert U.canonical_category("Open Ended Schemes(Equity Schemes - ELSS- Tax Saver Fund)") == "Equity Scheme - ELSS"
    assert (U.canonical_category("Open Ended Schemes(Income/Debt Oriented Schemes - Short Term Fund)")
            == "Debt Scheme - Short Duration Fund")
    assert U.canonical_category("Open Ended Schemes(Other Scheme - Index Funds)", "X Nifty 50 Index Fund") \
        == "Index Funds - Equity"
    assert U.canonical_category("Open Ended Schemes(Other Scheme - Index Funds)", "X CRISIL IBX Gilt Index") \
        == "Index Funds - Debt"
    assert U.canonical_category("") == "Uncategorised"


@pytest.mark.parametrize("plan,option,name,cat,ok", [
    ("Direct Plan", "Growth Option", "Acme Flexi Cap Fund", FLEXI, True),
    ("Direct Plan", "Growth", "Acme Flexi Cap Fund", FLEXI_NEW, True),
    ("Regular Plan", "Growth", "Acme Flexi Cap Fund", FLEXI, False),
    ("Direct Plan", "IDCW Option", "Acme Flexi Cap Fund", FLEXI, False),
    ("Direct Plan", "Growth", "Acme Flexi Cap Fund - Segregated Portfolio 1", FLEXI, False),
    ("Direct Plan", "Growth", "Acme FMP Series 12", "Close Ended Schemes(Income)", False),
    ("", "", "Acme Liquid Fund - Direct Plan - Growth", "Open Ended Schemes(Debt Scheme - Liquid Fund)", True),
    ("", "", "Acme Nifty ETF", "Open Ended Schemes(Exchange Traded Funds (ETFs) - Equity ETF)", False),
])
def test_static_universe_rules(plan, option, name, cat, ok):
    assert U.passes_static_rules(plan=plan, option=option, scheme_name=name, category=cat) is ok


def test_name_prefilter_for_dead_schemes():
    assert U.looks_like_direct_growth_name("DWS Equity Income Fund - Direct Plan - Growth")
    assert not U.looks_like_direct_growth_name("DWS Equity Income Fund - Direct Plan - IDCW")
    assert not U.looks_like_direct_growth_name("Indiabulls Liquid Fund- Direct Plan- Growth- Unclaimed Redemption")


# ── sources (pure) ───────────────────────────────────────────────────────

def test_parse_mfapi_json():
    h = parse_scheme_json("118989", {
        "meta": {"fund_house": "HDFC Mutual Fund", "scheme_type": "Open Ended Schemes",
                 "scheme_category": "Equity Scheme - Mid Cap Fund", "scheme_name": "HDFC Mid Cap - Direct - Growth",
                 "isin_growth": "INF179K01XQ0", "isin_div_reinvestment": None},
        "data": [{"date": "19-10-2026", "nav": "219.78400"}, {"date": "01-01-2013", "nav": "18.79900"},
                 {"date": "bad", "nav": "1"}, {"date": "02-01-2013", "nav": "N.A."}],
    })
    assert h.found and h.amfi_category == "Open Ended Schemes(Equity Scheme - Mid Cap Fund)"
    assert h.navs == [(date(2013, 1, 1), Decimal("18.79900")), (date(2026, 10, 19), Decimal("219.78400"))]
    assert h.bad_rows == 2
    empty = parse_scheme_json("999999", {"meta": {"scheme_code": 0, "scheme_name": ""}, "data": []})
    assert not empty.found


def test_parse_tri_json_and_chunks():
    rows = parse_tri_json([{"Date": "31 Jan 2024", "TotalReturnsIndex": "31,939.59"},
                           {"Date": "30 Jan 2024", "TotalReturnsIndex": "31640.24"},
                           {"Date": "29 Jan 2024", "TotalReturnsIndex": "-"}])
    assert rows == [(date(2024, 1, 30), Decimal("31640.24")), (date(2024, 1, 31), Decimal("31939.59"))]
    assert parse_tri_json({"d": json.dumps([{"Date": "01 Jan 2024", "TotalReturnsIndex": "1"}])})[0][1] == 1
    cs = chunks(date(2020, 1, 1), date(2022, 1, 10))
    assert cs[0] == (date(2020, 1, 1), date(2020, 12, 30)) and cs[-1][1] == date(2022, 1, 10)
    assert all((b - a).days <= 364 for a, b in cs)


# ── backfill (DB, network faked) ─────────────────────────────────────────

class FakeClient:
    def __init__(self, histories: dict[str, SchemeHistory | Exception]):
        self.h = histories
        self.calls = []

    def scheme_history(self, code, start=None, end=None):
        self.calls.append(code)
        v = self.h[code]
        if isinstance(v, Exception):
            raise v
        return v


def _hist(code, name, navs, cat="Equity Scheme - Flexi Cap Fund", typ="Open Ended Schemes"):
    return SchemeHistory(scheme_code=code, fund_house="Acme Mutual Fund", scheme_type=typ, scheme_category=cat,
                         scheme_name=name, navs=[(d, Decimal(str(v))) for d, v in navs], raw_bytes=100)


@pytest.mark.django_db
def test_backfill_inserts_resumes_and_records_issues():
    MFScheme.objects.create(scheme_code="1", amc_name="Acme", scheme_name="Acme Flexi", plan="Direct Plan",
                            option="Growth", category=FLEXI, is_direct=True)
    MFNav.objects.create(scheme_id="1", nav_date=date(2024, 1, 3), nav=Decimal("10.3"))   # AMFI daily row
    navs = [(date(2024, 1, 1), 10.0), (date(2024, 1, 2), 10.1), (date(2024, 1, 3), 10.2),   # 10.2 != AMFI 10.3
            (date(2024, 2, 1), 10.4), (date(2024, 2, 2), 20.0)]                              # gap + jump
    client = FakeClient({"1": _hist("1", "Acme Flexi - Direct - Growth", navs)})
    stats = BackfillStats()
    assert backfill_one(client, BackfillTarget("1", True), date(2020, 1, 1), stats=stats) == "done"
    assert MFNav.objects.filter(scheme_id="1").count() == 5
    assert MFNav.objects.get(scheme_id="1", nav_date=date(2024, 1, 3)).nav == Decimal("10.3")   # AMFI kept
    kinds = set(MFDataIssue.objects.filter(scheme_id="1").values_list("kind", flat=True))
    assert kinds == {"gap", "jump", "source_mismatch"}
    job = MFBackfillJob.objects.get(source="mfapi", key="1")
    assert job.rows_inserted == 4 and job.first_nav_date == date(2024, 1, 1)
    # Resumable: a second run at the same depth skips without an HTTP call.
    assert backfill_one(client, BackfillTarget("1", True), date(2020, 1, 1), stats=stats) == "skipped"
    assert client.calls == ["1"]
    # Deeper request re-fetches.
    assert backfill_one(client, BackfillTarget("1", True), date(2015, 1, 1), stats=stats) == "done"


@pytest.mark.django_db
def test_backfill_dead_scheme_created_inactive_and_out_of_universe_skipped():
    navs = [(date(2015, 1, 1) + timedelta(days=i), 10 + i * 0.01) for i in range(10)]
    client = FakeClient({
        "900": _hist("900", "Old Fund - Direct Plan - Growth", navs),
        "901": _hist("901", "Old FMP - Direct - Growth", navs, typ="Close Ended Schemes", cat="Income"),
        "902": RuntimeError("boom"),
    })
    stats = BackfillStats()
    assert backfill_one(client, BackfillTarget("900", False), date(2010, 1, 1), stats=stats) == "done"
    s = MFScheme.objects.get(scheme_code="900")
    assert s.is_active is False and s.catalog_source == "mfapi" and s.category == FLEXI
    assert backfill_one(client, BackfillTarget("901", False), date(2010, 1, 1), stats=stats) == "empty"
    assert not MFScheme.objects.filter(scheme_code="901").exists()
    assert backfill_one(client, BackfillTarget("902", False), date(2010, 1, 1), stats=stats) == "failed"
    assert stats.failed == 1 and stats.new_dead_schemes == 1 and stats.out_of_universe == 1
    assert "boom" in MFBackfillJob.objects.get(key="902").error


@pytest.mark.django_db
def test_refresh_active_flags():
    for code, last in (("a", date(2026, 10, 16)), ("b", date(2026, 1, 1))):
        MFScheme.objects.create(scheme_code=code, amc_name="A", scheme_name=code, category=FLEXI)
        MFNav.objects.create(scheme_id=code, nav_date=last, nav=Decimal("1"))
    MFNav.objects.create(scheme_id="a", nav_date=date(2026, 10, 15), nav=Decimal("1"))
    assert refresh_active_flags(date(2026, 10, 16)) == (1, 1)
    assert MFScheme.objects.get(scheme_code="b").is_active is False


# ── synthetic category for metrics / screener / backtest ─────────────────

def _make_fund(code, name, growth, years=7, cat=FLEXI, active=True, end=date(2026, 10, 16), noise=0.006, phase=0):
    MFScheme.objects.create(scheme_code=code, amc_name=f"{name} MF", scheme_name=name, plan="Direct Plan",
                            option="Growth", category=cat, is_direct=True, is_active=active)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=int(252 * years))
    rows = []
    for i, d in enumerate(idx):
        t = (d - idx[0]).days / 365.25
        v = 10 * (1 + growth) ** t * (1 + noise * math.sin(i / 9 + phase))
        rows.append(MFNav(scheme_id=code, nav_date=d.date(), nav=Decimal(f"{v:.4f}")))
    MFNav.objects.bulk_create(rows, batch_size=2000)


@pytest.fixture
def flexi_world(db):
    cache.clear()
    for i, g in enumerate([0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.11, 0.13, 0.15]):
        _make_fund(f"F{i}", f"Fund {i} Flexi Cap - Direct - Growth", g, phase=i)
    _make_fund("YOUNG", "Young Flexi - Direct - Growth", 0.30, years=1.5)
    _make_fund("DEAD", "Dead Flexi - Direct - Growth", 0.02, years=4, active=False, end=date(2023, 6, 30))
    _make_fund("IDX", "Acme Nifty 500 Index Fund - Direct - Growth", 0.125, cat=INDEX)
    idx = pd.bdate_range(end=pd.Timestamp(date(2026, 10, 16)), periods=252 * 7)
    MFBenchmarkNav.objects.bulk_create([
        MFBenchmarkNav(index_name="NIFTY 500", nav_date=d.date(),
                       value=Decimal(f"{1000 * 1.13 ** ((d - idx[0]).days / 365.25):.4f}")) for d in idx])
    return date(2026, 10, 16)


@pytest.mark.django_db
def test_compute_snapshots_and_screen(flexi_world):
    summary = compute_snapshots(flexi_world)
    assert summary["dead_in_medians"] == 1
    snaps = MFMetricSnapshot.objects.filter(category="Equity Scheme - Flexi Cap Fund")
    assert snaps.count() == 10                                   # 9 + YOUNG; DEAD only in medians
    f5 = snaps.get(scheme_id="F5")
    assert f5.cagr_5y == pytest.approx(0.18, abs=0.01) and f5.benchmark_name == "NIFTY 500"
    assert f5.beta is not None and f5.excess_cagr_3y_vs_benchmark > 0
    assert "history_lt_3y" in snaps.get(scheme_id="YOUNG").quality_flags

    res = screen(ScreenCriteria(categories=["Equity Scheme - Flexi Cap Fund"], top=3))
    cat = res["categories"][0]
    assert cat["excluded"] == {"history < 3.0y": 1} and cat["eligible_count"] == 9
    assert [r["scheme_code"] for r in cat["rows"]] == ["F5", "F4", "F8"]       # highest growth first
    top = cat["rows"][0]
    assert set(top["components"]) == {"consistency", "risk_adjusted", "drawdown", "cost", "size_age"}
    assert top["components"]["cost"] is None                     # TER not sourced -> n/a, not imputed
    assert "cost" not in cat["weights_used"] and sum(cat["weights_used"].values()) == pytest.approx(100, abs=0.05)
    assert any("TER" in n for n in cat["notes"])
    ic = cat["index_check"]
    assert ic["basis"] == "index_fund" and ic["index_funds_found"] == 1 and ic["verdict"] in ("yes", "no", "mixed")


@pytest.mark.django_db
def test_screen_filters(flexi_world):
    compute_snapshots(flexi_world)
    res = screen(ScreenCriteria(categories=["Equity Scheme - Flexi Cap Fund"], min_sharpe=100), with_index_check=False)
    assert res["categories"][0]["eligible_count"] == 0
    assert res["categories"][0]["excluded"]["Sharpe below min or n/a"] == 9


def test_percentile_scores_and_renormalisation():
    s = percentile_scores(pd.Series([1.0, 2.0, 3.0, None]))
    assert list(s.iloc[:3]) == [0.0, 50.0, 100.0] and pd.isna(s.iloc[3])
    assert list(percentile_scores(pd.Series([1.0, 2.0]), higher_is_better=False)) == [100.0, 0.0]
    assert list(percentile_scores(pd.Series([5.0]))) == [50.0]
    df = pd.DataFrame({"rolling_3y_pct_above_cat_median": [10, 90], "rolling_3y_min": [0.0, 0.1],
                       "sortino": [1, 2], "sharpe": [1, 2], "max_drawdown": [-0.3, -0.1],
                       "worst_1y": [-0.2, -0.1], "history_years": [5, 6], "rolling_3y_windows": [60, 60]})
    out, used, notes = score_frame(df, {"consistency": 30, "risk_adjusted": 25, "drawdown": 20, "cost": 15,
                                        "size_age": 10})
    assert "cost" not in used and sum(used.values()) == pytest.approx(100, abs=0.05)
    assert list(out["score"]) == pytest.approx([0.0, 100.0])


@pytest.mark.django_db
def test_backtest_point_in_time_and_dead_chaining(flexi_world):
    # Persistent growth differences -> the composite must rank them in order.
    res = backtest_category("Equity Scheme - Flexi Cap Fund", 12, date(2022, 3, 31), date(2026, 10, 16))
    assert res, "expected backtest dates"
    s = summarise(res)
    v = s["variants"]["trailing_3y_cagr"]
    assert v["top_mean"] > v["bottom_mean"] and v["ic_mean"] > 0.5
    assert any(r.dead_chained for r in res)          # DEAD died inside a horizon and was kept
    assert s["passive_basis"] == ["index_fund"]


def test_forward_return_alive_and_dead():
    idx = pd.bdate_range("2020-01-01", "2021-06-30")
    nav = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
    r, died = forward_return(nav, pd.Timestamp("2020-03-31"), 12)
    assert died is None and r == pytest.approx(nav.loc[:"2021-03-31"].iloc[-1] / nav.loc[:"2020-03-31"].iloc[-1] - 1)
    r2, died2 = forward_return(nav, pd.Timestamp("2020-12-31"), 12)
    assert died2 == idx[-1] and r2 > 0
    assert forward_return(nav, pd.Timestamp("2019-06-30"), 12) == (None, None)


@pytest.mark.django_db
def test_coverage_and_screen_commands(flexi_world):
    out = StringIO()
    call_command("compute_mf_metrics", as_of="2026-10-16", stdout=out)
    assert "snapshots written" in out.getvalue()
    out = StringIO()
    call_command("screen_mf", category=["Equity Scheme - Flexi Cap Fund"], top=3, stdout=out)
    text = out.getvalue()
    assert "F5" in text and "index-vs-active" in text and "n/a" in text
    out = StringIO()
    call_command("screen_mf", category=["Equity Scheme - Flexi Cap Fund"], top=2, json=True, no_index_check=True,
                 stdout=out)
    assert len(json.loads(out.getvalue())["categories"][0]["rows"]) == 2
    out = StringIO()
    call_command("mf_data_coverage", by_category=True, stdout=out)
    assert "live_universe" in out.getvalue() and "Flexi Cap" in out.getvalue()


def test_consistency_shrinks_thin_evidence_toward_50():
    df = pd.DataFrame({"rolling_3y_pct_above_cat_median": [100.0, 80.0], "rolling_3y_windows": [6, 96],
                       "rolling_3y_min": [0.1, 0.1]})
    out, _, _ = score_frame(df, {"consistency": 1})
    # young: (100*6 + 50*24)/30 = 60 ; seasoned: (80*96 + 50*24)/120 = 74 -> seasoned ranks higher
    assert list(out["consistency_pct_above_shrunk"]) == pytest.approx([60.0, 74.0])
    assert out["score"].iloc[1] > out["score"].iloc[0]
