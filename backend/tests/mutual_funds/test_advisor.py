"""MF advisor tests — fixtures only; there is no real holdings data."""
from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.events.models import Event
from apps.mutual_funds.models import MFHolding, MFNav, MFScheme, MFTransaction
from apps.mutual_funds.services import advisor
from tests.factories import PortfolioFactory, TenantFactory

TODAY = date(2026, 10, 15)
LARGE = "Open Ended Schemes(Equity Scheme - Large Cap Fund)"
FLEXI = "Open Ended Schemes(Equity Scheme - Flexi Cap Fund)"


def scheme(code, name, plan, option="Growth Option", amc="Acme Mutual Fund", category=LARGE):
    return MFScheme.objects.create(
        scheme_code=code, amc_name=amc, scheme_name=name, plan=plan, option=option,
        category=category, is_direct="direct" in plan.lower(),
    )


def nav(s, d, value):
    return MFNav.objects.create(scheme=s, nav_date=d, nav=Decimal(str(value)))


def holding(tenant, portfolio, s, units, folio="F1"):
    return MFHolding.objects.create(
        tenant=tenant, portfolio=portfolio, scheme=s, folio_number=folio, units=Decimal(str(units)),
    )


@pytest.fixture
def pf(db):
    t = TenantFactory()
    return t, PortfolioFactory(tenant=t)


# ── helpers ──────────────────────────────────────────────────────────────

def test_normalize_strips_plan_and_option_words():
    a = advisor.normalize_scheme_name("Acme Bluechip Fund - Direct Plan - Growth Option")
    b = advisor.normalize_scheme_name("Acme Bluechip Fund-Regular Growth")
    assert a == b == "acme bluechip fund"


def test_category_label_extracts_parenthesised_part():
    s = MFScheme(category=LARGE)
    assert advisor.category_label(s) == "Equity Scheme - Large Cap Fund"


# ── 1. direct vs regular ─────────────────────────────────────────────────

def test_regular_holding_flagged_with_twin_and_labelled_estimate(pf):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan")
    scheme("2", "Acme Bluechip Fund", "Direct Plan")
    nav(reg, TODAY, 100)
    holding(t, p, reg, 100)  # value 10,000

    rep = advisor.run_advisor(t, p, today=TODAY, emit_events=False, assumed_drag_pct=1.0)
    f = rep.by_kind("direct_vs_regular")
    assert len(f) == 1
    assert f[0].payload["direct_scheme_code"] == "2"
    assert f[0].payload["estimated_annual_drag_inr"] == 100.0
    assert f[0].payload["is_estimate"] is True
    assert "ESTIMATED" in f[0].text and "ASSUMED" in f[0].text


def test_regular_without_twin_still_flagged_but_no_twin(pf):
    t, p = pf
    reg = scheme("1", "Acme Solo Fund", "Regular Plan")
    nav(reg, TODAY, 50)
    holding(t, p, reg, 10)
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("direct_vs_regular")
    assert len(f) == 1 and f[0].payload["direct_scheme_code"] is None


def test_direct_holding_not_flagged(pf):
    t, p = pf
    d = scheme("2", "Acme Bluechip Fund", "Direct Plan")
    nav(d, TODAY, 100)
    holding(t, p, d, 10)
    assert advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("direct_vs_regular") == []


def test_twin_must_match_option_growth_vs_idcw(pf):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan", option="Growth Option")
    scheme("2", "Acme Bluechip Fund", "Direct Plan", option="IDCW Option")  # wrong option
    nav(reg, TODAY, 100)
    holding(t, p, reg, 1)
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("direct_vs_regular")
    assert f[0].payload["direct_scheme_code"] is None


def test_twin_must_match_amc(pf):
    t, p = pf
    reg = scheme("1", "Bluechip Fund", "Regular Plan", amc="A Mutual Fund")
    scheme("2", "Bluechip Fund", "Direct Plan", amc="B Mutual Fund")
    nav(reg, TODAY, 100)
    holding(t, p, reg, 1)
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("direct_vs_regular")
    assert f[0].payload["direct_scheme_code"] is None


def test_observed_gap_from_nav_history(pf):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan")
    d = scheme("2", "Acme Bluechip Fund", "Direct Plan")
    start = TODAY - timedelta(days=365)
    nav(reg, start, 100); nav(reg, TODAY, 110)   # +10%
    nav(d, start, 100); nav(d, TODAY, 111)       # +11%
    holding(t, p, reg, 10)
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("direct_vs_regular")[0]
    assert f.payload["observed_growth_gap_pct_pa"] == pytest.approx(1.0, abs=0.05)


def test_observed_gap_none_when_history_short(pf):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan")
    d = scheme("2", "Acme Bluechip Fund", "Direct Plan")
    for s in (reg, d):
        nav(s, TODAY - timedelta(days=30), 100); nav(s, TODAY, 101)
    holding(t, p, reg, 10)
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("direct_vs_regular")[0]
    assert f.payload["observed_growth_gap_pct_pa"] is None


# ── 2. allocation ────────────────────────────────────────────────────────

def test_allocation_drift_arithmetic(pf):
    t, p = pf
    a = scheme("1", "Large A", "Direct Plan", category=LARGE)
    b = scheme("2", "Flexi B", "Direct Plan", category=FLEXI, amc="Other Mutual Fund")
    nav(a, TODAY, 100); nav(b, TODAY, 100)
    holding(t, p, a, 70)   # 7,000
    holding(t, p, b, 30)   # 3,000
    target = {"Equity Scheme - Large Cap Fund": 50, "Equity Scheme - Flexi Cap Fund": 50}
    rep = advisor.run_advisor(t, p, target=target, today=TODAY, emit_events=False)
    rows = {f.payload["category"]: f for f in rep.by_kind("allocation") if "category" in f.payload}
    assert rows["Equity Scheme - Large Cap Fund"].payload["actual_pct"] == 70.0
    assert rows["Equity Scheme - Large Cap Fund"].payload["drift_pts"] == 20.0
    assert rows["Equity Scheme - Flexi Cap Fund"].payload["drift_pts"] == -20.0
    assert rows["Equity Scheme - Large Cap Fund"].payload["flagged"] is True


def test_allocation_untargeted_and_missing_categories(pf):
    t, p = pf
    a = scheme("1", "Large A", "Direct Plan", category=LARGE)
    nav(a, TODAY, 100)
    holding(t, p, a, 10)
    target = {"Debt Scheme - Liquid Fund": 100}
    rows = {f.payload["category"]: f.payload for f in
            advisor.run_advisor(t, p, target=target, today=TODAY, emit_events=False).by_kind("allocation")
            if "category" in f.payload}
    assert rows["Equity Scheme - Large Cap Fund"]["target_pct"] == 0.0
    assert rows["Equity Scheme - Large Cap Fund"]["actual_pct"] == 100.0
    assert rows["Debt Scheme - Liquid Fund"]["actual_pct"] == 0.0
    assert rows["Debt Scheme - Liquid Fund"]["drift_pts"] == -100.0


def test_allocation_warns_when_target_not_100(pf):
    t, p = pf
    a = scheme("1", "Large A", "Direct Plan")
    nav(a, TODAY, 100); holding(t, p, a, 1)
    f = advisor.run_advisor(t, p, target={"Equity Scheme - Large Cap Fund": 60}, today=TODAY,
                            emit_events=False).by_kind("allocation")
    assert any("sum to 60" in x.text for x in f)


def test_no_target_skips_allocation_with_note(pf):
    t, p = pf
    a = scheme("1", "Large A", "Direct Plan")
    nav(a, TODAY, 100); holding(t, p, a, 1)
    rep = advisor.run_advisor(t, p, today=TODAY, emit_events=False)
    assert rep.by_kind("allocation") == []
    assert any("No target" in n for n in rep.notes)


def test_holding_without_nav_is_unvalued_not_zero(pf):
    t, p = pf
    a = scheme("1", "Large A", "Direct Plan")
    holding(t, p, a, 100)
    rep = advisor.run_advisor(t, p, target={"Equity Scheme - Large Cap Fund": 100}, today=TODAY,
                              emit_events=False)
    assert rep.total_value == 0 and len(rep.unvalued) == 1


# ── 3. overlap ───────────────────────────────────────────────────────────

def test_same_scheme_in_two_plans_flagged(pf):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan")
    d = scheme("2", "Acme Bluechip Fund", "Direct Plan")
    nav(reg, TODAY, 10); nav(d, TODAY, 10)
    holding(t, p, reg, 1, "F1"); holding(t, p, d, 1, "F2")
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("overlap")
    assert any(x.payload.get("level") == "same_scheme" for x in f)


def test_same_amc_same_category_flagged(pf):
    t, p = pf
    a = scheme("1", "Acme Bluechip Fund", "Direct Plan")
    b = scheme("2", "Acme Top 100 Fund", "Direct Plan")
    nav(a, TODAY, 10); nav(b, TODAY, 10)
    holding(t, p, a, 1); holding(t, p, b, 1, "F2")
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("overlap")
    assert any(x.payload.get("level") == "category_amc" for x in f)


def test_different_amc_same_category_not_flagged_and_open_question_always_present(pf):
    t, p = pf
    a = scheme("1", "Fund One", "Direct Plan", amc="A Mutual Fund")
    b = scheme("2", "Fund Two", "Direct Plan", amc="B Mutual Fund")
    nav(a, TODAY, 10); nav(b, TODAY, 10)
    holding(t, p, a, 1); holding(t, p, b, 1, "F2")
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("overlap")
    assert not any(x.payload.get("level") in ("same_scheme", "category_amc") for x in f)
    oq = [x for x in f if x.payload.get("open_question")]
    assert len(oq) == 1 and "NOT computed" in oq[0].text


# ── 4. SIP continuity ────────────────────────────────────────────────────

def _sips(h, tenant, dates):
    for d in dates:
        MFTransaction.objects.create(tenant=tenant, holding=h, txn_type="sip", txn_date=d,
                                     units=Decimal("1"), nav=Decimal("10"), amount=Decimal("10"))


def test_lapsed_monthly_sip_flagged(pf):
    t, p = pf
    s = scheme("1", "Fund One", "Direct Plan"); nav(s, TODAY, 10)
    h = holding(t, p, s, 5)
    _sips(h, t, [date(2026, 3, 5), date(2026, 4, 5), date(2026, 5, 5), date(2026, 6, 5)])
    f = advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("sip_continuity")
    assert len(f) == 1 and f[0].payload["cadence"] == "monthly"
    assert f[0].payload["last_sip"] == "2026-06-05"


def test_active_sip_not_flagged(pf):
    t, p = pf
    s = scheme("1", "Fund One", "Direct Plan"); nav(s, TODAY, 10)
    h = holding(t, p, s, 5)
    _sips(h, t, [date(2026, 7, 5), date(2026, 8, 5), date(2026, 9, 5), date(2026, 10, 5)])
    assert advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("sip_continuity") == []


def test_too_little_sip_history_not_flagged(pf):
    t, p = pf
    s = scheme("1", "Fund One", "Direct Plan"); nav(s, TODAY, 10)
    h = holding(t, p, s, 5)
    _sips(h, t, [date(2026, 1, 5), date(2026, 2, 5)])
    assert advisor.run_advisor(t, p, today=TODAY, emit_events=False).by_kind("sip_continuity") == []


# ── events + command ─────────────────────────────────────────────────────

def test_findings_are_emitted_as_events(pf):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan")
    nav(reg, TODAY, 100); holding(t, p, reg, 10)
    rep = advisor.run_advisor(t, p, today=TODAY, emit_events=True)
    evs = Event.objects.filter(tenant=t, step_name__startswith="mf_advisor.")
    assert evs.count() == len(rep.findings) > 0
    drv = evs.get(step_name="mf_advisor.direct_vs_regular")
    assert drv.type == Event.Type.WORKFLOW_STEP_COMPLETED
    assert drv.payload["is_estimate"] is True


def test_emit_failure_never_stops_the_run(pf, monkeypatch):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan")
    nav(reg, TODAY, 100); holding(t, p, reg, 10)

    def boom(**kw):
        raise RuntimeError("event store down")

    monkeypatch.setattr("apps.events.services.event_writer.emit", boom)
    rep = advisor.run_advisor(t, p, today=TODAY, emit_events=True)
    assert len(rep.by_kind("direct_vs_regular")) == 1


def test_empty_portfolio_reports_note(pf):
    t, p = pf
    rep = advisor.run_advisor(t, p, today=TODAY, emit_events=False)
    assert rep.findings == [] and rep.notes


def test_holdings_are_tenant_scoped(pf):
    t, p = pf
    other_t = TenantFactory()
    other_p = PortfolioFactory(tenant=other_t)
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan"); nav(reg, TODAY, 100)
    holding(other_t, other_p, reg, 10)
    rep = advisor.run_advisor(t, None, today=TODAY, emit_events=False)
    assert rep.findings == []


def test_command_runs_end_to_end(pf, tmp_path, capsys):
    t, p = pf
    reg = scheme("1", "Acme Bluechip Fund", "Regular Plan"); nav(reg, date.today(), 100)
    holding(t, p, reg, 10)
    tf = tmp_path / "target.json"
    tf.write_text(json.dumps({"Equity Scheme - Large Cap Fund": 100}))
    call_command("run_mf_advisor", tenant=str(t.id), portfolio=str(p.id), target_file=str(tf),
                 assumed_drag_pct=0.5, no_events=True)
    out = capsys.readouterr().out
    assert "Direct vs Regular" in out and "ESTIMATED" in out
    assert "Category allocation" in out and "NOT computed" in out


def test_command_rejects_unknown_tenant_and_bad_target(pf, tmp_path):
    t, _ = pf
    with pytest.raises(CommandError):
        call_command("run_mf_advisor", tenant="does-not-exist")
    bad = tmp_path / "bad.json"
    bad.write_text("[1,2]")
    with pytest.raises(CommandError):
        call_command("run_mf_advisor", tenant=str(t.id), target_file=str(bad))
