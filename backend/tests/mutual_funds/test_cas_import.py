"""CAS import + command tests — synthetic data only (no real CAMS/KFin PDF)."""
from __future__ import annotations

from decimal import Decimal
from io import StringIO
from unittest.mock import patch

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.mutual_funds.models import MFHolding, MFScheme, MFTransaction
from apps.mutual_funds.services import cas_import
from apps.mutual_funds.services.cas_import import import_cas, match_scheme, normalize_name
from apps.mutual_funds.services.cas_parser import from_cas_data
from tests.factories import PortfolioFactory
from tests.mutual_funds.cas_fixtures import AXIS, make_cas_dict, seed_catalog


@pytest.fixture
def catalog(db):
    return seed_catalog()


@pytest.fixture
def pf(tenant):
    return PortfolioFactory(tenant=tenant)


def _run(tenant, pf, d=None, **kw):
    return import_cas(from_cas_data(d or make_cas_dict()), tenant=tenant, portfolio=pf, **kw)


# ── matching ───────────────────────────────────────────────────────────

def test_isin_match_prefers_exact_plan_over_decoy(catalog):
    ps = from_cas_data(make_cas_dict()).schemes[0]
    scheme, how = match_scheme(ps)
    assert scheme.scheme_code == "120503" and how == "isin"  # not the Regular-plan decoy


def test_dividend_reinvestment_isin_also_matches(catalog):
    ps = from_cas_data(make_cas_dict()).schemes[0]
    ps.isin = "INF846K01DQ6"
    assert match_scheme(ps)[0].scheme_code == "120503"


def test_unknown_isin_does_not_fall_back_to_name(catalog):
    ps = from_cas_data(make_cas_dict()).schemes[0]
    ps.isin, ps.amfi = "INF000000000", None  # name would match exactly, but an ISIN was printed
    assert match_scheme(ps) == (None, "isin_not_in_catalog")


def test_amfi_code_used_when_isin_not_in_catalog(catalog):
    ps = from_cas_data(make_cas_dict()).schemes[0]
    ps.isin = "INF000000000"  # amfi stays 120503
    scheme, how = match_scheme(ps)
    assert scheme.scheme_code == "120503" and how == "amfi_code"


def test_name_fallback_only_without_isin_and_is_exact(catalog):
    ps = from_cas_data(make_cas_dict()).schemes[0]
    ps.isin, ps.amfi = None, None
    scheme, how = match_scheme(ps)
    # RTA prefix, "(formerly …)" and "Option" suffix noise must not defeat the match,
    # and it must pick the Direct plan, not the Regular decoy.
    assert (scheme.scheme_code, how) == ("120503", "name")

    ps.name = "Axis Bluechip Fund"  # too loose -> no match, never a guess
    assert match_scheme(ps) == (None, "name_not_found")


def test_name_fallback_ambiguity_is_refused(catalog):
    MFScheme.objects.create(
        scheme_code="999999", amc_name="Axis Mutual Fund", plan="Direct Plan",
        scheme_name="Axis Bluechip Fund - Direct Plan - Growth Option",
    )
    ps = from_cas_data(make_cas_dict()).schemes[0]
    ps.isin, ps.amfi = None, None
    assert match_scheme(ps) == (None, "ambiguous_name")


def test_normalize_name():
    assert normalize_name("AXBC - Axis Bluechip Fund - Direct Plan - Growth (formerly X) ISIN: INF1") == (
        "axis bluechip fund direct growth"
    )
    assert normalize_name("Axis Bluechip Fund - Direct Growth Option") == "axis bluechip fund direct growth"
    assert normalize_name("Axis Bluechip Fund - Regular Plan - Growth") != normalize_name(
        "Axis Bluechip Fund - Direct Plan - Growth"
    )


# ── import ─────────────────────────────────────────────────────────────

def test_import_creates_holdings_and_transactions(catalog, tenant, pf):
    res = _run(tenant, pf)

    assert (res.matched, res.holdings_created, res.txns_created, res.unmatched, res.errors) == (2, 2, 6, [], [])
    h = MFHolding.objects.get(scheme_id="120503")
    assert h.source == "cas" and h.tenant_id == tenant.id and h.portfolio_id == pf.id
    assert h.units == Decimal("150.0000")
    assert h.avg_cost_per_unit == Decimal("50.0000")  # 7500 / 150
    assert h.folio_number == "91" + "1234567 / 0"
    assert sorted(t.txn_type for t in h.transactions.all()) == ["buy", "sell", "sip", "sip"]
    assert all(t.tenant_id == tenant.id for t in MFTransaction.objects.all())


def test_reimport_is_idempotent(catalog, tenant, pf):
    _run(tenant, pf)
    res = _run(tenant, pf)

    assert (res.holdings_created, res.holdings_updated, res.txns_created, res.txns_existing) == (0, 2, 0, 6)
    assert MFHolding.objects.count() == 2 and MFTransaction.objects.count() == 6


def test_identical_same_day_transactions_are_kept_not_collapsed(catalog, tenant, pf):
    d = make_cas_dict()
    txns = d["folios"][0]["schemes"][0]["transactions"]
    txns.append(dict(txns[0]))  # a second, identical SIP on the same day
    _run(tenant, pf, d)
    assert MFTransaction.objects.filter(txn_type="sip", txn_date="2026-01-05").count() == 2
    res = _run(tenant, pf, d)
    assert res.txns_created == 0 and MFTransaction.objects.filter(txn_type="sip", txn_date="2026-01-05").count() == 2


def test_reimport_picks_up_only_new_transactions_and_updates_units(catalog, tenant, pf):
    _run(tenant, pf)
    d = make_cas_dict()
    s = d["folios"][0]["schemes"][0]
    s["close"] = s["close_calculated"] = "200.000"
    s["transactions"].append({"date": "2026-09-05", "description": "SIP", "amount": "2000.00",
                              "units": "50.000", "nav": "40.0000", "balance": None,
                              "type": "PURCHASE_SIP", "dividend_rate": None, "gift_folio": None})
    res = _run(tenant, pf, d)
    assert res.txns_created == 1 and res.txns_existing == 6  # 4 Axis + 2 PPFAS already present
    assert MFHolding.objects.get(scheme_id="120503").units == Decimal("200.0000")


def test_unmatched_scheme_is_skipped_others_still_import(catalog, tenant, pf):
    d = make_cas_dict()
    d["folios"][1]["schemes"][0]["isin"] = "INF000000000"
    d["folios"][1]["schemes"][0]["amfi"] = None
    res = _run(tenant, pf, d)
    assert res.matched == 1 and len(res.unmatched) == 1
    assert res.unmatched[0]["reason"] == "isin_not_in_catalog"
    assert MFHolding.objects.count() == 1


def test_one_failing_scheme_is_isolated_and_leaves_no_partial_state(catalog, tenant, pf):
    real = cas_import._import_scheme

    def flaky(ps, scheme, **kw):
        if scheme.scheme_code == "122639":
            real(ps, scheme, **kw)  # does real writes + counting, then blows up
            raise RuntimeError("boom")
        return real(ps, scheme, **kw)

    with patch.object(cas_import, "_import_scheme", flaky):
        res = _run(tenant, pf)

    assert res.matched == 1 and len(res.errors) == 1 and "boom" in res.errors[0]
    assert res.holdings_created == 1 and res.txns_created == 4  # failed scheme not counted
    assert MFHolding.objects.count() == 1  # and its writes were rolled back
    assert "…" in res.errors[0] and "1234567" not in res.errors[0]  # folio masked


def test_dry_run_writes_nothing_but_reports_counts(catalog, tenant, pf):
    res = _run(tenant, pf, dry_run=True)
    assert res.dry_run and res.holdings_created == 2 and res.txns_created == 6
    assert MFHolding.objects.count() == 0 and MFTransaction.objects.count() == 0


def test_dry_run_after_import_reports_nothing_new(catalog, tenant, pf):
    _run(tenant, pf)
    res = _run(tenant, pf, dry_run=True)
    assert res.holdings_created == 0 and res.txns_created == 0 and res.txns_existing == 6


def test_closed_out_scheme_gets_zero_units_and_no_avg_cost(catalog, tenant, pf):
    d = make_cas_dict()
    s = d["folios"][0]["schemes"][0]
    s["close"] = s["close_calculated"] = "0.000"
    _run(tenant, pf, d)
    h = MFHolding.objects.get(scheme_id="120503")
    assert h.units == 0 and h.avg_cost_per_unit is None


def test_portfolio_from_another_tenant_is_rejected(catalog, two_tenants):
    parsed = from_cas_data(make_cas_dict())
    with pytest.raises(ValueError):
        import_cas(parsed, tenant=two_tenants.a, portfolio=two_tenants.portfolio_b)


def test_two_tenants_get_separate_holdings(catalog, two_tenants):
    parsed = from_cas_data(make_cas_dict())
    import_cas(parsed, tenant=two_tenants.a, portfolio=two_tenants.portfolio_a)
    import_cas(parsed, tenant=two_tenants.b, portfolio=two_tenants.portfolio_b)
    assert MFHolding.objects.filter(tenant=two_tenants.a).count() == 2
    assert MFHolding.objects.filter(tenant=two_tenants.b).count() == 2
    assert MFTransaction.objects.count() == 12


# ── management command (PDF parsing mocked; password handling checked) ──

def _cmd(*args, parsed=None):
    out, err = StringIO(), StringIO()
    with patch(
        "apps.mutual_funds.management.commands.import_cas_statement.parse_cas_pdf",
        return_value=parsed or from_cas_data(make_cas_dict()),
    ) as p:
        call_command("import_cas_statement", *args, stdout=out, stderr=err)
    return out.getvalue(), err.getvalue(), p


@pytest.fixture
def pdf(tmp_path):
    f = tmp_path / "cas.pdf"
    f.write_bytes(b"%PDF-stub")
    return str(f)


def test_command_imports_and_never_echoes_password(catalog, tenant, pf, pdf):
    secret = "ABCDE1234F01011990"
    out, err, p = _cmd("--file", pdf, "--password", secret, "--tenant", tenant.slug, "--portfolio", pf.name)
    assert "matched 2/2 schemes" in out and "transactions +6" in out
    assert secret not in out and secret not in err
    assert p.call_args.args[1] == secret  # but it did reach the parser
    assert MFHolding.objects.count() == 2


def test_command_dry_run(catalog, tenant, pf, pdf):
    out, _, _ = _cmd("--file", pdf, "--password", "x", "--tenant", str(tenant.id), "--dry-run")
    assert "[dry-run]" in out and MFHolding.objects.count() == 0


def test_command_reads_password_from_env(catalog, tenant, pf, pdf, monkeypatch):
    monkeypatch.setenv("CAS_PASSWORD", "FROMENV")
    _, _, p = _cmd("--file", pdf, "--tenant", tenant.slug)
    assert p.call_args.args[1] == "FROMENV"


def test_command_requires_explicit_tenant_when_ambiguous(catalog, two_tenants, pdf):
    with pytest.raises(CommandError, match="--tenant is required"):
        _cmd("--file", pdf, "--password", "x")


def test_command_requires_portfolio_when_tenant_has_none_or_many(catalog, tenant, pdf):
    with pytest.raises(CommandError, match="--portfolio is required"):
        _cmd("--file", pdf, "--password", "x", "--tenant", tenant.slug)  # tenant has no portfolio
    PortfolioFactory(tenant=tenant, name="A")
    PortfolioFactory(tenant=tenant, name="B")
    with pytest.raises(CommandError, match="--portfolio is required"):
        _cmd("--file", pdf, "--password", "x", "--tenant", tenant.slug)


def test_command_fails_clearly_on_empty_catalog(tenant, pf, pdf, db):
    with pytest.raises(CommandError, match="pull_amfi_nav"):
        _cmd("--file", pdf, "--password", "x", "--tenant", tenant.slug)


def test_command_errors_when_nothing_matches(catalog, tenant, pf, pdf):
    d = make_cas_dict()
    for f in d["folios"]:
        f["schemes"][0]["isin"], f["schemes"][0]["amfi"] = "INF000000000", None
    with pytest.raises(CommandError, match="No scheme"):
        _cmd("--file", pdf, "--password", "x", "--tenant", tenant.slug, parsed=from_cas_data(d))


def test_command_missing_file(catalog, tenant, pf):
    with pytest.raises(CommandError, match="File not found"):
        _cmd("--file", "/no/such.pdf", "--password", "x")
