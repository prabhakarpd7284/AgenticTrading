"""CAS parser adapter tests — synthetic data only (no real CAMS/KFin PDF)."""
from __future__ import annotations

from decimal import Decimal

import pytest

from apps.mutual_funds.services.cas_parser import CASParseError, from_cas_data, parse_cas_pdf
from tests.mutual_funds.cas_fixtures import make_cas_dict


def test_fixture_matches_casparser_real_model():
    """Guards the fixture itself: it must load into casparser's own CASData."""
    from casparser import CASData

    model = CASData.model_validate(make_cas_dict())
    parsed = from_cas_data(model)  # the typed-model path, not just the dict path
    assert len(parsed.schemes) == 2
    assert parsed.file_type == "CAMS" and parsed.cas_type == "DETAILED"


def test_maps_types_takes_absolute_values_and_skips_tax_lines():
    parsed = from_cas_data(make_cas_dict())
    axis = next(s for s in parsed.schemes if s.isin == "INF846K01DP8")

    assert [t.txn_type for t in axis.transactions] == ["sip", "sip", "buy", "sell"]
    sell = axis.transactions[-1]
    assert sell.units == Decimal("50.000") and sell.amount == Decimal("2750.00")  # sign stripped
    assert parsed.skipped_txn_counts == {"STT_TAX": 1}
    assert axis.close_units == Decimal("150.000") and axis.cost == Decimal("7500.00")
    assert axis.amfi == "120503"


def test_dividend_payout_without_units_is_zero_units():
    parsed = from_cas_data(make_cas_dict())
    pp = next(s for s in parsed.schemes if s.isin == "INF879O01027")
    div = pp.transactions[-1]
    assert div.txn_type == "dividend"
    assert div.units == 0 and div.nav == 0 and div.amount == Decimal("12.00")
    assert pp.transactions[0].txn_type == "switch_in"


def test_parse_warnings_are_carried_through():
    parsed = from_cas_data(make_cas_dict(warnings=["scheme X: balance mismatch"]))
    assert parsed.warnings == ["scheme X: balance mismatch"]


def test_demat_statement_is_rejected_clearly():
    with pytest.raises(CASParseError, match="demat"):
        from_cas_data({"accounts": [], "statement_period": {"from": "a", "to": "b"}})


def test_scheme_without_closing_balance_is_skipped_with_warning():
    d = make_cas_dict()
    d["folios"][0]["schemes"][0]["close"] = None
    parsed = from_cas_data(d)
    assert len(parsed.schemes) == 1
    assert any("no closing balance" in w for w in parsed.warnings)


def test_missing_file_raises_parse_error(tmp_path):
    with pytest.raises(CASParseError):
        parse_cas_pdf(str(tmp_path / "nope.pdf"), "PASSWORD1")


def test_garbage_pdf_raises_parse_error_without_leaking_password(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"this is not a pdf")
    secret = "ABCDE1234F01011990"
    with pytest.raises(CASParseError) as exc:
        parse_cas_pdf(str(f), secret)
    assert secret not in str(exc.value)
