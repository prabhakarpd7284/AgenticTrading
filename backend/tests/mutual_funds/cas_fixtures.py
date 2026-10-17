"""Synthetic CAS data for the CAS parser/import tests.

There is NO real CAMS/KFin statement behind these. The dict mirrors the shape
of ``casparser.types.CASData`` (and ``test_cas_parser`` validates that by
loading it into casparser's own pydantic model), but real-world layouts —
page breaks, wrapped scheme names, odd transaction descriptions — are untested.
"""
from __future__ import annotations

from copy import deepcopy

from apps.mutual_funds.models import MFScheme


def _txn(date, ttype, units, nav, amount, desc="", balance=None):
    return {
        "date": date, "description": desc or ttype.title(), "amount": amount,
        "units": units, "nav": nav, "balance": balance, "type": ttype,
        "dividend_rate": None, "gift_folio": None,
    }


def _scheme(name, isin, amfi, close, cost, nav, txns):
    return {
        "scheme": name, "advisor": "DIRECT", "rta_code": "AXBC", "rta": "CAMS",
        "type": "EQUITY", "isin": isin, "amfi": amfi, "nominees": [],
        "open": "0.000", "close": close, "close_calculated": close,
        "valuation": {"date": "2026-10-14", "nav": nav, "cost": cost,
                      "value": str(round(float(close) * float(nav), 2))},
        "transactions": txns,
    }


# Axis Bluechip: 2 SIPs, 1 lump sum, 1 STT line (skipped), 1 redemption.
AXIS = _scheme(
    "AXBC - Axis Bluechip Fund - Direct Plan - Growth (formerly Axis Bluechip Fund)",
    "INF846K01DP8", "120503", "150.000", "7500.00", "60.00",
    [
        _txn("2026-01-05", "PURCHASE_SIP", "50.000", "40.0000", "2000.00"),
        _txn("2026-02-05", "PURCHASE_SIP", "50.000", "40.0000", "2000.00"),
        _txn("2026-02-05", "PURCHASE", "100.000", "35.0000", "3500.00"),
        _txn("2026-02-05", "STT_TAX", None, None, "3.50"),
        _txn("2026-03-10", "REDEMPTION", "-50.000", "55.0000", "-2750.00"),
    ],
)

# Parag Parikh: IDCW payout (no units) + switch in.
PPFAS = _scheme(
    "PP001 - Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
    "INF879O01027", "122639", "20.000", "1000.00", "80.00",
    [
        _txn("2026-04-01", "SWITCH_IN", "20.000", "50.0000", "1000.00"),
        _txn("2026-06-30", "DIVIDEND_PAYOUT", None, None, "12.00"),
    ],
)


def make_cas_dict(schemes=None, *, cas_type="DETAILED", warnings=None) -> dict:
    schemes = schemes if schemes is not None else [AXIS, PPFAS]
    amc_of = {AXIS["isin"]: "Axis Mutual Fund", PPFAS["isin"]: "PPFAS Mutual Fund"}
    folios = [
        {
            "folio": f"91{i}234567 / 0", "amc": amc_of.get(s["isin"], "Unknown Mutual Fund"),
            "PAN": "AAAAA0000A", "KYC": "OK", "PANKYC": "OK",
            "schemes": [deepcopy(s)],
        }
        for i, s in enumerate(schemes, start=1)
    ]
    return {
        "statement_period": {"from": "2026-01-01", "to": "2026-10-14"},
        "folios": folios,
        "investor_info": {"name": "Test Investor", "email": "t@example.com",
                          "address": "Nowhere", "mobile": "0000000000"},
        "cas_type": cas_type,
        "file_type": "CAMS",
        "parse_warnings": warnings or [],
    }


def seed_catalog() -> dict[str, MFScheme]:
    """AMFI catalog rows matching the fixture schemes (plus one decoy)."""
    axis = MFScheme.objects.create(
        scheme_code="120503", isin_growth="INF846K01DP8", isin_div_reinvestment="INF846K01DQ6",
        amc_name="Axis Mutual Fund", scheme_name="Axis Bluechip Fund - Direct Plan - Growth Option",
        plan="Direct Plan", option="Growth", is_direct=True,
    )
    pp = MFScheme.objects.create(
        scheme_code="122639", isin_growth="INF879O01027", isin_div_reinvestment="",
        amc_name="PPFAS Mutual Fund", scheme_name="Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
        plan="Direct Plan", option="Growth", is_direct=True,
    )
    decoy = MFScheme.objects.create(
        scheme_code="120504", isin_growth="INF846K01DR4",
        amc_name="Axis Mutual Fund", scheme_name="Axis Bluechip Fund - Regular Plan - Growth Option",
        plan="Regular Plan", option="Growth", is_direct=False,
    )
    return {"axis": axis, "pp": pp, "decoy": decoy}
