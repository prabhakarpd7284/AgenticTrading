"""Enum fields on the order endpoint must reject bad input with 400, not 500.

`OrderCreateSerializer` declares `order_type`, `product` and `origin` as plain
CharFields, but the `OrderDraft` pydantic model they are splatted into declares
each as a `Literal`. Any value outside the literal set therefore sails through
DRF validation and raises `pydantic.ValidationError` inside the view — an
uncaught 500 on ordinary client input.

Found live on 2026-09-09: posting `origin="pyramid-backtest-2026-09-09"`
returned HTTP 500 with a stack trace instead of a 400 naming the bad field.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.django_db


def _payload(portfolio, **overrides):
    body = {
        "portfolio_id": str(portfolio.id),
        "symbol": "NIFTY15SEP2623500PE",
        "side": "BUY",
        "qty": 65,
        "order_type": "MARKET",
        "product": "INTRADAY",
        "price": 100.0,
        "sl": 90.0,
        "tp": 120.0,
        "origin": "ui",
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("origin", "pyramid-backtest-2026-09-09"),
        ("order_type", "ICEBERG"),
        ("product", "MARGIN"),
    ],
)
def test_bad_enum_value_is_a_400_not_a_500(
    auth_client, paper_portfolio, field, bad_value
):
    resp = auth_client.post(
        "/api/v1/orders/",
        _payload(paper_portfolio, **{field: bad_value}),
        format="json",
    )

    assert resp.status_code != 500, (
        f"{field}={bad_value!r} caused a server error; client input must never 500"
    )
    assert resp.status_code == 400
    # The error must name the offending field so the caller can fix it.
    assert field in str(resp.data).lower()


def test_valid_enum_values_still_accepted(auth_client, paper_portfolio):
    """Guard against over-tightening: the legitimate values must still pass
    serializer validation (whatever the risk engine then decides)."""
    resp = auth_client.post(
        "/api/v1/orders/",
        _payload(paper_portfolio, origin="strategy", order_type="LIMIT"),
        format="json",
    )

    assert resp.status_code != 400, (
        f"valid enum values were rejected by the serializer: {resp.data}"
    )
