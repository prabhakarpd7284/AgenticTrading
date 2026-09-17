"""Resolve the ATM option chain to stream.

AlphaDesk's live pipeline — tick feed, ltp cache, candle store, paper fills —
covers 98 equities and no options at all, even though options are where the
trading happens. This builds the option side of that universe.

Per CLAUDE.md: option lookup matches on **strike (in paisa) and expiry date
metadata**, never a substring search on the symbol, because BSE symbol encoding
can collide with strike digits.
"""
from __future__ import annotations

import pytest

from apps.market_data.services.option_universe import (
    STRIKE_STEP,
    atm_strike,
    strikes_around,
)


# ── ATM rounding ───────────────────────────────────────────────────────

@pytest.mark.parametrize("spot,expected", [
    (23427.60, 23450),     # nearer 23450 than 23400
    (23424.00, 23400),
    (23425.00, 23450),     # exact half rounds up
    (23400.00, 23400),
])
def test_nifty_atm_rounds_to_the_nearest_50(spot, expected):
    assert atm_strike("NIFTY", spot) == expected


def test_banknifty_uses_a_100_point_step():
    assert STRIKE_STEP["BANKNIFTY"] == 100
    assert atm_strike("BANKNIFTY", 56583.35) == 56600
    assert atm_strike("BANKNIFTY", 56549.00) == 56500


def test_sensex_uses_a_100_point_step():
    assert atm_strike("SENSEX", 78012.0) == 78000


def test_unknown_underlying_is_rejected_rather_than_guessed():
    """Guessing a strike step silently subscribes to contracts that do not
    exist, and the feed then looks alive while carrying nothing."""
    with pytest.raises(ValueError):
        atm_strike("NOTANINDEX", 100.0)


# ── Strike ladder ──────────────────────────────────────────────────────

def test_strikes_around_is_centred_on_atm():
    out = strikes_around("NIFTY", 23427.60, width=2)

    assert out == [23350, 23400, 23450, 23500, 23550]
    assert len(out) == 5           # width either side, plus ATM


def test_width_zero_is_just_the_atm_strike():
    assert strikes_around("NIFTY", 23427.60, width=0) == [23450]


def test_strikes_are_ordered_and_unique():
    out = strikes_around("BANKNIFTY", 56583.35, width=3)

    assert out == sorted(out)
    assert len(set(out)) == len(out)


def test_a_negative_width_is_rejected():
    with pytest.raises(ValueError):
        strikes_around("NIFTY", 23427.60, width=-1)


# ── Resolution must not fail silently ──────────────────────────────────
#
# Observed 2026-09-10: the chain resolved 0 contracts and logged
# "option_chain ... contracts=0" at INFO, so the session started with an
# equity-only universe while reporting success. Cause was an expiry-format
# mismatch — `iso_to_angel` gives "15SEP26" (the form embedded in symbol
# names) but the master's `expiry` field carries "15SEP2026".

def _live_expiry_forms():
    """The next real expiry, in both Angel spellings.

    Never pin a literal expiry in a test: 15SEP2026 was valid when this was
    written and gone from the master two days later, which failed the suite
    for reasons that had nothing to do with the code.
    """
    from trading.utils.expiry_utils import (
        iso_to_angel, iso_to_angel_long, next_expiry_date,
    )

    d = next_expiry_date("NIFTY")
    return iso_to_angel_long(d.isoformat()), iso_to_angel(d.isoformat())


def _live_spot():
    """A strike that exists for the current expiry — the ATM ladder has to
    line up with whatever NIFTY is trading near now."""
    from apps.market_data.services.option_universe import chain_contracts

    long_form, _ = _live_expiry_forms()
    from trading.services.ticker_service import TickerService
    ts = TickerService(); ts._ensure_loaded()
    strikes = sorted({
        int(round(float(r.get("strike", 0)) / 100.0))
        for r in (ts._instruments or ())
        if r.get("name") == "NIFTY" and r.get("expiry") == long_form
        and "OPT" in (r.get("instrumenttype") or "")
    })
    return float(strikes[len(strikes) // 2]) if strikes else 0.0


def test_chain_accepts_both_angel_expiry_forms():
    """Whichever helper a caller reaches for, the chain must resolve."""
    from apps.market_data.services.option_universe import chain_contracts

    long_form, short_form_exp = _live_expiry_forms()
    spot = _live_spot()
    if not spot:
        pytest.skip("symbol master has no NIFTY options for the next expiry")

    long_form_out = chain_contracts("NIFTY", spot, long_form, width=1)
    short_form_out = chain_contracts("NIFTY", spot, short_form_exp, width=1)

    assert len(long_form_out) > 0, "long form must resolve"
    assert {c["symbol"] for c in short_form_out} == {
        c["symbol"] for c in long_form_out
    }


def test_empty_chain_is_reported_as_a_problem():
    """A chain that resolves nothing is a failure, not a quiet zero."""
    from apps.market_data.services.option_universe import chain_is_healthy

    assert chain_is_healthy(contracts=[], expected_strikes=5) is False
    assert chain_is_healthy(contracts=[{}] * 2, expected_strikes=5) is False
    assert chain_is_healthy(contracts=[{}] * 10, expected_strikes=5) is True
