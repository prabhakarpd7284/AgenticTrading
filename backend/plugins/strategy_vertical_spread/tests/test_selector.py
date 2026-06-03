"""Unit tests for the strike selector.

Reproduces the April 2026 PoC selections on a synthetic chain calibrated
to the same spot + VIX regime, then exercises the max-width clamp.
"""
from __future__ import annotations

import pytest

from plugins.strategy_vertical_spread.selector import (
    Leg, MAX_WIDTH_POINTS, SpreadPick,
    atm_strike, select_bear_call, select_bull_put, select_iron_condor,
)


def make_chain(strikes_below_atm: int = 10, strikes_above_atm: int = 10,
                atm: int = 24150, step: int = 50,
                pe_atm_ltp: float = 180.0,
                ce_atm_ltp: float = 192.0,
                decay_per_strike: float = 0.85) -> list[Leg]:
    """Build a synthetic chain that decays geometrically OTM.

    decay_per_strike < 1 means each step further OTM is `decay` × prior price.
    Real chains decay faster on the OTM side than this, but this is enough
    for selector unit tests.
    """
    chain: list[Leg] = []
    # PEs at and below ATM (price drops as strike drops further OTM)
    price = pe_atm_ltp
    for i in range(strikes_below_atm + 1):
        k = atm - i * step
        chain.append(Leg(strike=k, opt="PE", ltp=price,
                          bid=price * 0.99, ask=price * 1.01, oi=10_000))
        price *= decay_per_strike
    # CEs at and above ATM
    price = ce_atm_ltp
    for i in range(strikes_above_atm + 1):
        k = atm + i * step
        chain.append(Leg(strike=k, opt="CE", ltp=price,
                          bid=price * 0.99, ask=price * 1.01, oi=10_000))
        price *= decay_per_strike
    return chain


def test_atm_rounds_correctly():
    assert atm_strike(24173.05) == 24150
    assert atm_strike(24176) == 24200
    assert atm_strike(24150) == 24150


def test_bull_put_picks_below_atm():
    chain = make_chain()
    pick = select_bull_put(chain, atm=24150, atm_pe_ltp=180.0)
    assert pick is not None
    assert pick.mode == "BULL_PUT"
    assert pick.sell.opt == "PE" and pick.buy.opt == "PE"
    assert pick.sell.strike < 24150
    assert pick.buy.strike < pick.sell.strike
    # credit must be positive (we collect more than we pay)
    assert pick.credit > 0


def test_bear_call_mirrors_bull_put():
    chain = make_chain()
    pick = select_bear_call(chain, atm=24150, atm_ce_ltp=192.0)
    assert pick is not None
    assert pick.mode == "BEAR_CALL"
    assert pick.sell.opt == "CE" and pick.buy.opt == "CE"
    assert pick.sell.strike > 24150
    assert pick.buy.strike > pick.sell.strike
    assert pick.credit > 0


def test_iron_condor_has_both_wings():
    chain = make_chain()
    pick = select_iron_condor(chain, atm=24150, atm_pe_ltp=180.0, atm_ce_ltp=192.0)
    assert pick is not None
    assert pick.mode == "IRON_CONDOR"
    assert pick.sell_2 is not None and pick.buy_2 is not None
    assert pick.sell.opt == "PE" and pick.buy.opt == "PE"
    assert pick.sell_2.opt == "CE" and pick.buy_2.opt == "CE"


def test_max_width_clamp_applies():
    """With a high-vol chain producing wide selections, clamp must shrink width."""
    # Slow decay → 80/60 rule wants to walk far OTM, so width balloons
    chain = make_chain(strikes_below_atm=30, atm=24150,
                       pe_atm_ltp=600.0, decay_per_strike=0.95)
    pick = select_bull_put(chain, atm=24150, atm_pe_ltp=600.0,
                            max_width=150)
    assert pick is not None
    assert (pick.sell.strike - pick.buy.strike) <= 150


def test_no_pick_when_chain_too_thin():
    """If no strike is at-or-under the buy target, selector returns None."""
    chain = [
        Leg(strike=24100, opt="PE", ltp=170.0, bid=169, ask=171),
        Leg(strike=24050, opt="PE", ltp=165.0, bid=164, ask=166),
        # gap — no further strikes
    ]
    pick = select_bull_put(chain, atm=24150, atm_pe_ltp=180.0)
    # 60% target = 108; nothing in the chain is that low → None
    assert pick is None


def test_sell_price_uses_bid_when_available():
    leg = Leg(strike=24050, opt="PE", ltp=180.0, bid=179.0, ask=181.0)
    # When shorting, we receive the bid
    assert leg.sell_price == 179.0
    # When buying, we pay the ask
    assert leg.buy_price == 181.0


def test_sell_price_falls_back_to_ltp_without_quotes():
    leg = Leg(strike=24050, opt="PE", ltp=180.0)
    assert leg.sell_price == 180.0
    assert leg.buy_price == 180.0
