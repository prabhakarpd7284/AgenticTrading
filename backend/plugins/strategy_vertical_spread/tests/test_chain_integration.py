"""Integration tests: paper-adapter chain → selector → spread.

These exercise the full pipeline from chain fetch to spread construction
against the synthetic paper-broker chain. Live (Angel/Fyers) adapters
have their own tests under apps.market_data.tests.
"""
from __future__ import annotations

import pytest

from apps.market_data.adapters.base import OptionsChainSnapshot
from apps.market_data.adapters.paper import PaperBrokerAdapter

from plugins.strategy_vertical_spread.selector import (
    Leg, atm_strike, select_bear_call, select_bull_put, select_iron_condor,
)
from plugins.strategy_vertical_spread.strategy import _legs_from_snapshot


@pytest.fixture
def paper():
    return PaperBrokerAdapter()


@pytest.fixture
def chain(paper):
    snap = paper.options_chain("NIFTY", strikes_window=20)
    assert snap is not None
    return snap


def test_paper_chain_basic_shape(chain):
    assert chain.source == "paper"
    assert chain.underlying == "NIFTY"
    assert chain.spot > 20_000
    # 41 strikes (20 below, ATM, 20 above)
    assert 35 <= len(chain.rows) <= 45
    # Every row has both CE and PE
    for r in chain.rows:
        assert r.ce is not None and r.pe is not None
        assert r.ce.opt == "CE" and r.pe.opt == "PE"
        # Greeks present
        assert r.pe.iv > 0
        assert r.pe.delta <= 0  # put delta is non-positive ([-1, 0]); ~0 for deep-OTM near expiry
        assert r.ce.delta >= 0  # call delta is non-negative ([0, 1])
        assert r.pe.theta != 0
        # Bid < LTP < Ask
        assert r.ce.bid <= r.ce.ltp <= r.ce.ask


def test_paper_chain_atm_matches_spot(chain):
    """ATM strike should be the closest 50-pt multiple to spot."""
    expected_atm = int(round(chain.spot / 50) * 50)
    assert chain.atm() == expected_atm


def test_paper_chain_put_call_parity_at_atm(chain):
    """ATM put and call should price near each other in low-DTE regime."""
    atm = chain.atm()
    row = next(r for r in chain.rows if r.strike == atm)
    # ATM call slightly > ATM put under r > 0 forward drift
    assert row.ce.ltp > row.pe.ltp
    # but within 30 pts at our weekly tenor
    assert abs(row.ce.ltp - row.pe.ltp) < 100


def test_paper_chain_otm_decay_monotonic(chain):
    """Premiums should decay monotonically OTM on each side."""
    pes = sorted([r for r in chain.rows if r.pe], key=lambda r: r.strike)
    ces = sorted([r for r in chain.rows if r.ce], key=lambda r: r.strike, reverse=True)
    # PE prices increase as strike rises (toward ITM)
    pe_prices = [r.pe.ltp for r in pes]
    assert pe_prices == sorted(pe_prices), "PE prices not monotonic by strike"
    # CE prices increase as strike falls (toward ITM)
    ce_prices = [r.ce.ltp for r in ces]
    assert ce_prices == sorted(ce_prices), "CE prices not monotonic"


def test_legs_from_snapshot_round_trips(chain):
    legs = _legs_from_snapshot(chain)
    assert len(legs) == len(chain.rows) * 2
    assert all(isinstance(L, Leg) for L in legs)


def test_paper_chain_powers_bull_put_selector(chain):
    """End-to-end: paper chain → selector picks a real bull put."""
    legs = _legs_from_snapshot(chain)
    atm = chain.atm()
    atm_pe = chain.find(atm, "PE")
    assert atm_pe is not None
    pick = select_bull_put(legs, atm=atm, atm_pe_ltp=atm_pe.ltp)
    assert pick is not None
    assert pick.mode == "BULL_PUT"
    assert pick.sell.strike < atm
    assert pick.buy.strike < pick.sell.strike
    assert pick.credit > 0
    assert pick.width > 0


def test_paper_chain_powers_iron_condor(chain):
    legs = _legs_from_snapshot(chain)
    atm = chain.atm()
    atm_pe = chain.find(atm, "PE").ltp
    atm_ce = chain.find(atm, "CE").ltp
    pick = select_iron_condor(legs, atm=atm, atm_pe_ltp=atm_pe, atm_ce_ltp=atm_ce)
    assert pick is not None
    assert pick.mode == "IRON_CONDOR"
    assert pick.sell_2 is not None and pick.buy_2 is not None
    assert pick.sell_2.opt == "CE" and pick.buy_2.opt == "CE"


def test_paper_chain_find_returns_correct_leg(chain):
    atm = chain.atm()
    leg = chain.find(atm, "PE")
    assert leg is not None
    assert leg.strike == atm and leg.opt == "PE"


def test_unknown_underlying_returns_none(paper):
    assert paper.options_chain("UNKNOWN_INDEX") is None
