"""Fyers symbol-master resolution — column pinning + uniqueness (#18).

Pure-logic tests with crafted master rows (no network, no CSV download): they
lock in that matching pins the known columns and refuses to resolve a contract
from a stray cell or an ambiguous match.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from plugins.broker_fyers import symbols as sym
from plugins.broker_fyers.symbols import IST

_EXP_EPOCH = 1782813600  # 2026-06-30 in IST
_EXP = datetime.fromtimestamp(_EXP_EPOCH, IST).date()


def _row(ticker, epoch, strike, opt, *, stray_strike_cell=None):
    """A 21-column master row matching the real Fyers FO layout."""
    cells = [""] * 21
    cells[8] = str(epoch)        # expiry epoch
    cells[9] = ticker            # tradeable ticker
    cells[13] = "NIFTY"          # root
    cells[15] = str(strike)      # strike
    cells[16] = opt              # CE / PE
    if stray_strike_cell is not None:
        cells[3] = str(stray_strike_cell)  # e.g. lot size equal to some strike
    return cells


def test_resolves_correct_contract(monkeypatch):
    rows = [_row("NSE:NIFTY26JUN27650CE", _EXP_EPOCH, 27650.0, "CE")]
    monkeypatch.setattr(sym, "_load_master", lambda exch: rows)
    assert sym.resolve_option_symbol("NIFTY", _EXP, 27650, "CE") == "NSE:NIFTY26JUN27650CE"


def test_stray_cell_equal_to_strike_does_not_resolve(monkeypatch):
    # The only place "65" appears is a stray col3 (lot size) — never col15.
    rows = [_row("NSE:NIFTY26JUN27650CE", _EXP_EPOCH, 27650.0, "CE", stray_strike_cell=65)]
    monkeypatch.setattr(sym, "_load_master", lambda exch: rows)
    with pytest.raises(LookupError):
        sym.resolve_option_symbol("NIFTY", _EXP, 65, "CE")


def test_ambiguous_match_is_refused(monkeypatch):
    rows = [
        _row("NSE:NIFTY26JUN27650CE", _EXP_EPOCH, 27650.0, "CE"),
        _row("NSE:NIFTY26JUNW27650CE", _EXP_EPOCH, 27650.0, "CE"),  # 2nd distinct ticker
    ]
    monkeypatch.setattr(sym, "_load_master", lambda exch: rows)
    with pytest.raises(LookupError, match="Ambiguous"):
        sym.resolve_option_symbol("NIFTY", _EXP, 27650, "CE")


def test_expiry_epoch_out_of_range_is_ignored(monkeypatch):
    # A 15-digit fytoken-like epoch must not be parsed as an expiry.
    rows = [_row("NSE:NIFTY26JUN27650CE", 101126063035191, 27650.0, "CE")]
    monkeypatch.setattr(sym, "_load_master", lambda exch: rows)
    with pytest.raises(LookupError):
        sym.resolve_option_symbol("NIFTY", _EXP, 27650, "CE")
