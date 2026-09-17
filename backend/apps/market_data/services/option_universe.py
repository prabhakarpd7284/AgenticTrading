"""Resolve which option contracts the live pipeline should stream.

The tick feed, ltp cache, candle store and paper-fill engine all cover 98
equities and no options — even though options are where the trading actually
happens. This is the option side of that universe: pick the strikes around ATM
for the current expiry, and resolve them to real contracts in the symbol master.

Two rules from CLAUDE.md are load-bearing here:

* **Strike lookup matches on strike-in-paisa plus expiry metadata**, never a
  substring search on the symbol — BSE encoding can collide with strike digits.
* **Lot sizes and expiry days are never hardcoded** beyond the strike *step*,
  which is an exchange contract specification rather than a tradeable quantity.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# Strike intervals per index. These are exchange contract specs — if an index
# is missing we refuse rather than guess, because a guessed step resolves to
# contracts that do not exist and the feed then looks alive while carrying
# nothing.
STRIKE_STEP = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
    "SENSEX": 100,
    "BANKEX": 100,
}

# Which exchange each underlying's options live on. BSE indices trade on BFO.
OPTION_EXCHANGE = {
    "NIFTY": "NFO",
    "BANKNIFTY": "NFO",
    "FINNIFTY": "NFO",
    "MIDCPNIFTY": "NFO",
    "SENSEX": "BFO",
    "BANKEX": "BFO",
}


def atm_strike(underlying: str, spot: float) -> int:
    """Nearest tradeable strike to ``spot``."""
    step = STRIKE_STEP.get(underlying.upper())
    if step is None:
        raise ValueError(
            f"No strike step known for {underlying!r}. Add it to STRIKE_STEP "
            f"rather than guessing — a wrong step resolves to contracts that "
            f"do not exist."
        )
    # Half-steps round up, matching how traders quote the ATM.
    return int((spot + step / 2) // step) * step


def strikes_around(underlying: str, spot: float, width: int = 2) -> list[int]:
    """``width`` strikes either side of ATM, inclusive, ascending."""
    if width < 0:
        raise ValueError("width must be >= 0")
    step = STRIKE_STEP[underlying.upper()]
    atm = atm_strike(underlying, spot)
    return [atm + i * step for i in range(-width, width + 1)]


def _expiry_forms(expiry: str) -> set[str]:
    """Both Angel expiry spellings for the same date.

    `iso_to_angel` yields the short form embedded in symbol names
    ("15SEP26") while the master's ``expiry`` column carries the long one
    ("15SEP2026"). Matching only one silently resolves an empty chain.
    """
    e = (expiry or "").upper().strip()
    if not e:
        return set()
    forms = {e}
    # DDMMMYY -> DDMMMYYYY and back, without parsing the month name.
    head, tail = e[:-2], e[-2:]
    if len(e) >= 7 and tail.isdigit():
        if len(e) == 7:                     # 15SEP26
            forms.add(f"{head}20{tail}")
        elif len(e) == 9 and head.endswith("20"):   # 15SEP2026
            forms.add(f"{head[:-2]}{tail}")
    return forms


def chain_is_healthy(*, contracts: list, expected_strikes: int) -> bool:
    """Did the chain actually resolve?

    Every strike should yield a CE and a PE. A chain far below that is a
    resolution failure — an expiry-format mismatch, a stale master — and
    must be reported rather than logged as a quiet zero.
    """
    if expected_strikes <= 0:
        return True
    return len(contracts) >= expected_strikes      # >= 1 leg per strike


def chain_contracts(
    underlying: str, spot: float, expiry: str, width: int = 2,
) -> list[dict]:
    """Resolve the ATM chain to real contracts.

    ``expiry`` is the Angel master's format (e.g. ``15SEP2026``). Returns dicts
    of ``{symbol, token, exchange, strike, option_type, lot_size}`` for every
    CE and PE that actually exists — silently skipping strikes the master does
    not list, since the ladder is generated arithmetically and the far wings
    may not be listed.
    """
    from trading.services.ticker_service import TickerService

    ts = TickerService()
    ts._ensure_loaded()

    underlying = underlying.upper()
    exchange = OPTION_EXCHANGE.get(underlying, "NFO")
    wanted_expiries = _expiry_forms(expiry)
    wanted = {s: None for s in strikes_around(underlying, spot, width)}

    out: list[dict] = []
    for row in ts._instruments or ():
        if row.get("name") != underlying:
            continue
        if "OPT" not in (row.get("instrumenttype") or ""):
            continue
        if row.get("expiry") not in wanted_expiries:
            continue
        # Strike is quoted in paisa in the master: 23500 -> 2350000.0
        try:
            strike = int(round(float(row.get("strike", 0)) / 100.0))
        except (TypeError, ValueError):
            continue
        if strike not in wanted:
            continue

        symbol = row.get("symbol") or ""
        option_type = "CE" if symbol.endswith("CE") else "PE" if symbol.endswith("PE") else ""
        if not option_type:
            continue

        out.append({
            "symbol": symbol,
            "token": str(row.get("token") or ""),
            "exchange": row.get("exch_seg") or exchange,
            "strike": strike,
            "option_type": option_type,
            "lot_size": int(row.get("lotsize") or 0),
        })

    out.sort(key=lambda c: (c["strike"], c["option_type"]))
    log.info(
        "option_universe.resolved",
        underlying=underlying, expiry=expiry, spot=spot,
        strikes=len(wanted), contracts=len(out),
    )
    return out
