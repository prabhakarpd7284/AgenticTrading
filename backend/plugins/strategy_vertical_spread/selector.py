"""Strike selector — 80%/60% premium-percentile rule with max-width clamp.

Given a fully-priced option chain (or LTP-only chain) and an ATM strike,
walks OTM finding the first strike at-or-under the sell-target premium,
then the first strike below that at-or-under the buy-target premium.

Includes the MAX_WIDTH_POINTS clamp learned from the April 2026 backtest:
in high-VIX regimes the rule picks unhealthily wide spreads (350pt was
₹18,521 max-loss / 3.7% of capital on April 1). The clamp ratchets the
buy-pct upward (toward the sell) until the spread fits the cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

__all__ = ["SpreadPick", "Leg", "select_bull_put", "select_bear_call",
           "select_iron_condor", "atm_strike"]


@dataclass(frozen=True)
class Leg:
    """One option leg with the data the selector needs.

    `ltp` is the close/last-traded price. `bid` / `ask` are optional;
    when present the selector uses them to compute a broker-truthful
    credit (sell @ bid, buy @ ask). When absent it falls back to LTP.
    """
    strike: int
    opt: Literal["CE", "PE"]
    ltp: float
    bid: float = 0.0
    ask: float = 0.0
    oi: int = 0

    @property
    def sell_price(self) -> float:
        """Price we'd receive when shorting this leg (broker-truthful)."""
        return self.bid if self.bid > 0 else self.ltp

    @property
    def buy_price(self) -> float:
        """Price we'd pay when buying this leg (broker-truthful)."""
        return self.ask if self.ask > 0 else self.ltp


@dataclass(frozen=True)
class SpreadPick:
    """Result of strike selection. credit is per-share (per-lot = credit × lot_size)."""
    mode: Literal["BULL_PUT", "BEAR_CALL", "IRON_CONDOR"]
    sell: Leg
    buy: Leg
    # For iron condor — second pair is the bear call side
    sell_2: Optional[Leg] = None
    buy_2: Optional[Leg] = None

    @property
    def credit(self) -> float:
        c = self.sell.sell_price - self.buy.buy_price
        if self.sell_2 and self.buy_2:
            c += self.sell_2.sell_price - self.buy_2.buy_price
        return c

    @property
    def width(self) -> int:
        w = abs(self.sell.strike - self.buy.strike)
        if self.sell_2 and self.buy_2:
            w = max(w, abs(self.sell_2.strike - self.buy_2.strike))
        return w


# Defaults match the PoC profile -------------------------------------
SELL_PCT: float = 0.80
BUY_PCT: float = 0.60
MAX_WIDTH_POINTS: int = 200          # safety clamp from April backtest learning
STRIKE_STEP_DEFAULT: int = 50


def atm_strike(spot: float, step: int = STRIKE_STEP_DEFAULT) -> int:
    """Round spot to the nearest strike step."""
    return int(round(spot / step) * step)


def select_bull_put(
    chain: Sequence[Leg],
    atm: int,
    atm_pe_ltp: float,
    *,
    sell_pct: float = SELL_PCT,
    buy_pct: float = BUY_PCT,
    max_width: int = MAX_WIDTH_POINTS,
) -> Optional[SpreadPick]:
    """Pick (sell PE, buy PE) for a bull-put spread.

    Targets are sell_pct·ATM_PE and buy_pct·ATM_PE. The selector walks
    strikes downward from ATM finding the first that satisfies each
    target. If the resulting width exceeds max_width, the buy leg is
    pulled in (closer to the sell) until the spread fits the clamp.
    """
    target_sell = atm_pe_ltp * sell_pct
    target_buy = atm_pe_ltp * buy_pct

    # PEs at-or-below ATM, sorted high-strike → low-strike
    pes_below = sorted(
        [leg for leg in chain if leg.opt == "PE" and leg.strike <= atm and leg.ltp > 0],
        key=lambda L: L.strike,
        reverse=True,
    )
    sell: Optional[Leg] = None
    buy: Optional[Leg] = None
    for leg in pes_below:
        if sell is None and leg.ltp <= target_sell:
            sell = leg
            continue
        if sell is not None and leg.strike < sell.strike and leg.ltp <= target_buy:
            buy = leg
            break
    if not sell or not buy:
        return None

    # Apply max-width clamp — narrow the spread by raising the buy strike
    # until the width fits the cap (lowest legal width = strike step).
    while (sell.strike - buy.strike) > max_width:
        higher_buys = [L for L in pes_below
                       if L.strike > buy.strike and L.strike < sell.strike]
        if not higher_buys:
            break
        # pull buy upward (smaller width)
        buy = min(higher_buys, key=lambda L: sell.strike - L.strike)

    return SpreadPick(mode="BULL_PUT", sell=sell, buy=buy)


def select_bear_call(
    chain: Sequence[Leg],
    atm: int,
    atm_ce_ltp: float,
    *,
    sell_pct: float = SELL_PCT,
    buy_pct: float = BUY_PCT,
    max_width: int = MAX_WIDTH_POINTS,
) -> Optional[SpreadPick]:
    """Pick (sell CE, buy CE) for a bear-call spread. Mirror of bull-put."""
    target_sell = atm_ce_ltp * sell_pct
    target_buy = atm_ce_ltp * buy_pct

    ces_above = sorted(
        [leg for leg in chain if leg.opt == "CE" and leg.strike >= atm and leg.ltp > 0],
        key=lambda L: L.strike,
    )
    sell: Optional[Leg] = None
    buy: Optional[Leg] = None
    for leg in ces_above:
        if sell is None and leg.ltp <= target_sell:
            sell = leg
            continue
        if sell is not None and leg.strike > sell.strike and leg.ltp <= target_buy:
            buy = leg
            break
    if not sell or not buy:
        return None

    while (buy.strike - sell.strike) > max_width:
        lower_buys = [L for L in ces_above
                      if L.strike < buy.strike and L.strike > sell.strike]
        if not lower_buys:
            break
        buy = min(lower_buys, key=lambda L: L.strike - sell.strike)

    return SpreadPick(mode="BEAR_CALL", sell=sell, buy=buy)


def select_iron_condor(
    chain: Sequence[Leg],
    atm: int,
    atm_pe_ltp: float,
    atm_ce_ltp: float,
    *,
    sell_pct: float = SELL_PCT,
    buy_pct: float = BUY_PCT,
    max_width: int = MAX_WIDTH_POINTS,
) -> Optional[SpreadPick]:
    """Bull put + bear call combined. Both wings must successfully pick."""
    bp = select_bull_put(chain, atm, atm_pe_ltp,
                          sell_pct=sell_pct, buy_pct=buy_pct, max_width=max_width)
    bc = select_bear_call(chain, atm, atm_ce_ltp,
                           sell_pct=sell_pct, buy_pct=buy_pct, max_width=max_width)
    if not bp or not bc:
        return None
    return SpreadPick(
        mode="IRON_CONDOR",
        sell=bp.sell, buy=bp.buy,
        sell_2=bc.sell, buy_2=bc.buy,
    )
