"""Paper fill simulation.

`PaperBrokerAdapter.place()` returns an order id and stops — nothing ever moved
a paper order past ``sent``. No Trade rows, no P&L, nothing to learn from. This
module supplies the missing half: decide when a resting order fills, and when an
open trade's stop or target is touched.

**Pessimistic by construction.** A paper book that flatters entries teaches the
wrong lesson, and the whole point of paper mode here is honest feedback. Where
a single tick leaves reality ambiguous we take the trader's worse side:

* a LIMIT order fills *at its limit*, never at a better touch price, so gapping
  through your price is not free money
* an unknown price (``reference_price()`` returns 0.0 for "no idea") never fills
  or exits anything — a fill at zero would book a fictional 100% gain
* if stop and target both look touched in one tick, the stop wins; a tick cannot
  tell us which came first inside the bar, and assuming the target would flatter
  every ambiguous trade

Everything here is pure: no DB, no clock, no broker. The driver in
`tasks/paper_fills.py` supplies prices and persists outcomes.
"""
from __future__ import annotations

# Exit reasons — mirror Trade.CloseReason.
SL_HIT = "SL_HIT"
TARGET_HIT = "TARGET_HIT"


def fill_price_for(order, ltp: float) -> float | None:
    """Price at which ``order`` fills on a tick of ``ltp``, or None to rest."""
    if ltp is None or ltp <= 0:
        return None

    order_type = (order.order_type or "MARKET").upper()
    if order_type == "MARKET":
        return float(ltp)

    # LIMIT / SL / SL-M all need a price to trigger against.
    if order.price is None:
        return None
    limit = float(order.price)

    if order.side == "BUY":
        return limit if ltp <= limit else None
    return limit if ltp >= limit else None


def exit_for(trade, ltp: float) -> tuple[float, str] | None:
    """``(price, reason)`` if ``trade`` should close on this tick, else None."""
    if ltp is None or ltp <= 0:
        return None

    stop = float(trade.stop_loss) if trade.stop_loss is not None else None
    target = float(trade.target) if trade.target is not None else None
    long_side = trade.side == "BUY"

    # Stop is evaluated first — see module docstring on ambiguity.
    if stop is not None:
        if (long_side and ltp <= stop) or (not long_side and ltp >= stop):
            return stop, SL_HIT

    if target is not None:
        if (long_side and ltp >= target) or (not long_side and ltp <= target):
            return target, TARGET_HIT

    return None
