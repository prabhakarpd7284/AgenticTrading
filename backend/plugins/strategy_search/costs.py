"""NSE intraday-equity (MIS) costs per executed order + slippage helpers.

Per executed order (one leg):
    brokerage  min(₹20, 0.1 % of turnover)
    STT        0.025 % of turnover, SELL side only
    NSE txn    0.00297 % of turnover
    SEBI       ₹10 per crore of turnover
    stamp      0.003 % of turnover, BUY side only
    GST        18 % on (brokerage + NSE txn + SEBI)

Not modelled: DP charges (not applicable to MIS), impact cost beyond the
slippage assumption, exchange-fee revisions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    txn: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.txn + self.sebi + self.stamp + self.gst


@dataclass(frozen=True)
class CostModel:
    brokerage_pct: float = 0.001
    brokerage_cap: float = 20.0
    stt_sell_pct: float = 0.00025
    txn_pct: float = 0.0000297
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.00003
    gst_pct: float = 0.18

    def breakdown(self, price: float, qty: int, is_buy: bool) -> CostBreakdown:
        turnover = price * qty
        brokerage = min(self.brokerage_cap, self.brokerage_pct * turnover)
        txn = self.txn_pct * turnover
        sebi = self.sebi_per_crore * turnover / 1e7
        return CostBreakdown(
            brokerage=brokerage,
            stt=0.0 if is_buy else self.stt_sell_pct * turnover,
            txn=txn,
            sebi=sebi,
            stamp=self.stamp_buy_pct * turnover if is_buy else 0.0,
            gst=self.gst_pct * (brokerage + txn + sebi),
        )

    def order_cost(self, price: float, qty: int, is_buy: bool) -> float:
        if qty <= 0 or price <= 0:
            return 0.0
        return self.breakdown(price, qty, is_buy).total

    def round_trip(self, entry: float, exit_: float, qty: int, side: int) -> float:
        """side +1 = long (buy, then sell), −1 = short (sell, then buy)."""
        return (self.order_cost(entry, qty, is_buy=side > 0)
                + self.order_cost(exit_, qty, is_buy=side < 0))


def slip(price: float, side: int, pct: float, *, entering: bool) -> float:
    """Fill price after slippage of ``pct`` % against the trade.

    Entering a long / exiting a short buys → price up; the other two sell → down.
    """
    buy = (side > 0) == entering
    return price * (1 + pct / 100.0) if buy else price * (1 - pct / 100.0)
