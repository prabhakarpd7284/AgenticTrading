"""Transaction-cost models for Studio backtests.

Intraday reuses the research harness's NSE MIS model unchanged
(``plugins.strategy_search.costs.CostModel``). Delivery (daily strategies,
held overnight) is the same structure with delivery rates: STT on both legs,
higher stamp duty, a per-sell DP charge. Rates are the published NSE /
SEBI / stamp-act schedules; brokerage is assumed min(₹20, 0.1 %) per order
(Angel One's flat plan) for both. Not modelled: impact cost beyond the
slippage assumption, exchange-fee revisions after the as-of date.
"""
from __future__ import annotations

from dataclasses import dataclass

from plugins.strategy_search.costs import CostBreakdown, CostModel

AS_OF = "2026-10"


@dataclass(frozen=True)
class DeliveryCostModel(CostModel):
    stt_buy_pct: float = 0.001          # 0.1 % on buy
    stt_sell_pct: float = 0.001         # 0.1 % on sell
    stamp_buy_pct: float = 0.00015      # 0.015 % on buy
    dp_per_sell: float = 15.93          # depository charge per scrip sold (incl. GST, approx.)

    def breakdown(self, price: float, qty: int, is_buy: bool) -> CostBreakdown:
        b = super().breakdown(price, qty, is_buy)
        turnover = price * qty
        stt = self.stt_buy_pct * turnover if is_buy else self.stt_sell_pct * turnover
        return CostBreakdown(brokerage=b.brokerage, stt=stt, txn=b.txn, sebi=b.sebi,
                             stamp=b.stamp, gst=b.gst + (0.0 if is_buy else self.dp_per_sell))


MODELS = {
    "nse_intraday_mis": CostModel(),
    "nse_delivery": DeliveryCostModel(),
}

DESCRIPTIONS = {
    "nse_intraday_mis": "NSE intraday (MIS): brokerage min(₹20, 0.1%), STT 0.025% sell, NSE txn "
                        "0.00297%, SEBI ₹10/cr, stamp 0.003% buy, GST 18% "
                        "(plugins/strategy_search/costs.py)",
    "nse_delivery": "NSE delivery (CNC): brokerage min(₹20, 0.1%), STT 0.1% both legs, NSE txn "
                    "0.00297%, SEBI ₹10/cr, stamp 0.015% buy, GST 18%, DP ≈₹15.93 per sell",
}


def resolve(model: str, daily: bool) -> str:
    if model in (None, "", "auto"):
        return "nse_delivery" if daily else "nse_intraday_mis"
    return model


def trade_cost_fn(model_name: str):
    """``cost_fn(trade) -> ₹`` for ``EngineConfig.cost_fn`` (round trip)."""
    model = MODELS[model_name]

    def fn(trade) -> float:
        side = 1 if trade.side.value == "BUY" else -1
        return model.round_trip(trade.entry_price, trade.exit_price, trade.quantity, side)
    return fn


def describe(model_name: str, slippage_pct: float) -> dict:
    return {"model": model_name, "description": DESCRIPTIONS[model_name], "as_of": AS_OF,
            "slippage_pct_per_side": slippage_pct}
