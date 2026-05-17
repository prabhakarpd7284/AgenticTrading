"""Margin + leverage estimator.

These are *formula-based estimates* for Indian-market instruments, not Angel
One's live `/order/margin` API. Use them for sizing, capital-allocation
splits, and leverage thinking. Swap with the broker's live calculator before
relying on these numbers for live order placement.

All amounts are in INR.

Bucket model
------------
The user's capital is conceptually split across these buckets so reporting
can answer "where is the money?". Each bucket has different margin
characteristics:

    equity_delivery   CNC product · settles T+1 · full notional locked
    equity_intraday   MIS product · 5× leverage typical (varies by stock)
    options_long      premium paid is the entire margin
    options_short     SPAN + exposure ≈ 12–18% notional + premium received
    options_weekly    same as options_short but flagged for weekly expiry
    options_monthly   same as options_short for monthly expiry
    straddle_weekly   weekly short straddle (both legs short)
    futures           SPAN + exposure ≈ 12–15% notional

Each function returns a `MarginEstimate` dataclass so the downstream code
gets typed fields rather than dict-keys.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

Bucket = Literal[
    "equity_delivery", "equity_intraday",
    "options_long", "options_short",
    "options_weekly", "options_monthly",
    "straddle_weekly",
    "futures",
]


# ── Tunables (rough Indian-broker defaults) ─────────────────────────────
EQUITY_MIS_LEVERAGE = 5.0           # 5× MIS typical
SHORT_OPTION_SPAN_PCT = 0.12        # SPAN ≈ 12% of notional
SHORT_OPTION_EXPOSURE_PCT = 0.04    # Exposure ≈ 4% of notional
FUTURES_MARGIN_PCT = 0.135          # ≈ 13.5%
# Straddle gets a small spread benefit because the legs are partially
# offsetting (different directions); broker still asks for the larger of
# the two legs' SPAN. Approximate as 1× CE-leg margin + 0.5× PE-leg margin.
STRADDLE_OFFSET_FACTOR = 1.5


@dataclass
class MarginEstimate:
    bucket: Bucket
    notional: float                 # gross exposure in INR
    span_margin: float = 0.0        # SPAN portion (initial)
    exposure_margin: float = 0.0    # Exposure portion (premium / extreme-move)
    premium_paid: float = 0.0       # long-option premium component
    premium_received: float = 0.0   # short-option premium received (reduces effective margin)
    total_margin: float = 0.0       # what's locked in your account
    leverage: float = 1.0           # notional / total_margin
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def equity_margin(
    side: Literal["BUY", "SELL"],
    quantity: int,
    price: float,
    product: Literal["CNC", "MIS"] = "CNC",
) -> MarginEstimate:
    """Equity delivery (CNC) or intraday (MIS)."""
    notional = float(quantity) * float(price)
    if product == "CNC":
        total = notional
        bucket: Bucket = "equity_delivery"
        note = "CNC · full notional locked"
    else:
        total = notional / EQUITY_MIS_LEVERAGE
        bucket = "equity_intraday"
        note = f"MIS · {EQUITY_MIS_LEVERAGE:g}× leverage assumed"
    return MarginEstimate(
        bucket=bucket,
        notional=notional,
        span_margin=total,           # bundled for equity
        total_margin=total,
        leverage=(notional / total) if total else 1.0,
        note=note,
    )


def option_margin(
    side: Literal["BUY", "SELL"],
    lots: int,
    lot_size: int,
    strike: float,
    premium: float,
    underlying_spot: Optional[float] = None,
    expiry_kind: Literal["weekly", "monthly"] = "weekly",
) -> MarginEstimate:
    """Single-leg option margin estimate.

    * Long: premium × qty (the most you can lose).
    * Short: SPAN + exposure on notional, minus premium received.
    """
    qty = lots * lot_size
    spot = underlying_spot or strike
    notional = spot * qty
    premium_value = premium * qty

    if side == "BUY":
        return MarginEstimate(
            bucket="options_long",
            notional=notional,
            premium_paid=premium_value,
            total_margin=premium_value,
            leverage=(notional / premium_value) if premium_value else 1.0,
            note="Long option · max loss = premium",
        )

    # Short option
    span = notional * SHORT_OPTION_SPAN_PCT
    exposure = notional * SHORT_OPTION_EXPOSURE_PCT
    gross = span + exposure
    net = max(0.0, gross - premium_value)
    bucket: Bucket = "options_weekly" if expiry_kind == "weekly" else "options_monthly"
    return MarginEstimate(
        bucket=bucket,
        notional=notional,
        span_margin=span,
        exposure_margin=exposure,
        premium_received=premium_value,
        total_margin=net,
        leverage=(notional / net) if net else 1.0,
        note=f"Short {expiry_kind} option · SPAN+exposure ≈ {(SHORT_OPTION_SPAN_PCT+SHORT_OPTION_EXPOSURE_PCT)*100:.0f}% notional, premium credit applied",
    )


def short_straddle_margin(
    lots: int,
    lot_size: int,
    strike: float,
    ce_premium: float,
    pe_premium: float,
    underlying_spot: Optional[float] = None,
) -> MarginEstimate:
    """Short straddle = short CE + short PE same strike. Broker grants a
    spread benefit because losses on one leg are bounded by gains on the
    other in many scenarios. Approximation: full margin on the harder leg
    + half on the softer leg.
    """
    ce = option_margin("SELL", lots, lot_size, strike, ce_premium, underlying_spot, "weekly")
    pe = option_margin("SELL", lots, lot_size, strike, pe_premium, underlying_spot, "weekly")
    # Take the higher leg's margin + 0.5 × the other
    hi, lo = (ce, pe) if ce.total_margin >= pe.total_margin else (pe, ce)
    total = hi.total_margin + lo.total_margin * 0.5
    notional = ce.notional + pe.notional
    return MarginEstimate(
        bucket="straddle_weekly",
        notional=notional,
        span_margin=hi.span_margin + lo.span_margin * 0.5,
        exposure_margin=hi.exposure_margin + lo.exposure_margin * 0.5,
        premium_received=ce.premium_received + pe.premium_received,
        total_margin=total,
        leverage=(notional / total) if total else 1.0,
        note=(
            f"Short straddle · SPAN-offset factor {STRADDLE_OFFSET_FACTOR}× · "
            f"premium received {ce.premium_received + pe.premium_received:,.0f}"
        ),
    )


def futures_margin(
    side: Literal["BUY", "SELL"],
    lots: int,
    lot_size: int,
    price: float,
) -> MarginEstimate:
    qty = lots * lot_size
    notional = qty * price
    total = notional * FUTURES_MARGIN_PCT
    return MarginEstimate(
        bucket="futures",
        notional=notional,
        span_margin=total,
        total_margin=total,
        leverage=(notional / total) if total else 1.0,
        note=f"Futures · {FUTURES_MARGIN_PCT*100:.1f}% margin assumed",
    )


def leverage_multiplier(notional: float, margin: float) -> float:
    """Standalone helper for callers that already know notional + margin."""
    return (notional / margin) if margin and margin > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────
# Bucket-level summary
# ─────────────────────────────────────────────────────────────────────────
def summarize_buckets(estimates: list[MarginEstimate]) -> dict:
    """Aggregate a list of margin estimates by bucket.

    Returns:
        {
          "by_bucket": {"equity_delivery": {count, notional, margin, leverage}, ...},
          "totals":    {"notional", "margin", "premium_received", "leverage"}
        }
    """
    out: dict[str, dict] = {}
    for e in estimates:
        b = out.setdefault(e.bucket, {
            "count": 0, "notional": 0.0, "total_margin": 0.0,
            "premium_received": 0.0, "premium_paid": 0.0,
        })
        b["count"] += 1
        b["notional"] += e.notional
        b["total_margin"] += e.total_margin
        b["premium_received"] += e.premium_received
        b["premium_paid"] += e.premium_paid
    for b in out.values():
        b["leverage"] = (b["notional"] / b["total_margin"]) if b["total_margin"] else 1.0
    notional = sum(b["notional"] for b in out.values())
    margin = sum(b["total_margin"] for b in out.values())
    return {
        "by_bucket": out,
        "totals": {
            "notional": notional,
            "total_margin": margin,
            "premium_received": sum(b["premium_received"] for b in out.values()),
            "premium_paid": sum(b["premium_paid"] for b in out.values()),
            "leverage": (notional / margin) if margin else 1.0,
        },
    }
