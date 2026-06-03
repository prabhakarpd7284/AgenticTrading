"""Paper broker adapter — simulates fills and returns empty positions/holdings.

Conforms to BrokerAdapterBase so it slots into the multi-broker factory +
combined positions endpoint. A paper link exists so the operator can dry-run
the linking UX without a real broker account.

Options chain support synthesizes a Black-Scholes + put-skew chain so
strategies tested against the paper adapter see deterministic, plausibly
priced quotes calibrated to typical India VIX regime.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from math import erf, exp, log, sqrt
from typing import Any, Optional

from apps.agents_core.domain.contracts import BrokerOrderId
from apps.market_data.adapters.base import (
    BrokerAdapterBase, BrokerHealth, Holding, Margin, OptionQuote,
    OptionsChainRow, OptionsChainSnapshot, Position,
)


# Paper-mode anchors so strategies run end-to-end without live feeds.
PAPER_SPOTS: dict[str, float] = {
    "NIFTY": 24173.05,
    "BANKNIFTY": 52000.0,
    "SENSEX": 79800.0,
}
PAPER_VIX: float = 18.59
PAPER_STRIKE_STEP: dict[str, int] = {
    "NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100,
}
RISK_FREE: float = 0.07     # India 10Y proxy


class PaperBrokerAdapter(BrokerAdapterBase):
    name = "paper"

    def __init__(self, credentials: dict[str, Any] | None = None, meta: dict[str, Any] | None = None):
        super().__init__(credentials or {}, meta or {})

    def authenticate(self) -> bool:
        return True

    def health_check(self) -> BrokerHealth:
        return BrokerHealth(ok=True, detail="paper", latency_ms=0.0)

    def fetch_positions(self) -> list[Position]:
        return []

    def fetch_holdings(self) -> list[Holding]:
        return []

    def fetch_margin(self) -> Margin:
        return Margin(available_cash=500_000.0, used=0.0, total=500_000.0)

    def place(self, order: dict) -> BrokerOrderId:
        return BrokerOrderId(broker="paper", id=f"PAPER-{uuid.uuid4().hex[:12]}")

    def cancel(self, order_id: BrokerOrderId) -> None:
        return None

    async def stream_ticks(self, tokens):
        # yields nothing — paper broker has no live feed
        return
        yield  # pragma: no cover

    # ── Options chain (BSM-synthesized) ───────────────────────────────
    def options_chain(
        self,
        underlying: str,
        expiry: Optional[str] = None,
        strikes_window: int = 20,
    ) -> Optional[OptionsChainSnapshot]:
        """Synthesize a put-skewed BSM chain anchored at PAPER_SPOTS.

        Picks the nearest weekly Tuesday expiry when none is supplied
        (matches NSE's post-2025 single-weekly convention). Greeks are
        analytical BSM (no smile adjustment beyond the put-skew applied
        on each leg's IV).
        """
        u = underlying.upper()
        spot = PAPER_SPOTS.get(u)
        if not spot:
            return None
        step = PAPER_STRIKE_STEP.get(u, 50)
        atm = int(round(spot / step) * step)

        if expiry is None:
            exp_date = _next_tuesday()
            expiry = exp_date.strftime("%d%b%Y").upper()
        else:
            try:
                exp_date = datetime.strptime(expiry, "%d%b%Y").date()
            except ValueError:
                exp_date = _next_tuesday()
                expiry = exp_date.strftime("%d%b%Y").upper()
        dte = max(0, (exp_date - date.today()).days)
        T = max(dte, 0) / 365.0

        # ATM vol = VIX/100 × weekly tenor adjustment
        sigma_atm = (PAPER_VIX / 100.0) * sqrt(max(T * 52, 0.5))
        sigma_atm = max(0.05, min(sigma_atm, 1.0))

        rows: list[OptionsChainRow] = []
        for i in range(-strikes_window, strikes_window + 1):
            k = atm + i * step
            if k <= 0:
                continue
            # Put-skew: OTM puts at higher IV, OTM calls at slightly lower IV.
            log_money = log(k / spot)
            iv_pe = max(0.05, sigma_atm + (0.45 if log_money < 0 else -0.08) * abs(log_money))
            iv_ce = max(0.05, sigma_atm + (-0.08 if log_money < 0 else 0.45) * abs(log_money))

            pe_ltp = _bsm_put(spot, k, T, RISK_FREE, iv_pe)
            ce_ltp = _bsm_call(spot, k, T, RISK_FREE, iv_ce)

            # synthetic ±2% spread, decreasing liquidity away from ATM
            spread_bps = 200 + 50 * abs(i)
            oi_curve = max(50_000 - 5_000 * abs(i), 100)

            pe = OptionQuote(
                token=f"PAPER-PE-{k}",
                symbol=f"{u}{exp_date.strftime('%d%b%y').upper()}{k}PE",
                strike=k, opt="PE",
                ltp=round(pe_ltp, 2),
                bid=round(pe_ltp * (1 - spread_bps / 20_000), 2),
                ask=round(pe_ltp * (1 + spread_bps / 20_000), 2),
                bid_qty=oi_curve, ask_qty=oi_curve,
                volume=oi_curve * 3, oi=oi_curve, oi_change=0,
                iv=round(iv_pe, 4),
                delta=round(_delta_put(spot, k, T, RISK_FREE, iv_pe), 4),
                gamma=round(_gamma(spot, k, T, RISK_FREE, iv_pe), 6),
                theta=round(_theta_put(spot, k, T, RISK_FREE, iv_pe) / 365, 4),
                vega=round(_vega(spot, k, T, RISK_FREE, iv_pe) / 100, 4),
            )
            ce = OptionQuote(
                token=f"PAPER-CE-{k}",
                symbol=f"{u}{exp_date.strftime('%d%b%y').upper()}{k}CE",
                strike=k, opt="CE",
                ltp=round(ce_ltp, 2),
                bid=round(ce_ltp * (1 - spread_bps / 20_000), 2),
                ask=round(ce_ltp * (1 + spread_bps / 20_000), 2),
                bid_qty=oi_curve, ask_qty=oi_curve,
                volume=oi_curve * 3, oi=oi_curve, oi_change=0,
                iv=round(iv_ce, 4),
                delta=round(_delta_call(spot, k, T, RISK_FREE, iv_ce), 4),
                gamma=round(_gamma(spot, k, T, RISK_FREE, iv_ce), 6),
                theta=round(_theta_call(spot, k, T, RISK_FREE, iv_ce) / 365, 4),
                vega=round(_vega(spot, k, T, RISK_FREE, iv_ce) / 100, 4),
            )
            rows.append(OptionsChainRow(strike=k, ce=ce, pe=pe))

        return OptionsChainSnapshot(
            underlying=u, spot=spot, expiry=expiry,
            fetched_at=datetime.now(timezone.utc),
            rows=rows, source="paper",
            vix=PAPER_VIX, atm_strike=atm,
        )


# ─────────────── BSM math (inlined to avoid the plugin import) ───────────────

def _N(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2)))


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))


def _bsm_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * _N(-d2) - S * _N(-d1)


def _bsm_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)


def _delta_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    return _N(_d1(S, K, T, r, sigma))


def _delta_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    return _N(_d1(S, K, T, r, sigma)) - 1.0


def _gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    from math import pi
    d1 = _d1(S, K, T, r, sigma)
    return (1.0 / sqrt(2 * pi) * exp(-0.5 * d1 * d1)) / (S * sigma * sqrt(T))


def _vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    from math import pi
    d1 = _d1(S, K, T, r, sigma)
    return S * sqrt(T) * (1.0 / sqrt(2 * pi)) * exp(-0.5 * d1 * d1)


def _theta_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    from math import pi
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    npdf = (1.0 / sqrt(2 * pi)) * exp(-0.5 * d1 * d1)
    return -(S * npdf * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * _N(d2)


def _theta_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    from math import pi
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    npdf = (1.0 / sqrt(2 * pi)) * exp(-0.5 * d1 * d1)
    return -(S * npdf * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * _N(-d2)


def _next_tuesday(after: date | None = None) -> date:
    """Next Tuesday on/after `after` (default: today). Matches NSE's
    post-2025 single-weekly convention."""
    d = after or date.today()
    days = (1 - d.weekday()) % 7
    if days == 0:
        days = 7
    return d + timedelta(days=days)
