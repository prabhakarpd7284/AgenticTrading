"""Black-Scholes-Merton pricing + IV solver.

Shared primitive — eventually moves to apps.market_data.services.bsm
(ADR-0006 P3) so the straddle plugin can reuse it too. Until then,
lives here as a self-contained module with zero Django dependencies.

All functions are pure: same inputs → same outputs, no I/O, no globals.
"""
from __future__ import annotations

from math import erf, exp, log, sqrt

__all__ = [
    "bsm_call", "bsm_put", "iv_from_price",
    "delta_call", "delta_put", "theta_call", "theta_put",
]


def _norm_cdf(x: float) -> float:
    """N(x) — standard normal cumulative distribution."""
    return 0.5 * (1.0 + erf(x / sqrt(2)))


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))


def bsm_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price under BSM. S spot, K strike, T years, r risk-free, sigma IV."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def bsm_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price under BSM."""
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def iv_from_price(
    target: float, S: float, K: float, T: float, r: float,
    is_call: bool = False, tol: float = 1e-4, max_iter: int = 80,
) -> float:
    """Bisection solver for implied volatility.

    Returns IV (volatility, e.g. 0.18 = 18%). Robust under all market
    regimes — converges within ~50 iterations even for far-OTM strikes.
    Returns the bracket midpoint at convergence; caller is responsible
    for sanity-checking against typical ranges (≈0.05 — 1.0 for index).
    """
    lo, hi = 0.001, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = bsm_call(S, K, T, r, mid) if is_call else bsm_put(S, K, T, r, mid)
        if price > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """N(d1) — call delta. Range [0, 1]."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    return _norm_cdf(_d1(S, K, T, r, sigma))


def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """N(d1) − 1 — put delta. Range [−1, 0]."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    return _norm_cdf(_d1(S, K, T, r, sigma)) - 1.0


def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Call theta in price-units per year. Divide by 365 for daily decay."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    # N'(d1) = (1/√2π) e^(−d1²/2)
    from math import pi
    npdf_d1 = (1.0 / sqrt(2 * pi)) * exp(-0.5 * d1 * d1)
    return -(S * npdf_d1 * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * _norm_cdf(d2)


def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put theta in price-units per year. Divide by 365 for daily decay."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    from math import pi
    npdf_d1 = (1.0 / sqrt(2 * pi)) * exp(-0.5 * d1 * d1)
    return -(S * npdf_d1 * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * _norm_cdf(-d2)
