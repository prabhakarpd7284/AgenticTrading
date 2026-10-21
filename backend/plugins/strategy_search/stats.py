"""Small statistics helpers for honest reporting (power, break-even slippage).

Normal approximations — good enough to size a forward test, not to claim
significance.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def trades_for_t(sd: float, mu: float, t: float = 2.0) -> int:
    """Trades needed for the expected t-stat of a per-trade mean ``mu`` (SD
    ``sd``) to reach ``t``: n = (t·sd/mu)²."""
    if mu == 0:
        return math.inf  # type: ignore[return-value]
    return math.ceil((t * sd / abs(mu)) ** 2)


def p_mean_above(threshold: float, mu: float, sd: float, n: int) -> float:
    """P(sample mean of n trades > threshold) when the true mean is ``mu``."""
    se = sd / math.sqrt(n)
    return 1.0 - _phi((threshold - mu) / se)


def detectable_edge(sd: float, n: int, z: float = 1.645) -> float:
    """Smallest true mean a one-sided test at ``z`` would flag 50 % of the time."""
    return z * sd / math.sqrt(n)


def zero_crossing(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Linear interpolation of the first x where y crosses 0 (xs ascending).
    Used for break-even slippage: xs = slippage % per side, ys = net R/trade.
    None if y never changes sign across the grid; extrapolates linearly from
    the first two points if all ys are positive and falling."""
    pts = list(zip(xs, ys, strict=True))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if y0 == 0:
            return x0
        if (y0 > 0) != (y1 > 0) and y1 != y0:
            return x0 + (x1 - x0) * y0 / (y0 - y1)
    if len(pts) >= 2 and all(y > 0 for y in ys) and ys[-1] < ys[0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return x1 + (x1 - x0) * y1 / (y0 - y1) if y0 != y1 else None
    return None
