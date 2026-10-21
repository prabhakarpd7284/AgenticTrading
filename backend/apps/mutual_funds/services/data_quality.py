"""NAV-series data-quality checks (S1). Pure functions over a sorted list of
``(date, nav)`` pairs; the caller persists results as ``MFDataIssue`` rows.

Checks:
- ``gap``          > ``GAP_TRADING_DAYS`` weekdays between consecutive NAVs.
                   Weekdays are a proxy for trading days (exchange holidays are
                   not subtracted), so a reported gap is never shorter than real.
- ``jump``         |NAV_t / NAV_t-1 - 1| > ``JUMP_PCT`` (splits, bonus, unit
                   consolidation, or plain data errors).
- ``nonpositive``  NAV <= 0.

Nothing is fixed or interpolated here; metrics flag windows that overlap an
issue instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

GAP_TRADING_DAYS = 10
JUMP_PCT = 0.25


@dataclass(frozen=True)
class Issue:
    kind: str
    start_date: date
    end_date: date
    detail: dict = field(default_factory=dict, hash=False, compare=False)


def weekdays_between(a: date, b: date) -> int:
    """Weekdays strictly after ``a`` and strictly before ``b``."""
    if b <= a:
        return 0
    return int(np.busday_count(a + timedelta(days=1), b))


def find_issues(navs: list[tuple[date, Decimal | float]], gap_days: int = GAP_TRADING_DAYS,
                jump_pct: float = JUMP_PCT) -> list[Issue]:
    issues: list[Issue] = []
    prev_d = prev_v = None
    for d, v in navs:
        fv = float(v)
        if fv <= 0:
            issues.append(Issue("nonpositive", d, d, {"nav": fv}))
        if prev_d is not None:
            missing = weekdays_between(prev_d, d)
            if missing > gap_days:
                issues.append(Issue("gap", prev_d, d, {"missing_weekdays": missing}))
            if prev_v is not None and prev_v > 0 and fv > 0:
                move = fv / prev_v - 1
                if abs(move) > jump_pct:
                    issues.append(Issue("jump", prev_d, d, {"from": prev_v, "to": fv, "move_pct": round(move * 100, 2)}))
        prev_d, prev_v = d, fv
    return issues
