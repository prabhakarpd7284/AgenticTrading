"""Risk:Reward must be judged at the precision it is reported.

The engine reports `rr` rounded to 2dp but compared the raw float against the
minimum, producing the self-contradictory rejection:

    "Risk:Reward ratio 1.50 below minimum 1.5"

Two ways a legitimately-1.5 signal lands under the bar:

  * float representation — 2136.93/1424.62 == 1.4999999999999973
  * producer rounding — the swing_v2 scanner rounds its target to 2dp, so a
    target designed for exactly 1.5 computes as 1.499946791529215

Since swing_v2 targets are *constructed* for RR 1.5, this rejected every one
of its signals, permanently. Observed live on 2026-09-09 with PAYTM and
PAGEIND.
"""
from __future__ import annotations

import pytest

from apps.trading.services.risk_engine import RiskEngine


def _rr_of(entry: float, sl: float, target: float) -> float:
    return abs(target - entry) / abs(entry - sl)


@pytest.mark.parametrize(
    "name,entry,sl,target",
    [
        # Real signals that were rejected in production.
        ("PAGEIND", 35225.00, 36649.62, 33088.07),  # rr = 1.4999999999999973
        ("PAYTM", 1740.10, 1646.13, 1881.05),       # rr = 1.499946791529215
    ],
)
def test_rr_at_the_boundary_is_accepted(name, entry, sl, target):
    """A signal whose reported RR is 1.50 must not be rejected for being
    'below minimum 1.5'."""
    rr = _rr_of(entry, sl, target)
    assert round(rr, 2) == 1.5, f"{name} should report as 1.50, got {rr}"

    engine = RiskEngine()
    min_rr = float(engine.limits.get("MIN_RISK_REWARD_RATIO", 1.5))

    assert not engine._rr_below_minimum(rr, min_rr), (
        f"{name}: rr={rr} reports as {rr:.2f} but was judged below {min_rr}"
    )


def test_genuinely_low_rr_is_still_rejected():
    """Guard against over-loosening — a real sub-1.5 ratio must still fail."""
    engine = RiskEngine()
    min_rr = float(engine.limits.get("MIN_RISK_REWARD_RATIO", 1.5))

    assert engine._rr_below_minimum(1.44, min_rr)
    assert engine._rr_below_minimum(1.0, min_rr)
    # 1.494 reports as 1.49 — below the bar at the reported precision.
    assert engine._rr_below_minimum(1.494, min_rr)


def test_comfortably_high_rr_still_passes():
    engine = RiskEngine()
    min_rr = float(engine.limits.get("MIN_RISK_REWARD_RATIO", 1.5))

    assert not engine._rr_below_minimum(2.0, min_rr)
    assert not engine._rr_below_minimum(1.51, min_rr)
