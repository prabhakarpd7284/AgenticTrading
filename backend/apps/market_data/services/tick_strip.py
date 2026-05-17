"""Micro Advance/Decline & TICK Internal Strip.

For the NIFTY-500 universe (proxied here by the SECTOR_CONSTITUENTS
union — the universe we actually have intraday quotes for), compute:

  advancers       constituents with %-change > 0 today
  decliners       constituents with %-change < 0
  unchanged       constituents flat
  ad_line         advancers − decliners
  tick            ad_line normalised to a NYSE-style scale
                  (×800 / universe_size, so a full-flush day pins ~±800)
  ad_ratio        advancers / max(1, decliners)
  extreme         flag ∈ {EXTREME_GREEN, EXTREME_RED, NEUTRAL}
  leaders         top-5 by %-change
  laggards        bottom-5 by %-change

Uses the same `_batch_changes` cache that sector_dispersion already
populates, so this endpoint is cheap when paired with the dispersion
panel.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


_EXTREME_TICK = 400          # NYSE convention; tune as needed
_EXTREME_AD_RATIO = 3.0      # 3× more advancers than decliners (or vice-versa)


def build_tick_strip(tenant=None) -> dict[str, Any]:
    """Snapshot TICK reading for the held universe."""
    from apps.market_data.services.rotation_service import SECTOR_CONSTITUENTS
    from apps.market_data.services.sector_dispersion import _batch_changes

    # Dedupe constituents across sectors
    universe: list[str] = []
    seen: set[str] = set()
    for syms in SECTOR_CONSTITUENTS.values():
        for s in syms:
            if s not in seen:
                seen.add(s); universe.append(s)
    if not universe:
        return {
            "universe_size": 0, "advancers": 0, "decliners": 0, "unchanged": 0,
            "ad_line": 0, "ad_ratio": 0.0, "tick": 0, "extreme": "NEUTRAL",
            "leaders": [], "laggards": [], "as_of": datetime.now(timezone.utc).isoformat(),
            "note": "No constituents available — check rotation_service.SECTOR_CONSTITUENTS.",
        }

    pct = _batch_changes(universe)
    valid = [(sym, v) for sym, v in pct.items() if v is not None]

    advancers = sum(1 for _, v in valid if v > 0.05)
    decliners = sum(1 for _, v in valid if v < -0.05)
    unchanged = sum(1 for _, v in valid if -0.05 <= v <= 0.05)
    ad_line = advancers - decliners
    universe_size = max(1, len(valid))
    tick = round(ad_line * (800.0 / universe_size))
    ad_ratio = round(advancers / max(1, decliners), 2)

    if tick >= _EXTREME_TICK or ad_ratio >= _EXTREME_AD_RATIO:
        extreme = "EXTREME_GREEN"
    elif tick <= -_EXTREME_TICK or (decliners > 0 and decliners / max(1, advancers) >= _EXTREME_AD_RATIO):
        extreme = "EXTREME_RED"
    else:
        extreme = "NEUTRAL"

    sorted_pct = sorted(valid, key=lambda kv: kv[1], reverse=True)
    leaders = [{"symbol": s, "pct": round(v, 2)} for s, v in sorted_pct[:5]]
    laggards = [{"symbol": s, "pct": round(v, 2)} for s, v in sorted_pct[-5:][::-1]]

    return {
        "universe_size": len(valid),
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "ad_line": ad_line,
        "ad_ratio": ad_ratio,
        "tick": tick,
        "extreme": extreme,
        "extreme_threshold": _EXTREME_TICK,
        "leaders": leaders,
        "laggards": laggards,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "note": (
            "TICK is scaled to NYSE convention (±800 ≈ full-flush). "
            "EXTREME_GREEN = late-stage chase; fade or trim longs. "
            "EXTREME_RED = capitulation; first counter-trend bounce is "
            "usually playable. NEUTRAL = trade your normal edges."
        ),
    }
