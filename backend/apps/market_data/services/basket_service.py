"""Basket service — morning basket status for AlphaDesk UI.

Caches mood+signals for 60s during market hours (mood changes slowly).
Force-refresh bypasses cache.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from django.core.cache import cache

logger = logging.getLogger(__name__)

CACHE_KEY = "market:basket:v1"
CACHE_TTL = 60  # 1 minute — mood doesn't flip every second


def build_basket_status(force: bool = False) -> dict:
    """Return current basket state: mood + signals.

    Cached 60s during market hours. Use force=True to bypass.
    """
    if not force:
        cached = cache.get(CACHE_KEY)
        if cached is not None:
            return cached

    from trading.basket.config import BasketConfig
    from trading.basket.mood import MarketMoodAssessor, MarketMood

    cfg = BasketConfig()

    try:
        assessor = MarketMoodAssessor(cfg)
        mood = assessor.assess()
    except Exception as e:
        logger.warning("Basket mood assessment failed: %s", e)
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "mood": "UNKNOWN",
            "mood_details": {},
            "signals": [],
            "errors": [str(e)],
        }

    result = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "mood": mood.mood.value,
        "mood_details": mood.to_dict(),
        "signals": [],
        "errors": [],
    }

    if mood.mood != MarketMood.NEUTRAL:
        try:
            from trading.basket.signals import BasketSignalGenerator
            gen = BasketSignalGenerator(cfg)
            signals = gen.generate(mood)
            result["signals"] = [s.to_dict() for s in signals]
        except Exception as e:
            logger.warning("Basket signal generation failed: %s", e)
            result["errors"].append(f"Signal error: {e}")

    cache.set(CACHE_KEY, result, CACHE_TTL)
    return result
