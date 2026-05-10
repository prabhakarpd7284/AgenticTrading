"""Market mood assessment — morning tone for basket direction.

Inputs:
  1. Advance/decline ratio of NIFTY 50 components
  2. NIFTY spot vs previous close (gap direction)
  3. India VIX level (calm / elevated / spike)

Output:
  MoodAssessment with BULLISH / BEARISH / NEUTRAL + confidence + reasons
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from logzero import logger

from trading.basket.config import BasketConfig


class MarketMood(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class MoodAssessment:
    mood: MarketMood
    confidence: float                  # 0-1
    advance: int = 0
    decline: int = 0
    ad_ratio: float = 0.0
    nifty_spot: float = 0.0
    nifty_prev_close: float = 0.0
    gap_pct: float = 0.0
    vix: float = 0.0
    vix_tier: str = "normal"           # calm / normal / elevated / spike
    reasons: List[str] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.mood == MarketMood.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.mood == MarketMood.BEARISH

    def to_dict(self) -> dict:
        return {
            "mood": self.mood.value,
            "confidence": self.confidence,
            "advance": self.advance,
            "decline": self.decline,
            "ad_ratio": round(self.ad_ratio, 2),
            "nifty_spot": self.nifty_spot,
            "nifty_prev_close": self.nifty_prev_close,
            "gap_pct": round(self.gap_pct, 2),
            "vix": self.vix,
            "vix_tier": self.vix_tier,
            "reasons": self.reasons,
        }


class MarketMoodAssessor:
    """Assess morning market mood from live data."""

    def __init__(self, cfg: BasketConfig = None):
        from trading.basket.config import BasketConfig
        self.cfg = cfg or BasketConfig()

    def assess(self) -> MoodAssessment:
        """Fetch live data and compute mood. Works during market hours."""
        from trading.services.data_service import BrokerClient
        from dashboard_utils.market_scanner import NIFTY_50_SYMBOLS, fetch_nifty50_ltp

        broker = BrokerClient.get_instance()
        broker.ensure_login()

        # 1. NIFTY spot + prev close
        nifty_data = broker.ltp("NSE", "NIFTY", "99926000")
        nifty_spot = nifty_data.get("ltp", 0)
        nifty_prev = nifty_data.get("close", nifty_spot)
        gap_pct = ((nifty_spot - nifty_prev) / nifty_prev * 100) if nifty_prev else 0

        # 2. VIX
        vix_data = broker.ltp("NSE", "INDIA VIX", "99926017")
        vix = vix_data.get("ltp", 0)

        # 3. Advance / Decline (batch fetch — 1 API call)
        stocks = fetch_nifty50_ltp(broker, NIFTY_50_SYMBOLS)
        advance = sum(1 for s in stocks if s.get("change_pct", 0) > 0)
        decline = sum(1 for s in stocks if s.get("change_pct", 0) < 0)
        flat = len(stocks) - advance - decline
        ad_ratio = advance / max(decline, 1)

        # 4. VIX tier
        if vix < 13:
            vix_tier = "calm"
        elif vix < 18:
            vix_tier = "normal"
        elif vix < self.cfg.vix_elevated:
            vix_tier = "elevated"
        else:
            vix_tier = "spike"

        # 5. Score mood
        bull_score = 0
        bear_score = 0
        reasons: List[str] = []

        # A/D ratio
        if ad_ratio >= self.cfg.ad_bullish:
            bull_score += 2
            reasons.append(f"A/D {advance}/{decline} ({ad_ratio:.1f}) bullish")
        elif ad_ratio <= self.cfg.ad_bearish:
            bear_score += 2
            reasons.append(f"A/D {advance}/{decline} ({ad_ratio:.1f}) bearish")
        else:
            reasons.append(f"A/D {advance}/{decline} ({ad_ratio:.1f}) mixed")

        # Gap direction
        if gap_pct > 0.3:
            bull_score += 1
            reasons.append(f"Gap up {gap_pct:+.2f}%")
        elif gap_pct < -0.3:
            bear_score += 1
            reasons.append(f"Gap down {gap_pct:+.2f}%")
        else:
            reasons.append(f"Flat open {gap_pct:+.2f}%")

        # VIX
        if vix_tier == "spike":
            bear_score += 1
            reasons.append(f"VIX spike {vix:.1f}")
        elif vix_tier == "calm":
            bull_score += 0.5
            reasons.append(f"VIX calm {vix:.1f}")
        else:
            reasons.append(f"VIX {vix:.1f} ({vix_tier})")

        # Final mood
        total = bull_score + bear_score
        if bull_score > bear_score and bull_score >= 2:
            mood = MarketMood.BULLISH
            confidence = min(bull_score / max(total, 1), 1.0)
        elif bear_score > bull_score and bear_score >= 2:
            mood = MarketMood.BEARISH
            confidence = min(bear_score / max(total, 1), 1.0)
        else:
            mood = MarketMood.NEUTRAL
            confidence = 0.3

        logger.info(
            f"Mood: {mood.value} (conf={confidence:.0%}) | "
            f"A/D={advance}/{decline} gap={gap_pct:+.2f}% VIX={vix:.1f}"
        )

        return MoodAssessment(
            mood=mood,
            confidence=round(confidence, 2),
            advance=advance,
            decline=decline,
            ad_ratio=round(ad_ratio, 2),
            nifty_spot=round(nifty_spot, 2),
            nifty_prev_close=round(nifty_prev, 2),
            gap_pct=round(gap_pct, 2),
            vix=round(vix, 2),
            vix_tier=vix_tier,
            reasons=reasons,
        )
