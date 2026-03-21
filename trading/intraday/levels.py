"""
Level Map — the foundation of profitable trading.

A level is a price where supply and demand have historically shifted.
The more independent sources confirm a level, the higher the probability
of a reaction when price arrives there.

Sources:
  - Previous day high/low/close (universal S/R)
  - Camarilla pivots (mathematical, widely watched)
  - Swing points from daily chart (multi-day supply/demand zones)
  - Round numbers (psychological + options strike clustering)
  - VWAP (institutional average, intraday gravity)
  - Opening range (first 15 min high/low)
  - Gap zones (open vs prev close)

Usage:
    level_map = LevelMap.build(daily_candles, today_open, today_candles)
    nearest, score = level_map.find_nearest(price)
    support = level_map.find_support_below(price)
    resistance = level_map.find_resistance_above(price)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from logzero import logger


@dataclass
class Level:
    """A single price level with metadata."""
    price: float
    source: str          # e.g. "PDH", "Cam_R3", "Swing_High", "Round", "VWAP"
    strength: int        # 0-100 confluence score
    level_type: str      # "support" | "resistance" | "pivot"
    description: str = ""


@dataclass
class LevelMap:
    """
    All significant price levels for a trading session.

    Built pre-market from historical data, then updated intraday
    as VWAP and ORB become available.
    """
    levels: List[Level] = field(default_factory=list)

    # Quick-access named levels
    pdh: float = 0
    pdl: float = 0
    pdc: float = 0
    cam_s3: float = 0
    cam_s4: float = 0
    cam_r3: float = 0
    cam_r4: float = 0
    cam_pivot: float = 0
    vwap: float = 0
    orb_high: float = 0
    orb_low: float = 0
    day_high: float = 0
    day_low: float = 0

    @classmethod
    def build(
        cls,
        daily_candles: list,
        today_open: float = 0,
        intraday_candles: list = None,
    ) -> "LevelMap":
        """
        Build the full level map from historical + intraday data.

        Args:
            daily_candles: List of dicts with {date, open, high, low, close, volume}
                           Last entry = most recent completed day (yesterday).
                           Need at least 5 days for swing detection.
            today_open: Today's opening price (0 if pre-market).
            intraday_candles: Today's 5-min candles so far (for VWAP, ORB, day extremes).

        Returns:
            LevelMap with all levels scored.
        """
        lm = cls()
        all_levels: List[Level] = []

        if not daily_candles:
            return lm

        prev = daily_candles[-1]
        ph = float(prev.get("high", prev[2] if isinstance(prev, (list, tuple)) else 0))
        pl = float(prev.get("low", prev[3] if isinstance(prev, (list, tuple)) else 0))
        pc = float(prev.get("close", prev[4] if isinstance(prev, (list, tuple)) else 0))

        lm.pdh = ph
        lm.pdl = pl
        lm.pdc = pc

        # ── 1. Previous day high/low/close ──
        from trading.config import config as _cfg
        lc = _cfg.levels

        all_levels.append(Level(ph, "PDH", lc.pdh_pdl_base_score, "resistance", "Previous day high"))
        all_levels.append(Level(pl, "PDL", lc.pdh_pdl_base_score, "support", "Previous day low"))
        all_levels.append(Level(pc, "PDC", 15, "pivot", "Previous day close"))

        # ── 2. Camarilla pivots ──
        r = ph - pl
        cam_r3 = pc + r * 1.1 / 4
        cam_r4 = pc + r * 1.1 / 2
        cam_s3 = pc - r * 1.1 / 4
        cam_s4 = pc - r * 1.1 / 2
        cam_p = (ph + pl + pc) / 3

        lm.cam_r3 = cam_r3
        lm.cam_r4 = cam_r4
        lm.cam_s3 = cam_s3
        lm.cam_s4 = cam_s4
        lm.cam_pivot = cam_p

        all_levels.append(Level(cam_r3, "Cam_R3", lc.camarilla_base_score, "resistance", "Camarilla R3 — sell zone"))
        all_levels.append(Level(cam_r4, "Cam_R4", lc.camarilla_base_score, "resistance", "Camarilla R4 — stop hunt"))
        all_levels.append(Level(cam_s3, "Cam_S3", lc.camarilla_base_score, "support", "Camarilla S3 — buy zone"))
        all_levels.append(Level(cam_s4, "Cam_S4", lc.camarilla_base_score, "support", "Camarilla S4 — stop hunt"))
        all_levels.append(Level(cam_p, "Cam_Pivot", 15, "pivot", "Camarilla pivot"))

        # ── 3. Swing points from daily chart (past 10 days) ──
        swings = _find_swing_points(daily_candles, lookback=2)
        for stype, price, dt in swings:
            ltype = "resistance" if stype == "SWING_HIGH" else "support"
            all_levels.append(Level(price, stype, lc.swing_base_score, ltype, f"Swing {stype.lower()} from {dt}"))

        # ── 4. Round numbers (within 500 pts of prev close) ──
        # NIFTY: every 100 pts. Stocks: every 50 or 100 depending on price.
        step = lc.round_step_nifty if pc > 5000 else lc.round_step_stock if pc > 500 else 10
        base = int(pc / step) * step
        for rn in range(base - 5 * step, base + 6 * step, step):
            if abs(rn - pc) < 5 * step:
                all_levels.append(Level(float(rn), "Round", lc.round_base_score, "pivot",
                                        f"Round number {rn}"))

        # ── 5. Week high/low (from daily data) ──
        if len(daily_candles) >= 5:
            week_data = daily_candles[-5:]
            week_high = max(_get_high(c) for c in week_data)
            week_low = min(_get_low(c) for c in week_data)
            all_levels.append(Level(week_high, "Week_High", lc.week_hl_base_score, "resistance", "5-day high"))
            all_levels.append(Level(week_low, "Week_Low", lc.week_hl_base_score, "support", "5-day low"))

        # ── 6. Intraday levels (if candles available) ──
        if intraday_candles and len(intraday_candles) >= 3:
            # VWAP
            vwap_val = _calc_vwap(intraday_candles)
            if vwap_val > 0:
                lm.vwap = vwap_val
                all_levels.append(Level(vwap_val, "VWAP", lc.vwap_base_score, "pivot",
                                        "Volume weighted average price"))

            # ORB (first 3 candles = 15 min)
            orb_candles = intraday_candles[:3]
            orb_high = max(_get_high(c) for c in orb_candles)
            orb_low = min(_get_low(c) for c in orb_candles)
            lm.orb_high = orb_high
            lm.orb_low = orb_low
            all_levels.append(Level(orb_high, "ORB_High", lc.orb_base_score, "resistance", "Opening range high"))
            all_levels.append(Level(orb_low, "ORB_Low", lc.orb_base_score, "support", "Opening range low"))

            # Day high/low (running)
            lm.day_high = max(_get_high(c) for c in intraday_candles)
            lm.day_low = min(_get_low(c) for c in intraday_candles)

        # ── 7. Gap zone ──
        if today_open > 0 and pc > 0:
            gap_pct = abs(today_open - pc) / pc * 100
            if gap_pct > 0.3:
                all_levels.append(Level(pc, "Gap_Fill_Target", 15, "pivot",
                                        f"Gap fill target (prev close)"))
                all_levels.append(Level(today_open, "Gap_Open", 10, "pivot",
                                        f"Today's open (gap edge)"))

        # ── Score levels by confluence ──
        scored_levels = _score_confluence(all_levels)

        lm.levels = sorted(scored_levels, key=lambda x: x.price)
        return lm

    def find_nearest(self, price: float, max_dist: float = 50) -> Tuple[Optional[Level], int]:
        """
        Find the nearest level to a price. Returns (level, score) or (None, 0).
        max_dist: maximum distance in price points to consider.
        """
        best = None
        best_dist = max_dist + 1
        for lvl in self.levels:
            dist = abs(lvl.price - price)
            if dist < best_dist:
                best_dist = dist
                best = lvl
        if best and best_dist <= max_dist:
            return best, best.strength
        return None, 0

    def find_support_below(self, price: float) -> Optional[Level]:
        """Find the strongest support level below price."""
        supports = [l for l in self.levels if l.price < price and l.level_type in ("support", "pivot")]
        if not supports:
            return None
        return max(supports, key=lambda x: x.strength)

    def find_resistance_above(self, price: float) -> Optional[Level]:
        """Find the strongest resistance level above price."""
        resistances = [l for l in self.levels if l.price > price and l.level_type in ("resistance", "pivot")]
        if not resistances:
            return None
        return max(resistances, key=lambda x: x.strength)

    def levels_in_range(self, low: float, high: float) -> List[Level]:
        """All levels between low and high, sorted by strength."""
        return sorted(
            [l for l in self.levels if low <= l.price <= high],
            key=lambda x: -x.strength,
        )

    def summary(self, current_price: float = 0) -> str:
        """Human-readable summary for LLM context or logging."""
        lines = ["LEVEL MAP:"]
        for lvl in self.levels:
            if lvl.strength >= 20:  # only show significant levels
                dist = f" ({lvl.price - current_price:+.0f})" if current_price else ""
                marker = ""
                if current_price and abs(lvl.price - current_price) < 30:
                    marker = " ← PRICE HERE"
                lines.append(
                    f"  {lvl.source:15s} {lvl.price:>8.0f} "
                    f"str={lvl.strength:>3d} {lvl.level_type:>10s}{dist}{marker}"
                )
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _get_high(candle) -> float:
    if isinstance(candle, dict):
        return float(candle.get("high", 0))
    if isinstance(candle, (list, tuple)):
        return float(candle[2]) if len(candle) > 2 else 0
    return 0


def _get_low(candle) -> float:
    if isinstance(candle, dict):
        return float(candle.get("low", 0))
    if isinstance(candle, (list, tuple)):
        return float(candle[3]) if len(candle) > 3 else 0
    return 0


def _find_swing_points(daily_candles: list, lookback: int = 2) -> list:
    """
    Find swing highs and lows from daily candle data.

    A swing high = a candle whose high is higher than N candles on each side.
    A swing low = a candle whose low is lower than N candles on each side.

    Returns: list of (type, price, date_str)
    """
    swings = []
    n = len(daily_candles)

    for i in range(lookback, n - lookback):
        c = daily_candles[i]
        h = _get_high(c)
        l = _get_low(c)
        dt = c.get("date", c[0][:10] if isinstance(c, (list, tuple)) else f"day_{i}")

        # Check swing high
        is_high = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i or j < 0 or j >= n:
                continue
            if _get_high(daily_candles[j]) > h:
                is_high = False
                break
        if is_high:
            swings.append(("SWING_HIGH", h, dt))

        # Check swing low
        is_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i or j < 0 or j >= n:
                continue
            if _get_low(daily_candles[j]) < l:
                is_low = False
                break
        if is_low:
            swings.append(("SWING_LOW", l, dt))

    return swings


def _calc_vwap(candles: list) -> float:
    """VWAP from intraday candles."""
    cum_pv = 0.0
    cum_vol = 0
    for c in candles:
        if isinstance(c, dict):
            tp = (c.get("high", 0) + c.get("low", 0) + c.get("close", 0)) / 3
            vol = c.get("volume", 0)
        elif isinstance(c, (list, tuple)):
            tp = (float(c[2]) + float(c[3]) + float(c[4])) / 3 if len(c) > 4 else 0
            vol = int(c[5]) if len(c) > 5 else 0
        else:
            continue
        cum_pv += tp * vol
        cum_vol += vol
    return cum_pv / cum_vol if cum_vol > 0 else 0


def _score_confluence(levels: List[Level]) -> List[Level]:
    """
    Score each level by how many other levels cluster near it.

    If PDH and Cam_R3 and a round number are all within 30 pts,
    each gets a boost because 3 independent sources confirm the zone.
    """
    from trading.config import config as _cfg
    CLUSTER_DIST = _cfg.levels.cluster_distance

    for i, lvl in enumerate(levels):
        cluster_boost = 0
        for j, other in enumerate(levels):
            if i == j:
                continue
            if abs(lvl.price - other.price) <= CLUSTER_DIST:
                # Boost based on the OTHER level's base strength
                cluster_boost += other.strength // 3  # ~7-10 pts per clustering level

        # Cap total boost at 40 to prevent runaway scores
        lvl.strength = min(100, lvl.strength + min(cluster_boost, 40))

    return levels
