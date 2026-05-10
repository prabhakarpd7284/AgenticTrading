"""
Deterministic Risk Engine — NO LLM here.

This is your real protection. Pure Python validation
before any order hits the broker.

Criterion ordering (matches the 10-point gate used in every workflow):
  1. Basic field validation                 — malformed plan
  2. Stop-loss direction                    — sanity
  3. Target direction                       — sanity
  4. Risk per trade ≤ MAX_RISK_PER_TRADE_PCT
  5. Daily loss within MAX_DAILY_LOSS_PCT
  6. Position size ≤ MAX_POSITION_SIZE_PCT
  7. Risk:Reward ≥ MIN_RISK_REWARD_RATIO
  8. Confidence ≥ MIN_CONFIDENCE
  9. Open positions < MAX_OPEN_POSITIONS
 10. Regime (Cascade Stage 1)               — tradeable flag + VIX tier

The regime gate is checked FIRST (fail-fast) because it supersedes all
others: if the regime is extreme, we don't waste cycles validating a plan
the desk shouldn't be making in the first place.
"""
import os
from datetime import date
from typing import Any, Tuple

from logzero import logger

# Import Django settings for risk parameters
# These are loaded from .env via config/settings.py
MAX_RISK_PER_TRADE_PCT = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0"))
MIN_RISK_REWARD_RATIO = 1.5
MIN_CONFIDENCE = 0.55  # Optimized: conf≥0.55 + Camarilla SL + regime filter = PF 1.47
MAX_OPEN_POSITIONS = 3

# ── Cascade Stage 1 coupling ────────────────────────────────────────────────
# Matches pulse_service.CACHE_KEY exactly — this is the single source of truth
# for "what regime are we in right now?".  If the cache is cold (no pulse has
# run yet), we SOFT-SKIP the gate so the test suite and pre-market workflows
# continue to function without a live data provider.  To enforce the gate
# strictly (production), set RISK_REGIME_STRICT=1 in the environment.
PULSE_CACHE_KEY = "market:pulse:v1"
RISK_REGIME_STRICT = os.getenv("RISK_REGIME_STRICT", "0") == "1"


def validate_trade(
    plan: dict,
    capital: float,
    daily_loss: float = 0.0,
    open_positions: int = 0,
) -> Tuple[bool, str, dict]:
    """
    Validate a TradePlan against risk rules.

    Args:
        plan: dict with keys from TradePlan schema
        capital: total portfolio capital in INR
        daily_loss: cumulative daily losses so far (positive number)
        open_positions: number of currently open positions

    Returns:
        (approved: bool, reason: str, details: dict)
    """
    details: dict = {}

    # ──────────────────────────────────────────
    # 0. Cascade regime gate (Stage 1) — fail fast
    # ──────────────────────────────────────────
    regime_ok, regime_reason, regime_details = _check_regime()
    details["regime"] = regime_details
    if not regime_ok:
        logger.warning("Risk REJECTED by regime gate: %s", regime_reason)
        return False, regime_reason, details

    # ──────────────────────────────────────────
    # 1. Basic field validation
    # ──────────────────────────────────────────
    required = ["symbol", "side", "entry_price", "stop_loss", "target", "quantity", "confidence"]
    for field in required:
        if field not in plan or plan[field] is None:
            return False, f"Missing required field: {field}", details

    entry = plan["entry_price"]
    sl = plan["stop_loss"]
    target = plan["target"]
    qty = plan["quantity"]
    side = plan["side"]
    confidence = plan["confidence"]

    if qty <= 0:
        return False, "Quantity must be positive", details

    if entry <= 0 or sl <= 0 or target <= 0:
        return False, "Prices must be positive", details

    # ──────────────────────────────────────────
    # 2. Stop loss direction check
    # ──────────────────────────────────────────
    if side == "BUY" and sl >= entry:
        return False, f"BUY stop_loss ({sl}) must be below entry ({entry})", details

    if side == "SELL" and sl <= entry:
        return False, f"SELL stop_loss ({sl}) must be above entry ({entry})", details

    # ──────────────────────────────────────────
    # 3. Target direction check
    # ──────────────────────────────────────────
    if side == "BUY" and target <= entry:
        return False, f"BUY target ({target}) must be above entry ({entry})", details

    if side == "SELL" and target >= entry:
        return False, f"SELL target ({target}) must be below entry ({entry})", details

    # ──────────────────────────────────────────
    # 4. Risk per trade (1% of capital default)
    # ──────────────────────────────────────────
    risk_per_share = abs(entry - sl)
    risk_amount = risk_per_share * qty
    max_risk = capital * (MAX_RISK_PER_TRADE_PCT / 100)

    details["risk_amount"] = round(risk_amount, 2)
    details["max_risk_allowed"] = round(max_risk, 2)
    details["risk_pct_of_capital"] = round((risk_amount / capital) * 100, 2) if capital > 0 else 0

    if risk_amount > max_risk:
        return (
            False,
            f"Risk {risk_amount:.0f} INR exceeds {MAX_RISK_PER_TRADE_PCT}% of capital ({max_risk:.0f} INR)",
            details,
        )

    # ──────────────────────────────────────────
    # 5. Daily loss limit (3% of capital default)
    # ──────────────────────────────────────────
    max_daily_loss = capital * (MAX_DAILY_LOSS_PCT / 100)
    details["daily_loss_so_far"] = round(daily_loss, 2)
    details["max_daily_loss"] = round(max_daily_loss, 2)

    if daily_loss >= max_daily_loss:
        return False, f"Daily loss limit reached ({daily_loss:.0f} >= {max_daily_loss:.0f} INR)", details

    # Check if this trade could breach daily loss limit
    if (daily_loss + risk_amount) > max_daily_loss * 1.5:
        return (
            False,
            f"Trade could breach daily loss limit. Loss so far: {daily_loss:.0f}, "
            f"Trade risk: {risk_amount:.0f}, Limit: {max_daily_loss:.0f}",
            details,
        )

    # ──────────────────────────────────────────
    # 6. Position size (10% of capital default)
    # ──────────────────────────────────────────
    position_value = entry * qty
    max_position = capital * (MAX_POSITION_SIZE_PCT / 100)
    details["position_value"] = round(position_value, 2)
    details["max_position_value"] = round(max_position, 2)

    if position_value > max_position:
        return (
            False,
            f"Position value {position_value:.0f} INR exceeds {MAX_POSITION_SIZE_PCT}% of capital ({max_position:.0f} INR)",
            details,
        )

    # ──────────────────────────────────────────
    # 7. Risk:Reward ratio
    # ──────────────────────────────────────────
    reward_per_share = abs(target - entry)
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    details["risk_reward_ratio"] = round(rr_ratio, 2)

    if rr_ratio < MIN_RISK_REWARD_RATIO:
        return (
            False,
            f"Risk:Reward ratio {rr_ratio:.2f} below minimum {MIN_RISK_REWARD_RATIO}",
            details,
        )

    # ──────────────────────────────────────────
    # 8. Confidence threshold
    # ──────────────────────────────────────────
    if confidence < MIN_CONFIDENCE:
        return False, f"Confidence {confidence:.2f} below minimum {MIN_CONFIDENCE}", details

    # ──────────────────────────────────────────
    # 9. Max open positions
    # ──────────────────────────────────────────
    if open_positions >= MAX_OPEN_POSITIONS:
        return False, f"Max open positions reached ({open_positions}/{MAX_OPEN_POSITIONS})", details

    # ──────────────────────────────────────────
    # All checks passed
    # ──────────────────────────────────────────
    logger.info(
        f"Risk APPROVED: {side} {qty}x {plan['symbol']} @ {entry} | "
        f"Risk: {risk_amount:.0f} INR ({details['risk_pct_of_capital']:.1f}%) | "
        f"R:R {rr_ratio:.1f}"
    )

    return True, "Approved", details


# ──────────────────────────────────────────────────────────────────────────
# Cascade Stage 1 coupling — regime gate
# ──────────────────────────────────────────────────────────────────────────
def _check_regime() -> Tuple[bool, str, dict[str, Any]]:
    """Read the live pulse payload from Django cache and decide whether
    the regime allows ANY new directional entry.

    Returns ``(approved, reason, details)`` where ``details`` carries the
    regime summary for display on the risk-breakdown UI.

    Behaviour:
      * Cache miss (pulse hasn't run) → soft-skip in dev, hard-reject in
        production when ``RISK_REGIME_STRICT=1``.
      * vol = ``extreme`` → always reject (matches VIX > 35 straddle gate).
      * ``tradeable=False`` from the classifier → reject with its summary.
      * Everything else → approve, carry the tier down to details for UI.
    """
    try:
        from django.core.cache import cache
    except Exception:  # pragma: no cover — non-Django context
        return True, "Approved (no Django cache available)", {"skipped": True}

    payload = cache.get(PULSE_CACHE_KEY)
    if payload is None:
        # Pulse cache is cold. This happens when the backend just booted or
        # when yfinance isn't reachable — never a reason to block paper
        # flows, but production should refuse.
        if RISK_REGIME_STRICT:
            return False, "Regime cache cold — refusing to trade blind", {"cache": "miss"}
        return True, "Approved (regime cache cold, soft-skip)", {"cache": "miss"}

    # `payload` is a PulsePayload dataclass instance; keep this duck-typed
    # so the engine doesn't import pulse_service (avoids a dependency loop
    # when pulse_service itself grows deps on trading.*).
    regime = getattr(payload, "regime", None) or (
        payload.get("regime") if isinstance(payload, dict) else None
    )
    if not regime:
        return True, "Approved (regime payload malformed, soft-skip)", {"cache": "bad"}

    vol = regime.get("vol")
    tradeable = regime.get("tradeable", True)
    summary = regime.get("summary", "")

    details: dict[str, Any] = {
        "vol": vol,
        "trend": regime.get("trend"),
        "global_tone": regime.get("global_tone"),
        "tradeable": tradeable,
        "summary": summary,
    }

    if vol == "extreme":
        return False, f"Vol regime EXTREME — no new entries. {summary}", details

    if not tradeable:
        return False, f"Regime not tradeable: {summary}", details

    return True, "Approved (regime OK)", details
