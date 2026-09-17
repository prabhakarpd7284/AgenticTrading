"""
Entry Advisor — decide whether to OPEN a short straddle.

Companion to analyzer.py (which manages already-open positions). Pure Python,
deterministic. No LLM. Same Angel One data path the rest of the options module
uses (BrokerClient singleton → batch market_data + daily candles).

Given (underlying, expiry_iso), returns ENTER / WAIT / SKIP with a scored
breakdown, the suggested ATM strike, resolved CE/PE legs, expected premium,
breakevens, and the 1.5× hard-stop risk in INR.

Decision rubric (deterministic — change the constants here, not in callers):

  VIX phase        — premium environment. <11 cheap, 14-24 sweet, ≥24 risky.
  IV vs RV edge    — implied move (combined premium / spot) vs realized N-day
                     move (close-to-close stdev × sqrt(DTE)). >1.2× means
                     the market is paying us more than recent reality warrants.
  DTE              — 3-7 sweet, 1-2 gamma-trap, 0 reject.
  Trend strength   — N consecutive same-direction closes is a red flag for
                     a delta-neutral premium-sell strategy.
  Strike fit       — how close ATM lands to spot (proxy for initial delta).

Aggregated to a 0-100 score:
  ≥ 55 → ENTER
  35..54 → WAIT (right setup, wrong timing — re-check next session)
  < 35 → SKIP
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from logzero import logger

from trading.utils.expiry_utils import (
    days_to_expiry as _dte,
    iso_to_angel,
    normalize_expiry,
)
from trading.utils.time_utils import last_trading_day


# ──────────────────────────────────────────────
# Per-underlying constants
# Mirrors frontend/src/lib/market-config.ts — kept here for backend use only.
# If these drift, fix market-config.ts first and copy values back.
# ──────────────────────────────────────────────
UNDERLYINGS: dict[str, dict] = {
    "NIFTY": {
        "spot_token": "99926000",
        "spot_exchange": "NSE",
        "option_exchange": "NFO",
        "lot_size": 65,         # NSE circular FAOP70616 (Jan 2026)
        "strike_step": 50,
        "weekly": True,
    },
    "BANKNIFTY": {
        "spot_token": "99926009",
        "spot_exchange": "NSE",
        "option_exchange": "NFO",
        "lot_size": 30,
        "strike_step": 100,
        "weekly": False,        # monthly only post Sep-2025
    },
    "SENSEX": {
        "spot_token": "99919000",
        "spot_exchange": "BSE",
        "option_exchange": "BFO",
        "lot_size": 20,
        "strike_step": 100,
        "weekly": True,
    },
}

VIX_TOKEN = "99926017"
VIX_EXCHANGE = "NSE"

# ── Scoring weights (sum tunable; total max ≈ 100) ──
SCORE_ENTER = 55
SCORE_WAIT = 35
HARD_STOP_MULT = 1.5            # matches analyzer.HARD_STOP_MULTIPLIER
REALIZED_VOL_LOOKBACK = 15      # trading days for RV
TREND_LOOKBACK = 3              # consecutive same-direction closes

# ── Hard vetoes (override the numeric score; analogous to @RiskGuard rules) ──
# IV/RV below this means the market is paying us less than recent realized
# movement would warrant. A short straddle in that environment is asymmetric
# risk in the wrong direction — never ENTER regardless of other inputs.
VETO_IV_RV_RATIO = 0.9


@dataclass
class EntryDecision:
    decision: str                       # "ENTER" | "WAIT" | "SKIP"
    score: int
    underlying: str
    expiry_iso: str
    expiry_angel: str
    dte: int

    spot: float
    vix: float
    vix_prev: float

    suggested_strike: int
    ce_symbol: str
    ce_token: str
    ce_ltp: float
    pe_symbol: str
    pe_token: str
    pe_ltp: float

    combined_premium_pts: float
    combined_premium_inr: float         # per lot
    breakeven_low: float
    breakeven_high: float
    implied_move_pct: float              # premium / spot

    realized_move_pct: float             # daily stdev × √DTE (over lookback)
    iv_rv_ratio: float                   # implied_move / realized_move
    trend_streak: int                    # signed: +N up, -N down, 0 mixed

    max_loss_inr: float                  # combined_premium × 0.5 × lot_size (= 1.5× stop hit)
    score_components: list[tuple[str, int, str]] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Realized vol from daily candles
# ──────────────────────────────────────────────
def _fetch_daily_closes(broker, token: str, exchange: str, lookback_days: int) -> list[float]:
    """Pull `lookback_days` worth of daily closes via broker.fetch_candles.

    We pad the calendar window so weekends/holidays don't starve the lookback.
    Returns closes oldest→newest.
    """
    end = last_trading_day()
    start = end - timedelta(days=lookback_days * 2 + 5)   # generous pad
    raw = broker.fetch_candles(
        symbol_token=token,
        start=f"{start.isoformat()} 09:15",
        end=f"{end.isoformat()} 15:30",
        interval="ONE_DAY",
        exchange=exchange,
    ) or []
    closes = [float(c[4]) for c in raw if len(c) >= 5]
    return closes[-lookback_days:] if len(closes) > lookback_days else closes


def _daily_return_stdev_pct(closes: list[float]) -> float:
    """Sample stdev of daily log returns, expressed as % (e.g. 0.85 = 0.85%/day)."""
    if len(closes) < 3:
        return 0.0
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * 100


def _trend_streak(closes: list[float], lookback: int) -> int:
    """Signed streak of consecutive same-direction daily closes over the tail.

    +N → N up-days in a row, -N → N down-days in a row, 0 → mixed.
    """
    if len(closes) < lookback + 1:
        return 0
    tail = closes[-(lookback + 1):]
    diffs = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if all(d > 0 for d in diffs):
        return lookback
    if all(d < 0 for d in diffs):
        return -lookback
    return 0


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────
def _score_vix(vix: float) -> tuple[int, str]:
    if vix < 11:
        return -25, f"VIX {vix:.1f} too cheap — straddle premium will be thin"
    if vix < 14:
        return -5,  f"VIX {vix:.1f} below the sweet spot"
    if vix < 18:
        return +15, f"VIX {vix:.1f} in the sweet spot for premium-sellers"
    if vix < 24:
        return +20, f"VIX {vix:.1f} elevated — premium is fat"
    return +5, f"VIX {vix:.1f} very high — fat premium but breakout risk is real"


def _score_iv_rv(ratio: float) -> tuple[int, str]:
    if ratio <= 0:
        return 0, "IV/RV ratio unavailable (insufficient candles)"
    if ratio < 0.9:
        return -25, f"Implied move only {ratio:.2f}× realized — market is underpricing vol"
    if ratio < 1.1:
        return 0, f"Implied move {ratio:.2f}× realized — fair premium, no edge"
    if ratio < 1.3:
        return +10, f"Implied move {ratio:.2f}× realized — modest edge to the seller"
    return +20, f"Implied move {ratio:.2f}× realized — premium is rich vs reality"


def _score_dte(dte: int) -> tuple[int, str]:
    if dte == 0:
        return -100, "Expiry today — never open a new straddle at 0 DTE"
    if dte <= 2:
        return -15, f"{dte} DTE — gamma is too hot to open"
    if dte <= 7:
        return +15, f"{dte} DTE — sweet spot for theta capture"
    if dte <= 14:
        return +5, f"{dte} DTE — manageable but theta is still slow"
    return -5, f"{dte} DTE — too far out; capital tied up for weak decay"


def _score_trend(streak: int) -> tuple[int, str]:
    if streak == 0:
        return +5, "No 3-day directional streak — range-bound posture intact"
    direction = "up" if streak > 0 else "down"
    return -15, f"{abs(streak)} consecutive {direction}-days — trending market, straddles bleed"


def _score_strike_fit(spot: float, strike: int) -> tuple[int, str]:
    drift_pct = abs(spot - strike) / spot * 100
    if drift_pct < 0.2:
        return +5, f"ATM strike {strike} within {drift_pct:.2f}% of spot — clean delta"
    return 0, f"ATM strike {strike} offset by {drift_pct:.2f}% — small starting delta"


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def decide_entry(underlying: str, expiry_iso: str) -> EntryDecision:
    """Decide whether to open a short straddle for (underlying, expiry).

    Reads live data from Angel One. Raises ValueError on bad inputs;
    network/data errors are surfaced via warnings in the result.
    """
    underlying = underlying.upper()
    if underlying not in UNDERLYINGS:
        raise ValueError(f"Unknown underlying {underlying!r}. Supported: {list(UNDERLYINGS)}")

    expiry_angel = iso_to_angel(expiry_iso)
    if not expiry_angel:
        raise ValueError(f"Bad expiry {expiry_iso!r} — expected YYYY-MM-DD")

    cfg = UNDERLYINGS[underlying]
    dte = _dte(expiry_iso)

    # ── Resolve broker + ticker
    from trading.services.data_service import BrokerClient
    from trading.services.ticker_service import ticker_service

    broker = BrokerClient.get_instance()
    broker.ensure_login()

    warnings: list[str] = []

    # ── Step 1: snapshot spot + VIX in one call
    snap_tokens = {cfg["spot_exchange"]: [cfg["spot_token"]]}
    if VIX_EXCHANGE == cfg["spot_exchange"]:
        snap_tokens[VIX_EXCHANGE].append(VIX_TOKEN)
    else:
        snap_tokens.setdefault(VIX_EXCHANGE, []).append(VIX_TOKEN)

    snap = broker.market_data_batch(snap_tokens, mode="OHLC") or []
    spot = vix = vix_prev = 0.0
    for item in snap:
        tok = str(item.get("symbolToken", ""))
        if tok == cfg["spot_token"]:
            spot = float(item.get("ltp", 0))
        elif tok == VIX_TOKEN:
            vix = float(item.get("ltp", 0))
            vix_prev = float(item.get("close", 0))
    if spot <= 0:
        raise RuntimeError(f"Failed to fetch {underlying} spot from broker")
    if vix <= 0:
        warnings.append("VIX unavailable — scoring assumes neutral VIX")
        vix = 14.0
        vix_prev = vix

    # ── Step 2: ATM strike + leg resolution
    step = cfg["strike_step"]
    lower = (int(spot) // step) * step
    upper = lower + step
    suggested_strike = lower if abs(spot - lower) <= abs(spot - upper) else upper

    legs = ticker_service.get_nfo_options(underlying, suggested_strike, expiry_angel)
    if "CE" not in legs or "PE" not in legs:
        raise RuntimeError(
            f"No CE/PE legs found for {underlying} {suggested_strike} {expiry_angel} — "
            f"check scrip master + expiry date"
        )
    ce_symbol, ce_token = legs["CE"]
    pe_symbol, pe_token = legs["PE"]

    # ── Step 3: leg LTPs (batched)
    leg_data = broker.market_data_batch(
        {cfg["option_exchange"]: [ce_token, pe_token]}, mode="OHLC"
    ) or []
    ce_ltp = pe_ltp = 0.0
    for item in leg_data:
        tok = str(item.get("symbolToken", ""))
        if tok == ce_token:
            ce_ltp = float(item.get("ltp", 0))
        elif tok == pe_token:
            pe_ltp = float(item.get("ltp", 0))
    if ce_ltp <= 0 or pe_ltp <= 0:
        warnings.append("Option LTPs incomplete — premium math may be off")

    combined_pts = ce_ltp + pe_ltp
    combined_inr = combined_pts * cfg["lot_size"]
    implied_move_pct = (combined_pts / spot * 100) if spot else 0.0

    # ── Step 4: realized vol + trend from daily candles
    try:
        closes = _fetch_daily_closes(
            broker, cfg["spot_token"], cfg["spot_exchange"], REALIZED_VOL_LOOKBACK
        )
    except Exception as e:
        logger.warning(f"Daily candle fetch failed: {e}")
        closes = []
        warnings.append("Daily candle fetch failed — RV scoring skipped")

    daily_stdev_pct = _daily_return_stdev_pct(closes)
    realized_move_pct = daily_stdev_pct * math.sqrt(max(dte, 1))
    iv_rv_ratio = (implied_move_pct / realized_move_pct) if realized_move_pct > 0 else 0.0
    streak = _trend_streak(closes, TREND_LOOKBACK)

    # ── Step 5: score
    components: list[tuple[str, int, str]] = []
    for name, fn, arg in [
        ("VIX",         _score_vix,         vix),
        ("IV/RV edge",  _score_iv_rv,       iv_rv_ratio),
        ("DTE",         _score_dte,         dte),
        ("Trend",       _score_trend,       streak),
    ]:
        pts, why = fn(arg)
        components.append((name, pts, why))
    pts, why = _score_strike_fit(spot, suggested_strike)
    components.append(("Strike fit", pts, why))

    score = max(0, min(100, 50 + sum(p for _, p, _ in components)))

    vetoed = False
    if iv_rv_ratio and iv_rv_ratio < VETO_IV_RV_RATIO:
        vetoed = True
        warnings.append(
            f"VETO: IV/RV {iv_rv_ratio:.2f} < {VETO_IV_RV_RATIO} — "
            "market is underpricing vol; selling premium has negative edge"
        )

    if dte <= 0:
        decision = "SKIP"
    elif vetoed:
        decision = "WAIT"          # right setup, wrong premium environment
    elif score >= SCORE_ENTER:
        decision = "ENTER"
    elif score >= SCORE_WAIT:
        decision = "WAIT"
    else:
        decision = "SKIP"

    # ── Step 6: risk math
    breakeven_low = suggested_strike - combined_pts
    breakeven_high = suggested_strike + combined_pts
    # If we stop out at 1.5× sold, we lose 0.5× sold per lot in points.
    max_loss_inr = combined_pts * (HARD_STOP_MULT - 1.0) * cfg["lot_size"]

    reasoning = [w for _, _, w in components]

    return EntryDecision(
        decision=decision,
        score=score,
        underlying=underlying,
        expiry_iso=expiry_iso,
        expiry_angel=expiry_angel,
        dte=dte,
        spot=spot,
        vix=vix,
        vix_prev=vix_prev,
        suggested_strike=suggested_strike,
        ce_symbol=ce_symbol,
        ce_token=ce_token,
        ce_ltp=ce_ltp,
        pe_symbol=pe_symbol,
        pe_token=pe_token,
        pe_ltp=pe_ltp,
        combined_premium_pts=combined_pts,
        combined_premium_inr=combined_inr,
        breakeven_low=breakeven_low,
        breakeven_high=breakeven_high,
        implied_move_pct=implied_move_pct,
        realized_move_pct=realized_move_pct,
        iv_rv_ratio=iv_rv_ratio,
        trend_streak=streak,
        max_loss_inr=max_loss_inr,
        score_components=components,
        reasoning=reasoning,
        warnings=warnings,
    )


# ──────────────────────────────────────────────
# CLI pretty-printer
# ──────────────────────────────────────────────
def format_decision(d: EntryDecision) -> str:
    badge = {"ENTER": "✅ ENTER", "WAIT": "⏸  WAIT", "SKIP": "⛔ SKIP"}[d.decision]
    head = (
        f"\n{'═' * 64}\n"
        f"{badge}  |  Score {d.score}/100  |  {d.underlying}  |  Expiry {d.expiry_iso} ({d.dte} DTE)\n"
        f"{'═' * 64}"
    )

    market = (
        f"\nMARKET\n"
        f"  Spot              : {d.spot:,.2f}\n"
        f"  VIX               : {d.vix:.2f}  (prev close {d.vix_prev:.2f})\n"
        f"  Realized vol      : {d.realized_move_pct:.2f}% expected over {d.dte}d "
        f"({d.realized_move_pct / max(math.sqrt(d.dte), 1):.2f}% daily stdev)\n"
        f"  Implied move      : {d.implied_move_pct:.2f}% (from combined premium)\n"
        f"  IV / RV           : {d.iv_rv_ratio:.2f}×\n"
        f"  Trend streak      : {d.trend_streak:+d} day(s)"
    )

    legs = (
        f"\n\nSUGGESTED LEGS (ATM)\n"
        f"  Strike            : {d.suggested_strike}\n"
        f"  {d.ce_symbol:<28}  CE LTP {d.ce_ltp:>8.2f}  (token {d.ce_token})\n"
        f"  {d.pe_symbol:<28}  PE LTP {d.pe_ltp:>8.2f}  (token {d.pe_token})\n"
        f"  Combined premium  : {d.combined_premium_pts:.2f} pts  →  ₹{d.combined_premium_inr:,.0f} / lot\n"
        f"  Breakevens        : {d.breakeven_low:,.0f}  ↔  {d.breakeven_high:,.0f}\n"
        f"  Max loss (1.5× SL): ₹{d.max_loss_inr:,.0f} / lot"
    )

    score_lines = "\n".join(
        f"  {name:<14} {pts:+4d}   {why}" for name, pts, why in d.score_components
    )

    warn_block = ""
    if d.warnings:
        warn_block = "\n\nWARNINGS\n" + "\n".join(f"  ⚠ {w}" for w in d.warnings)

    return f"{head}{market}{legs}\n\nSCORE BREAKDOWN\n{score_lines}{warn_block}\n"
