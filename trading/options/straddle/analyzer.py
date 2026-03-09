"""
Straddle Analyzer — pure Python, zero LLM.

Computes everything the LLM needs to make a management decision:
- Real-time P&L per lot
- Delta exposure (approximate from moneyness)
- VIX phase classification
- Market phase from 5-min candle structure
- Expiry scenario table
- Risk flags (underwater, stop triggered, expiry-day)
- Human-readable summary text injected into the straddle prompt
"""
from datetime import date, datetime
from typing import List, Optional

from .state import StraddleAnalysis, ExpiryScenario


# ──────────────────────────────────────────────
# Delta approximation (Black-Scholes heuristic)
# ──────────────────────────────────────────────
def _approx_delta(spot: float, strike: float, option_type: str, days_to_expiry: int) -> float:
    """
    Approximate option delta from moneyness and DTE.
    Not Black-Scholes — a practical heuristic for directional management.

    Returns delta from the LONG perspective (short position is negative of this).
    """
    moneyness = (spot - strike) / strike  # % in/out of money

    if option_type == "CE":
        if days_to_expiry <= 1:
            # Gamma spikes near expiry — binary behavior
            if spot >= strike:
                return min(0.95, 0.5 + abs(moneyness) * 5)
            else:
                return max(0.05, 0.5 - abs(moneyness) * 5)
        else:
            base = 0.5 + moneyness * 2.5
            return max(0.05, min(0.95, base))

    else:  # PE
        if days_to_expiry <= 1:
            if spot <= strike:
                return max(-0.95, -0.5 - abs(moneyness) * 5)
            else:
                return min(-0.05, -0.5 + abs(moneyness) * 5)
        else:
            base = -0.5 + moneyness * 2.5
            return max(-0.95, min(-0.05, base))


# ──────────────────────────────────────────────
# VIX phase classifier
# ──────────────────────────────────────────────
def _classify_vix(vix: float) -> str:
    if vix < 15:
        return "CALM"
    elif vix < 22:
        return "ELEVATED"
    else:
        return "SPIKE"


# ──────────────────────────────────────────────
# Market phase classifier (from 5-min candles)
# ──────────────────────────────────────────────
def _classify_market_phase(
    candles: list,
    nifty_spot: float,
    nifty_prev_close: float,
    vix_current: float,
    vix_prev_close: float,
) -> str:
    """
    Classify intraday market phase from candle structure.

    candles: list of [timestamp, open, high, low, close, volume]
    Returns: CRASH | CHOP | RECOVERY | RALLY | CLOSE
    """
    now = datetime.now()
    if now.hour >= 14 and now.minute >= 45:
        return "CLOSE"

    if not candles or len(candles) < 3:
        return "CHOP"

    # Last 6 candles (30 minutes of price action)
    recent = candles[-6:] if len(candles) >= 6 else candles

    highs  = [c[2] for c in recent]
    lows   = [c[3] for c in recent]
    closes = [c[4] for c in recent]

    day_change_pct = (nifty_spot - nifty_prev_close) / nifty_prev_close * 100
    vix_change_pct = (vix_current - vix_prev_close) / vix_prev_close * 100

    # Day low (all candles)
    all_lows = [c[3] for c in candles]
    day_low = min(all_lows)
    recovery_from_low = (nifty_spot - day_low) / day_low * 100

    # Trend: are recent candles making higher highs and higher lows?
    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
    lower_lows   = sum(1 for i in range(1, len(lows))  if lows[i]  < lows[i - 1])

    # CRASH: big gap down + VIX spiking + making new lows
    if day_change_pct < -1.5 and vix_change_pct > 10 and lower_lows >= 3:
        return "CRASH"

    # RECOVERY: bounced from lows, higher highs forming, VIX falling/stable
    if recovery_from_low > 0.5 and higher_highs >= 3 and vix_change_pct < 5:
        return "RECOVERY"

    # RALLY: NIFTY approaching or above previous close, upward momentum
    if day_change_pct > 0.5 and higher_highs >= 4:
        return "RALLY"

    # CHOP: everything else
    return "CHOP"


# ──────────────────────────────────────────────
# Expiry scenario table
# ──────────────────────────────────────────────
def _build_scenarios(
    strike: int,
    ce_sell_price: float,
    pe_sell_price: float,
    lot_size: int,
    nifty_spot: float,
) -> List[ExpiryScenario]:
    """Build expiry P&L table at key NIFTY levels."""
    levels = [
        ("Bull rally", nifty_spot + 300, "NIFTY up 300 pts from now"),
        ("Mild recovery", nifty_spot + 150, "NIFTY up 150 pts"),
        ("Strike (best case)", strike, "Exact ATM at expiry"),
        ("Current level", nifty_spot, "NIFTY unchanged"),
        ("Mild drop", nifty_spot - 150, "NIFTY down 150 pts"),
        ("Major drop", nifty_spot - 400, "NIFTY down 400 pts"),
        ("Crash", nifty_spot - 700, "NIFTY down 700 pts"),
    ]

    scenarios = []
    for label, nifty_level, _ in levels:
        ce_exp = max(0.0, nifty_level - strike)
        pe_exp = max(0.0, strike - nifty_level)
        net_pnl_pts = (ce_sell_price - ce_exp) + (pe_sell_price - pe_exp)
        scenarios.append(ExpiryScenario(
            label=label,
            nifty_level=round(nifty_level, 0),
            ce_expiry_value=round(ce_exp, 2),
            pe_expiry_value=round(pe_exp, 2),
            net_pnl_inr=round(net_pnl_pts * lot_size, 0),
        ))

    return scenarios


# ──────────────────────────────────────────────
# Days to expiry
# ──────────────────────────────────────────────
def _days_to_expiry(expiry_str: str) -> int:
    """Return calendar days to expiry. 0 = expiry today."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return max(0, (expiry_date - date.today()).days)
    except Exception:
        return 999


# ──────────────────────────────────────────────
# Summary text builder (injected into LLM prompt)
# ──────────────────────────────────────────────
def _build_summary_text(a: dict) -> str:
    """Build a dense, readable analysis block for the LLM context."""

    dte  = a["days_to_expiry"]
    expiry_warning = "⚠️  EXPIRY TOMORROW — Gamma is extreme. Prefer closing over holding." if a["expiry_tomorrow"] else ""
    underwater_warning = "🚨 POSITION UNDERWATER — Combined premium > sold. Hard stop triggered." if a["is_underwater"] else ""

    scenario_lines = "\n".join([
        f"  {s['label']:<25} NIFTY {s['nifty_level']:<8.0f}  P&L: {s['net_pnl_inr']:>+,.0f} INR"
        for s in a["scenarios"]
    ])

    return f"""STRADDLE POSITION SNAPSHOT
==========================
Underlying : {a.get('underlying', 'NIFTY')}  |  Strike: {a.get('strike', '?')}  |  Expiry: {a.get('expiry', '?')}  |  DTE: {dte} day(s)
{expiry_warning}
{underwater_warning}

MARKET DATA (LIVE)
------------------
NIFTY Spot     : {a['nifty_spot']:,.2f}  (prev close: {a['nifty_prev_close']:,.2f}  |  gap: {a['nifty_gap_pts']:+.0f} pts = {a['nifty_gap_pct']:+.2f}%)
India VIX      : {a['vix_current']:.2f}  (prev close: {a['vix_prev_close']:.2f}  |  {a['vix_change_pct']:+.1f}%)  →  Phase: {a['vix_phase']}
Market Phase   : {a['market_phase']}

OPTION LEGS
-----------
24200 CE  |  Sold: {a['ce_sell_price']:.2f}  |  Now: {a['ce_ltp']:.2f}  |  P&L: {(a['ce_sell_price'] - a['ce_ltp']):+.2f} pts
24200 PE  |  Sold: {a['pe_sell_price']:.2f}  |  Now: {a['pe_ltp']:.2f}  |  P&L: {(a['pe_sell_price'] - a['pe_ltp']):+.2f} pts
Combined  |  Sold: {a['combined_sold']:.2f}  |  Now: {a['combined_current']:.2f}  |  Decayed: {a['premium_decayed_pct']:.1f}%

NET P&L    : {a['net_pnl_pts']:+.2f} pts  →  {a['net_pnl_inr']:>+,.0f} INR per lot

DELTA EXPOSURE
--------------
CE delta   : {a['ce_delta']:+.2f}  (short → net {-a['ce_delta']:+.2f})
PE delta   : {a['pe_delta']:+.2f}  (short → net {-a['pe_delta']:+.2f})
Net delta  : {a['net_delta']:+.2f}  →  Bias: {a['delta_bias']}
Per 100pt NIFTY move: {abs(a['net_delta']) * 100 * a.get('lot_size', 65):,.0f} INR impact

MONEYNESS
---------
PE is ITM by  : {a['pe_itm_by']:.0f} pts  (put is in-the-money — dangerous leg)
CE is OTM by  : {a['ce_itm_by']:.0f} pts  (call is out-of-money — safe leg)
Tested leg    : {a['nearest_itm_leg']}

EXPIRY SCENARIOS (if held to expiry)
------------------------------------
{scenario_lines}"""


# ──────────────────────────────────────────────
# Main analyzer entry point
# ──────────────────────────────────────────────
def analyze_straddle(
    underlying: str,
    strike: int,
    expiry: str,
    lot_size: int,
    lots: int,
    ce_sell_price: float,
    pe_sell_price: float,
    ce_ltp: float,
    pe_ltp: float,
    nifty_spot: float,
    nifty_prev_close: float,
    vix_current: float,
    vix_prev_close: float,
    candles: Optional[list] = None,
) -> StraddleAnalysis:
    """
    Full straddle analysis. Pure Python — no network calls, no LLM.

    Args:
        underlying: "NIFTY"
        strike: 24200
        expiry: "2026-03-10"
        lot_size: 65
        lots: number of lots (1, 2, ...)
        ce_sell_price: price CE was sold at (from StraddlePosition)
        pe_sell_price: price PE was sold at (from StraddlePosition)
        ce_ltp: current CE last traded price
        pe_ltp: current PE last traded price
        nifty_spot: current NIFTY 50 spot
        nifty_prev_close: NIFTY previous close
        vix_current: India VIX current
        vix_prev_close: India VIX previous close
        candles: list of [ts, open, high, low, close, volume] — 5-min NIFTY candles

    Returns:
        StraddleAnalysis (Pydantic model)
    """
    # ── Market ──
    nifty_gap_pts = nifty_spot - nifty_prev_close
    nifty_gap_pct = nifty_gap_pts / nifty_prev_close * 100
    vix_change_pct = (vix_current - vix_prev_close) / vix_prev_close * 100
    vix_phase = _classify_vix(vix_current)

    # ── P&L ──
    combined_sold    = ce_sell_price + pe_sell_price
    combined_current = ce_ltp + pe_ltp
    net_pnl_pts  = combined_sold - combined_current
    net_pnl_inr  = net_pnl_pts * lot_size * lots
    premium_decayed_pct = (net_pnl_pts / combined_sold * 100) if combined_sold > 0 else 0

    # ── Moneyness ──
    pe_itm_by = max(0.0, strike - nifty_spot)  # PE ITM if spot below strike
    ce_itm_by = max(0.0, nifty_spot - strike)  # CE ITM if spot above strike

    if pe_itm_by > ce_itm_by:
        nearest_itm_leg = "PE"
    elif ce_itm_by > pe_itm_by:
        nearest_itm_leg = "CE"
    else:
        nearest_itm_leg = "BOTH_OTM"

    # ── Delta ──
    dte = _days_to_expiry(expiry)
    long_ce_delta = _approx_delta(nifty_spot, strike, "CE", dte)
    long_pe_delta = _approx_delta(nifty_spot, strike, "PE", dte)

    # We are SHORT both options → flip delta sign
    short_ce_delta = -long_ce_delta   # short call = negative delta
    short_pe_delta = -long_pe_delta   # short put = positive delta (since long PE delta is negative)
    net_delta = short_ce_delta + short_pe_delta

    if net_delta > 0.15:
        delta_bias = "LONG"
    elif net_delta < -0.15:
        delta_bias = "SHORT"
    else:
        delta_bias = "NEUTRAL"

    # ── Risk flags ──
    is_underwater = combined_current > combined_sold
    stop_triggered = combined_current > (combined_sold * 1.0)  # Any loss = stop consideration
    expiry_tomorrow = dte <= 1

    # ── Market phase ──
    market_phase = _classify_market_phase(
        candles or [], nifty_spot, nifty_prev_close, vix_current, vix_prev_close
    )

    # ── Scenarios ──
    scenarios = _build_scenarios(strike, ce_sell_price, pe_sell_price, lot_size * lots, nifty_spot)

    # ── Assemble analysis dict for summary text ──
    analysis_dict = {
        "underlying": underlying,
        "strike": strike,
        "expiry": expiry,
        "days_to_expiry": dte,
        "expiry_tomorrow": expiry_tomorrow,
        "nifty_spot": nifty_spot,
        "nifty_prev_close": nifty_prev_close,
        "nifty_gap_pts": nifty_gap_pts,
        "nifty_gap_pct": nifty_gap_pct,
        "vix_current": vix_current,
        "vix_prev_close": vix_prev_close,
        "vix_change_pct": vix_change_pct,
        "vix_phase": vix_phase,
        "ce_ltp": ce_ltp,
        "pe_ltp": pe_ltp,
        "ce_sell_price": ce_sell_price,
        "pe_sell_price": pe_sell_price,
        "combined_sold": combined_sold,
        "combined_current": combined_current,
        "net_pnl_pts": net_pnl_pts,
        "net_pnl_inr": net_pnl_inr,
        "premium_decayed_pct": premium_decayed_pct,
        "ce_delta": short_ce_delta,
        "pe_delta": short_pe_delta,
        "net_delta": net_delta,
        "delta_bias": delta_bias,
        "is_underwater": is_underwater,
        "stop_triggered": stop_triggered,
        "pe_itm_by": pe_itm_by,
        "ce_itm_by": ce_itm_by,
        "nearest_itm_leg": nearest_itm_leg,
        "lot_size": lot_size * lots,
        "scenarios": [s.model_dump() for s in scenarios],
        "market_phase": market_phase,
    }

    summary_text = _build_summary_text(analysis_dict)
    analysis_dict["summary_text"] = summary_text

    return StraddleAnalysis(
        nifty_spot=nifty_spot,
        nifty_prev_close=nifty_prev_close,
        nifty_gap_pts=nifty_gap_pts,
        nifty_gap_pct=nifty_gap_pct,
        vix_current=vix_current,
        vix_prev_close=vix_prev_close,
        vix_change_pct=vix_change_pct,
        vix_phase=vix_phase,
        ce_ltp=ce_ltp,
        pe_ltp=pe_ltp,
        ce_sell_price=ce_sell_price,
        pe_sell_price=pe_sell_price,
        combined_sold=combined_sold,
        combined_current=combined_current,
        net_pnl_pts=net_pnl_pts,
        net_pnl_inr=net_pnl_inr,
        premium_decayed_pct=premium_decayed_pct,
        ce_delta=short_ce_delta,
        pe_delta=short_pe_delta,
        net_delta=net_delta,
        delta_bias=delta_bias,
        is_underwater=is_underwater,
        stop_triggered=stop_triggered,
        expiry_tomorrow=expiry_tomorrow,
        days_to_expiry=dte,
        market_phase=market_phase,
        pe_itm_by=pe_itm_by,
        ce_itm_by=ce_itm_by,
        nearest_itm_leg=nearest_itm_leg,
        scenarios=scenarios,
        summary_text=summary_text,
    )
