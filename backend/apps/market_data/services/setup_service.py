"""Setup preview service — Stage 5 of The Cascade.

Stage 4 (shortlist) hands the desk 10-15 ranked names.  Stage 5 (setup) is
where the operator clicks one of those names and asks: *"would the desk
actually take this trade right now, and if not, why not?"*  The answer must
be deterministic, instant, and **fully audited** — every one of the ten
@RiskGuard criteria has to be visible, with the threshold and the actual
value side-by-side.

Reuse-first architecture
────────────────────────
* The single source of truth for *whether* a trade clears risk lives in
  ``trading/services/risk_engine.py::validate_trade``.  We import that and
  call it for the **overall verdict** — there is exactly one production
  risk gate and the UI must agree with what would actually run live.
* The same module exposes the constants (MIN_RISK_REWARD_RATIO,
  MAX_RISK_PER_TRADE_PCT, etc.) and the regime-cache key, so this service
  pulls them from there rather than re-declaring them.  If those numbers
  ever change, the UI updates automatically.
* ``DefaultMarketData`` (already used by the agents) provides LTP and
  candles — we reuse it instead of opening a second pipe.
* Symbol metadata (lot size, exchange, token) comes from the same
  ``Symbol`` table the rest of the platform reads.

What this service computes (the *interesting* part)
───────────────────────────────────────────────────
Because ``validate_trade`` is fail-fast (returns on the first failed
criterion), it can't power a UI breakdown card on its own.  We add a thin
``evaluate_criteria`` helper that runs each of the ten criteria
**independently** against the same plan / capital / context — passes
through every gate so the operator sees ALL of them at once, not just the
first one that tripped.  The gates themselves are the same constants and
the same comparisons; this is presentation, not policy.

The proposed plan is built deterministically from the symbol's recent
candles (no LLM):

* entry          = last traded price
* stop           = entry ∓ STOP_ATR_MULT × ATR(14)
* target         = entry ± TARGET_ATR_MULT × ATR(14)   (so R:R = 2.0)
* quantity       = floor(MAX_RISK_PER_TRADE × capital / risk_per_share)
* confidence     = 0.6 (above the 0.55 floor — neutral baseline)

The output payload is built to render directly into a "would-this-trade-
clear-the-desk?" card on the frontend, with a 10-row criterion breakdown.

API surface
───────────
* ``build_setup(symbol, side, *, capital, daily_loss, open_positions,
  data_port=None)`` → ``SetupPayload``  (synchronous, no Celery)
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from django.core.cache import cache

# REUSE the production risk engine. Single source of truth.
from trading.services.risk_engine import (
    MAX_DAILY_LOSS_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    MAX_RISK_PER_TRADE_PCT,
    MIN_CONFIDENCE,
    MIN_RISK_REWARD_RATIO,
    PULSE_CACHE_KEY,
    _check_regime,
    validate_trade,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning knobs — env-overridable for stress testing
# ---------------------------------------------------------------------------
STOP_ATR_MULT = float(os.getenv("SETUP_STOP_ATR_MULT", "1.5"))
TARGET_ATR_MULT = float(os.getenv("SETUP_TARGET_ATR_MULT", "3.0"))
DEFAULT_CONFIDENCE = float(os.getenv("SETUP_DEFAULT_CONFIDENCE", "0.6"))
ATR_LOOKBACK_CANDLES = int(os.getenv("SETUP_ATR_LOOKBACK", "50"))
ATR_PERIOD = 14


# ---------------------------------------------------------------------------
# Output dataclasses — match the JSON the frontend renders
# ---------------------------------------------------------------------------
@dataclass
class CriterionResult:
    """One row in the @RiskGuard breakdown card."""
    key: str             # stable identifier (used by the UI as a React key)
    label: str           # human-readable name
    passed: bool
    detail: str          # short fact: "Risk ₹4,800 / ₹5,000 (1% cap)"
    severity: str = "info"  # info | warning | danger — drives badge tone


@dataclass
class ProposedPlan:
    symbol: str
    side: str            # BUY | SELL
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    confidence: float
    risk_per_share: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    notes: list[str] = field(default_factory=list)


@dataclass
class MarketSnapshot:
    last: float | None
    atr: float | None
    atr_pct: float | None
    change_pct: float | None
    candle_count: int


@dataclass
class RegimeSnapshot:
    tradeable: bool
    vol: str | None
    trend: str | None
    summary: str
    cached: bool        # was the pulse cache warm at lookup time?


@dataclass
class RiskBreakdown:
    approved: bool
    reason: str         # the validate_trade authoritative one-liner
    criteria: list[CriterionResult] = field(default_factory=list)


@dataclass
class SetupPayload:
    as_of: str
    symbol: str
    side: str
    market: MarketSnapshot
    plan: ProposedPlan | None
    regime: RegimeSnapshot
    risk: RiskBreakdown
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wilder_atr(highs: list[float], lows: list[float], closes: list[float],
                period: int = ATR_PERIOD) -> float | None:
    """Standard ATR — same formula the shortlist screener uses, kept local
    so this service stays self-contained."""
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return None
    trs: list[float] = []
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    tail = trs[-period:] if len(trs) >= period else trs
    if not tail:
        return None
    return sum(tail) / len(tail)


def _market_snapshot(data_port, symbol: str) -> tuple[MarketSnapshot, list[str]]:
    """Pull LTP + ATR from the market data port.  Errors degrade to None
    fields, never raise — the UI shows a 'data thin' state instead of 500."""
    errors: list[str] = []
    last: float | None = None
    atr: float | None = None
    change_pct: float | None = None
    candle_count = 0

    try:
        ltp = data_port.ltp(symbol)
        last = float(ltp) if ltp else None
    except Exception as e:  # noqa: BLE001
        errors.append(f"ltp: {e}")

    try:
        candles = data_port.candles(symbol, "5m", ATR_LOOKBACK_CANDLES) or []
        candle_count = len(candles)
        if candles:
            highs = [float(c["h"]) for c in candles]
            lows = [float(c["l"]) for c in candles]
            closes = [float(c["c"]) for c in candles]
            atr = _wilder_atr(highs, lows, closes)
            # Day change vs the first candle of the window — rough but
            # honest given we don't always have a prev_close handy.
            if len(closes) >= 2 and closes[0]:
                change_pct = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
            # Prefer last candle close when LTP cache is cold.
            if last is None:
                last = float(closes[-1])
    except Exception as e:  # noqa: BLE001
        errors.append(f"candles: {e}")

    atr_pct = round(atr / last * 100, 2) if (atr and last) else None
    snap = MarketSnapshot(
        last=last,
        atr=round(atr, 4) if atr is not None else None,
        atr_pct=atr_pct,
        change_pct=change_pct,
        candle_count=candle_count,
    )
    return snap, errors


def _regime_snapshot() -> RegimeSnapshot:
    """Read the same pulse cache the @RiskGuard regime gate uses, then
    summarise it for the UI.  Never raises."""
    payload = cache.get(PULSE_CACHE_KEY)
    cached = payload is not None
    ok, reason, details = _check_regime()
    return RegimeSnapshot(
        tradeable=ok,
        vol=details.get("vol"),
        trend=details.get("trend"),
        summary=details.get("summary") or reason,
        cached=cached,
    )


def _build_plan(symbol: str, side: str, last: float, atr: float, capital: float
                ) -> tuple[ProposedPlan, list[str]]:
    """Deterministic ATR-anchored plan builder — no LLM.

    Symmetry: SL = STOP_ATR_MULT × ATR away, target = TARGET_ATR_MULT × ATR
    away.  Quantity sized so risk == 1% of capital (rounded down).
    """
    notes: list[str] = []
    side = side.upper()

    if side == "BUY":
        stop_loss = round(last - STOP_ATR_MULT * atr, 2)
        target = round(last + TARGET_ATR_MULT * atr, 2)
    elif side == "SELL":
        stop_loss = round(last + STOP_ATR_MULT * atr, 2)
        target = round(last - TARGET_ATR_MULT * atr, 2)
    else:
        raise ValueError(f"Unknown side: {side!r} (expected BUY or SELL)")

    risk_per_share = abs(last - stop_loss)
    if risk_per_share <= 0:
        raise ValueError("Computed risk_per_share is 0 — ATR too small for entry.")

    max_risk_amount = capital * MAX_RISK_PER_TRADE_PCT / 100
    raw_qty = max_risk_amount / risk_per_share
    # Cap by position-size rule too (10% of capital).
    qty_position_cap = (capital * MAX_POSITION_SIZE_PCT / 100) / last
    qty = int(math.floor(min(raw_qty, qty_position_cap)))
    if qty <= 0:
        notes.append(
            f"Sized down to 0 — risk-per-share ₹{risk_per_share:.2f} > "
            f"1%-of-capital cap ₹{max_risk_amount:.0f}."
        )
        qty = 1   # keep plan well-formed; risk gate will reject it.

    risk_amount = round(risk_per_share * qty, 2)
    reward_amount = round(abs(target - last) * qty, 2)
    rr = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0.0

    plan = ProposedPlan(
        symbol=symbol,
        side=side,
        entry_price=round(last, 2),
        stop_loss=stop_loss,
        target=target,
        quantity=qty,
        confidence=DEFAULT_CONFIDENCE,
        risk_per_share=round(risk_per_share, 2),
        risk_amount=risk_amount,
        reward_amount=reward_amount,
        risk_reward_ratio=rr,
        notes=notes,
    )
    return plan, notes


# ---------------------------------------------------------------------------
# Per-criterion breakdown — runs each gate independently for the UI.
# These mirror trading/services/risk_engine.validate_trade exactly.
# ---------------------------------------------------------------------------
def evaluate_criteria(
    plan: dict,
    capital: float,
    daily_loss: float = 0.0,
    open_positions: int = 0,
) -> list[CriterionResult]:
    """Return one ``CriterionResult`` for every gate in ``validate_trade``,
    even the ones that wouldn't have been reached by the fail-fast
    validator.  Order matches the engine.

    The intent is purely presentational: the canonical decision still comes
    from ``validate_trade`` — this gives the operator a complete X-ray.
    """
    rows: list[CriterionResult] = []

    # 0 — Cascade regime
    ok, reason, details = _check_regime()
    rows.append(CriterionResult(
        key="regime",
        label="Cascade regime",
        passed=ok,
        detail=details.get("summary") or reason,
        severity="info" if ok else "danger",
    ))

    # 1 — basic fields present + positive numbers
    required = ("symbol", "side", "entry_price", "stop_loss", "target",
                "quantity", "confidence")
    missing = [k for k in required if plan.get(k) in (None, "")]
    qty = plan.get("quantity") or 0
    entry = plan.get("entry_price") or 0
    sl = plan.get("stop_loss") or 0
    target = plan.get("target") or 0
    side = plan.get("side", "")
    confidence = plan.get("confidence", 0.0) or 0.0

    field_ok = not missing and qty > 0 and entry > 0 and sl > 0 and target > 0
    rows.append(CriterionResult(
        key="fields",
        label="Plan well-formed",
        passed=field_ok,
        detail=("All required fields present; sizes positive."
                if field_ok
                else f"Issues: {', '.join(missing) or 'non-positive price/qty'}"),
        severity="info" if field_ok else "danger",
    ))

    # 2 — stop direction
    sl_ok = (
        (side == "BUY" and sl < entry) or (side == "SELL" and sl > entry)
        if entry and sl else False
    )
    rows.append(CriterionResult(
        key="stop_direction",
        label="Stop direction",
        passed=sl_ok,
        detail=(f"{side} SL {sl} vs entry {entry} — "
                f"{'correct side' if sl_ok else 'WRONG side of entry'}"),
        severity="info" if sl_ok else "danger",
    ))

    # 3 — target direction
    tgt_ok = (
        (side == "BUY" and target > entry) or (side == "SELL" and target < entry)
        if entry and target else False
    )
    rows.append(CriterionResult(
        key="target_direction",
        label="Target direction",
        passed=tgt_ok,
        detail=(f"{side} target {target} vs entry {entry} — "
                f"{'correct side' if tgt_ok else 'WRONG side of entry'}"),
        severity="info" if tgt_ok else "danger",
    ))

    # 4 — risk per trade ≤ MAX_RISK_PER_TRADE_PCT of capital
    risk_per_share = abs(entry - sl) if entry and sl else 0.0
    risk_amount = risk_per_share * qty
    max_risk = capital * MAX_RISK_PER_TRADE_PCT / 100 if capital else 0.0
    risk_ok = capital > 0 and risk_amount <= max_risk
    risk_pct = (risk_amount / capital * 100) if capital else 0.0
    rows.append(CriterionResult(
        key="risk_per_trade",
        label=f"Risk ≤ {MAX_RISK_PER_TRADE_PCT:.0f}% of capital",
        passed=risk_ok,
        detail=(f"Risk ₹{risk_amount:,.0f} ({risk_pct:.2f}%) "
                f"vs cap ₹{max_risk:,.0f}"),
        severity="info" if risk_ok else "danger",
    ))

    # 5 — daily loss within MAX_DAILY_LOSS_PCT
    max_daily = capital * MAX_DAILY_LOSS_PCT / 100 if capital else 0.0
    breached = daily_loss >= max_daily
    near_breach = (daily_loss + risk_amount) > max_daily * 1.5
    daily_ok = capital > 0 and not breached and not near_breach
    rows.append(CriterionResult(
        key="daily_loss",
        label=f"Daily loss ≤ {MAX_DAILY_LOSS_PCT:.0f}% of capital",
        passed=daily_ok,
        detail=(f"Loss so far ₹{daily_loss:,.0f} | this trade risks "
                f"₹{risk_amount:,.0f} | cap ₹{max_daily:,.0f}"),
        severity="info" if daily_ok else "danger",
    ))

    # 6 — position size ≤ MAX_POSITION_SIZE_PCT
    position_value = entry * qty
    max_pos = capital * MAX_POSITION_SIZE_PCT / 100 if capital else 0.0
    pos_ok = capital > 0 and position_value <= max_pos
    rows.append(CriterionResult(
        key="position_size",
        label=f"Position ≤ {MAX_POSITION_SIZE_PCT:.0f}% of capital",
        passed=pos_ok,
        detail=(f"Notional ₹{position_value:,.0f} vs cap ₹{max_pos:,.0f}"),
        severity="info" if pos_ok else "warning",
    ))

    # 7 — risk:reward ≥ MIN_RISK_REWARD_RATIO
    reward_per_share = abs(target - entry) if entry and target else 0.0
    rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0.0
    rr_ok = rr >= MIN_RISK_REWARD_RATIO
    rows.append(CriterionResult(
        key="risk_reward",
        label=f"R:R ≥ {MIN_RISK_REWARD_RATIO:.1f}",
        passed=rr_ok,
        detail=f"R:R {rr:.2f}:1",
        severity="info" if rr_ok else "warning",
    ))

    # 8 — confidence
    conf_ok = confidence >= MIN_CONFIDENCE
    rows.append(CriterionResult(
        key="confidence",
        label=f"Confidence ≥ {MIN_CONFIDENCE:.2f}",
        passed=conf_ok,
        detail=f"Plan confidence {confidence:.2f}",
        severity="info" if conf_ok else "warning",
    ))

    # 9 — open positions
    pos_count_ok = open_positions < MAX_OPEN_POSITIONS
    rows.append(CriterionResult(
        key="open_positions",
        label=f"Open positions < {MAX_OPEN_POSITIONS}",
        passed=pos_count_ok,
        detail=f"{open_positions} of {MAX_OPEN_POSITIONS} slots in use",
        severity="info" if pos_count_ok else "warning",
    ))

    return rows


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_setup(
    symbol: str,
    side: str,
    *,
    capital: float,
    daily_loss: float = 0.0,
    open_positions: int = 0,
    data_port: Any = None,
    tenant_id: Any = None,
) -> SetupPayload:
    """Synchronous Stage 5 setup-preview builder.

    Args:
        symbol: NSE tradingsymbol (e.g. ``"TCS"``).  Resolved against the
            ``Symbol`` table for metadata; not strictly required to exist
            (the data port handles missing rows by returning empty data).
        side: ``"BUY"`` or ``"SELL"``.
        capital: portfolio capital in INR — used by every cap criterion.
        daily_loss: cumulative loss so far today (positive number).
        open_positions: open-position count.
        data_port: pluggable port for tests.  Defaults to
            ``DefaultMarketData(tenant_id)``.
        tenant_id: passed through when creating the default port.

    Returns:
        ``SetupPayload`` with the proposed plan, the regime snapshot, the
        per-criterion breakdown, and the authoritative validate_trade verdict.
    """
    symbol = symbol.strip().upper()
    side = side.strip().upper()
    errors: list[str] = []

    # ── Market snapshot ────────────────────────────────────────────────
    if data_port is None:
        from apps.market_data.services.data_port import DefaultMarketData
        data_port = DefaultMarketData(tenant_id)

    market, market_errors = _market_snapshot(data_port, symbol)
    errors.extend(market_errors)

    regime = _regime_snapshot()

    # ── Build plan (or fail soft if data is too thin) ──────────────────
    plan_obj: ProposedPlan | None = None
    plan_dict: dict[str, Any] = {}
    if not market.last or not market.atr:
        errors.append(
            "insufficient market data — need both LTP and ATR(14) "
            "to size a plan"
        )
    else:
        try:
            plan_obj, _notes = _build_plan(
                symbol=symbol,
                side=side,
                last=market.last,
                atr=market.atr,
                capital=capital,
            )
            plan_dict = asdict(plan_obj)
        except Exception as e:  # noqa: BLE001
            errors.append(f"plan_builder: {e}")

    # ── Run the production validator + the per-criterion breakdown ────
    if plan_obj is not None:
        approved, reason, _details = validate_trade(
            plan=plan_dict,
            capital=capital,
            daily_loss=daily_loss,
            open_positions=open_positions,
        )
        criteria = evaluate_criteria(
            plan=plan_dict,
            capital=capital,
            daily_loss=daily_loss,
            open_positions=open_positions,
        )
    else:
        approved = False
        reason = "Cannot evaluate: insufficient data to build a plan."
        criteria = evaluate_criteria(
            plan={"side": side, "confidence": DEFAULT_CONFIDENCE,
                  "entry_price": market.last or 0,
                  "stop_loss": 0, "target": 0, "quantity": 0,
                  "symbol": symbol},
            capital=capital,
            daily_loss=daily_loss,
            open_positions=open_positions,
        )

    risk = RiskBreakdown(approved=approved, reason=reason, criteria=criteria)

    return SetupPayload(
        as_of=datetime.now(tz=timezone.utc).isoformat(),
        symbol=symbol,
        side=side,
        market=market,
        plan=plan_obj,
        regime=regime,
        risk=risk,
        errors=errors,
    )
