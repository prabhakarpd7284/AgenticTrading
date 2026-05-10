"""Swing Scanner service — Oliver Kell Cycle of Price Action.

Scans the NIFTY 100 universe on daily/weekly charts to detect which of
8 cycle phases each stock is in, with multi-timeframe trend confirmation.

Contract:
  * Reuses the ``trading.swing`` module for cycle detection
  * Returns a dataclass payload matching the frontend's SwingScanPayload
  * Cached 600s (cycle phases change once per trading day)
  * Use ``?force=1`` to bypass the cache

Cycle Phases (Oliver Kell):
  Bullish: RE (Reversal Extension) → WP (Wedge Pop) → EC (EMA Crossback) → BB (Basin Break)
  Bearish: EX (Exhaustion Extension) → WD (Wedge Drop) → EC_BEAR → BB_BEAR

Best setups: WP/EC/BB when daily + weekly trends are both bullish (aligned).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from django.core.cache import cache

logger = logging.getLogger(__name__)

CACHE_KEY = "market:swing-scanner:v1"
CACHE_TTL = 600  # 10 minutes — cycle phases are daily


# ---------------------------------------------------------------------------
# Payload dataclasses — mirror frontend types
# ---------------------------------------------------------------------------

@dataclass
class SwingStock:
    """One stock's cycle analysis result."""
    symbol: str
    phase: str                          # RE, WP, EC, BB, EX, WD, EC_BEAR, BB_BEAR, NONE
    phase_label: str                    # Human-readable
    action: str                         # BUY, SELL, WATCH, AVOID, SHORT, —
    trend_daily: str                    # bullish, bearish, neutral
    trend_weekly: str                   # bullish, bearish, neutral
    aligned: bool                       # daily + weekly trends agree
    confidence: float                   # 0-1
    close: float
    ema10: float
    ema20: float
    ema50: float
    upper_ext: float
    lower_ext: float
    volume_ratio: float
    error: str = ""


@dataclass
class SwingScanPayload:
    """Full scan result for the frontend."""
    as_of: str
    scan_date: str
    total: int = 0
    active: int = 0
    buy_aligned: int = 0
    short_aligned: int = 0
    watch: int = 0
    stocks: list[dict[str, Any]] = field(default_factory=list)
    phase_distribution: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# Phase code → human label
PHASE_LABELS = {
    "RE": "Reversal Extension",
    "WP": "Wedge Pop",
    "EC": "EMA Crossback",
    "BB": "Basin Break",
    "EX": "Exhaustion Extension",
    "WD": "Wedge Drop",
    "EC_BEAR": "Bear EMA Crossback",
    "BB_BEAR": "Bear Basin Break",
    "NONE": "No Phase",
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_swing_scanner(
    force: bool = False,
    symbols: list[str] | None = None,
    scan_date: str | None = None,
) -> SwingScanPayload:
    """Run the Oliver Kell cycle scanner.

    Args:
        force: Bypass the cache.
        symbols: Override symbol list (default: NIFTY 100).
        scan_date: Date to scan (default: today).

    Returns:
        SwingScanPayload with all results and summary metrics.
    """
    if not force:
        cached = cache.get(CACHE_KEY)
        if cached is not None:
            return cached

    scan_date = scan_date or date.today().strftime("%Y-%m-%d")
    errors: list[str] = []

    # Resolve symbols
    if symbols is None:
        try:
            from dashboard_utils.market_scanner import SCREENER_UNIVERSE
            symbols = list(SCREENER_UNIVERSE)
        except Exception as e:
            logger.warning("Could not load SCREENER_UNIVERSE: %s", e)
            symbols = []
            errors.append(f"Symbol universe unavailable: {e}")

    if not symbols:
        return SwingScanPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            scan_date=scan_date,
            errors=errors or ["No symbols to scan"],
        )

    # Run the scanner
    try:
        from trading.swing.ok_scanner import OKScanner
        from trading.swing.ok_cycles import (
            BULLISH_ACTIONABLE,
            BEARISH_ACTIONABLE,
            CyclePhase,
        )

        scanner = OKScanner()
        results = scanner.scan(symbols, scan_date=scan_date)
    except Exception as e:
        logger.exception("Swing scanner failed: %s", e)
        return SwingScanPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            scan_date=scan_date,
            errors=[f"Scanner error: {e}"],
        )

    # Build stock list
    stocks: list[dict[str, Any]] = []
    active_count = 0
    buy_aligned = 0
    short_aligned = 0
    watch_count = 0
    phase_dist: dict[str, int] = {}

    for r in results:
        phase_code = r.phase.value
        phase_dist[phase_code] = phase_dist.get(phase_code, 0) + 1

        if r.phase != CyclePhase.NONE:
            active_count += 1

        if r.phase in BULLISH_ACTIONABLE and r.aligned:
            buy_aligned += 1
        if r.phase in BEARISH_ACTIONABLE and r.aligned:
            short_aligned += 1
        if r.phase == CyclePhase.REVERSAL_EXTENSION:
            watch_count += 1

        # Only include stocks with active phases in the response
        if r.phase == CyclePhase.NONE:
            continue

        stocks.append({
            "symbol": r.symbol,
            "phase": phase_code,
            "phase_label": PHASE_LABELS.get(phase_code, phase_code),
            "action": r.action,
            "trend_daily": r.trend_daily.value,
            "trend_weekly": r.trend_weekly.value,
            "aligned": r.aligned,
            "confidence": r.confidence,
            "close": r.last_close,
            "ema10": r.ema_fast,
            "ema20": r.ema_mid,
            "ema50": r.ema_slow,
            "upper_ext": r.upper_ext,
            "lower_ext": r.lower_ext,
            "volume_ratio": r.volume_ratio,
            "error": r.error,
        })

    payload = SwingScanPayload(
        as_of=datetime.now(timezone.utc).isoformat(),
        scan_date=scan_date,
        total=len(results),
        active=active_count,
        buy_aligned=buy_aligned,
        short_aligned=short_aligned,
        watch=watch_count,
        stocks=stocks,
        phase_distribution={k: v for k, v in phase_dist.items() if k != "NONE"},
        errors=errors,
    )

    if not force:
        cache.set(CACHE_KEY, payload, CACHE_TTL)

    return payload
