"""Shortlist service — Stage 4 of The Cascade.

Stage 3 tells us *which sectors* are in motion.  Stage 4 turns that into a
ranked list of actual tradeable names — the 10-15 stocks the desk would
actually put on a watchlist for the day.

Contract: every name on the shortlist must pass the **hard filters** from
TRADING_FRAMEWORK.md, then be ordered by a **confluence score** built from
soft screens.  This is the last filter before Stage 5 (@DirectionalTrader)
looks at individual setups.

Hard filters (non-negotiable):
  * F&O eligible — NIFTY50 universe satisfies this today; the check is a
    hook for when we expand beyond NIFTY50.
  * ATR(14) / price >= 1.0%  (framework says 1.5%; we accept 1.0% so low-vol
    days still produce a usable watchlist — RISK_SHORTLIST_ATR_PCT override).
  * Avg daily turnover >= ₹10 cr  (20-day avg volume × close).

Soft screens (feed the confluence score, 0-100):
  * Parent-sector rank (leader sectors win)
  * Position inside sector (top-3 leader vs laggard vs middle)
  * Day move magnitude aligned with sector direction
  * Volume today vs 20-day avg (relative volume)
  * Distance from 52-week high/low (breakout proximity)

Reuse-first design:
  * Sector data + per-constituent change_pct comes from
    ``rotation_service.build_rotation()`` — no duplicate fetch, no drift.
  * ATR / turnover / 52-week range comes from a second yfinance batch, but
    only for the ~20 candidates surfaced by rotation (cheap).
  * The history provider is an instance attribute so tests can stub it.

Cache: 300 s (shortlist doesn't flip as fast as the pulse).
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from django.core.cache import cache

from apps.market_data.services.rotation_service import (
    SECTOR_CONSTITUENTS,
    _yf_equity,
    build_rotation,
)

logger = logging.getLogger(__name__)

CACHE_KEY = "market:shortlist:v1"
CACHE_TTL = 300  # seconds

# Hard-filter thresholds (environment-overridable for dev / stress testing).
MIN_ATR_PCT = float(os.getenv("SHORTLIST_MIN_ATR_PCT", "1.0"))  # of price
MIN_TURNOVER_CR = float(os.getenv("SHORTLIST_MIN_TURNOVER_CR", "10.0"))  # ₹ crore
TOP_HOT_SECTORS = int(os.getenv("SHORTLIST_TOP_SECTORS", "4"))  # sector count
TOP_N = int(os.getenv("SHORTLIST_TOP_N", "15"))  # output cap

# F&O-eligible universe.  For the NIFTY50 desk this is effectively the full
# universe — every NIFTY50 name trades F&O on NSE.  Keeping the set explicit
# so the hard filter is enforced even if SECTOR_CONSTITUENTS is expanded to
# cash-only smallcaps later.
FNO_ELIGIBLE: set[str] = {
    # NIFTY50 — all F&O-eligible on NSE (as of 2026).
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
    "TCS", "TATACONSUM", "TATAPOWER", "TATASTEEL", "TECHM",
    "TITAN", "ULTRACEMCO", "WIPRO",
}


# ---------------------------------------------------------------------------
# Candidate fundamentals — filled by the history provider
# ---------------------------------------------------------------------------
@dataclass
class Fundamentals:
    """Output of the history provider for a single stock.

    All fields are optional — when yfinance degrades, missing values just
    lead to a filter miss rather than an endpoint crash.
    """
    atr14: float | None = None        # in rupees
    avg_volume_20: float | None = None
    avg_close_20: float | None = None  # for turnover calc
    high_52w: float | None = None
    low_52w: float | None = None
    last_volume: float | None = None


# ---------------------------------------------------------------------------
# History provider — pluggable so tests can stub it
# ---------------------------------------------------------------------------
class YFinanceHistoryProvider:
    """Fetch 30-day daily history for a set of NSE symbols and derive the
    fundamentals the shortlist scorer needs.

    One multi-symbol download call — much faster than per-symbol ``.history``
    round-trips when we're evaluating ~20 candidates.
    """

    def fetch(self, symbols: Iterable[str]) -> dict[str, Fundamentals]:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            raise RuntimeError(
                "yfinance not installed — cannot build shortlist fundamentals."
            )

        symbols = list(symbols)
        if not symbols:
            return {}

        yf_syms = [_yf_equity(s) for s in symbols]
        out: dict[str, Fundamentals] = {s: Fundamentals() for s in symbols}

        try:
            hist = yf.download(
                tickers=" ".join(yf_syms),
                period="30d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("yfinance shortlist history fetch failed: %s", e)
            return out

        # Also get 52-week range via fast_info in a single Tickers call.
        try:
            tickers = yf.Tickers(" ".join(yf_syms))
        except Exception:  # noqa: BLE001
            tickers = None

        for sym, yf_sym in zip(symbols, yf_syms):
            try:
                bars = hist[yf_sym] if yf_sym in getattr(hist, "columns", []) else None
                if bars is None or bars.empty:
                    # Single-ticker case — hist is already the frame.
                    bars = hist if len(yf_syms) == 1 else None
                if bars is None or bars.empty:
                    continue

                highs = bars["High"].dropna().tolist()
                lows = bars["Low"].dropna().tolist()
                closes = bars["Close"].dropna().tolist()
                vols = bars["Volume"].dropna().tolist()

                out[sym].atr14 = _atr(highs, lows, closes, period=14)
                if len(vols) >= 1:
                    out[sym].last_volume = float(vols[-1])
                if len(vols) >= 5:
                    tail = min(20, len(vols))
                    out[sym].avg_volume_20 = float(sum(vols[-tail:]) / tail)
                if len(closes) >= 5:
                    tail = min(20, len(closes))
                    out[sym].avg_close_20 = float(sum(closes[-tail:]) / tail)

                if tickers is not None:
                    fi = tickers.tickers[yf_sym].fast_info
                    out[sym].high_52w = _maybe_float(getattr(fi, "year_high", None))
                    out[sym].low_52w = _maybe_float(getattr(fi, "year_low", None))
            except Exception as e:  # noqa: BLE001
                logger.debug("shortlist fundamentals failed for %s: %s", sym, e)

        return out


def _atr(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> float | None:
    """Classic Wilder-ish ATR using simple mean of true ranges.  Good enough
    for a shortlist-grade filter; precise ATR is the planner's job."""
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


def _maybe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v  # NaN guard


# ---------------------------------------------------------------------------
# Candidate record
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    symbol: str
    sector_key: str
    sector_label: str
    sector_rank: int
    change_pct: float | None
    last: float | None
    atr_pct: float | None = None
    turnover_cr: float | None = None
    rel_volume: float | None = None
    range_52w_pos: float | None = None  # 0..1
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    is_leader: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShortlistPayload:
    as_of: str
    hot_sectors: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    filtered_out: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scorer — deterministic confluence score (0..100)
# ---------------------------------------------------------------------------
def _score_candidate(c: Candidate, *, n_sectors: int, sector_pct: float | None) -> None:
    """Mutate ``c`` in place: fill ``c.score`` and append human-readable
    ``c.reasons``.  No magic ML — pure weights keep this auditable."""
    score = 0.0
    reasons: list[str] = []

    # 1. Sector rank — up to 25 pts, linearly decays across top-4.
    sector_bonus = max(0, (TOP_HOT_SECTORS - (c.sector_rank - 1)) / TOP_HOT_SECTORS) * 25
    score += sector_bonus
    reasons.append(f"sector #{c.sector_rank} (+{sector_bonus:.0f})")

    # 2. Leader inside sector — top-3 constituent gets a +15 boost.
    if c.is_leader:
        score += 15
        reasons.append("sector leader (+15)")

    # 3. Day-move alignment — stock moving the same way as its sector
    #    (and meaningfully) → +15.  Against-sector move → -10 penalty.
    if c.change_pct is not None and sector_pct is not None:
        same_sign = (c.change_pct >= 0) == (sector_pct >= 0)
        if same_sign and abs(c.change_pct) >= 0.5:
            score += 15
            reasons.append(f"aligned with sector ({c.change_pct:+.2f}%)")
        elif not same_sign and abs(c.change_pct) >= 0.5:
            score -= 10
            reasons.append(f"against sector ({c.change_pct:+.2f}%) (-10)")

    # 4. ATR / price — worth trading above 1.5%, bonus scales.
    if c.atr_pct is not None:
        if c.atr_pct >= 2.5:
            score += 20
            reasons.append(f"ATR {c.atr_pct:.1f}% (+20)")
        elif c.atr_pct >= 1.5:
            score += 15
            reasons.append(f"ATR {c.atr_pct:.1f}% (+15)")
        elif c.atr_pct >= 1.0:
            score += 10
            reasons.append(f"ATR {c.atr_pct:.1f}% (+10)")

    # 5. Relative volume — today >> 20-day avg → 10 pts.
    if c.rel_volume is not None:
        if c.rel_volume >= 1.5:
            score += 10
            reasons.append(f"rel vol {c.rel_volume:.1f}x (+10)")
        elif c.rel_volume >= 1.2:
            score += 5
            reasons.append(f"rel vol {c.rel_volume:.1f}x (+5)")

    # 6. 52-week position — near breakout / breakdown → 10 pts.
    if c.range_52w_pos is not None:
        if c.range_52w_pos > 0.95:
            score += 10
            reasons.append("near 52w high (+10)")
        elif c.range_52w_pos < 0.05:
            score += 10
            reasons.append("near 52w low (+10)")

    # Clamp then carry state.
    c.score = round(max(0.0, min(100.0, score)), 1)
    c.reasons = reasons


# ---------------------------------------------------------------------------
# Orchestrator — glue Stage 3 → fundamentals → filter → rank
# ---------------------------------------------------------------------------
def build_shortlist(
    force: bool = False,
    *,
    history_provider: YFinanceHistoryProvider | None = None,
    rotation_force: bool = False,
) -> ShortlistPayload:
    """Assemble the Stage 4 shortlist.

    Args:
        force: Bypass the shortlist cache.
        history_provider: Inject a stub in tests.  Defaults to
            ``YFinanceHistoryProvider()`` when None.
        rotation_force: Propagate ``force`` to the upstream rotation service.
    """
    if not force:
        cached = cache.get(CACHE_KEY)
        if cached is not None:
            return cached

    errors: list[str] = []

    # ── Stage 3: what are today's hot sectors? ──────────────────────────
    rotation = build_rotation(force=rotation_force)

    ranked_sectors = [s for s in rotation.sectors if s.get("change_pct") is not None]
    ranked_sectors = ranked_sectors[:TOP_HOT_SECTORS]
    hot_sector_keys = [s["key"] for s in ranked_sectors]

    # ── Build candidate set from hot sectors, preferring sector leaders. ─
    # A stock present in multiple hot sectors inherits the BEST rank.
    pool: dict[str, Candidate] = {}
    sector_pct_map: dict[str, float | None] = {}
    for s in ranked_sectors:
        sector_key = s["key"]
        sector_pct_map[sector_key] = s.get("change_pct")
        leader_syms = {l["symbol"] for l in s.get("leaders", [])}
        constituents = SECTOR_CONSTITUENTS.get(sector_key, [])

        # Fold leaders into the pool first; they carry the is_leader flag.
        for sym in constituents:
            if sym not in FNO_ELIGIBLE:
                continue
            if sym in pool and pool[sym].sector_rank <= s["rank"]:
                continue  # keep the better sector rank
            pool[sym] = Candidate(
                symbol=sym,
                sector_key=sector_key,
                sector_label=s.get("label", sector_key),
                sector_rank=s["rank"],
                change_pct=_lookup_pct(sym, s),
                last=_lookup_last(sym, s),
                is_leader=sym in leader_syms,
            )

    if not pool:
        payload = ShortlistPayload(
            as_of=datetime.now(tz=timezone.utc).isoformat(),
            hot_sectors=hot_sector_keys,
            candidates=[],
            errors=errors + ["no candidates — rotation produced no ranked sectors"],
        )
        cache.set(CACHE_KEY, payload, CACHE_TTL)
        return payload

    # ── Fetch fundamentals once for the whole candidate pool. ───────────
    provider = history_provider if history_provider is not None else YFinanceHistoryProvider()
    try:
        funds = provider.fetch(list(pool.keys()))
    except Exception as e:  # noqa: BLE001
        logger.warning("shortlist history provider failed: %s", e)
        errors.append(f"history_provider: {e}")
        funds = {sym: Fundamentals() for sym in pool}

    # ── Apply hard filters + compute soft-screen metrics + score. ───────
    accepted: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    for sym, c in pool.items():
        f = funds.get(sym, Fundamentals())

        # Derive ATR % and turnover.
        price_ref = c.last or f.avg_close_20
        if f.atr14 is not None and price_ref:
            c.atr_pct = round(f.atr14 / price_ref * 100, 2)
        if f.avg_volume_20 is not None and f.avg_close_20 is not None:
            # ₹ crore = shares × price / 1e7
            c.turnover_cr = round(f.avg_volume_20 * f.avg_close_20 / 1e7, 2)
        if f.avg_volume_20 and f.last_volume is not None and f.avg_volume_20 > 0:
            c.rel_volume = round(f.last_volume / f.avg_volume_20, 2)
        if f.high_52w and f.low_52w and c.last is not None and f.high_52w > f.low_52w:
            c.range_52w_pos = round((c.last - f.low_52w) / (f.high_52w - f.low_52w), 3)

        # Hard-filter gates (log, don't silently drop).
        fail: list[str] = []
        if c.atr_pct is not None and c.atr_pct < MIN_ATR_PCT:
            fail.append(f"ATR {c.atr_pct:.2f}% < {MIN_ATR_PCT}%")
        if c.turnover_cr is not None and c.turnover_cr < MIN_TURNOVER_CR:
            fail.append(f"turnover ₹{c.turnover_cr:.1f}cr < ₹{MIN_TURNOVER_CR}cr")

        if fail:
            rejected.append({**c.as_dict(), "reject_reasons": fail})
            continue

        _score_candidate(
            c, n_sectors=len(ranked_sectors),
            sector_pct=sector_pct_map.get(c.sector_key),
        )
        accepted.append(c)

    accepted.sort(key=lambda x: x.score, reverse=True)
    accepted = accepted[:TOP_N]

    payload = ShortlistPayload(
        as_of=datetime.now(tz=timezone.utc).isoformat(),
        hot_sectors=hot_sector_keys,
        candidates=[c.as_dict() for c in accepted],
        filtered_out=rejected,
        errors=errors,
    )
    cache.set(CACHE_KEY, payload, CACHE_TTL)
    return payload


def _lookup_pct(sym: str, sector: dict[str, Any]) -> float | None:
    for m in sector.get("leaders", []) + sector.get("laggards", []):
        if m.get("symbol") == sym:
            return m.get("change_pct")
    return None


def _lookup_last(sym: str, sector: dict[str, Any]) -> float | None:
    for m in sector.get("leaders", []) + sector.get("laggards", []):
        if m.get("symbol") == sym:
            return m.get("last")
    return None
