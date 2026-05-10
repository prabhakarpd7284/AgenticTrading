"""Sector Rotation service — Stage 3 of The Cascade.

Given the regime read from Stage 1/2, which *sectors* are leading today,
and inside those sectors who are the stock-level leaders / laggards?  This
is the drill-in from the pulse-page heatmap.

Design choices (deliberately simple, reuse-first):

* **Reuse** ``SECTOR_TICKERS`` and ``YFinanceProvider`` from
  ``pulse_service`` rather than inventing a second data pipe.  One provider,
  one cache topology, one set of failure modes.
* **Reuse** the ``NIFTY50`` universe from ``trading.intraday.universe`` — we
  only care about liquid, tradeable names.  Non-liquid smallcaps have no
  business on a rotation screen that feeds the shortlist.
* Static ``SECTOR_CONSTITUENTS`` mapping: NSE's official constituent lists
  are slow-moving.  Hard-coding the big 3–6 liquid names per sector is more
  reliable than scraping the NSE FTP at runtime, and keeps the endpoint
  usable when yfinance partly fails.
* **60-second cache** — sector rotation doesn't flip in ticks.  Polling
  every minute from the UI is plenty.

The shape mirrors the pulse-page sector heatmap so the frontend can reuse
tone / colour helpers:

    {
      "as_of": "...",
      "sectors": [
        {
          "key": "NIFTY_IT", "label": "IT", "rank": 1,
          "change_pct": 1.42, "last": 42310.5,
          "leaders":  [{"symbol": "TCS",  "change_pct": 2.10, "last": 4150.3, ...}, ...],
          "laggards": [{"symbol": "WIPRO","change_pct": -0.80, "last":  452.1, ...}],
          "breadth": {"up": 4, "down": 1, "flat": 0},
        },
        ...
      ],
      "errors": [...],
    }
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from django.core.cache import cache

# Reuse existing infra — one provider, one pattern.
from apps.market_data.services.pulse_service import (
    SECTOR_TICKERS,
    YFinanceProvider,
    _safe_float,
)

logger = logging.getLogger(__name__)

CACHE_KEY = "market:rotation:v1"
CACHE_TTL = 60  # seconds — rotation is a slower signal than pulse

# ---------------------------------------------------------------------------
# Sector → liquid NIFTY50 constituents.
#
# Hand-curated top 3–5 names per sector by free-float market cap.  These are
# the stocks an intraday trader would actually look at when a sector runs —
# and they're all in the NIFTY50 universe, which means they clear @RiskGuard's
# liquidity bar automatically.
#
# Kept deliberately short: long lists = noise.  Stage 4 (shortlist) is the
# right place to expand to the full sector index.
# ---------------------------------------------------------------------------
SECTOR_CONSTITUENTS: dict[str, list[str]] = {
    "NIFTY_BANK": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
    ],
    "NIFTY_IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    ],
    "NIFTY_AUTO": [
        "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TATAPOWER",
    ],
    "NIFTY_PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "APOLLOHOSP",
    ],
    "NIFTY_FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    ],
    "NIFTY_METAL": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA",
    ],
    "NIFTY_ENERGY": [
        "RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID", "COALINDIA",
    ],
    "NIFTY_REALTY": [
        # NIFTY50 has no pure-play realty names — leave empty rather than
        # lie.  Frontend will show "—" gracefully.
    ],
    "NIFTY_PSUBANK": [
        "SBIN",
    ],
    "NIFTY_FIN": [
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
    ],
    "NIFTY_MEDIA": [
        # No liquid media names in NIFTY50 currently.
    ],
}

# yfinance ticker suffix for NSE cash-market equities.  M&M → ``M%26M.NS`` is
# not needed; yfinance handles the ampersand directly as ``M%26M.NS`` but the
# library URL-encodes for us.  Empirically ``M&M.NS`` works.
def _yf_equity(symbol: str) -> str:
    return f"{symbol}.NS"


@dataclass
class StockMove:
    symbol: str
    last: float | None
    change_pct: float | None
    change: float | None
    prev_close: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SectorRotation:
    key: str
    label: str
    rank: int
    change_pct: float | None
    last: float | None
    leaders: list[dict[str, Any]] = field(default_factory=list)
    laggards: list[dict[str, Any]] = field(default_factory=list)
    breadth: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RotationPayload:
    as_of: str
    sectors: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sector_label(key: str) -> str:
    """Cosmetic — strip the NIFTY_ prefix and Title-Case what's left.

    Matches the label produced by ``pulse_service.build_pulse`` so the UI
    can reuse the same tone helpers without mapping.
    """
    return key.replace("NIFTY_", "").replace("_", " ").title()


def _classify_breadth(moves: list[StockMove]) -> dict[str, int]:
    up = sum(1 for m in moves if (m.change_pct or 0) > 0.1)
    down = sum(1 for m in moves if (m.change_pct or 0) < -0.1)
    flat = len(moves) - up - down
    return {"up": up, "down": down, "flat": flat}


def _rank_leaders(moves: list[StockMove], *, n: int = 3) -> tuple[list[dict], list[dict]]:
    """Top-N winners and bottom-N losers. Skips stocks with no data so the
    UI never renders an empty cell.
    """
    rated = [m for m in moves if m.change_pct is not None]
    rated.sort(key=lambda m: m.change_pct, reverse=True)  # type: ignore[arg-type]
    leaders = [m.as_dict() for m in rated[:n]]
    laggards = [m.as_dict() for m in rated[-n:]][::-1]  # reverse so worst first
    return leaders, laggards


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_rotation(force: bool = False) -> RotationPayload:
    """Build the rotation payload.  One yfinance fetch for all sectors +
    all constituents, then classify locally.

    Cached for ``CACHE_TTL`` seconds.
    """
    if not force:
        cached = cache.get(CACHE_KEY)
        if cached is not None:
            return cached

    errors: list[str] = []
    provider = YFinanceProvider()

    # Collect every yfinance symbol we need in one batch — indices +
    # constituents — then index by yfinance ticker.
    sector_yf: dict[str, str] = dict(SECTOR_TICKERS)           # sector → yf
    equity_yf: dict[str, str] = {}                             # stock → yf
    for names in SECTOR_CONSTITUENTS.values():
        for sym in names:
            equity_yf.setdefault(sym, _yf_equity(sym))

    all_yf = list(set(list(sector_yf.values()) + list(equity_yf.values())))

    try:
        raw = provider.fetch(all_yf)
    except Exception as e:  # noqa: BLE001
        logger.warning("rotation provider failed: %s", e)
        errors.append(f"data_provider: {e}")
        raw = {}

    # Build a per-stock StockMove map so we can look up constituents cheaply.
    stock_moves: dict[str, StockMove] = {}
    for sym, yf_sym in equity_yf.items():
        data = raw.get(yf_sym, {})
        stock_moves[sym] = StockMove(
            symbol=sym,
            last=_safe_float(data.get("last")),
            prev_close=_safe_float(data.get("prev_close")),
            change=_safe_float(data.get("change")),
            change_pct=_safe_float(data.get("change_pct")),
        )

    # Build the per-sector rotation.
    sectors: list[SectorRotation] = []
    for key, yf_sym in sector_yf.items():
        idx = raw.get(yf_sym, {})
        members = SECTOR_CONSTITUENTS.get(key, [])
        moves = [stock_moves[m] for m in members if m in stock_moves]
        leaders, laggards = _rank_leaders(moves)
        sectors.append(
            SectorRotation(
                key=key,
                label=_sector_label(key),
                rank=0,  # filled in after sort
                change_pct=_safe_float(idx.get("change_pct")),
                last=_safe_float(idx.get("last")),
                leaders=leaders,
                laggards=laggards,
                breadth=_classify_breadth(moves),
            )
        )

    # Rank sectors by % move (missing values sink to the bottom).
    sectors.sort(
        key=lambda s: s.change_pct if s.change_pct is not None else -999,
        reverse=True,
    )
    for i, s in enumerate(sectors, start=1):
        s.rank = i

    payload = RotationPayload(
        as_of=datetime.now(tz=timezone.utc).isoformat(),
        sectors=[s.as_dict() for s in sectors],
        errors=errors,
    )
    cache.set(CACHE_KEY, payload, CACHE_TTL)
    return payload
