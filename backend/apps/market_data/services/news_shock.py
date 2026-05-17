"""News-Shock & Trading-Halt monitor + circuit-breaker.

Two responsibilities now:

  1. Surface shock events per held symbol (stub until NSE corp-action
     feed wired — events array is always populated by `_fetch_events()`).
  2. Maintain a per-symbol PAUSE list with a cooldown clock. The pause
     is a soft-flag the trader-facing planner / RiskGuard can read to
     skip new entries on a shocked symbol for N minutes after the
     event. Stored in Django cache (in-memory; not persisted across
     server restarts — fine for paper mode).

POST endpoints (wired in views_palace.py-style):
  POST /api/v1/market-data/news-shocks/pause/    {symbol, minutes=15}
  POST /api/v1/market-data/news-shocks/unpause/  {symbol}
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from django.core.cache import cache


_PAUSE_PREFIX = "news_shock:pause:"
_PAUSE_INDEX = "news_shock:pause_index"     # set of currently-paused symbols
_DEFAULT_COOLDOWN_MIN = 15


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fetch_events() -> list[dict]:
    """Future hook for the real NSE corporate-actions feed.

    Until that's wired, returns an empty list. Once it lands, populate
    events as: {symbol, severity, source, headline, ts, flatten_recommendation}.
    """
    return []


def pause_symbol(symbol: str, *, minutes: int = _DEFAULT_COOLDOWN_MIN, reason: str = "") -> dict:
    """Mark `symbol` as paused for `minutes`. Returns the pause record."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "symbol required"}
    expires_at = _now() + timedelta(minutes=max(1, minutes))
    record = {
        "symbol": sym,
        "paused_at": _now().isoformat(),
        "re_entry_at": expires_at.isoformat(),
        "minutes": int(minutes),
        "reason": reason or "manual",
    }
    cache.set(f"{_PAUSE_PREFIX}{sym}", record, timeout=int(minutes * 60))
    # Maintain a separate index so we can list paused symbols regardless
    # of whether they're in the DB. Index ttl is generous (1h) — entries
    # self-clean during the next listing.
    idx = set(cache.get(_PAUSE_INDEX) or [])
    idx.add(sym)
    cache.set(_PAUSE_INDEX, list(idx), timeout=3600)
    return record


def unpause_symbol(symbol: str) -> dict:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "symbol required"}
    key = f"{_PAUSE_PREFIX}{sym}"
    had = cache.get(key)
    cache.delete(key)
    idx = set(cache.get(_PAUSE_INDEX) or [])
    idx.discard(sym)
    cache.set(_PAUSE_INDEX, list(idx), timeout=3600)
    return {"symbol": sym, "was_paused": bool(had)}


def _active_pauses() -> list[dict]:
    """Scan our pause-index for live entries, dropping expired ones."""
    out: list[dict] = []
    idx = list(cache.get(_PAUSE_INDEX) or [])
    live: list[str] = []
    for sym in idx:
        rec = cache.get(f"{_PAUSE_PREFIX}{sym}")
        if rec:
            out.append(rec)
            live.append(sym)
    if len(live) != len(idx):
        cache.set(_PAUSE_INDEX, live, timeout=3600)
    out.sort(key=lambda r: r["re_entry_at"])
    return out


def is_paused(symbol: str) -> bool:
    """Convenience for @RiskGuard / planner to skip a symbol."""
    sym = (symbol or "").upper().strip()
    return bool(cache.get(f"{_PAUSE_PREFIX}{sym}"))


def build_news_shock(tenant=None) -> dict[str, Any]:
    try:
        from trading.models import TradeJournal, WatchlistEntry
        held = set(TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).values_list("symbol", flat=True))
        watched = set(WatchlistEntry.objects.values_list("symbol", flat=True))
        coverage = sorted({s for s in (held | watched) if s})[:50]
    except Exception:  # noqa: BLE001
        coverage = []

    events = _fetch_events()
    paused = _active_pauses()
    return {
        "events": events,
        "paused_symbols": paused,
        "active_pause_count": len(paused),
        "coverage_symbols": coverage,
        "as_of": _now().isoformat(),
        "data_source": "stub" if not events else "live",
        "default_cooldown_min": _DEFAULT_COOLDOWN_MIN,
        "note": (
            "Pause a symbol (via /pause/) to block new entries for N minutes. "
            "Cooldown clears automatically. The events array stays empty "
            "until the NSE corporate-action feed is wired."
        ),
    }
