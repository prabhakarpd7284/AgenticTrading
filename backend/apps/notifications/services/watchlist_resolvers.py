"""Watchlist resolver functions — one per Kind.

A resolver takes the watchlist row (so it can read `config`) and returns a
list of symbols. Pure functions over existing DB tables — no broker calls,
no network IO. That's the design constraint that lets the periodic refresh
sweep complete in <100ms for hundreds of watchlists.

Each resolver MUST:
  * read its tuning from `watchlist.config` with sane defaults
  * cap output (top_n or hard ceiling) so a runaway query never blows the
    JSON column up to MB-scale
  * uppercase + dedup is unnecessary — TradingViewWatchlist.save handles it

Adding a new kind is a four-step ritual:
  1. Add a Kind enum value to TradingViewWatchlist.Kind
  2. Write a resolver function here
  3. Register in RESOLVERS below
  4. Add the kind to the frontend kind picker
"""
from __future__ import annotations

from datetime import timedelta
from typing import Callable

import structlog
from django.db.models import Count, Q
from django.utils import timezone

from apps.notifications.models import TradingViewWatchlist
from apps.strategies.models import Signal, WatchlistEntry
from apps.trading.models import Trade

log = structlog.get_logger()


# ── Resolver helpers ─────────────────────────────────────────────────────

def _window_start(config: dict, key: str, default: int, unit: str = "days"):
    """Read a positive int from config with a default. unit = "days"|"hours"."""
    raw = config.get(key, default)
    try:
        n = max(1, int(raw))
    except (TypeError, ValueError):
        n = default
    kwargs = {unit: n}
    return timezone.now() - timedelta(**kwargs), n


def _top_n(config: dict, default: int = 20) -> int:
    try:
        return max(1, min(200, int(config.get("top_n", default))))
    except (TypeError, ValueError):
        return default


# ── SIGNAL_RANK — top-N by signal count across all sources ──────────────

def resolve_signal_rank(wl: TradingViewWatchlist) -> list[str]:
    """Top-N symbols by signal count in the last `window_days` (default 7).
    Sorted by count desc, then by recency desc as the tie-breaker so the
    list is stable across refreshes when counts collide."""
    since, _ = _window_start(wl.config, "window_days", 7, "days")
    top_n = _top_n(wl.config)
    qs = (
        Signal.objects
        .filter(tenant=wl.tenant, signal_time__gte=since)
        .values("symbol")
        .annotate(c=Count("id"))
        .order_by("-c", "-symbol")[:top_n]
    )
    return [row["symbol"] for row in qs]


# ── SOURCE_HOT — top-N filtered to one source ───────────────────────────

def resolve_source_hot(wl: TradingViewWatchlist) -> list[str]:
    """Like SIGNAL_RANK but pinned to a single Source. config.source must be
    a valid Signal.Source value (TRADINGVIEW, SCREENER, OK_SCANNER, PREMARKET).
    Empty/unknown source falls back to the default RANK behaviour rather
    than producing 0 rows — operator forgetting to set the source shouldn't
    silently empty the watchlist."""
    source = str(wl.config.get("source") or "").upper().strip()
    valid = {choice for choice, _ in Signal.Source.choices}
    if source not in valid:
        log.warning("watchlist.source_hot.invalid_source",
                    wl_id=str(wl.id), got=source)
        return resolve_signal_rank(wl)

    since, _ = _window_start(wl.config, "window_days", 7, "days")
    top_n = _top_n(wl.config)
    qs = (
        Signal.objects
        .filter(tenant=wl.tenant, source=source, signal_time__gte=since)
        .values("symbol")
        .annotate(c=Count("id"))
        .order_by("-c", "-symbol")[:top_n]
    )
    return [row["symbol"] for row in qs]


# ── RECENT_ACTIVE — anything that fired in the last N hours ─────────────

def resolve_recent_active(wl: TradingViewWatchlist) -> list[str]:
    """Distinct symbols with any Signal in the last `window_hours` (default 24).
    No top-N cap because the natural cap is "however many symbols actually
    fired" — typically 10-50 even on a noisy day. Hard ceiling at 500 for
    safety. Sorted by most-recent-fire desc."""
    since, _ = _window_start(wl.config, "window_hours", 24, "hours")
    qs = (
        Signal.objects
        .filter(tenant=wl.tenant, signal_time__gte=since)
        .values("symbol")
        .annotate(latest=Count("id"))
        .order_by("-latest")[:500]
    )
    return [row["symbol"] for row in qs]


# ── TRADED_RECENTLY — symbols you actually executed ─────────────────────

def resolve_traded_recently(wl: TradingViewWatchlist) -> list[str]:
    """Symbols on Trade rows that crossed into a real-money state in the last
    `window_days` (default 30). 'Real-money' = SENT/PARTIAL/FILLED/CLOSED;
    we exclude PLAN/APPROVED/REJECTED/CANCELLED because they never touched
    the broker. Useful for 'what am I still managing?' workflows."""
    since, _ = _window_start(wl.config, "window_days", 30, "days")
    realised = [
        Trade.Status.SENT,
        Trade.Status.PARTIAL,
        Trade.Status.FILLED,
        Trade.Status.CLOSED,
    ]
    qs = (
        Trade.objects
        .filter(
            tenant=wl.tenant,
            created_at__gte=since,
            status__in=realised,
        )
        .values("symbol")
        .annotate(c=Count("id"))
        .order_by("-c", "symbol")[:200]
    )
    return [row["symbol"] for row in qs]


# ── SHORTLIST_TODAY — the premarket scanner's daily output ──────────────

def resolve_shortlist_today(wl: TradingViewWatchlist) -> list[str]:
    """Today's WatchlistEntry rows from the premarket scanner — Cascade
    Stage 4. config.outcomes (optional list) filters by outcome; default
    keeps WATCHING and TRIGGERED (excluding SKIPPED/NO_SIGNAL etc.)."""
    today = timezone.now().date()
    wanted_outcomes = wl.config.get("outcomes") or [
        WatchlistEntry.Outcome.WATCHING,
        WatchlistEntry.Outcome.TRIGGERED,
        WatchlistEntry.Outcome.TRADED,
    ]
    qs = (
        WatchlistEntry.objects
        .filter(
            tenant=wl.tenant,
            scan_date=today,
            outcome__in=wanted_outcomes,
        )
        .order_by("-score")
        .values_list("symbol", flat=True)[:200]
    )
    return list(qs)


# ── Dispatcher ──────────────────────────────────────────────────────────

Resolver = Callable[[TradingViewWatchlist], list[str]]

RESOLVERS: dict[str, Resolver] = {
    TradingViewWatchlist.Kind.SIGNAL_RANK:     resolve_signal_rank,
    TradingViewWatchlist.Kind.SOURCE_HOT:      resolve_source_hot,
    TradingViewWatchlist.Kind.RECENT_ACTIVE:   resolve_recent_active,
    TradingViewWatchlist.Kind.TRADED_RECENTLY: resolve_traded_recently,
    TradingViewWatchlist.Kind.SHORTLIST_TODAY: resolve_shortlist_today,
}


def resolve_symbols(wl: TradingViewWatchlist) -> list[str]:
    """Return the symbols for this watchlist. MANUAL kind returns the stored
    list unchanged. Auto kinds dispatch to their resolver. Unknown kinds log
    a warning and return the empty list so the periodic sweep keeps going."""
    if wl.kind == TradingViewWatchlist.Kind.MANUAL:
        return list(wl.symbols or [])
    fn = RESOLVERS.get(wl.kind)
    if fn is None:
        log.warning("watchlist.resolve.unknown_kind",
                    wl_id=str(wl.id), kind=wl.kind)
        return []
    return fn(wl)


def refresh_watchlist(wl: TradingViewWatchlist) -> int:
    """Resolve + persist for one auto watchlist. Returns the new symbol count.
    Safe for MANUAL — no-ops without writing. Errors propagate; the periodic
    task wraps the loop in per-row try/except."""
    if wl.kind == TradingViewWatchlist.Kind.MANUAL:
        return len(wl.symbols or [])
    symbols = resolve_symbols(wl)
    # Bypass save() normalisation cost — resolvers already return uppercase
    # uniques. Use .objects.filter().update() so we don't overwrite a
    # concurrent UI edit on the description/name field.
    now = timezone.now()
    TradingViewWatchlist.objects.filter(pk=wl.pk).update(
        symbols=symbols,
        symbols_refreshed_at=now,
    )
    wl.symbols = symbols
    wl.symbols_refreshed_at = now
    return len(symbols)
