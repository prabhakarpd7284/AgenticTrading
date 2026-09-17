"""Swing-aware forward enrichment for multi-day Signals.

The legacy ``enrich_signals`` command measures **same-day intraday** MFE/MAE
(5-minute candles between signal_time and 15:30). That is correct for intraday
screener/scalp signals but *wrong* for swing/positional ideas (e.g. StockEdge
composite momentum), whose payoff plays out over many sessions — same-day it
looks flat and gets marked EXPIRED with ~0 capture.

This module enriches swing signals with **forward daily candles** over a holding
window: max favorable / adverse excursion across the next N trading days, plus
per-horizon (5/10/20d) snapshots. It fills the same ``Signal`` fields the monthly
capture matrix reads (``max_favorable_move`` / ``max_adverse_move`` / ``eod_price``)
so swing signals score meaningfully.

``SWING_SOURCES`` is the single source of truth for "which sources are swing" —
the intraday enricher excludes these so the two never clobber each other.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Callable

import structlog

log = structlog.get_logger(__name__)

# Sources whose signals are multi-day. Enriched here with forward daily candles;
# the legacy intraday enricher excludes them.
SWING_SOURCES = ["STOCKEDGE"]

DEFAULT_HORIZON_DAYS = 20            # trading days held
DEFAULT_HORIZON_SNAPSHOTS = (5, 10, 20)

# candle_fetcher(symbol, start_date, end_date) -> chronological list of daily
# {"ts": datetime, "h": float, "l": float, "c": float}
CandleFetcher = Callable[[str, date, date], list[dict]]


def forward_excursion(entry: float, side: str, candles: list[dict],
                      horizons: tuple[int, ...] = DEFAULT_HORIZON_SNAPSHOTS) -> dict | None:
    """Max favorable/adverse excursion over forward daily candles.

    ``candles`` — chronological daily bars (``h``/``l``/``c``) AFTER the signal
    date. Returns ``{mfe, mae, eod_price, days, horizons:{fwd_5d_mfe,...}}`` or
    ``None`` when empty. Moves are clamped at 0 (an excursion never goes
    "negative" — that's just the other side's excursion).
    """
    if not candles:
        return None
    is_buy = side == "BUY"

    def excursion(window: list[dict]) -> tuple[float, float]:
        hi = max(c["h"] for c in window)
        lo = min(c["l"] for c in window)
        if is_buy:
            return round(max(0.0, hi - entry), 2), round(max(0.0, entry - lo), 2)
        return round(max(0.0, entry - lo), 2), round(max(0.0, hi - entry), 2)

    mfe, mae = excursion(candles)
    per: dict[str, float] = {}
    for h in horizons:
        w = candles[:h]
        if w:
            m, a = excursion(w)
            per[f"fwd_{h}d_mfe"] = m
            per[f"fwd_{h}d_mae"] = a
    return {
        "mfe": mfe, "mae": mae,
        "eod_price": round(candles[-1]["c"], 2),
        "days": len(candles), "horizons": per,
    }


def enrich_swing_signal(sig, candles: list[dict], *, horizon: int, finalize: bool) -> bool:
    """Apply forward enrichment to one Signal. Returns True if it was updated.

    ``finalize`` (window fully elapsed) also sets ``eod_price`` and flips a
    PENDING outcome to EXPIRED — which is the sentinel that stops re-enrichment.
    A non-final (mid-window) pass fills MFE/MAE + horizon snapshots but leaves
    ``eod_price`` NULL so the next run refreshes it.
    """
    window = candles[:horizon] if horizon else candles
    fx = forward_excursion(sig.entry_price, sig.side, window)
    if fx is None:
        return False

    sig.max_favorable_move = fx["mfe"]
    sig.max_adverse_move = fx["mae"]
    ind = dict(sig.indicators or {})
    ind.update(fx["horizons"])
    ind["fwd_days"] = fx["days"]
    sig.indicators = ind
    fields = ["max_favorable_move", "max_adverse_move", "indicators"]

    if finalize:
        sig.eod_price = fx["eod_price"]
        fields.append("eod_price")
        if sig.outcome == sig.__class__.Outcome.PENDING:
            sig.outcome = sig.__class__.Outcome.EXPIRED
            fields.append("outcome")

    sig.save(update_fields=fields)
    return True


def enrich_swing_signals(
    tenant, *,
    source: str = "STOCKEDGE",
    horizon: int = DEFAULT_HORIZON_DAYS,
    as_of: date | None = None,
    candle_fetcher: CandleFetcher | None = None,
    partial: bool = False,
) -> dict:
    """Enrich a tenant's un-enriched swing signals with forward daily candles.

    Selects ``source`` signals with ``eod_price IS NULL`` and ``signal_date <
    as_of``. The holding window is "complete" once ``horizon`` forward daily
    bars exist (or enough calendar time has passed); complete windows finalize,
    incomplete ones are skipped unless ``partial=True``. Fully non-blocking:
    a per-symbol fetch error skips that signal, never raises.
    """
    from apps.strategies.models import Signal

    as_of = as_of or date.today()
    fetch = candle_fetcher or default_daily_fetcher()

    qs = (
        Signal.objects.filter(
            tenant=tenant, source=source,
            eod_price__isnull=True, signal_date__lt=as_of,
        )
        .order_by("signal_date", "symbol")
    )

    enriched = skipped = 0
    for sig in qs:
        start = sig.signal_date + timedelta(days=1)
        # widen the fetch window to cover weekends/holidays for `horizon` bars
        end = min(as_of, sig.signal_date + timedelta(days=int(horizon * 1.6) + 5))
        try:
            raw = fetch(sig.symbol, start, end) or []
        except Exception as exc:  # noqa: BLE001 - never break the loop
            log.warning("swing_enrich.fetch_failed", symbol=sig.symbol, error=str(exc))
            skipped += 1
            continue

        candles = sorted(
            (c for c in raw if c["ts"].date() > sig.signal_date),
            key=lambda c: c["ts"],
        )
        if not candles:
            skipped += 1
            continue

        window_complete = (
            len(candles) >= horizon
            or (sig.signal_date + timedelta(days=int(horizon * 1.5))) <= as_of
        )
        if not window_complete and not partial:
            skipped += 1
            continue

        if enrich_swing_signal(sig, candles, horizon=horizon, finalize=window_complete):
            enriched += 1

    log.info("swing_enrich.done", tenant=str(tenant), source=source,
             enriched=enriched, skipped=skipped)
    return {"enriched": enriched, "skipped": skipped, "source": source, "as_of": as_of}


# ── real broker-backed daily candle fetcher ───────────────────────────
def default_daily_fetcher() -> CandleFetcher:
    """Daily-candle fetcher backed by the broker (same path as enrich_signals)."""
    from trading.services.data_service import DataService
    from trading.services.ticker_service import ticker_service

    ds = DataService()

    def fetch(symbol: str, start: date, end: date) -> list[dict]:
        token = ticker_service.get_token(symbol)
        if not token:
            log.warning("swing_enrich.no_token", symbol=symbol)
            return []
        ds._ensure_broker()
        raw = ds._broker.fetch_candles(
            token, f"{start:%Y-%m-%d} 09:15", f"{end:%Y-%m-%d} 15:30",
            interval="ONE_DAY",
        )
        return _normalise_daily(raw)

    return fetch


def _normalise_daily(raw) -> list[dict]:
    out: list[dict] = []
    for c in raw or []:
        if isinstance(c, list):
            ts_str, h, lo, cl = c[0], float(c[2]), float(c[3]), float(c[4])
        else:
            ts_str = c.get("timestamp") or c.get("time", "")
            h = float(c.get("high", 0))
            lo = float(c.get("low", 0))
            cl = float(c.get("close", 0))
        if isinstance(ts_str, str):
            ts = datetime.fromisoformat(ts_str.replace("+05:30", ""))
        elif isinstance(ts_str, datetime):
            ts = ts_str
        else:
            ts = datetime.now()
        out.append({"ts": ts, "h": h, "l": lo, "c": cl})
    return out
