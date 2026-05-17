"""Earnings + Dividend calendar overlay on open positions.

For each open position we now try to pull the next-earnings date and
forward dividend date from yfinance. yfinance is best-effort — when
it's offline or the symbol isn't recognised we fall back to the stub
contract so the FE schema is stable.

Per-symbol responses are cached for 6 hours (earnings don't move
intraday).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from django.core.cache import cache


_TTL_HOURS = 6


def _next_event(symbol: str) -> dict:
    """Return {earnings_date, ex_div_date, consensus_eps} via yfinance.

    Each field may be None if yfinance doesn't have it.
    """
    key = f"earnings:{symbol}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    out: dict[str, Any] = {
        "earnings_date": None,
        "ex_div_date": None,
        "consensus_eps": None,
        "avg_post_earn_gap_pct": None,
    }
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        # Earnings dates
        cal = getattr(ticker, "calendar", None)
        if cal is not None and "Earnings Date" in (cal.index if hasattr(cal, "index") else []):
            edate = cal.loc["Earnings Date"][0]
            if edate is not None:
                out["earnings_date"] = edate.strftime("%Y-%m-%d") if hasattr(edate, "strftime") else str(edate)[:10]
        if cal is not None and "Earnings Average" in (cal.index if hasattr(cal, "index") else []):
            val = cal.loc["Earnings Average"][0]
            try:
                out["consensus_eps"] = round(float(val), 2)
            except (TypeError, ValueError):
                pass
        # Forward dividend date
        info = getattr(ticker, "info", {}) or {}
        ts = info.get("exDividendDate")
        if ts:
            try:
                out["ex_div_date"] = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                pass
        # Historical avg gap — uses last 4 quarterly earnings if available
        try:
            hist_e = ticker.earnings_history if hasattr(ticker, "earnings_history") else None
            if hist_e is not None and not hist_e.empty:
                surprise = hist_e.get("epsActual") - hist_e.get("epsEstimate")
                gap_estimate = (surprise / hist_e.get("epsEstimate").abs() * 100).tail(4).mean()
                if gap_estimate == gap_estimate:  # not NaN
                    out["avg_post_earn_gap_pct"] = round(float(gap_estimate), 2)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    cache.set(key, out, _TTL_HOURS * 3600)
    return out


def _days_to(dt_str: str | None) -> int | None:
    if not dt_str:
        return None
    try:
        d = date.fromisoformat(dt_str[:10])
    except ValueError:
        return None
    return (d - date.today()).days


def build_earnings_overlay(tenant=None) -> dict[str, Any]:
    rows: list[dict] = []
    seen: set[str] = set()
    try:
        from trading.models import TradeJournal
        open_trades = TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED", "FILLED")
        ).order_by("-created_at")[:50]
        for t in open_trades:
            sym = (t.symbol or "").upper()
            # Skip if we already pulled this symbol — no point hitting
            # yfinance twice for the same name.
            if sym in seen:
                continue
            seen.add(sym)
            event = _next_event(sym) if sym and not sym.startswith(("NIFTY", "BANKNIFTY", "SENSEX")) else {
                "earnings_date": None, "ex_div_date": None,
                "consensus_eps": None, "avg_post_earn_gap_pct": None,
            }
            rows.append({
                "trade_id": t.id,
                "symbol": sym,
                "side": t.side,
                "qty": int(t.quantity or 0),
                "earnings_date": event["earnings_date"],
                "ex_div_date": event["ex_div_date"],
                "consensus_eps": event["consensus_eps"],
                "avg_post_earn_gap_pct": event["avg_post_earn_gap_pct"],
                "days_to_event": _days_to(event["earnings_date"]) or _days_to(event["ex_div_date"]),
            })
    except Exception:  # noqa: BLE001
        pass

    populated = sum(1 for r in rows if r["earnings_date"] or r["ex_div_date"])
    return {
        "count": len(rows),
        "rows": rows,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_source": "yfinance" if populated > 0 else "yfinance-empty",
        "populated_count": populated,
        "note": (
            "Earnings + dividend dates via yfinance (best-effort; some "
            "Indian equities don't have full coverage). 3 days before "
            "earnings: decide keep / hedge / close before the binary print."
        ),
    }
