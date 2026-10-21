"""Next-day sentiment filter — TRADING_RULES.md §3 (deterministic).

    strong_negative          → no longs  (BUY blocked)
    strong_positive          → no shorts (SELL blocked)
    results/major event due  → skip the stock (both blocked)
    unscored / no scan       → never blocks (and says so)

Direction verdicts are computed once by the nightly scan and stored on
``NightlyScanRow.allow_long`` / ``allow_short``; this module only reads them.
"""
from __future__ import annotations

from datetime import date

from apps.news.models import NightlyScan, NightlyScanRow

USABLE_STATUSES = (NightlyScan.Status.SUCCESS, NightlyScan.Status.PARTIAL)


def allowed_directions(*, label: str, event_tomorrow: bool, event_purpose: str = "",
                       score: float | None = None) -> tuple[bool, bool, list[str]]:
    """(allow_long, allow_short, reasons) for one stock."""
    reasons: list[str] = []
    allow_long = allow_short = True
    if event_tomorrow:
        allow_long = allow_short = False
        reasons.append(f"event due: {event_purpose or 'corporate event'} → skip stock")
    s = f" ({score:+.2f})" if score is not None else ""
    if label == NightlyScanRow.Label.STRONG_NEGATIVE:
        allow_long = False
        reasons.append(f"strong negative sentiment{s} → no longs")
    elif label == NightlyScanRow.Label.STRONG_POSITIVE:
        allow_short = False
        reasons.append(f"strong positive sentiment{s} → no shorts")
    elif label == NightlyScanRow.Label.UNSCORED:
        reasons.append("sentiment could not be computed → filter not applied")
    return allow_long, allow_short, reasons


def allowed_label(allow_long: bool, allow_short: bool) -> str:
    if allow_long and allow_short:
        return "both"
    if allow_long:
        return "long"
    if allow_short:
        return "short"
    return "none"


def check_direction(symbol: str, side: str, trading_day: date) -> dict:
    scan = (NightlyScan.objects.filter(for_date=trading_day, status__in=USABLE_STATUSES)
            .only("id", "for_date", "status").first())
    if scan is None:
        return {"allowed": True, "status": "no_scan",
                "reason": f"no nightly scan for {trading_day.isoformat()} — sentiment filter not applied"}
    row = scan.rows.filter(symbol=symbol).first()
    if row is None:
        return {"allowed": True, "status": "not_in_scan", "scan_id": scan.id,
                "reason": f"{symbol} not in the nightly scan universe — filter not applied"}
    base = {"scan_id": scan.id, "label": row.sentiment_label,
            "event_tomorrow": row.event_tomorrow, "allowed_directions": row.allowed}
    ok = row.allow_long if side == "BUY" else row.allow_short
    if not ok:
        why = "; ".join(row.reasons) or row.allowed
        return {**base, "allowed": False, "status": "blocked",
                "reason": f"{symbol} {side} not allowed on {trading_day.isoformat()}: {why}"}
    status = "unscored" if row.sentiment_label == NightlyScanRow.Label.UNSCORED else "pass"
    return {**base, "allowed": True, "status": status, "reason": "; ".join(row.reasons)}
