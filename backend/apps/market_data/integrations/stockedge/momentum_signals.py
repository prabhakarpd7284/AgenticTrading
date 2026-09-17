"""Turn the StockEdge momentum shortlist into first-class AlphaDesk Signals.

Bridges the pure momentum analytics (:mod:`momentum`) into the strategies +
events + feedback pipeline: for the strongest momentum names it writes
``apps.strategies.Signal`` rows (``source=STOCKEDGE``, a dedicated ``strategy``
string so it gets its own monthly-audit line) and emits one ``signal.fired``
event each. The EOD ``enrich_signals`` job then fills outcomes and the monthly
report scores it automatically — exactly like the screener / OK-scanner.

Signals are long ("BUY") momentum ideas fired off the end-of-day score snapshot:
entry = the snapshot LTP, a fixed-% swing stop, and an R-multiple target.

Background firing resolves the "first tenant" (single-trader convention) and is
fully non-blocking — a missing tenant or an event-emit failure never raises.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import structlog

from . import momentum as M

log = structlog.get_logger(__name__)

STRATEGY_NAME = "StockEdge Composite Momentum"   # <=40 chars → its own audit line
IST = ZoneInfo("Asia/Kolkata")

DEFAULTS = {
    "top": 20,
    "min_composite": 70.0,
    "min_mcap_cr": 5000.0,                # avoid illiquid micro-caps
    "classes": ("sustained", "emerging"),  # actionable longs only
    "stop_pct": 0.04,                     # 4% swing stop
    "rr": 2.0,                            # 2:1 target
}


def _signal_time(as_of_date) -> datetime:
    """EOD (15:30 IST close) timestamp on the snapshot's trading day."""
    return datetime.combine(as_of_date, time(15, 30), tzinfo=IST)


def build_signal_rows(snapshot, **params) -> list[dict]:
    """Select + shape the momentum names into signal dicts (no DB writes).

    Returns at most ``top`` rows, already ranked by composite (desc), each with
    entry/stop/target and the enriched momentum record under ``record``.
    """
    p = {**DEFAULTS, **params}
    classes = tuple(p["classes"])
    rows: list[dict] = []
    for r in M.momentum_universe(snapshot):  # ranked by composite desc
        if r["classification"] not in classes:
            continue
        if (r["composite"] or 0) < p["min_composite"]:
            continue
        if (r["market_cap_cr"] or 0) < p["min_mcap_cr"]:
            continue
        if not r["ltp"] or r["ltp"] <= 0:
            continue
        entry = round(float(r["ltp"]), 2)
        stop = round(entry * (1 - p["stop_pct"]), 2)
        target = round(entry + (entry - stop) * p["rr"], 2)
        rows.append({
            "symbol": r["symbol"], "side": "BUY",
            "entry": entry, "stop": stop, "target": target,
            "confidence": round((r["composite"] or 0) / 100.0, 3),
            "rr": p["rr"], "record": r,
        })
        if len(rows) >= p["top"]:
            break
    return rows


def fire_momentum_signals(snapshot, *, force: bool = False,
                          dry_run: bool = False, **params) -> dict:
    """Persist Signal rows + emit ``signal.fired`` events for the shortlist.

    Idempotent per (tenant, strategy, signal_date): a same-day re-run is a no-op
    unless ``force`` (which replaces the day's momentum signals). ``dry_run``
    builds + returns the rows without touching the DB.
    """
    from apps.strategies.models import Signal
    from apps.tenants.models import Tenant

    rows = build_signal_rows(snapshot, **params)
    if dry_run:
        return {"dry_run": True, "would_fire": len(rows), "rows": rows, "tenant": None}

    tenant = Tenant.objects.order_by("created_at").first()
    if tenant is None:
        log.warning("stockedge.momentum.no_tenant")
        return {"fired": 0, "skipped": 0, "reason": "no_tenant", "rows": rows}

    as_of = snapshot.as_of_date
    existing = Signal.objects.filter(
        tenant=tenant, source=Signal.Source.STOCKEDGE,
        strategy=STRATEGY_NAME, signal_date=as_of,
    )
    if existing.exists():
        if not force:
            return {"fired": 0, "skipped": existing.count(),
                    "already": True, "rows": rows, "tenant": tenant}
        existing.delete()

    st = _signal_time(as_of)
    created = []
    for row in rows:
        rec = row["record"]
        sig = Signal.objects.create(
            tenant=tenant, symbol=row["symbol"], signal_date=as_of, signal_time=st,
            source=Signal.Source.STOCKEDGE, strategy=STRATEGY_NAME, side="BUY",
            entry_price=row["entry"], stoploss=row["stop"], target=row["target"],
            confidence=row["confidence"], risk_reward=row["rr"],
            reasons=[
                f"momentum:{rec['classification']}",
                f"composite {rec['composite']}",
                f"1M/3M/6M {rec['score_1m']}/{rec['score_3m']}/{rec['score_6m']}",
                f"acceleration {rec['acceleration']}",
            ],
            indicators={
                "composite": rec["composite"], "acceleration": rec["acceleration"],
                "classification": rec["classification"],
                "score_1m": rec["score_1m"], "score_3m": rec["score_3m"],
                "score_6m": rec["score_6m"], "zone_1m": rec["zone_1m"],
                "zone_3m": rec["zone_3m"], "zone_6m": rec["zone_6m"],
                "sector": rec["sector"], "market_cap_cr": rec["market_cap_cr"],
            },
        )
        created.append(sig)
        _emit_fired(tenant, sig, rec, st)

    log.info("stockedge.momentum.fired", count=len(created), as_of=str(as_of))
    return {"fired": len(created), "skipped": 0, "signals": created,
            "rows": rows, "tenant": tenant}


def _emit_fired(tenant, sig, rec: dict, ts) -> None:
    """Non-blocking signal.fired event linked to the Signal row."""
    try:
        from apps.events.models import Event
        from apps.events.services.event_writer import emit

        emit(
            tenant=tenant,
            type=Event.Type.SIGNAL_FIRED,
            actor_kind=Event.ActorKind.SYSTEM,
            text=(f"BUY {sig.symbol} @ {sig.entry_price:.2f} "
                  f"(StockEdge momentum · {rec['classification']} · comp {rec['composite']})"),
            signal_id=sig.id,
            payload={
                "source": "STOCKEDGE", "symbol": sig.symbol, "side": "BUY",
                "entry": sig.entry_price, "strategy": STRATEGY_NAME,
                "score": rec["composite"], "classification": rec["classification"],
                "acceleration": rec["acceleration"],
                "score_1m": rec["score_1m"], "score_3m": rec["score_3m"],
                "score_6m": rec["score_6m"],
            },
            ts=ts,
            broadcast=False,
        )
    except Exception:  # noqa: BLE001 - events must never break signal firing
        log.warning("stockedge.momentum.emit_failed", symbol=sig.symbol)
