"""Action Board track record + "missed" replay (UX redesign R3).

``simulate_plan`` replays a card on that day's stored 1m candles (captured
after the close at 15:50): LIMIT entry at the plan price after the card's
as-of time, then the initial stop or target — whichever the bars hit first,
stop first when one bar touches both (same convention as the paper fill
engine) — else exit at the last close before 15:14 IST. The trailing stop is
NOT simulated (initial stop/target only); the result says so.

``track_record`` reports, per strategy: paper results from closed trades that
came from approved cards, and what skipped / expired (and blocked) cards
would have done. Sample sizes are always shown; everything is "unproven".
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import structlog
from django.utils import timezone

from apps.trading.services.action_board import (
    IST,
    SESSION_OPEN,
    STAGE_LABEL,
    at,
    board_setting,
    db_bars,
    hhmm,
)

log = structlog.get_logger(__name__)

SIM_NOTE = ("LIMIT entry at the plan price after the card's as-of time; initial stop/target only "
            "(no trailing stop); stop first if one bar hits both; else exit at the last 1m close "
            "before {eod}.")


def simulate_plan(*, side: str, entry: float, stop: float, target: float, qty: int,
                  start: datetime, bars: list[list], eod: datetime) -> dict:
    """Pure replay. ``bars`` = ``[ts, o, h, l, c, v]`` rows for the whole session."""
    window = [b for b in bars if start <= b[0] < eod]
    base = {"basis": "1m candles", "note": SIM_NOTE.format(eod=f"{eod.astimezone(IST):%H:%M}"),
            "bars": len(window)}
    if not window:
        return {**base, "status": "no_data", "entry_hit": None}
    long_side = side == "BUY"
    risk_ps = abs(entry - stop)
    filled_at = None
    exit_price, exit_time, status = None, None, None
    for b in window:
        t, _o, h, low, _c = b[0], b[1], b[2], b[3], b[4]
        if filled_at is None:
            if (long_side and low <= entry) or (not long_side and h >= entry):
                filled_at = t
                # Fill bar: only the stop can be judged (order inside the bar unknown).
                if (long_side and low <= stop) or (not long_side and h >= stop):
                    exit_price, exit_time, status = stop, t, "stop"
                    break
            continue
        if (long_side and low <= stop) or (not long_side and h >= stop):
            exit_price, exit_time, status = stop, t, "stop"
            break
        if (long_side and h >= target) or (not long_side and low <= target):
            exit_price, exit_time, status = target, t, "target"
            break
    if filled_at is None:
        return {**base, "status": "no_entry", "entry_hit": False}
    if exit_price is None:
        exit_price, exit_time, status = window[-1][4], window[-1][0], "eod"
    direction = 1 if long_side else -1
    r = ((exit_price - entry) * direction / risk_ps) if risk_ps > 0 else None
    return {**base, "status": status, "entry_hit": True,
            "entry_time": filled_at.astimezone(IST).strftime("%H:%M"),
            "exit_time": exit_time.astimezone(IST).strftime("%H:%M"),
            "exit_price": round(float(exit_price), 2),
            "r": round(r, 3) if r is not None else None,
            "pnl": round((exit_price - entry) * direction * qty, 2)}


SIM_STATUSES = ("skipped", "expired", "rejected")


def evaluate_outcomes(day: date, *, run=None, force: bool = False, now: datetime | None = None) -> dict:
    """EOD: expire open cards, then replay every non-executed card of ``day``."""
    from apps.system.services import run_checks as rc
    from apps.system.services.run_receipts import current_run
    from apps.trading.models import TodayAction
    from apps.trading.services.action_decisions import refresh_statuses

    run = run or current_run()
    now = now or timezone.now()
    qs = TodayAction.objects.filter(date=day)
    expired = refresh_statuses(qs, now)
    cards = list(qs.filter(status__in=SIM_STATUSES))
    if not force:
        cards = [c for c in cards if not c.outcome_computed_at]
    eod = at(day, hhmm(board_setting("EOD_EXIT")))
    lo = at(day, SESSION_OPEN)
    cache: dict[str, list] = {}
    counts: dict[str, int] = {}
    no_data = 0
    for c in cards:
        try:
            if c.symbol not in cache:
                cache[c.symbol] = db_bars(c.symbol, lo, eod)
            out = simulate_plan(side=c.side, entry=float(c.entry_price), stop=float(c.stoploss),
                                target=float(c.target), qty=int(c.suggested_qty), start=c.planned_as_of,
                                bars=cache[c.symbol], eod=eod)
            c.outcome = out
            c.outcome_computed_at = now
            c.save(update_fields=["outcome", "outcome_computed_at", "updated_at"])
            counts[out["status"]] = counts.get(out["status"], 0) + 1
            if out["status"] == "no_data":
                no_data += 1
            run.unit_ok()
        except Exception as e:  # noqa: BLE001
            run.unit_failed(error=f"{c.symbol}: {e}")
    run.output(date=day.isoformat(), expired=expired, evaluated=len(cards), outcomes=counts,
               symbols=len(cache))
    if cards and no_data == len(cards):
        run.check(rc.make("candles_present", rc.FAIL,
                          f"no stored 1m candles for {day} — run the after-close capture first"))
    elif no_data:
        run.check(rc.make("candles_present", rc.WARN, f"{no_data} of {len(cards)} card(s) had no 1m candles"))
    elif cards:
        run.check(rc.make("candles_present", rc.PASS, f"1m candles found for all {len(cards)} card(s)"))
    else:
        run.check(rc.make("candles_present", rc.INFO, "no skipped/expired/blocked cards to replay"))
    return {"date": day.isoformat(), "expired": expired, "evaluated": len(cards), "outcomes": counts}


def _stats(rs: list[float], pnls: list[float]) -> dict:
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return {"n": n, "wins": wins, "losses": sum(1 for p in pnls if p < 0),
            "win_rate": round(wins / n, 3) if n else None,
            "avg_r": round(sum(rs) / len(rs), 3) if rs else None,
            "net_pnl": round(sum(pnls), 2) if n else None}


def _sample_label(n: int) -> str:
    return f"n={n} · {STAGE_LABEL.split(' — ')[0]}"


def track_record(tenant, strategy: str = "directional") -> dict:
    """Paper + missed stats for one strategy on one tenant."""
    from apps.trading.models import TodayAction, Trade
    from apps.trading.services.cooldown import trade_net_pnl

    cards = TodayAction.objects.filter(tenant=tenant, strategy=strategy)
    executed = {c.order_id: c for c in cards.filter(status=TodayAction.Status.EXECUTED, order__isnull=False)}
    trades = list(Trade.objects.filter(tenant=tenant, primary_order_id__in=list(executed)))
    closed = [t for t in trades if t.status == Trade.Status.CLOSED]
    rs, pnls = [], []
    for t in closed:
        card = executed.get(t.primary_order_id)
        net = float(trade_net_pnl(t))
        pnls.append(net)
        risk = abs(float(card.entry_price) - float(card.stoploss)) * int(t.quantity) if card else 0
        if risk > 0:
            rs.append(float(t.realized_pnl or Decimal("0")) / risk)
    paper = _stats(rs, pnls)
    paper.update({"open": sum(1 for t in trades if t.status != Trade.Status.CLOSED),
                  "approved_cards": len(executed), "label": _sample_label(paper["n"]),
                  "basis": "closed paper trades from approved cards; P&L net of estimated costs, "
                           "R on gross P&L"})

    def what_if(statuses: tuple[str, ...], shadow: bool | None = None) -> dict:
        qs = cards.filter(status__in=statuses, outcome_computed_at__isnull=False)
        if shadow is not None:
            qs = qs.filter(shadow=shadow)
        entered = [c.outcome for c in qs if (c.outcome or {}).get("entry_hit")]
        s = _stats([o["r"] for o in entered if o.get("r") is not None], [float(o.get("pnl") or 0) for o in entered])
        s.update({"cards": qs.count(), "entered": len(entered),
                  "no_entry": sum(1 for c in qs if (c.outcome or {}).get("status") == "no_entry"),
                  "no_data": sum(1 for c in qs if (c.outcome or {}).get("status") == "no_data"),
                  "label": _sample_label(len(entered))})
        return s

    missed = what_if(("skipped", "expired"))
    missed["basis"] = "skipped + expired cards replayed on that day's 1m candles (gross, before costs)"
    blocked = what_if(("rejected",))
    blocked["basis"] = "risk-rejected and cooldown-shadow cards, same replay"
    total = cards.count()
    first = cards.order_by("date").values_list("date", flat=True).first()
    return {
        "strategy": strategy, "stage": STAGE_LABEL, "proven": False,
        "cards_total": total, "first_date": first.isoformat() if first else None,
        "paper": paper, "missed": missed, "blocked": blocked,
        "summary": f"{_sample_label(paper['n'])} paper trades · {missed['entered']} missed replayed",
        "as_of": timezone.now().isoformat(),
    }
