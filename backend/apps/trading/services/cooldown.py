"""Losing-streak cooldown — docs/strategies/TRADING_RULES.md §1.

    loss day      = a trading day whose portfolio net P&L (realized, after
                    estimated costs) is < 0. Days without trades don't count.
    trigger       = 3 consecutive loss days (no-trade days are skipped).
    cooldown      = the next 2 NSE trading days: no new automated entries
                    (manual/UI orders still allowed; strategies run in shadow).
    re-entry      = risk per trade × 0.5 until the first profitable day after
                    the cooldown, then back to 100 %.

Deterministic, no LLM. ``evaluate_cooldown`` is a pure function over
``(day, net_pnl)`` pairs so the state machine can be tested without a DB;
``evaluate_for_portfolio`` feeds it from closed ``Trade`` rows and
``refresh_cooldown`` persists the result on ``CooldownState`` (a cache for
the UI / Telegram — the order path always re-evaluates from trades).

Assumptions (the Trade model carries no cost columns):
  * Net P&L = ``Trade.realized_pnl`` − an *estimate* of Indian statutory
    costs + discount-broker brokerage (``estimate_costs``). Disable with
    ``TRADING_RULES["DEDUCT_ESTIMATED_COSTS"] = False``.
  * A trade belongs to its ``trade_date`` (entry day; intraday = exit day).
  * Trade days that fall *inside* a cooldown (manual trades only, since
    automated entries are blocked) neither extend the streak nor clear the
    reduced-risk mode — the streak restarts after the cooldown.
  * A trade day with net P&L exactly 0 is "not a loss" and breaks a streak,
    but it is not "profitable" so it does not end reduced-risk mode.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from django.conf import settings

DEFAULTS = {
    "COOLDOWN_LOSS_DAYS": 3,
    "COOLDOWN_TRADING_DAYS": 2,
    "REDUCED_RISK_MULTIPLIER": 0.5,
    "COOLDOWN_LOOKBACK_DAYS": 365,
    "DEDUCT_ESTIMATED_COSTS": True,
}


def rules_setting(key: str):
    return getattr(settings, "TRADING_RULES", {}).get(key, DEFAULTS.get(key))


NORMAL, COOLDOWN, REDUCED = "normal", "cooldown", "reduced"


@dataclass
class DayPnl:
    day: date
    net_pnl: float


@dataclass
class CooldownStatus:
    as_of: date
    status: str = NORMAL
    loss_streak: int = 0
    trigger_date: date | None = None
    cooldown_until: date | None = None
    risk_multiplier: float = 1.0
    reason: str = ""
    history: list[dict] = field(default_factory=list)

    @property
    def blocks_entries(self) -> bool:
        return self.status == COOLDOWN

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("as_of", "trigger_date", "cooldown_until"):
            d[k] = d[k].isoformat() if d[k] else None
        return d


def _default_is_trading_day(d: date) -> bool:
    from apps.market_data.services.market_calendar import is_trading_day
    return is_trading_day(d)


def nth_trading_day_after(d: date, n: int,
                          is_trading_day: Callable[[date], bool] = _default_is_trading_day) -> date:
    cur, seen = d, 0
    while seen < n:
        cur = cur + timedelta(days=1)
        if is_trading_day(cur):
            seen += 1
    return cur


def evaluate_cooldown(
    days: Iterable[DayPnl],
    as_of: date,
    *,
    is_trading_day: Callable[[date], bool] = _default_is_trading_day,
    loss_days: int | None = None,
    cooldown_days: int | None = None,
    reduced_multiplier: float | None = None,
) -> CooldownStatus:
    """Status for trading day ``as_of`` from completed days strictly before it."""
    loss_days = int(loss_days or rules_setting("COOLDOWN_LOSS_DAYS"))
    cooldown_days = int(cooldown_days or rules_setting("COOLDOWN_TRADING_DAYS"))
    mult = float(reduced_multiplier if reduced_multiplier is not None
                 else rules_setting("REDUCED_RISK_MULTIPLIER"))

    ordered = sorted((d for d in days if d.day < as_of), key=lambda d: d.day)
    streak = 0
    trigger: date | None = None
    until: date | None = None
    reduced = False
    history: list[dict] = []

    for d in ordered:
        kind = "loss" if d.net_pnl < 0 else ("profit" if d.net_pnl > 0 else "flat")
        in_cooldown = trigger is not None and until is not None and trigger < d.day <= until
        history.append({"date": d.day.isoformat(), "net_pnl": round(d.net_pnl, 2),
                        "kind": kind, "in_cooldown": in_cooldown})
        if in_cooldown:
            continue
        if kind == "profit":
            streak = 0
            reduced = False
        elif kind == "loss":
            streak += 1
            if streak >= loss_days:
                trigger = d.day
                until = nth_trading_day_after(d.day, cooldown_days, is_trading_day)
                reduced = True
                streak = 0
        else:
            streak = 0

    st = CooldownStatus(as_of=as_of, loss_streak=streak, trigger_date=trigger,
                        cooldown_until=until, history=history[-15:])
    if trigger is not None and until is not None and trigger < as_of <= until:
        st.status = COOLDOWN
        st.risk_multiplier = 0.0
        st.reason = (f"{loss_days} consecutive loss days ending {trigger.isoformat()} — "
                     f"no automated entries until {until.isoformat()} (inclusive)")
    elif reduced:
        st.status = REDUCED
        st.risk_multiplier = mult
        st.reason = (f"post-cooldown: risk per trade ×{mult:g} until the first "
                     f"profitable day")
    else:
        st.reason = (f"normal (current loss streak {streak}/{loss_days})" if streak
                     else "normal")
    return st


# ── Costs ────────────────────────────────────────────────────────────────

def estimate_costs(*, side: str, entry: float, exit: float, units: float,
                   exchange: str = "NSE", product: str = "INTRADAY") -> float:
    """Rough Indian round-trip costs in INR (discount broker, 2026 rates).

    An estimate, deliberately simple: brokerage, STT, exchange txn charge,
    SEBI fee, stamp duty and 18 % GST. Good enough to tell a scratch day from
    a real loss day; not a contract note.
    """
    if units <= 0 or entry <= 0 or exit <= 0:
        return 0.0
    buy_val, sell_val = (entry * units, exit * units) if side == "BUY" else (exit * units, entry * units)
    turnover = buy_val + sell_val
    exch = (exchange or "NSE").upper()
    if exch in ("NFO", "BFO"):              # options (premium turnover)
        brokerage = 40.0
        stt = 0.001 * sell_val
        txn = 0.0003503 * turnover
        stamp = 0.00003 * buy_val
    elif (product or "INTRADAY").upper() == "DELIVERY":
        brokerage = 0.0
        stt = 0.001 * turnover
        txn = 0.0000297 * turnover
        stamp = 0.00015 * buy_val
    else:                                    # equity intraday
        brokerage = min(20.0, 0.0003 * buy_val) + min(20.0, 0.0003 * sell_val)
        stt = 0.00025 * sell_val
        txn = 0.0000297 * turnover
        stamp = 0.00003 * buy_val
    sebi = turnover * 10 / 1e7
    gst = 0.18 * (brokerage + txn + sebi)
    return round(brokerage + stt + txn + sebi + stamp + gst, 2)


def trade_net_pnl(trade) -> float:
    pnl = float(trade.realized_pnl or 0)
    if not rules_setting("DEDUCT_ESTIMATED_COSTS"):
        return pnl
    if trade.exit_price is None:
        return pnl
    units = float((trade.exit_quantity or trade.quantity or 0) * (trade.lot_size or 1))
    return pnl - estimate_costs(
        side=trade.side, entry=float(trade.fill_price or trade.entry_price),
        exit=float(trade.exit_price), units=units,
        exchange=trade.exchange, product=trade.product,
    )


def daily_net_pnl(portfolio, *, until: date, lookback_days: int | None = None) -> list[DayPnl]:
    from apps.trading.models import Trade

    lookback = int(lookback_days or rules_setting("COOLDOWN_LOOKBACK_DAYS"))
    qs = Trade.objects.filter(
        portfolio=portfolio, status=Trade.Status.CLOSED, realized_pnl__isnull=False,
        trade_date__lt=until, trade_date__gte=until - timedelta(days=lookback),
    ).only("trade_date", "realized_pnl", "side", "entry_price", "fill_price", "exit_price",
           "quantity", "exit_quantity", "lot_size", "exchange", "product")
    per_day: dict[date, float] = defaultdict(float)
    for t in qs.iterator():
        per_day[t.trade_date] += trade_net_pnl(t)
    return [DayPnl(day=d, net_pnl=v) for d, v in sorted(per_day.items())]


def evaluate_for_portfolio(portfolio, as_of: date | None = None) -> CooldownStatus:
    from django.utils import timezone

    as_of = as_of or timezone.localdate()
    return evaluate_cooldown(daily_net_pnl(portfolio, until=as_of), as_of)


def refresh_cooldown(portfolio, as_of: date | None = None) -> CooldownStatus:
    """Evaluate and persist on ``CooldownState`` (one row per portfolio)."""
    from apps.trading.models import CooldownState

    st = evaluate_for_portfolio(portfolio, as_of)
    CooldownState.objects.update_or_create(
        portfolio=portfolio,
        defaults=dict(
            tenant_id=portfolio.tenant_id, as_of=st.as_of, status=st.status,
            loss_streak=st.loss_streak, trigger_date=st.trigger_date,
            cooldown_until=st.cooldown_until,
            risk_multiplier=Decimal(str(st.risk_multiplier)),
            history=st.history, reason=st.reason[:255],
        ),
    )
    return st


def trading_portfolios(tenant=None):
    """Portfolios the rules apply to: capital > 0 (excludes MF holding books)."""
    from apps.trading.models import Portfolio

    qs = Portfolio.objects.filter(capital__gt=0)
    return qs.filter(tenant=tenant) if tenant is not None else qs
