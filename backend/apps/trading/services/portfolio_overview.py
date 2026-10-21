"""Portfolio overview + attribution (UX redesign R2, plan §5).

One read model over the tenant's whole book, built from the sources chosen in
``docs/REDESIGN_PORTFOLIO_SOURCE.md``:

* equity  — open/closed ``Trade`` rows of the trading-book portfolios;
* options — ``OptionsPosition`` / ``OptionsLeg``;
* mutual funds — ``MFHolding`` at the latest ``MFNav`` (own staleness flag);
* history — ``PortfolioSnapshot`` of the trading-book portfolios only;
* broker — ``BrokerPositionSnapshot``, shown beside the book for reconciliation,
  never merged into it.

Honesty rules: a component with no data source is ``not_connected`` and its
numbers are ``None`` (never 0); a component that is connected but holds
nothing is ``empty``. Every component carries its own freshness.

Accounting (paper book): ``Portfolio.capital`` is the starting capital and is
not compounded by fills, so realized P&L is summed from ``Trade`` /
``OptionsPosition`` rows.

  trading value = capital + realized P&L + open mark-to-market (equity + options)
  deployed      = cost of open equity positions + premium paid on long option legs
  cash          = capital + realized P&L - deployed
  total value   = trading value + MF current value (when connected)
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from django.conf import settings
from django.db.models import Max
from django.db.models.functions import TruncDate
from django.utils import timezone

IST = ZoneInfo("Asia/Kolkata")

OPEN_POSITION = ("FILLED", "PARTIAL")      # in the market
PENDING_ORDER = ("SENT", "QUEUED", "APPROVED")
OPEN_OPTIONS = ("ACTIVE", "PARTIAL", "HEDGED")
LONG_LEGS = ("LONG_CE", "LONG_PE")
SHORT_LEGS = ("SHORT_CE", "SHORT_PE")

# Freshness limits.
SNAPSHOT_MAX_AGE = timedelta(minutes=5)        # beat writes one every 60 s
LIVE_MARK_MAX_AGE = timedelta(minutes=5)       # during market hours
BROKER_MAX_AGE = timedelta(minutes=15)

ASSET_LABELS = {"equity": "Equity", "options": "Options", "mutual_funds": "Mutual funds", "cash": "Cash"}


# ─── small helpers ───────────────────────────────────────────────────────

def _f(x) -> float | None:
    return None if x is None else round(float(x), 2)


def _pct(num, den) -> float | None:
    if num is None or not den:
        return None
    return round(float(num) / float(den) * 100, 2)


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def market_open(now: datetime) -> bool:
    from apps.market_data.services.market_calendar import is_trading_day

    local = now.astimezone(IST)
    return is_trading_day(local.date()) and time(9, 15) <= local.time() < time(15, 30)


def freshness(as_of: datetime | None, *, max_age: timedelta, now: datetime, source: str,
              note: str = "") -> dict:
    """``{"as_of", "age_seconds", "stale", "source"}`` — stale when unknown or too old."""
    if as_of is None:
        return {"as_of": None, "age_seconds": None, "stale": True, "source": source,
                "note": note or "no timestamp available"}
    age = (now - as_of).total_seconds()
    out = {"as_of": as_of.isoformat(), "age_seconds": round(age), "stale": age > max_age.total_seconds(),
           "source": source}
    if note:
        out["note"] = note
    return out


def mark_price(symbol: str, now: datetime) -> tuple[float | None, str, datetime | None]:
    """Server-side mark for an open position without touching the broker:
    Redis ``ltp:<symbol>`` (live tick cache) → newest local Candle close."""
    from django.core.cache import cache

    from apps.market_data.models import Candle

    try:
        v = cache.get(f"ltp:{symbol}")
        if v is not None and float(v) > 0:
            return float(v), "live tick cache", now
    except Exception:  # noqa: BLE001 — cache down → fall through to candles
        pass
    row = (Candle.objects.filter(symbol__tradingsymbol=symbol)
           .order_by("-t").values("c", "t", "interval").first())
    if row and row["c"]:
        # A daily candle is stamped at session midnight; its close is known at 15:30.
        t = row["t"]
        if row["interval"] == "1d":
            t = datetime.combine(t.astimezone(IST).date(), time(15, 30), tzinfo=IST)
        return float(row["c"]), f"{row['interval']} candle close", t
    return None, "unpriced", None


def period_starts(now: datetime) -> dict[str, datetime | None]:
    local = now.astimezone(IST)
    day0 = datetime.combine(local.date(), time(0), tzinfo=IST)
    return {
        "day": day0,
        "week": day0 - timedelta(days=local.weekday()),
        "month": day0.replace(day=1),
        "since_start": None,
    }


def sector_of(symbol: str) -> str:
    from apps.trading.services.monthly_report import SECTOR_MAP

    return SECTOR_MAP.get(symbol.upper().removesuffix("-EQ"), "Other")


def _units(t) -> int:
    qty = t.fill_quantity or t.quantity or 0
    return int(qty) * int(t.lot_size or 1)


def _entry(t) -> float:
    return float(t.fill_price if t.fill_price is not None else t.entry_price)


def _strategy_label(run, origin: str) -> str:
    if run is not None and getattr(run, "strategy_name", ""):
        return run.strategy_name
    return {"manual": "Manual", "api": "API / webhook", "broker_sync": "Broker sync"}.get(origin, "Unattributed")


def _closed_at(t) -> datetime:
    if t.closed_at:
        return t.closed_at
    return datetime.combine(t.trade_date, time(15, 30), tzinfo=IST)


# ─── components ──────────────────────────────────────────────────────────

def _equity(tenant, book_ids, now) -> dict:
    from apps.trading.models import Trade

    opens = list(Trade.objects.filter(tenant=tenant, portfolio_id__in=book_ids, status__in=OPEN_POSITION)
                 .select_related("strategy_run").order_by("symbol"))
    pending = Trade.objects.filter(tenant=tenant, portfolio_id__in=book_ids, status__in=PENDING_ORDER).count()
    ever = Trade.objects.filter(tenant=tenant, portfolio_id__in=book_ids).exists()

    rows, oldest_mark, unpriced = [], None, 0
    cost = market = unreal = risk = 0.0
    stops_missing = 0
    for t in opens:
        units = _units(t)
        entry = _entry(t)
        direction = 1 if t.side == "BUY" else -1
        price, src, as_of = mark_price(t.symbol, now)
        if price is None and t.last_ltp:
            price, src, as_of = float(t.last_ltp), "last stored LTP", t.updated_at
        notional = entry * units
        cost += notional
        pnl = None
        if price is None:
            unpriced += 1
        else:
            pnl = (price - entry) * units * direction
            unreal += pnl
            market += price * units
            if as_of and (oldest_mark is None or as_of < oldest_mark):
                oldest_mark = as_of
        stop = t.trail_stop if t.trail_stop is not None else t.stop_loss
        ref = price if price is not None else entry
        stop_risk = None
        if stop:
            stop_risk = max(0.0, (ref - float(stop)) * units * direction)
            risk += stop_risk
        else:
            stops_missing += 1
        rows.append({
            "id": str(t.id), "symbol": t.symbol, "side": t.side, "units": units, "entry": round(entry, 2),
            "mark": None if price is None else round(price, 2), "mark_source": src,
            "mark_as_of": _iso(as_of), "cost": round(notional, 2),
            "unrealized_pnl": None if pnl is None else round(pnl, 2),
            "stop": _f(stop), "risk_to_stop": None if stop_risk is None else round(stop_risk, 2),
            "strategy": _strategy_label(t.strategy_run, t.origin), "sector": sector_of(t.symbol),
            "product": t.product,
        })

    is_open = market_open(now)
    if not opens:
        fresh = {"as_of": None, "age_seconds": None, "stale": False, "source": "no open positions"}
    else:
        # Outside market hours the last close is the right mark; only flag
        # marks that predate the latest completed session.
        from apps.system.services.run_checks import expected_session

        if is_open:
            fresh = freshness(oldest_mark, max_age=LIVE_MARK_MAX_AGE, now=now, source="tick cache / candles")
        else:
            session_close = datetime.combine(expected_session(now), time(15, 29), tzinfo=IST)
            fresh = freshness(oldest_mark, max_age=timedelta(days=30), now=now, source="tick cache / candles")
            fresh["stale"] = oldest_mark is None or oldest_mark < session_close
        if unpriced:
            fresh["stale"] = True
            fresh["note"] = f"{unpriced} position(s) have no price"
    return {
        "status": "ok" if opens else "empty",
        "label": f"{len(opens)} open position(s)" if opens else "no open equity positions",
        "open_count": len(opens),
        "pending_orders": pending,
        "has_history": ever,
        "cost": round(cost, 2) if opens else None,
        "market_value": round(market, 2) if opens and not unpriced else (round(market, 2) if market else None),
        "unrealized_pnl": round(unreal, 2) if opens else None,
        "open_risk": round(risk, 2) if opens else None,
        "positions_without_stop": stops_missing,
        "unpriced": unpriced,
        "positions": rows,
        "freshness": fresh,
    }


def _options(tenant, book_ids, now) -> dict:
    from apps.trading.models import OptionsPosition

    opens = list(OptionsPosition.objects.filter(tenant=tenant, portfolio_id__in=book_ids,
                                                status__in=OPEN_OPTIONS)
                 .select_related("strategy_run").prefetch_related("legs"))
    rows, newest = [], None
    unreal = long_premium = short_premium = net_liq = 0.0
    for p in opens:
        legs = [leg for leg in p.legs.all() if leg.closed_at is None]
        premium = sum(float(leg.open_price) * leg.qty for leg in legs if leg.leg_role != "HEDGE_FUT")
        for leg in legs:
            if newest is None or leg.updated_at > newest:
                newest = leg.updated_at
            px = float(leg.current_price or 0)
            if leg.leg_role in LONG_LEGS:
                long_premium += float(leg.open_price) * leg.qty
                net_liq += px * leg.qty
            elif leg.leg_role in SHORT_LEGS:
                short_premium += float(leg.open_price) * leg.qty
                net_liq -= px * leg.qty
        unreal += float(p.current_pnl_inr or 0)
        rows.append({
            "id": str(p.id), "type": p.position_type, "underlying": p.underlying,
            "expiry": p.expiry.isoformat(), "lots": p.lots, "status": p.status,
            "open_pnl": _f(p.current_pnl_inr), "realized_pnl": _f(p.realized_pnl),
            "legs": len(legs), "premium": round(premium, 2), "strategy": _strategy_label(p.strategy_run, "") if p.strategy_run_id
            else p.get_position_type_display(),
        })
    if not opens:
        fresh = {"as_of": None, "age_seconds": None, "stale": False, "source": "no open option positions"}
    else:
        max_age = LIVE_MARK_MAX_AGE if market_open(now) else timedelta(hours=18)
        fresh = freshness(newest, max_age=max_age, now=now, source="option leg LTP (OptionsLeg.updated_at)")
    return {
        "status": "ok" if opens else "empty",
        "label": f"{len(opens)} open option position(s)" if opens else "no open option positions",
        "open_count": len(opens),
        "unrealized_pnl": round(unreal, 2) if opens else None,
        "long_premium_paid": round(long_premium, 2) if opens else None,
        "short_premium_sold": round(short_premium, 2) if opens else None,
        "net_liquidation": round(net_liq, 2) if opens else None,
        # Short options have no stop distance in the model — their risk is open-ended.
        "positions_without_stop": sum(1 for p in opens if any(
            leg.leg_role in SHORT_LEGS for leg in p.legs.all() if leg.closed_at is None)),
        "positions": rows,
        "freshness": fresh,
    }


def _mutual_funds(tenant, now) -> dict:
    from apps.mutual_funds.api.views import build_holdings_payload, catalog_nav_date
    from apps.mutual_funds.models import MFHolding, MFNav

    any_holding = MFHolding.objects.filter(tenant=tenant).exists()
    max_nav = MFNav.objects.aggregate(m=Max("nav_date"))["m"]
    nav_date = catalog_nav_date(max_nav)
    nav_fresh = freshness(datetime.combine(nav_date, time(23, 0), tzinfo=IST) if nav_date else None,
                          max_age=timedelta(days=4), now=now, source="AMFI NAV catalog")
    if not any_holding:
        return {"status": "not_connected", "label": "no CAS imported yet",
                "reason": "No mutual-fund holdings on file — import a CAS statement to connect them.",
                "holding_count": 0, "current_value": None, "invested": None, "gain": None,
                "catalog_nav_date": _iso(nav_date), "nav_is_stale": None, "stale_holdings": 0,
                "allocation": [], "freshness": nav_fresh}
    payload = build_holdings_payload(tenant)
    s = payload["summary"]
    rows = payload["results"]
    stale = sum(1 for r in rows if r["nav_is_stale"])
    oldest = min((r["nav_date"] for r in rows if r["nav_date"]), default=None)
    fresh = freshness(datetime.combine(date.fromisoformat(oldest), time(23, 0), tzinfo=IST) if oldest else None,
                      max_age=timedelta(days=4), now=now, source="latest NAV of each held scheme")
    fresh["stale"] = bool(stale) or fresh["stale"]
    if not rows:
        return {"status": "empty", "label": "all folios closed (0 units)", "holding_count": 0,
                "current_value": None, "invested": None, "gain": None, "catalog_nav_date": _iso(nav_date),
                "nav_is_stale": None, "stale_holdings": 0, "allocation": [], "freshness": nav_fresh}
    valued = s["holding_count"] - s["unvalued_count"]
    return {
        "status": "ok",
        "label": f"{s['holding_count']} folio(s)",
        "holding_count": s["holding_count"],
        "current_value": s["current_value"] if valued else None,
        "invested": s["invested"] if s["missing_cost_count"] < s["holding_count"] else None,
        "gain": s["gain"] if s["gain_basis_invested"] else None,
        "gain_pct": s["gain_pct"],
        "unvalued_count": s["unvalued_count"],
        "missing_cost_count": s["missing_cost_count"],
        "catalog_nav_date": _iso(nav_date),
        "oldest_held_nav_date": oldest,
        "nav_is_stale": bool(stale),
        "stale_holdings": stale,
        "allocation": s["allocation"],
        "freshness": fresh,
    }


def _broker(tenant, now) -> dict:
    """Reconciliation panel: what the broker account holds, beside the book."""
    from apps.market_data.models import BrokerLink, BrokerPositionSnapshot

    links = list(BrokerLink.objects.filter(tenant=tenant).exclude(broker_name="paper").order_by("broker_name"))
    out = []
    margin_used = margin_avail = None
    margin_as_of = None
    for link in links:
        snap = (BrokerPositionSnapshot.objects.filter(tenant=tenant, link=link)
                .order_by("-fetched_at").first())
        row = {"broker": link.broker_name, "link_status": link.status, "snapshot": None}
        if snap is not None:
            m = snap.margin or {}
            row["snapshot"] = {
                "fetched_at": snap.fetched_at.isoformat(), "ok": snap.ok, "error": snap.error,
                "positions": len(snap.positions or []), "holdings": len(snap.holdings or []),
                "margin_used": m.get("used"), "margin_available": m.get("available_cash"),
                "freshness": freshness(snap.fetched_at, max_age=BROKER_MAX_AGE, now=now,
                                       source=f"{link.broker_name} refresh"),
            }
            if link.status == BrokerLink.Status.ACTIVE and snap.ok and m.get("used") is not None:
                margin_used = (margin_used or 0.0) + float(m.get("used") or 0)
                margin_avail = (margin_avail or 0.0) + float(m.get("available_cash") or 0)
                margin_as_of = snap.fetched_at if margin_as_of is None else min(margin_as_of, snap.fetched_at)
        out.append(row)
    active = [r for r in out if r["link_status"] == "active"]
    if not links:
        status, label = "not_connected", "no broker linked"
    elif not active:
        status, label = "not_connected", "no active broker link (" + ", ".join(
            f"{r['broker']} {r['link_status']}" for r in out) + ")"
    else:
        status, label = "ok", ", ".join(r["broker"] for r in active) + " active"
    return {"status": status, "label": label, "links": out,
            "positions": sum((r["snapshot"] or {}).get("positions", 0) for r in active),
            "holdings": sum((r["snapshot"] or {}).get("holdings", 0) for r in active),
            "margin_used": None if margin_used is None else round(margin_used, 2),
            "margin_available": None if margin_avail is None else round(margin_avail, 2),
            "freshness": freshness(margin_as_of, max_age=BROKER_MAX_AGE, now=now, source="broker snapshot")
            if active else {"as_of": None, "age_seconds": None, "stale": True, "source": "no active link"}}


def _realized(tenant, book_ids, now) -> dict:
    """Realized P&L per period from closed equity trades + options positions."""
    from apps.trading.models import OptionsPosition, Trade

    trades = list(Trade.objects.filter(tenant=tenant, portfolio_id__in=book_ids, status="CLOSED",
                                       realized_pnl__isnull=False)
                  .values("realized_pnl", "closed_at", "trade_date"))
    opts = list(OptionsPosition.objects.filter(tenant=tenant, portfolio_id__in=book_ids)
                .exclude(realized_pnl=0).values("realized_pnl", "closed_at", "trade_date", "status"))
    events = []
    for r in trades + opts:
        when = r["closed_at"] or datetime.combine(r["trade_date"], time(15, 30), tzinfo=IST)
        events.append((when, float(r["realized_pnl"])))
    starts = period_starts(now)
    out = {k: round(sum(v for w, v in events if s is None or w >= s), 2) for k, s in starts.items()}
    return {"by_period": out, "count": len(events), "total": out["since_start"],
            "first_at": _iso(min((w for w, _ in events), default=None))}


def _history(tenant, book, now) -> dict:
    """Equity curve on daily closes from ``PortfolioSnapshot`` (book portfolios only)."""
    from apps.trading.models import PortfolioSnapshot, Trade

    ids = [p.id for p in book]
    qs = PortfolioSnapshot.objects.filter(tenant=tenant, portfolio_id__in=ids)
    last = qs.order_by("-captured_at").values("captured_at").first()
    if last is None:
        return {"status": "not_connected", "label": "no portfolio snapshots yet", "points": [],
                "max_drawdown": None, "snapshot_count": 0,
                "freshness": {"as_of": None, "age_seconds": None, "stale": True, "source": "PortfolioSnapshot"}}

    closes = (qs.annotate(d=TruncDate("captured_at", tzinfo=IST)).values("d", "portfolio_id")
              .annotate(last=Max("captured_at")))
    wanted = {(c["portfolio_id"], c["last"]) for c in closes}
    rows = (qs.filter(captured_at__in=[w[1] for w in wanted])
            .values("portfolio_id", "captured_at", "unrealized_pnl"))
    by_day: dict[date, float] = defaultdict(float)
    for r in rows:
        if (r["portfolio_id"], r["captured_at"]) in wanted:
            by_day[r["captured_at"].astimezone(IST).date()] += float(r["unrealized_pnl"])

    # Realized P&L by close day, compounded into the curve (Portfolio.capital is
    # never updated by fills, so snapshot.equity alone stays flat).
    realized_by_day: dict[date, float] = defaultdict(float)
    for r in Trade.objects.filter(tenant=tenant, portfolio_id__in=ids, status="CLOSED",
                                  realized_pnl__isnull=False).values("realized_pnl", "closed_at", "trade_date"):
        d = r["closed_at"].astimezone(IST).date() if r["closed_at"] else r["trade_date"]
        realized_by_day[d] += float(r["realized_pnl"])
    capital = sum(float(p.capital) for p in book)
    points, cum, peak = [], 0.0, None
    dd_abs, dd_pct, dd_peak_day, dd_trough_day = 0.0, 0.0, None, None
    peak_day = None
    days = sorted(set(by_day) | {d for d in realized_by_day if by_day and d >= min(by_day)})
    pre = sum(v for d, v in realized_by_day.items() if by_day and d < min(by_day))
    cum = pre
    for d in days:
        cum += realized_by_day.get(d, 0.0)
        eq = capital + cum + by_day.get(d, 0.0)
        points.append({"date": d.isoformat(), "equity": round(eq, 2), "realized_cum": round(cum, 2),
                       "unrealized": round(by_day.get(d, 0.0), 2)})
        if peak is None or eq > peak:
            peak, peak_day = eq, d
        if peak and peak - eq > dd_abs:
            dd_abs, dd_pct = peak - eq, (peak - eq) / peak * 100
            dd_peak_day, dd_trough_day = peak_day, d
    return {
        "status": "ok",
        "basis": "daily close per portfolio: capital + cumulative realized P&L + open mark-to-market",
        "points": points,
        "snapshot_count": qs.count(),
        "first_date": points[0]["date"] if points else None,
        "max_drawdown": {"abs": round(dd_abs, 2), "pct": round(dd_pct, 2),
                         "peak_date": _iso(dd_peak_day), "trough_date": _iso(dd_trough_day),
                         "note": "on daily closes"},
        "freshness": freshness(last["captured_at"], max_age=SNAPSHOT_MAX_AGE, now=now,
                               source="PortfolioSnapshot (beat, every 60 s)"),
    }


# ─── overview ────────────────────────────────────────────────────────────

def build_overview(tenant, now: datetime | None = None) -> dict:
    from apps.trading.services.portfolio_roles import book_portfolios, mf_portfolio_ids

    now = now or timezone.now()
    book = book_portfolios(tenant)
    book_ids = [p.id for p in book]
    capital = sum(float(p.capital) for p in book) if book else None
    mode = "live" if any(p.mode == "live" for p in book) else "paper"

    eq = _equity(tenant, book_ids, now)
    op = _options(tenant, book_ids, now)
    mf = _mutual_funds(tenant, now)
    br = _broker(tenant, now)
    realized = _realized(tenant, book_ids, now)
    hist = _history(tenant, book, now) if book else {
        "status": "not_connected", "label": "no trading portfolio", "points": [], "max_drawdown": None,
        "snapshot_count": 0, "freshness": {"as_of": None, "age_seconds": None, "stale": True,
                                           "source": "PortfolioSnapshot"}}

    unreal = (eq["unrealized_pnl"] or 0.0) + (op["unrealized_pnl"] or 0.0)
    has_trading = capital is not None
    deployed = (eq["cost"] or 0.0) + (op["long_premium_paid"] or 0.0)
    trading_value = None if not has_trading else round(capital + realized["total"] + unreal, 2)
    cash = None if not has_trading else round(capital + realized["total"] - deployed, 2)

    included, missing = [], []
    for key, comp in (("equity", eq), ("options", op), ("mutual_funds", mf)):
        (missing if comp["status"] == "not_connected" else included).append(key)
    mf_value = mf["current_value"] if mf["status"] == "ok" else None
    total_value = None
    if trading_value is not None or mf_value is not None:
        total_value = round((trading_value or 0.0) + (mf_value or 0.0), 2)
    invested = round(deployed + (mf["invested"] or 0.0), 2) if (has_trading or mf["invested"] is not None) else None

    # P&L per period: realized in the period + today's open mark (open MTM is
    # not split by period — it is the current mark of what is still open).
    has_trades = eq["has_history"] or op["open_count"] or realized["count"]
    pnl = {}
    for k, v in realized["by_period"].items():
        pnl[k] = {"realized": v if has_trades else None,
                  "unrealized_open": round(unreal, 2) if (eq["open_count"] or op["open_count"]) else None,
                  "total": round(v + unreal, 2) if has_trades else None}
    pnl["status"] = "ok" if has_trades else "empty"
    pnl["label"] = "" if has_trades else "no trades yet — P&L starts with the first fill"
    if mf["status"] == "ok":
        pnl["mutual_funds_gain"] = mf["gain"]

    # Allocation by asset class (market value; cash shown separately).
    by_class = []
    eq_val = eq["market_value"] if eq["status"] == "ok" else None
    op_val = op["net_liquidation"] if op["status"] == "ok" else None
    for key, val, comp in (("equity", eq_val, eq), ("options", op_val, op), ("mutual_funds", mf_value, mf),
                           ("cash", cash, None)):
        by_class.append({"key": key, "label": ASSET_LABELS[key], "value": val,
                         "status": comp["status"] if comp else ("ok" if cash is not None else "not_connected")})
    denom = sum(abs(r["value"]) for r in by_class if r["value"])
    for r in by_class:
        r["pct"] = _pct(abs(r["value"]), denom) if r["value"] is not None else None

    by_strategy: dict[str, float] = defaultdict(float)
    by_sector: dict[str, float] = defaultdict(float)
    for r in eq["positions"]:
        v = r["cost"] if r["mark"] is None else r["mark"] * r["units"]
        by_strategy[r["strategy"]] += v
        by_sector[r["sector"]] += v
    for r in op["positions"]:
        # Options exposure = premium notional of the open legs.
        by_strategy[r["strategy"]] += r["premium"]
        by_sector["Index options" if r["underlying"] in ("NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY")
                  else sector_of(r["underlying"])] += r["premium"]
    if mf_value:
        by_strategy["Mutual funds"] += mf_value

    def _rows(d):
        tot = sum(d.values())
        return [{"key": k, "value": round(v, 2), "pct": _pct(v, tot)} for k, v in
                sorted(d.items(), key=lambda kv: -kv[1])]

    # Risk.
    rconf = getattr(settings, "ALPHADESK", {}) or {}
    loss_pct = float(rconf.get("MAX_DAILY_LOSS_PCT", 3.0))
    loss_limit = round(capital * loss_pct / 100, 2) if capital else None
    today_total = pnl["day"]["total"]
    loss_used = None if (loss_limit is None or today_total is None) else round(max(0.0, -today_total), 2)
    risk = {
        "open_risk": eq["open_risk"],
        "open_risk_pct_of_capital": _pct(eq["open_risk"], capital),
        "positions_without_stop": eq["positions_without_stop"] + op["positions_without_stop"],
        "margin_used_book": round(deployed + (op["short_premium_sold"] or 0.0), 2)
        if (eq["open_count"] or op["open_count"]) else None,
        "margin_used_broker": br["margin_used"],
        "margin_available_broker": br["margin_available"],
        "daily_loss": {"used": loss_used, "limit": loss_limit, "limit_pct_of_capital": loss_pct,
                       "pct_used": _pct(loss_used, loss_limit) if loss_used is not None else None,
                       "basis": "today's realized + open mark-to-market"},
        "note": "" if eq["open_count"] or op["open_count"] else "nothing open — no risk on the book",
    }

    freshness_all = {
        "equity": eq["freshness"], "options": op["freshness"], "mutual_funds": mf["freshness"],
        "snapshots": hist["freshness"], "broker": br["freshness"],
    }
    mf_ids = mf_portfolio_ids(tenant)
    return {
        "as_of": now.isoformat(),
        "mode": mode,
        "portfolios": [{"id": str(p.id), "name": p.name, "mode": p.mode, "capital": _f(p.capital),
                        "role": "trading"} for p in book]
        + [{"id": str(pid), "name": n, "mode": m, "capital": _f(c), "role": "mutual_funds"}
           for pid, n, m, c in _portfolio_rows(tenant, mf_ids)],
        "total": {
            "value": total_value, "invested": invested, "cash": cash, "capital": _f(capital),
            "trading_value": trading_value, "mutual_funds_value": mf_value,
            "components_included": included, "components_missing": missing,
        },
        "pnl": pnl,
        "components": {"equity": eq, "options": op, "mutual_funds": mf},
        "allocation": {"by_asset_class": by_class, "by_strategy": _rows(by_strategy),
                       "by_sector": _rows(by_sector)},
        "risk": risk,
        "history": hist,
        "broker_reconciliation": _reconcile(eq, br),
        "freshness": freshness_all,
    }


def _portfolio_rows(tenant, ids):
    from apps.trading.models import Portfolio

    return Portfolio.objects.filter(tenant=tenant, id__in=ids).values_list("id", "name", "mode", "capital")


def _reconcile(eq: dict, br: dict) -> dict:
    out = dict(br)
    if br["status"] != "ok":
        out["discrepancy"] = None
        return out
    book_n = eq["open_count"]
    broker_n = br["positions"]
    out["book_positions"] = book_n
    out["discrepancy"] = None if book_n == broker_n else (
        f"book has {book_n} open equity position(s), broker reports {broker_n} "
        "(expected in paper mode: paper fills never reach the broker)")
    return out


# ─── attribution ─────────────────────────────────────────────────────────

def build_attribution(tenant, *, months: int = 12, now: datetime | None = None) -> dict:
    """Realized P&L by strategy / symbol / month from closed trades + options."""
    from apps.trading.models import OptionsPosition, Trade
    from apps.trading.services.portfolio_roles import book_portfolios

    now = now or timezone.now()
    ids = [p.id for p in book_portfolios(tenant)]
    since = (now.astimezone(IST).date().replace(day=1) - timedelta(days=31 * (months - 1))).replace(day=1)
    rows = []
    for t in (Trade.objects.filter(tenant=tenant, portfolio_id__in=ids, status="CLOSED",
                                   realized_pnl__isnull=False)
              .select_related("strategy_run")):
        when = _closed_at(t)
        if when.astimezone(IST).date() < since:
            continue
        rows.append({"strategy": _strategy_label(t.strategy_run, t.origin), "symbol": t.symbol,
                     "month": when.astimezone(IST).strftime("%Y-%m"), "pnl": float(t.realized_pnl),
                     "asset": "options" if t.exchange in ("NFO", "BFO") else "equity"})
    for p in (OptionsPosition.objects.filter(tenant=tenant, portfolio_id__in=ids, status="CLOSED")
              .select_related("strategy_run")):
        when = p.closed_at or datetime.combine(p.trade_date, time(15, 30), tzinfo=IST)
        if when.astimezone(IST).date() < since:
            continue
        rows.append({"strategy": _strategy_label(p.strategy_run, "") if p.strategy_run_id
                     else p.get_position_type_display(), "symbol": p.underlying,
                     "month": when.astimezone(IST).strftime("%Y-%m"), "pnl": float(p.realized_pnl),
                     "asset": "options"})

    def group(key):
        acc: dict[str, dict] = {}
        for r in rows:
            g = acc.setdefault(r[key], {"key": r[key], "pnl": 0.0, "trades": 0, "wins": 0, "losses": 0})
            g["pnl"] += r["pnl"]
            g["trades"] += 1
            g["wins"] += r["pnl"] > 0
            g["losses"] += r["pnl"] < 0
        out = []
        for g in acc.values():
            g["pnl"] = round(g["pnl"], 2)
            g["win_rate"] = _pct(g["wins"], g["trades"])
            out.append(g)
        return out

    by_month = sorted(group("month"), key=lambda g: g["key"])
    return {
        "as_of": now.isoformat(),
        "since": since.isoformat(),
        "status": "ok" if rows else "empty",
        "label": "" if rows else "no closed trades in this window yet",
        "trade_count": len(rows),
        "total_pnl": round(sum(r["pnl"] for r in rows), 2) if rows else None,
        "by_strategy": sorted(group("strategy"), key=lambda g: -g["pnl"]),
        "by_symbol": sorted(group("symbol"), key=lambda g: -g["pnl"]),
        "by_month": by_month,
    }

