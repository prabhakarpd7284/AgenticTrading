"""Action Board builder (docs/ALPHADESK_UX_REDESIGN_PLAN.md §4, UX redesign R3).

Owner decision 2026-10-22: the only idea source is the existing
**directional** strategy (Claude planner, ``trading.agents.planner``), run in
PAPER mode only and labelled "unproven — forward paper test"
(docs/strategies/STRATEGY_SEARCH_RESULTS.md: nothing has an edge yet).

Pipeline for one trading day (``build_action_board``):

  1. Nightly scan for the day (``NightlyScan.for_date``). No usable scan →
     the run fails its ``scan_present`` check and builds nothing.
  2. Universe: NIFTY 50 rows the scan allows in at least one direction with
     no corporate event (``allowed != "none"`` and not ``event_tomorrow``).
  3. 1m bars from 09:15 up to the as-of time (local ``Candle`` rows when they
     cover the window, else Angel SmartAPI — the directional strategy's own
     data path). If fewer than ``MIN_FRESH_FRACTION`` of candidates have
     fresh bars, the ``data_fresh`` check FAILS and no planner is called.
  4. **Ranking rule (deterministic):** ``|early move %|`` = |last 1m close at
     the as-of time − previous session close (scan ``prev_hlc.c``)| / prev
     close × 100, descending; ties by symbol. The biggest gap-plus-first-
     minutes movers are planned first.
  5. Claude planner on the top ``MAX_PLANNER_CALLS`` (default 10). A failure
     or timeout is recorded for that symbol and never stops the run.
  6. Plans whose side the scan does not allow are dropped (and journaled).
  7. Per tenant (its oldest capital>0 portfolio, PAPER only): quantity from
     the existing sizer (``calc_position_size``; the LLM's quantity is
     ignored) × the cooldown risk multiplier (0.5 after a cooldown); then a
     **dry run** of the order gates (kill switch, AI pause, cooldown,
     sentiment, notional backstop, 10-criterion RiskEngine). Nothing is
     placed. During a cooldown the card is saved as a *shadow* (blocked).
  8. One card per (tenant, date, strategy, symbol, side); a card the owner
     already decided is never overwritten. Every card emits an event.
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from django.conf import settings
from django.utils import timezone

log = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
EARLIEST_LIVE = time(9, 20)     # need a few 1m bars before planning anything

STRATEGY = "directional"
STAGE_LABEL = "unproven — forward paper test"


def strategy_version() -> str:
    try:
        from plugins.strategy_directional.strategy import DirectionalStrategy

        return str(DirectionalStrategy.version)
    except Exception:  # noqa: BLE001
        return "1.0.0"


DEFAULTS: dict[str, Any] = {
    "TIME": "09:35",
    "MAX_PLANNER_CALLS": 10,
    "EXPIRES_AT": "15:00",
    "EOD_EXIT": "15:14",
    "MIN_BARS": 10,
    "MAX_BAR_LAG_MIN": 5,
    "MIN_FRESH_FRACTION": 0.8,
    "TELEGRAM": True,
    "APP_URL": "http://localhost:5173",
}


def board_setting(key: str):
    return (getattr(settings, "ACTION_BOARD", {}) or {}).get(key, DEFAULTS.get(key))


def hhmm(value: str) -> time:
    hh, mm = str(value).split(":")
    return time(int(hh), int(mm))


def at(day: date, t: time) -> datetime:
    return datetime.combine(day, t, tzinfo=IST)


def expires_at_for(day: date) -> datetime:
    return at(day, hhmm(board_setting("EXPIRES_AT")))


# ─── Market data ────────────────────────────────────────────────────────

def db_bars(symbol: str, lo: datetime, hi: datetime) -> list[list]:
    """Stored 1m candles ``lo <= t < hi`` as ``[ts, o, h, l, c, v]`` rows."""
    from apps.market_data.models import Candle

    rows = (Candle.objects.filter(interval="1m", symbol__tradingsymbol=symbol,
                                  symbol__exchange="NSE", t__gte=lo, t__lt=hi)
            .order_by("t").values_list("t", "o", "h", "l", "c", "v"))
    return [[t, float(o), float(h), float(l_), float(c), int(v or 0)] for t, o, h, l_, c, v in rows]


def _angel_bars(symbol: str, day: date, until: datetime) -> list[list]:
    """Angel SmartAPI 1m candles — the legacy BrokerClient the directional
    workflow's ``DataService`` uses (real broker data; never synthesised)."""
    from trading.services.data_service import BrokerClient
    from trading.services.ticker_service import ticker_service

    token = ticker_service.get_token(symbol)
    if not token:
        raise LookupError(f"no Angel token for {symbol}")
    raw = BrokerClient.get_instance().fetch_candles(
        token, f"{day.isoformat()} 09:15", f"{day.isoformat()} {until:%H:%M}", "ONE_MINUTE")
    out = []
    for r in raw or []:
        ts = r[0]
        t = datetime.fromisoformat(str(ts)) if not isinstance(ts, datetime) else ts
        if t.tzinfo is None:
            t = t.replace(tzinfo=IST)
        if t < until:
            out.append([t, float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[5] or 0)])
    return out


def load_bars(symbol: str, day: date, until: datetime) -> tuple[list[list], str]:
    """1m bars for ``day`` from 09:15 up to (not incl.) ``until``.

    Local candles win when they cover the window (no broker calls); otherwise
    Angel. Returns ``(bars, source)``.
    """
    lo = at(day, SESSION_OPEN)
    expected = max(1, int((until - lo).total_seconds() // 60))
    local = db_bars(symbol, lo, until)
    if len(local) >= int(expected * 0.9):
        return local, "db_1m"
    return _angel_bars(symbol, day, until), "angel_1m"


def bars_fresh(bars: list[list], until: datetime) -> tuple[bool, str]:
    if len(bars) < int(board_setting("MIN_BARS")):
        return False, f"only {len(bars)} 1m bars"
    last = bars[-1][0]
    lag = (until - last).total_seconds() / 60
    if lag > float(board_setting("MAX_BAR_LAG_MIN")) + 1:
        return False, f"last bar {last.astimezone(IST):%H:%M}, {lag:.0f} min before {until:%H:%M}"
    return True, ""


def market_summary(symbol: str, day: date, bars: list[list], as_of: datetime) -> str:
    """The directional workflow's own LLM data summary over these bars, with
    a header that says exactly what the bars are: the session *so far*. (Its
    stock "last completed session" header made the planner treat a 09:35
    replay as a finished day.)"""
    from trading.services.data_service import DataService, enrich_ohlcv

    df = enrich_ohlcv([[b[0].isoformat(), b[1], b[2], b[3], b[4], b[5]] for b in bars])
    text = DataService()._build_text_summary(df, symbol, day.isoformat(), is_live=True,
                                             session_phase="OPENING", days_old=0)
    header = (f"Market Data for {symbol} on {day.isoformat()} — session in progress: the first "
              f"{len(bars)} one-minute bars, 09:15 to {as_of.astimezone(IST):%H:%M} IST. Nothing after "
              f"{as_of.astimezone(IST):%H:%M} is known yet:")
    return header + "\n" + text.split("\n", 1)[1] if "\n" in text else text


def levels_stale(row, scan) -> bool:
    """Scan levels not built from the session right before the trading day."""
    lv = row.levels or {}
    return bool(lv.get("stale")) or row.data_date is None or row.data_date < scan.session_date


def scan_context(row) -> str:
    lv = row.levels or {}
    prev = lv.get("prev_hlc") or {}
    floor = lv.get("floor") or {}
    cpr = lv.get("cpr") or {}
    lines = [
        f"NIGHTLY SCAN for {row.symbol} (deterministic levels + news filter; levels from the "
        f"{row.data_date} session):",
        f"  Allowed directions today: {row.allowed} (BUY allowed={row.allow_long}, SELL allowed={row.allow_short})",
        f"  Previous session H/L/C: {prev.get('h')} / {prev.get('l')} / {prev.get('c')}",
        f"  Floor pivots: PP {floor.get('pp')} | R1 {floor.get('r1')} R2 {floor.get('r2')} | "
        f"S1 {floor.get('s1')} S2 {floor.get('s2')}",
        f"  CPR: BC {cpr.get('bc')} TC {cpr.get('tc')} ({row.cpr_class or cpr.get('class') or '?'}); "
        f"two-day relationship: {row.two_day_relationship or '?'}; bias: {row.bias or '?'}",
        f"  News sentiment: {row.sentiment_label}"
        + (f" ({float(row.sentiment_score):+.2f})" if row.sentiment_score is not None else ""),
    ]
    if row.reasons:
        lines.append("  Filter notes: " + "; ".join(str(r) for r in row.reasons))
    return "\n".join(lines)


def prev_close(row) -> float | None:
    try:
        c = ((row.levels or {}).get("prev_hlc") or {}).get("c")
        return float(c) if c else None
    except (TypeError, ValueError):
        return None


def rank_candidates(items: list[dict]) -> list[dict]:
    """Deterministic: |early move %| desc, then symbol. Items without a
    previous close rank last (move unknown)."""
    def key(it):
        m = it.get("move_pct")
        return (0 if m is not None else 1, -abs(m) if m is not None else 0.0, it["symbol"])
    ranked = sorted(items, key=key)
    for i, it in enumerate(ranked, 1):
        it["rank"] = i
    return ranked


# ─── Planner ────────────────────────────────────────────────────────────

def call_planner(user_intent: str, market_data_summary: str, rag_context: str) -> dict:
    """The directional planner (Claude CLI on the Max plan when
    ``PLANNER_MODE=cli``). Returns the plan dict or ``{"error": ...}``.
    Tests patch this — never call the real CLI from tests."""
    from trading.agents.planner import run_planner

    return run_planner(user_intent, market_data_summary, rag_context)


def rag_context(symbol: str) -> str:
    try:
        from trading.rag.retriever import retrieve_context, retrieve_portfolio_context

        return f"{retrieve_context(symbol)}\n\n---\n\n{retrieve_portfolio_context()}"
    except Exception as e:  # noqa: BLE001
        return f"RAG context unavailable: {e}"


def planner_intent(symbol: str, allowed: str, as_of: datetime) -> str:
    return (f"Plan an INTRADAY equity trade for {symbol} on NSE, entered after "
            f"{as_of.astimezone(IST):%H:%M} IST and flat by 15:14 IST. The nightly scan allows: "
            f"{allowed} (long = BUY, short = SELL). Only propose an allowed side. Use the first "
            f"minutes' structure and the scan levels. Quantity is ignored — sizing is computed "
            f"deterministically. If there is no clean setup, set confidence to 0.0.")


def validate_plan(plan: dict, symbol: str) -> tuple[dict | None, str]:
    """Normalise a planner dict. Returns (plan, "") or (None, why)."""
    if not isinstance(plan, dict):
        return None, "planner returned no plan"
    if plan.get("error"):
        return None, str(plan["error"])[:300]
    if str(plan.get("symbol", "")).upper().replace("-EQ", "") != symbol:
        return None, f"planner returned a different symbol ({plan.get('symbol')})"
    try:
        side = str(plan["side"]).upper()
        entry, stop, target = float(plan["entry_price"]), float(plan["stop_loss"]), float(plan["target"])
        conf = float(plan.get("confidence") or 0)
    except (KeyError, TypeError, ValueError) as e:
        return None, f"malformed plan: {e}"
    if side not in ("BUY", "SELL"):
        return None, f"invalid side {side!r}"
    if conf < 0.01:
        return None, "declined: " + str(plan.get("reasoning") or "no setup")[:240]
    if min(entry, stop, target) <= 0:
        return None, "non-positive prices"
    return {"side": side, "entry": entry, "stop": stop, "target": target, "confidence": conf,
            "reasoning": str(plan.get("reasoning") or "").strip(), "llm_qty": plan.get("quantity")}, ""


# ─── Sizing + dry-run gates ─────────────────────────────────────────────

def size_qty(capital: float, entry: float, stop: float, risk_multiplier: float) -> int:
    """The existing deterministic sizer — never the LLM's quantity."""
    from trading.utils.pnl_utils import calc_position_size

    lim = settings.ALPHADESK
    return int(calc_position_size(
        capital, entry, stop,
        max_risk_pct=float(lim["MAX_RISK_PER_TRADE_PCT"]) * float(risk_multiplier),
        max_position_pct=float(lim["MAX_POSITION_SIZE_PCT"]),
    ))


def dry_run_gates(portfolio, *, symbol: str, side: str, entry: float, stop: float, target: float,
                  qty: int, confidence: float, risk_multiplier: float, day: date,
                  cooldown=None) -> dict:
    """Every gate ``PlaceOrder`` applies to an automated order, evaluated
    without placing anything. ``approved`` is True only if all would pass;
    ``failed_criterion`` names the first that would not."""
    from apps.system.services.flags import is_ai_paused, is_kill_switch_on
    from apps.trading.services.place_order import reference_price
    from apps.trading.services.risk_engine import RiskEngine, TradeDraft
    from apps.trading.services.trading_rules import check_sentiment

    gates: dict[str, Any] = {}
    failed, reason = "", ""

    def fail(name: str, why: str) -> None:
        nonlocal failed, reason
        if not failed:
            failed, reason = name, why

    gates["kill_switch"] = bool(is_kill_switch_on(portfolio.tenant_id))
    if gates["kill_switch"]:
        fail("kill_switch", "Trading halted — operator kill switch is ON")
    gates["ai_paused"] = bool(is_ai_paused(portfolio.tenant_id))
    if gates["ai_paused"]:
        fail("ai_paused", "AI trading is paused by the operator")
    if cooldown is not None:
        gates["cooldown"] = {"status": cooldown.status, "risk_multiplier": cooldown.risk_multiplier,
                             "reason": cooldown.reason}
        if cooldown.blocks_entries:
            fail("cooldown", f"Losing-streak cooldown: {cooldown.reason}")
    sent = check_sentiment(symbol, side, day)
    gates["sentiment"] = sent
    if not sent.get("allowed", True):
        fail("sentiment", f"Sentiment filter: {sent.get('reason', '')}")
    if qty <= 0:
        fail("size", "Deterministic size is 0 shares at this stop distance")

    lim = settings.ALPHADESK
    ref = reference_price(symbol, portfolio.tenant_id)
    px = ref if ref > 0 else entry
    gates["notional"] = {"qty": qty, "ref_price": round(ref, 2) if ref else None,
                         "notional": round(px * qty, 2),
                         "max_qty": int(lim["MAX_ORDER_QTY"]), "max_notional": float(lim["MAX_ORDER_NOTIONAL"])}
    if qty > int(lim["MAX_ORDER_QTY"]) or px * qty > float(lim["MAX_ORDER_NOTIONAL"]):
        fail("notional", "Order exceeds the absolute qty/notional backstop")

    criteria: dict = {}
    risk_reason = ""
    if qty > 0:
        decision = RiskEngine().validate(
            TradeDraft(symbol=symbol, side=side, entry_price=entry, stop_loss=stop, target=target,
                       quantity=qty, confidence=confidence),
            portfolio_id=portfolio.id, risk_multiplier=risk_multiplier)
        criteria = decision.criteria
        risk_reason = decision.reason
        if not decision.approved:
            fail(decision.failed_criterion or "risk_engine", decision.reason)
    return {
        "mode": "dry_run",
        "approved": not failed,
        "reason": reason or risk_reason or "Approved",
        "failed_criterion": failed,
        "criteria": criteria,
        "gates": gates,
        "risk_multiplier": risk_multiplier,
        "confidence_used": confidence,
        "evaluated_at": timezone.now().isoformat(),
    }


# ─── Board ──────────────────────────────────────────────────────────────

def board_portfolios() -> list:
    """One trading portfolio per tenant: the oldest with capital > 0."""
    from apps.trading.services.cooldown import trading_portfolios

    seen, out = set(), []
    for p in trading_portfolios().select_related("tenant").order_by("tenant_id", "created_at"):
        if p.tenant_id in seen:
            continue
        seen.add(p.tenant_id)
        out.append(p)
    return out


def resolve_as_of(day: date, now: datetime) -> tuple[datetime | None, bool, str]:
    """(as_of, replay, skip_reason). Past days and after-expiry runs replay
    the configured board time; a live run plans from the current minute."""
    now_ist = now.astimezone(IST)
    board_t = hhmm(board_setting("TIME"))
    if day > now_ist.date():
        return None, False, f"{day} is in the future"
    if day < now_ist.date():
        return at(day, board_t), True, ""
    if now_ist.time() < EARLIEST_LIVE:
        return None, False, f"too early — the board needs the first minutes of 1m data (runs {board_t:%H:%M})"
    if now_ist >= expires_at_for(day):
        return at(day, board_t), True, ""
    return now_ist.replace(second=0, microsecond=0), False, ""


def _reasons(plan: dict, item: dict, row) -> list[str]:
    out = []
    text = plan.get("reasoning") or ""
    if text:
        out.append(text[:280])
    if item.get("move_pct") is not None:
        out.append(f"Early move {item['move_pct']:+.2f}% vs previous close {item['prev_close']:.2f} "
                   f"(rank {item['rank']} of {item['ranked_of']} by |move|)")
    sent = row.sentiment_label.replace("_", " ")
    out.append(f"Nightly scan: {row.allowed} allowed · sentiment {sent} · CPR {row.cpr_class or '?'}"
               f" · {row.two_day_relationship or 'no two-day pattern'}")
    return out[:3]


def build_action_board(day: date | None = None, *, max_planner_calls: int | None = None,
                       telegram: bool | None = None, run=None, now: datetime | None = None) -> dict:
    from apps.events.models import Event
    from apps.events.services.event_writer import emit
    from apps.market_data.constants import NIFTY_50_SYMBOLS
    from apps.market_data.services.market_calendar import is_trading_day
    from apps.news.models import NightlyScan
    from apps.news.services.sentiment_filter import USABLE_STATUSES
    from apps.system.services import run_checks as rc
    from apps.system.services.run_receipts import current_run
    from apps.trading.services.cooldown import evaluate_for_portfolio

    run = run or current_run()
    now = now or timezone.now()
    day = day or now.astimezone(IST).date()
    cap = int(max_planner_calls if max_planner_calls is not None else board_setting("MAX_PLANNER_CALLS"))
    run.output(date=day.isoformat(), planner_cap=cap)

    if not is_trading_day(day):
        run.skip(f"{day} is not an NSE trading day")
        return {"skipped": f"{day} is not an NSE trading day"}
    as_of, replay, why = resolve_as_of(day, now)
    if as_of is None:
        run.skip(why)
        return {"skipped": why}
    expires = expires_at_for(day)
    telegram = (not replay and bool(board_setting("TELEGRAM"))) if telegram is None else telegram
    run.output(as_of=as_of.isoformat(), replay=replay)
    run.link("Today — Action Board", "/")

    portfolios = board_portfolios()
    paper = [p for p in portfolios if p.mode == "paper"]
    for p in portfolios:
        if p.mode != "paper":
            run.warn(f"tenant {p.tenant_id}: portfolio {p.name} is {p.mode} — the board is PAPER-only in R3")
    if not paper:
        run.skip("no paper trading portfolio")
        return {"skipped": "no paper trading portfolio"}

    # 1. Nightly scan ---------------------------------------------------
    scan = NightlyScan.objects.filter(for_date=day, status__in=USABLE_STATUSES).first()
    if scan is None:
        run.check(rc.make("scan_present", rc.FAIL,
                          f"no usable nightly scan for {day} — the board needs the scan's allowed directions"))
        return {"date": day.isoformat(), "error": "no nightly scan", "cards": 0}
    run.check(rc.make("scan_present", rc.PASS, f"nightly scan #{scan.id} ({scan.status}) for {day}",
                      scan_id=scan.id))
    run.set_provenance(nightly_scan_id=scan.id, scan_session_date=str(scan.session_date))

    rows = {r.symbol: r for r in scan.rows.filter(symbol__in=list(NIFTY_50_SYMBOLS))}
    excluded = {"no_direction": 0, "event": 0, "stale_levels": 0}
    candidates = []
    for sym, r in sorted(rows.items()):
        if r.event_tomorrow:
            excluded["event"] += 1
        elif r.allowed == "none" or not (r.allow_long or r.allow_short):
            excluded["no_direction"] += 1
        elif levels_stale(r, scan):
            excluded["stale_levels"] += 1
        else:
            candidates.append(sym)
    run.output(scan_id=scan.id, universe=len(rows), candidates=len(candidates), excluded=excluded)
    stale_lv = excluded["stale_levels"]
    eligible = stale_lv + len(candidates)
    if eligible and stale_lv > (1 - float(board_setting("MIN_FRESH_FRACTION"))) * eligible:
        dates = sorted({str(r.data_date) for r in rows.values() if levels_stale(r, scan)})
        run.check(rc.make("levels_fresh", rc.FAIL,
                          f"{stale_lv} of {eligible} scan rows have levels from {', '.join(dates)}, not the "
                          f"{scan.session_date} session — not planning on stale levels",
                          stale=stale_lv, expected_session=str(scan.session_date)))
        return {"date": day.isoformat(), "error": "stale scan levels", "cards": 0}
    run.check(rc.make("levels_fresh", rc.WARN if stale_lv else rc.PASS,
                      f"scan levels from the {scan.session_date} session"
                      + (f" ({stale_lv} stale row(s) excluded)" if stale_lv else "")))

    # 2. Data -------------------------------------------------------------
    items, stale, sources = [], [], {}
    for sym in candidates:
        try:
            bars, src = load_bars(sym, day, as_of)
        except Exception as e:  # noqa: BLE001 — one symbol's data never stops the board
            stale.append({"symbol": sym, "why": f"{type(e).__name__}: {e}"[:200]})
            continue
        ok, why_stale = bars_fresh(bars, as_of)
        if not ok:
            stale.append({"symbol": sym, "why": why_stale, "source": src})
            continue
        sources[src] = sources.get(src, 0) + 1
        pc = prev_close(rows[sym])
        last = bars[-1][4]
        items.append({"symbol": sym, "bars": bars, "source": src, "last": last, "prev_close": pc,
                      "move_pct": round((last - pc) / pc * 100, 3) if pc else None,
                      "last_bar": bars[-1][0].astimezone(IST).strftime("%H:%M")})
    fresh_n = len(items)
    run.set_provenance(bars_interval="1m", bar_sources=sources, data_source="Angel SmartAPI (1m)")
    run.output(fresh=fresh_n, stale=len(stale), stale_examples=stale[:8])
    min_frac = float(board_setting("MIN_FRESH_FRACTION"))
    if candidates and fresh_n < min_frac * len(candidates):
        run.check(rc.make("data_fresh", rc.FAIL,
                          f"only {fresh_n} of {len(candidates)} candidates have fresh 1m bars up to "
                          f"{as_of.astimezone(IST):%H:%M} — not planning on stale data",
                          fresh=fresh_n, candidates=len(candidates), examples=stale[:5]))
        return {"date": day.isoformat(), "error": "stale data", "cards": 0}
    run.check(rc.make("data_fresh", rc.PASS if not stale else rc.WARN,
                      f"{fresh_n} of {len(candidates)} candidates have 1m bars up to "
                      f"{as_of.astimezone(IST):%H:%M} ({', '.join(f'{k}: {v}' for k, v in sources.items()) or 'none'})"))

    # 3. Rank + plan ------------------------------------------------------
    ranked = rank_candidates(items)
    for it in ranked:
        it["ranked_of"] = len(ranked)
    run.output(ranking_rule="|early move %| vs previous close, desc; ties by symbol",
               ranked=[{"symbol": it["symbol"], "move_pct": it["move_pct"]} for it in ranked[:max(cap, 10)]])
    per_symbol: dict[str, dict] = {}
    plans: list[tuple[dict, dict]] = []
    for it in ranked[:cap]:
        sym, row = it["symbol"], rows[it["symbol"]]
        entry = {"rank": it["rank"], "move_pct": it["move_pct"], "source": it["source"]}
        t0 = _time.monotonic()
        try:
            summary = (market_summary(sym, day, it["bars"], as_of)
                       + "\n\n" + scan_context(row))
            raw = call_planner(planner_intent(sym, row.allowed, as_of), summary, rag_context(sym))
        except Exception as e:  # noqa: BLE001 — planner failures never crash the board
            raw = {"error": f"{type(e).__name__}: {e}"}
        entry["latency_ms"] = int((_time.monotonic() - t0) * 1000)
        plan, why = validate_plan(raw, sym)
        if plan is None:
            failed_call = isinstance(raw, dict) and bool(raw.get("error"))
            entry["result"] = "failed" if failed_call else "no_plan"
            entry["detail"] = why
            if failed_call:
                run.unit_failed(error=f"{sym}: {why}")
            else:
                run.unit_ok()
            per_symbol[sym] = entry
            continue
        run.unit_ok()
        allowed = row.allow_long if plan["side"] == "BUY" else row.allow_short
        if not allowed:
            entry["result"] = "direction_blocked"
            entry["detail"] = f"{plan['side']} not allowed by the nightly scan ({row.allowed})"
            per_symbol[sym] = entry
            for p in paper:
                emit(tenant=p.tenant, type=Event.Type.RISK_REJECTED, severity=Event.Severity.WARN,
                     actor_kind=Event.ActorKind.RISK_ENGINE, step_name="action_board.scan_direction",
                     text=f"Action Board dropped {plan['side']} {sym}: scan allows {row.allowed}",
                     payload={"symbol": sym, "plan": plan, "allowed": row.allowed, "scan_id": scan.id,
                              "strategy": STRATEGY, "date": day.isoformat()})
            continue
        entry["result"] = "planned"
        entry["side"] = plan["side"]
        entry["confidence"] = plan["confidence"]
        per_symbol[sym] = entry
        plans.append((it, plan))

    # 4. Cards per tenant ------------------------------------------------
    receipt_id = str(run.receipt.id) if getattr(run, "receipt", None) is not None else None
    statuses: dict[str, int] = {}
    passed_risk = 0
    cards_out: list = []
    for portfolio in paper:
        try:
            cd = evaluate_for_portfolio(portfolio, day)
        except Exception as e:  # noqa: BLE001
            run.swallowed_error(f"cooldown for {portfolio.id}: {e}")
            cd = None
        for it, plan in plans:
            try:
                card, state = _save_card(portfolio, day=day, as_of=as_of, expires=expires, now=now,
                                         item=it, plan=plan, row=rows[it["symbol"]], scan=scan,
                                         cooldown=cd, receipt_id=receipt_id, replay=replay)
            except Exception as e:  # noqa: BLE001
                run.swallowed_error(f"card {it['symbol']} for tenant {portfolio.tenant_id}: {e}")
                continue
            if card is None:
                continue
            statuses[state] = statuses.get(state, 0) + 1
            if card.risk_passed:
                passed_risk += 1
            cards_out.append(card)

    dropped: dict[str, int] = {}
    for e in per_symbol.values():
        if e["result"] != "planned":
            dropped[e["result"]] = dropped.get(e["result"], 0) + 1
    planner_calls = min(cap, len(ranked))
    run.output(planner_calls=planner_calls, planned=len(plans), dropped=dropped, per_symbol=per_symbol,
               cards=len(cards_out), card_states=statuses, passed_risk=passed_risk, tenants=len(paper))
    failed_calls = sum(1 for e in per_symbol.values() if e["result"] == "failed")
    if planner_calls and failed_calls == planner_calls:
        run.check(rc.make("planner", rc.FAIL, f"all {planner_calls} planner call(s) failed",
                          examples=[e.get("detail") for e in per_symbol.values()][:3]))
    elif failed_calls:
        run.check(rc.make("planner", rc.WARN, f"{failed_calls} of {planner_calls} planner call(s) failed"))
    else:
        run.check(rc.make("planner", rc.PASS, f"{planner_calls} planner call(s), {len(plans)} plan(s)"))
    run.check(rc.make("cards_written", rc.INFO,
                      f"{len(cards_out)} card(s), {passed_risk} passed the risk pre-check "
                      "(0 is a legitimate outcome)"))

    result = {"date": day.isoformat(), "as_of": as_of.isoformat(), "replay": replay,
              "candidates": len(candidates), "fresh": fresh_n, "planner_calls": planner_calls,
              "planned": len(plans), "cards": len(cards_out), "passed_risk": passed_risk,
              "card_states": statuses, "per_symbol": per_symbol}
    if telegram:
        ok, err = send_board_telegram(day, result, [c for c in cards_out])
        run.output(telegram_sent=ok)
        if not ok:
            run.warn(f"Telegram summary: {err}")
        result["telegram_sent"] = ok
    return result


def _save_card(portfolio, *, day, as_of, expires, now, item, plan, row, scan, cooldown, receipt_id,
               replay) -> tuple[Any, str]:
    from django.db import transaction

    from apps.events.models import Event
    from apps.events.services.event_writer import emit
    from apps.trading.domain.entities import OrderDraft
    from apps.trading.models import TodayAction
    from apps.trading.services import trading_rules

    sym, side = item["symbol"], plan["side"]
    existing = list(TodayAction.objects.filter(tenant_id=portfolio.tenant_id, date=day,
                                               strategy=STRATEGY, symbol=sym))
    same = next((c for c in existing if c.side == side), None)
    other = next((c for c in existing if c.side != side and c.status != TodayAction.Status.EXPIRED), None)
    if other is not None and same is None:
        return None, "opposite_exists"       # one live idea per symbol per day
    if same is not None and same.status != TodayAction.Status.PROPOSED:
        return same, f"kept_{same.status}"   # never overwrite an owner decision

    shadow = bool(cooldown and cooldown.blocks_entries)
    mult = 1.0 if (cooldown is None or shadow) else float(cooldown.risk_multiplier)
    capital = float(portfolio.capital)
    qty = size_qty(capital, plan["entry"], plan["stop"], mult)
    check = dry_run_gates(portfolio, symbol=sym, side=side, entry=plan["entry"], stop=plan["stop"],
                          target=plan["target"], qty=qty, confidence=plan["confidence"],
                          risk_multiplier=mult, day=day, cooldown=cooldown)
    passed = bool(check["approved"])
    if shadow:
        status = TodayAction.Status.REJECTED
    elif not passed:
        status = TodayAction.Status.REJECTED
    elif now >= expires:
        status = TodayAction.Status.EXPIRED
    else:
        status = TodayAction.Status.PROPOSED
    d = lambda v: Decimal(str(round(float(v), 4)))  # noqa: E731
    fields = dict(
        portfolio=portfolio, kind=side, strategy_version=strategy_version(),
        horizon=TodayAction.Horizon.INTRADAY,
        entry_price=d(plan["entry"]), entry_low=d(plan["entry"]), entry_high=d(plan["entry"]),
        stoploss=d(plan["stop"]), target=d(plan["target"]), suggested_qty=qty,
        capital_required=Decimal(str(round(plan["entry"] * qty, 2))),
        risk_at_stake=Decimal(str(round(abs(plan["entry"] - plan["stop"]) * qty, 2))),
        risk_multiplier=Decimal(str(mult)), risk_check=check, risk_passed=passed,
        confidence=Decimal(str(round(plan["confidence"], 3))), reasons=_reasons(plan, item, row),
        source_refs={
            "run_receipt_id": receipt_id, "nightly_scan_id": scan.id, "nightly_scan_row_id": row.id,
            "planner": {"mode": _planner_mode(), "llm_quantity_ignored": plan.get("llm_qty")},
            "data": {"source": item["source"], "bars": len(item["bars"]), "last_bar": item["last_bar"],
                     "last_price": item["last"], "prev_close": item["prev_close"], "move_pct": item["move_pct"]},
            "rank": item["rank"], "replay": replay, "stage": STAGE_LABEL,
        },
        shadow=shadow, blocked_reason=(check["reason"][:255] if shadow else ""),
        planned_as_of=as_of, expires_at=expires, status=status,
    )
    with transaction.atomic():
        card, created = TodayAction.objects.update_or_create(
            tenant_id=portfolio.tenant_id, date=day, strategy=STRATEGY, symbol=sym, side=side,
            defaults=fields)

    emit(tenant=portfolio.tenant,
         type=Event.Type.TRADE_PLANNED if passed and not shadow else Event.Type.RISK_REJECTED,
         severity=Event.Severity.INFO if passed and not shadow else Event.Severity.WARN,
         actor_kind=Event.ActorKind.WORKFLOW, step_name="action_board.card",
         text=(f"Action Board {'card' if passed and not shadow else 'card (blocked)'}: {side} {qty} {sym} @ "
               f"{plan['entry']:.2f} SL {plan['stop']:.2f} T {plan['target']:.2f} — "
               f"{'risk pre-check passed' if passed else check['reason']}")[:500],
         payload={"action_id": str(card.id), "date": day.isoformat(), "strategy": STRATEGY,
                  "status": card.status, "risk_check": {k: check[k] for k in ("approved", "reason",
                                                                              "failed_criterion")},
                  "shadow": shadow, "created": created})
    if shadow:
        try:
            draft = OrderDraft(symbol=sym, side=side, qty=max(qty, 1), order_type="LIMIT", product="INTRADAY",
                               price=plan["entry"], sl=plan["stop"], tp=plan["target"], origin="strategy")
            trading_rules.record_block(
                portfolio.tenant, portfolio, draft,
                trading_rules.RulesDecision(False, reason=check["reason"], rule="cooldown",
                                            tags={"source": "action_board", "action_id": str(card.id)}))
        except Exception as e:  # noqa: BLE001
            log.warning("action_board.shadow_event_failed", error=str(e))
    return card, card.status if not shadow else "shadow"


def _planner_mode() -> str:
    import os

    return os.getenv("PLANNER_MODE", "cli").lower()


# ─── Telegram ───────────────────────────────────────────────────────────

def format_board_telegram(day: date, result: dict, cards: list) -> str:
    import html

    url = str(board_setting("APP_URL")).rstrip("/") + "/"
    head = (f"<b>Action Board {day:%d %b}</b> · directional · PAPER · <i>{STAGE_LABEL}</i>\n"
            f"{result.get('candidates', 0)} candidates · {result.get('planner_calls', 0)} planned by Claude · "
            f"{result.get('planned', 0)} plan(s) · {result.get('passed_risk', 0)} passed risk")
    lines = [head]
    live = [c for c in cards if not c.shadow]
    for c in sorted(live, key=lambda c: (not c.risk_passed, c.symbol))[:10]:
        mark = "risk OK" if c.risk_passed else f"risk FAIL ({c.risk_check.get('failed_criterion') or '?'})"
        lines.append(f"• <b>{html.escape(c.symbol)}</b> {c.side} {c.suggested_qty} @ {float(c.entry_price):.2f}"
                     f" · SL {float(c.stoploss):.2f} · T {float(c.target):.2f}"
                     f" · ₹{float(c.risk_at_stake):,.0f} at risk · {mark}")
    shadow = [c for c in cards if c.shadow]
    if shadow:
        lines.append(f"{len(shadow)} shadow card(s) blocked by the losing-streak cooldown.")
    if not cards:
        lines.append("No actions today.")
    lines.append(f"Approve / skip on Today: {html.escape(url)}")
    return "\n".join(lines)[:3900]


def send_board_telegram(day: date, result: dict, cards: list) -> tuple[bool, str]:
    """Synchronous send through the existing Telegram service (same path as
    the nightly scan). Outbound only — the bot has no inbound callbacks."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        from plugins.strategy_screener.telegram import TelegramAlertService

        svc = TelegramAlertService()
        if not svc.is_configured:
            return False, "telegram not configured"
        before = svc.total_errors
        svc._do_send(format_board_telegram(day, result, cards), "HTML")
        if svc.total_errors > before:
            return False, "telegram send failed (see logs)"
        return True, ""
    except Exception as e:  # noqa: BLE001
        log.warning("action_board.telegram_failed", error=str(e))
        return False, str(e)[:200]
