"""21:00 IST nightly scan — TRADING_RULES.md §5.

    1. refresh data      bhavcopy (daily, all NSE) + NIFTY 50/indices 1m candles
    2. levels            floor / CPR (+width) / Camarilla / two-day relationship
    3. news              Google News, ET, Livemint, Business Standard, NSE filings
                         + NSE event calendar / board meetings
    4. sentiment         Claude CLI per headline → per-stock rollup
    5. verdict           allowed directions per stock (sentiment_filter rules)
    6. persist           NightlyScan + NightlyScanRow (idempotent per for_date)
    7. cooldown          refresh every trading portfolio's CooldownState
    8. report            Event per tenant + Telegram summary

Every step is timed and wrapped: a failing step/source degrades the scan to
``partial`` instead of aborting it. Only the final DB write can fail it.
"""
from __future__ import annotations

import html
import re
import time
from datetime import date, datetime, timedelta

import structlog
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.news import constants
from apps.news.models import CorporateEvent, NewsItem, NightlyScan, NightlyScanRow
from apps.news.services import collectors, indexing, levels, sentiment
from apps.news.services.sentiment_filter import allowed_directions, allowed_label
from apps.news.services.store import MARKET

log = structlog.get_logger(__name__)

DEFAULT_MAJOR_EVENT_RE = (
    r"result|fund ?rais|bonus|split|buy ?back|merger|amalgamation|demerger|"
    r"scheme of arrangement|rights issue|delisting|qip"
)
HEADLINES_PER_ROW = 8


def scan_conf(key: str, default=None):
    return getattr(settings, "NIGHTLY_SCAN", {}).get(key, default)


# ── Dates ────────────────────────────────────────────────────────────────

def previous_trading_day(d: date) -> date:
    from apps.market_data.services.market_calendar import is_trading_day

    cur = d - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def resolve_dates(session_date: date | None = None, now: datetime | None = None) -> tuple[date, date]:
    """(session_date, for_date).

    ``session_date`` = the evening's session (data the levels come from);
    ``for_date`` = the next NSE trading day the filter governs. With no
    ``session_date``: before 09:15 on a trading day the scan is for today
    (a late-night run), otherwise for the next trading day.
    """
    from apps.market_data.services.market_calendar import is_trading_day, next_trading_day

    if session_date is not None:
        for_date = next_trading_day(session_date)
        if not is_trading_day(session_date):
            session_date = previous_trading_day(session_date)
        return session_date, for_date
    now = timezone.localtime(now) if now else timezone.localtime()
    today = now.date()
    if (now.hour, now.minute) < (9, 15) and is_trading_day(today):
        for_date = today
    else:
        for_date = next_trading_day(today)
    return previous_trading_day(for_date), for_date


# ── Steps ────────────────────────────────────────────────────────────────

def refresh_data(session_date: date) -> dict:
    """Bhavcopy for the last week up to the session + 1m NIFTY 50/indices
    (both commands are resumable: they only fetch what's missing)."""
    from django.core.management import call_command

    out = {}
    for key, kwargs in (
        ("bhavcopy", {"date_from": (session_date - timedelta(days=7)).isoformat(),
                      "date_to": session_date.isoformat()}),
        ("candles_1m", {"universe": "nifty50", "interval": "1m", "months": 1}),
    ):
        cmd = "backfill_bhavcopy" if key == "bhavcopy" else "backfill_candles"
        t0 = time.monotonic()
        try:
            call_command(cmd, **kwargs)
            out[key] = {"ok": True}
        except BaseException as e:  # noqa: BLE001 — CommandError/SystemExit included
            if isinstance(e, KeyboardInterrupt):
                raise
            out[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            log.warning("nightly_scan.refresh_failed", step=key, error=str(e))
        out[key]["secs"] = round(time.monotonic() - t0, 1)
    return out


def compute_all_levels(symbols: list[str], session_date: date) -> dict[str, dict]:
    out = {}
    for sym in symbols:
        try:
            last, prev = levels.last_two_sessions(sym, session_date)
            if last is None:
                out[sym] = {"error": "no daily candles"}
                continue
            lv = levels.compute_levels(last, prev)
            lv["stale"] = last.day < session_date
            out[sym] = lv
        except Exception as e:  # noqa: BLE001
            out[sym] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def window_items(symbols: list[str], *, since: datetime, until: datetime) -> list[NewsItem]:
    """Headlines in the window that mention a universe symbol, plus market-wide."""
    from django.db.models import Q

    qs = (NewsItem.objects.filter(published_at__gte=since, published_at__lte=until)
          .filter(Q(mentions__symbol__in=symbols) | Q(is_market_wide=True))
          .distinct().prefetch_related("mentions").order_by("-published_at"))
    return list(qs)


def _scoring_candidates(items: list[NewsItem], symbols: list[str]) -> list[NewsItem]:
    """Cap what goes to the LLM: newest N per stock + newest M market-wide."""
    per_sym = int(scan_conf("MAX_SCORED_PER_SYMBOL", 10))
    max_macro = int(scan_conf("MAX_SCORED_MACRO", 40))
    chosen: dict[int, NewsItem] = {}
    counts: dict[str, int] = {}
    macro = 0
    for it in items:  # newest first
        if it.materiality == NewsItem.Materiality.NOISE:
            continue
        syms = [m.symbol for m in it.mentions.all() if m.symbol in symbols]
        if syms:
            if any(counts.get(s, 0) < per_sym for s in syms):
                chosen[it.id] = it
                for s in syms:
                    counts[s] = counts.get(s, 0) + 1
        elif it.is_market_wide and macro < max_macro:
            chosen[it.id] = it
            macro += 1
    return list(chosen.values())


def upcoming_events(symbols: list[str], session_date: date, for_date: date) -> dict[str, list[str]]:
    """Major events after the last session and up to (incl.) the next one."""
    pat = re.compile(scan_conf("MAJOR_EVENT_REGEX", DEFAULT_MAJOR_EVENT_RE), re.IGNORECASE)
    out: dict[str, list[str]] = {}
    for ev in CorporateEvent.objects.filter(symbol__in=symbols, event_date__gt=session_date,
                                            event_date__lte=for_date):
        if pat.search(f"{ev.purpose} {ev.description}"):
            label = f"{ev.purpose} on {ev.event_date.isoformat()}"
            if label not in out.setdefault(ev.symbol, []):
                out[ev.symbol].append(label)
    return out


def _headline_snapshot(it: NewsItem) -> dict:
    return {
        "id": it.id, "title": it.title, "url": it.url, "source": it.source,
        "publisher": it.publisher, "kind": it.kind,
        "published_at": it.published_at.isoformat(),
        "score": float(it.sentiment_score) if it.sentiment_score is not None else None,
        "materiality": it.materiality or None,
        "confidence": float(it.confidence) if it.confidence is not None else None,
    }


# ── Orchestration ────────────────────────────────────────────────────────

class ScanAlreadyRunning(Exception):
    pass


def run_nightly_scan(*, session_date: date | None = None, telegram: bool = True,
                     refresh: bool = True, symbols: list[str] | None = None,
                     now: datetime | None = None, fetch=None, runner=None) -> NightlyScan:
    """Run (or re-run — idempotent per ``for_date``) the nightly scan."""
    from django.core.cache import cache

    from apps.market_data.constants import NIFTY_50_SYMBOLS

    now = now or timezone.now()
    session_date, for_date = resolve_dates(session_date, now if session_date is None else None)
    lock = f"nightly_scan:lock:{for_date.isoformat()}"
    if not cache.add(lock, 1, 2 * 3600):
        raise ScanAlreadyRunning(f"a nightly scan for {for_date} is already running")
    try:
        return _run(session_date, for_date, now=now, telegram=telegram, refresh=refresh,
                    symbols=list(symbols or NIFTY_50_SYMBOLS), fetch=fetch, runner=runner)
    except Exception as e:
        mark_failed(for_date, f"{type(e).__name__}: {e}")
        raise
    finally:
        cache.delete(lock)


def _run(session_date: date, for_date: date, *, now: datetime, telegram: bool, refresh: bool,
         symbols: list[str], fetch, runner) -> NightlyScan:
    t_start = time.monotonic()
    lookback = int(sentiment.cfg("LOOKBACK_DAYS"))

    scan, _ = NightlyScan.objects.update_or_create(
        for_date=for_date,
        defaults=dict(session_date=session_date, status=NightlyScan.Status.RUNNING,
                      started_at=timezone.now(), finished_at=None, error="",
                      timings={}, sources={}, summary={}, telegram_sent=False),
    )
    timings: dict[str, float] = {}
    degraded: list[str] = []

    def step(name):
        class _T:
            def __enter__(self_):
                self_.t0 = time.monotonic()

            def __exit__(self_, *exc):
                timings[name] = round(time.monotonic() - self_.t0, 1)
                return False
        return _T()

    # 1. refresh
    refresh_status = {}
    if refresh:
        with step("refresh"):
            refresh_status = refresh_data(session_date)
        if not all(v.get("ok") for v in refresh_status.values()):
            degraded.append("data refresh")

    # 2. levels (stocks + indices)
    with step("levels"):
        lv = compute_all_levels(symbols + constants.INDEX_SYMBOLS, session_date)
    if any("error" in v for k, v in lv.items() if k in symbols):
        degraded.append("levels")

    # 3. news
    with step("news"):
        sources = collectors.collect_all(symbols, now=now, lookback_days=lookback, fetch=fetch)
    failed_sources = [k for k, v in sources.items() if not v.get("ok")]
    if failed_sources:
        degraded.append("sources: " + ", ".join(failed_sources))

    # 4. sentiment
    since = now - timedelta(days=lookback)
    items = window_items(symbols, since=since, until=now + timedelta(hours=1))
    with step("sentiment"):
        score_status = sentiment.score_items(_scoring_candidates(items, symbols), runner=runner)
    if score_status.failed_batches or score_status.aborted:
        degraded.append("sentiment")
    sources["claude_sentiment"] = {"ok": not score_status.aborted and not score_status.failed_batches,
                                   **score_status.as_dict()}

    # 4b. RAG index (best effort)
    with step("indexing"):
        try:
            idx = indexing.index_items([it for it in items if it.scored_at is not None])
        except Exception as e:  # noqa: BLE001
            idx = {"ok": False, "error": str(e)[:200]}
    sources["voyage_index"] = idx

    # 5. rollup + verdicts
    with step("rollup"):
        events = upcoming_events(symbols, session_date, for_date)
        by_sym: dict[str, list[NewsItem]] = {s: [] for s in symbols}
        macro_items: list[NewsItem] = []
        for it in items:
            syms = [m.symbol for m in it.mentions.all() if m.symbol in by_sym]
            for s in syms:
                by_sym[s].append(it)
            if not syms and it.is_market_wide:
                macro_items.append(it)
        rows = []
        for sym in symbols:
            ru = sentiment.rollup(by_sym[sym], now=now)
            ev = events.get(sym, [])
            al, ash, reasons = allowed_directions(label=ru.label, event_tomorrow=bool(ev),
                                                  event_purpose="; ".join(ev), score=ru.score)
            L = lv.get(sym, {})
            if "error" in L:
                reasons.append(f"levels unavailable: {L['error']}")
            elif L.get("stale"):
                reasons.append(f"levels from {L.get('data_date')} (latest session data missing)")
            shown = [it for it in by_sym[sym] if it.materiality != NewsItem.Materiality.NOISE]
            shown.sort(key=lambda it: (abs(float(it.sentiment_score or 0)) *
                                       sentiment.MATERIALITY_WEIGHT.get(it.materiality or "other", 0.4),
                                       it.published_at), reverse=True)
            rows.append(NightlyScanRow(
                symbol=sym,
                data_date=date.fromisoformat(L["data_date"]) if L.get("data_date") else None,
                levels=L if "error" not in L else {},
                cpr_width_pct=(L.get("cpr") or {}).get("width_pct"),
                cpr_class=(L.get("cpr") or {}).get("class", ""),
                two_day_relationship=L.get("cpr_relationship", ""),
                camarilla_relationship=L.get("camarilla_relationship", ""),
                bias=L.get("bias", ""),
                sentiment_score=ru.score, sentiment_label=ru.label, sentiment_weight=ru.weight,
                headline_count=ru.headline_count, scored_count=ru.scored_count,
                event_tomorrow=bool(ev), event_purpose="; ".join(ev)[:255],
                allow_long=al, allow_short=ash, allowed=allowed_label(al, ash),
                reasons=reasons,
                headlines=[_headline_snapshot(it) for it in shown[:HEADLINES_PER_ROW]],
            ))
        market = sentiment.rollup(macro_items, now=now)

    # 6. cooldown (as of the trading day the scan governs)
    with step("cooldown"):
        cooldowns = refresh_cooldowns(for_date)

    summary = build_summary(rows, market, macro_items, lv, cooldowns, sources, score_status,
                            refresh_status)
    status = NightlyScan.Status.PARTIAL if degraded else NightlyScan.Status.SUCCESS
    summary["degraded"] = degraded

    with transaction.atomic():
        scan.rows.all().delete()
        for r in rows:
            r.scan = scan
        NightlyScanRow.objects.bulk_create(rows)
        timings["total"] = round(time.monotonic() - t_start, 1)
        scan.status = status
        scan.timings = timings
        scan.sources = {**sources, **{f"refresh_{k}": v for k, v in refresh_status.items()}}
        scan.summary = summary
        scan.finished_at = timezone.now()
        scan.save()

    _emit_scan_event(scan)

    if telegram:
        with step("telegram"):
            ok, err = send_telegram_summary(scan)
        scan.telegram_sent = ok
        if err:
            scan.summary = {**scan.summary, "telegram_error": err}
        timings["total"] = round(time.monotonic() - t_start, 1)
        scan.timings = timings
        scan.save(update_fields=["telegram_sent", "summary", "timings"])
    return scan


def mark_failed(for_date: date, error: str) -> None:
    NightlyScan.objects.filter(for_date=for_date, status=NightlyScan.Status.RUNNING).update(
        status=NightlyScan.Status.FAILED, error=error[:4000], finished_at=timezone.now())


def refresh_cooldowns(as_of: date) -> list[dict]:
    from apps.trading.services.cooldown import refresh_cooldown, trading_portfolios

    out = []
    for p in trading_portfolios().select_related("tenant"):
        try:
            st = refresh_cooldown(p, as_of)
            out.append({"tenant_id": str(p.tenant_id), "portfolio_id": str(p.id),
                        "portfolio": p.name, **st.as_dict()})
        except Exception as e:  # noqa: BLE001
            out.append({"portfolio_id": str(p.id), "portfolio": p.name, "error": str(e)[:200]})
    return out


def build_summary(rows, market, macro_items, lv, cooldowns, sources, score_status,
                  refresh_status) -> dict:
    scored = [r for r in rows if r.sentiment_score is not None]
    top_pos = sorted([r for r in scored if r.sentiment_score > 0],
                     key=lambda r: r.sentiment_score, reverse=True)[:5]
    top_neg = sorted([r for r in scored if r.sentiment_score < 0],
                     key=lambda r: r.sentiment_score)[:5]

    def brief(r):
        h = r.headlines[0]["title"] if r.headlines else ""
        return {"symbol": r.symbol, "score": float(r.sentiment_score), "label": r.sentiment_label,
                "headline": h[:140]}

    labels: dict[str, int] = {}
    for r in rows:
        labels[r.sentiment_label] = labels.get(r.sentiment_label, 0) + 1
    breadth = labels.get("positive", 0) + labels.get("strong_positive", 0) \
        - labels.get("negative", 0) - labels.get("strong_negative", 0)
    indices = {}
    for sym in constants.INDEX_SYMBOLS:
        L = lv.get(sym) or {}
        if "cpr" in L:
            indices[sym] = {"cpr_class": L["cpr"]["class"], "cpr_width_pct": L["cpr"]["width_pct"],
                            "relationship": L.get("cpr_relationship", ""), "bias": L.get("bias", ""),
                            "pp": L["cpr"]["pp"], "data_date": L.get("data_date")}
    return {
        "universe": len(rows),
        "pivot_source": levels.PIVOT_SOURCE,
        "market": {"score": market.score, "label": market.label,
                   "headlines": market.headline_count, "scored": market.scored_count,
                   "breadth": breadth, "labels": labels},
        "indices": indices,
        "headlines_total": sum(r.headline_count for r in rows),
        "headlines_scored": sum(r.scored_count for r in rows),
        "sentiment_available": not score_status.aborted and score_status.scored > 0
        or score_status.candidates == 0,
        "top_positive": [brief(r) for r in top_pos],
        "top_negative": [brief(r) for r in top_neg],
        "skip": [{"symbol": r.symbol, "allowed": r.allowed, "reasons": r.reasons}
                 for r in rows if r.allowed != "both"],
        "unscored": [r.symbol for r in rows if r.sentiment_label == "unscored"],
        "cooldowns": cooldowns,
        "refresh": refresh_status,
    }


def _emit_scan_event(scan: NightlyScan) -> None:
    from apps.events.models import Event
    from apps.events.services.event_writer import emit
    from apps.tenants.models import Tenant
    from apps.trading.services.cooldown import trading_portfolios

    tenant_ids = set(trading_portfolios().values_list("tenant_id", flat=True))
    s = scan.summary
    for tenant in Tenant.objects.filter(id__in=tenant_ids):
        emit(
            tenant=tenant, type=Event.Type.WORKFLOW_COMPLETED, actor_kind=Event.ActorKind.SYSTEM,
            step_name="nightly_scan",
            severity=Event.Severity.INFO if scan.status == NightlyScan.Status.SUCCESS
            else Event.Severity.WARN,
            text=(f"nightly scan for {scan.for_date.isoformat()}: {scan.status} — "
                  f"{len(s.get('skip', []))} stock(s) restricted, market "
                  f"{s.get('market', {}).get('label')}"),
            payload={"scan_id": scan.id, "for_date": scan.for_date.isoformat(),
                     "status": scan.status, "skip": s.get("skip", []),
                     "degraded": s.get("degraded", [])},
        )


# ── Telegram ─────────────────────────────────────────────────────────────

def _e(s) -> str:
    return html.escape(str(s), quote=False)


def format_telegram(scan: NightlyScan) -> str:
    s = scan.summary or {}
    m = s.get("market", {})
    lines = [f"<b>🌙 Nightly scan — for {scan.for_date.strftime('%a %d %b %Y')}</b>"]
    if scan.status != NightlyScan.Status.SUCCESS:
        lines.append(f"Status: {_e(scan.status)} ({_e(', '.join(s.get('degraded', [])))})")
    score = m.get("score")
    lines.append(f"Market mood: <b>{_e(m.get('label', '?'))}</b>"
                 f"{f' ({score:+.2f})' if score is not None else ''} from {m.get('scored', 0)} "
                 f"market headlines · stock breadth {m.get('breadth', 0):+d}")
    for sym, ix in (s.get("indices") or {}).items():
        lines.append(f"{_e(sym)}: CPR {_e(ix['cpr_class'])} ({ix['cpr_width_pct']:.2f}%), "
                     f"{_e(ix['relationship'] or '—')} → {_e(ix['bias'] or '—')}")
    if not s.get("sentiment_available", True):
        lines.append("⚠️ Sentiment could NOT be computed — the filter will not block anything.")
    else:
        lines.append(f"Headlines scored: {s.get('headlines_scored', 0)}/{s.get('headlines_total', 0)}")

    def fmt(items):
        return "\n".join(f"• {_e(i['symbol'])} {i['score']:+.2f} — {_e(i['headline'][:90])}"
                         for i in items)

    if s.get("top_positive"):
        lines += ["", "<b>Top positive</b>", fmt(s["top_positive"][:3])]
    if s.get("top_negative"):
        lines += ["", "<b>Top negative</b>", fmt(s["top_negative"][:3])]
    skip = s.get("skip") or []
    lines += ["", f"<b>Restricted tomorrow ({len(skip)})</b>"]
    if skip:
        tag = {"none": "SKIP", "long": "long only", "short": "short only"}
        for r in skip[:15]:
            lines.append(f"• {_e(r['symbol'])} [{tag.get(r['allowed'], r['allowed'])}] — "
                         f"{_e('; '.join(r['reasons'])[:120])}")
        if len(skip) > 15:
            lines.append(f"… and {len(skip) - 15} more")
    else:
        lines.append("none — all NIFTY 50 stocks tradable both ways")
    if s.get("unscored"):
        lines.append(f"Unscored (filter off): {_e(', '.join(s['unscored'][:12]))}")
    cds = s.get("cooldowns") or []
    lines += ["", "<b>Cooldown</b>"]
    if not cds:
        lines.append("no trading portfolios")
    for c in cds:
        if c.get("error"):
            lines.append(f"• {_e(c['portfolio'])}: error {_e(c['error'][:80])}")
            continue
        st = c["status"]
        txt = {"normal": f"normal (loss streak {c['loss_streak']}/3)",
               "cooldown": f"COOLDOWN — no automated entries until {c['cooldown_until']}",
               "reduced": f"reduced risk ×{c['risk_multiplier']:g} until a profitable day"}[st]
        lines.append(f"• {_e(c['portfolio'])}: {_e(txt)}")
    src = []
    for k, v in (scan.sources or {}).items():
        if k.startswith("refresh_") or k in ("claude_sentiment", "voyage_index"):
            continue
        src.append(f"{k} {'ok' if v.get('ok') else 'FAIL'} {v.get('new', 0)}/{v.get('seen', 0)}")
    lines += ["", f"<i>Sources: {_e(' · '.join(src))}</i>",
              f"<i>Took {scan.timings.get('total', 0):.0f}s</i>"]
    text = "\n".join(lines)
    return text[:4000]


def send_telegram_summary(scan: NightlyScan) -> tuple[bool, str]:
    """Synchronous send (the screener service's own send is fire-and-forget on
    a daemon thread, which a short-lived command would kill before it posts)."""
    try:
        from plugins.strategy_screener.telegram import TelegramAlertService

        svc = TelegramAlertService()
        if not svc.is_configured:
            return False, "telegram not configured"
        before = svc.total_errors
        svc._do_send(format_telegram(scan), "HTML")
        if svc.total_errors > before:
            return False, "telegram send failed (see logs)"
        return True, ""
    except Exception as e:  # noqa: BLE001
        log.warning("nightly_scan.telegram_failed", error=str(e))
        return False, str(e)[:200]
