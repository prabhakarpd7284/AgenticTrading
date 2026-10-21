"""News / filing / event-calendar collectors.

Every collector returns a ``SourceStatus`` and never raises — one source
failing must not abort the nightly scan. Headlines + links only.

Sources (verified 2026-10-21 with a browser User-Agent):
  google_news         news.google.com RSS search, one query per stock
  et_markets          Economic Times markets RSS
  et_stocks           Economic Times stocks RSS
  livemint_markets    Livemint markets RSS
  bs_markets          Business Standard markets RSS
  nse_announcements   NSE corporate-announcements JSON, per symbol, date-bounded
  nse_event_calendar  NSE event-calendar JSON (upcoming results/board meetings)
  nse_board_meetings  NSE corporate-board-meetings JSON
(The NSE JSON APIs answer with UA + Accept + Referer; no cookie needed.
Moneycontrol is deliberately not a source — it blocks automated access.)
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import structlog
from django.utils import timezone

from apps.news import constants
from apps.news.models import CorporateEvent, NewsItem
from apps.news.services import http
from apps.news.services.store import upsert_item

log = structlog.get_logger(__name__)

RSS_FEEDS = {
    "et_markets": ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    "et_stocks": ("Economic Times", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    "livemint_markets": ("Livemint", "https://www.livemint.com/rss/markets"),
    "bs_markets": ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
}
GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
NSE_REFERER = "https://www.nseindia.com/"
NSE_ANNOUNCEMENTS_URL = ("https://www.nseindia.com/api/corporate-announcements?index=equities"
                         "&symbol={sym}&from_date={frm}&to_date={to}")
NSE_EVENT_CALENDAR_URL = "https://www.nseindia.com/api/event-calendar?index=equities"
NSE_BOARD_MEETINGS_URL = "https://www.nseindia.com/api/corporate-board-meetings?index=equities"

# Routine NSE filings with no price relevance — stored as materiality=noise
# without an LLM call.
NSE_NOISE_RE = re.compile(
    r"newspaper|certificate|trading window|investor meet|con\.? ?call|share certificate|"
    r"esop|esos|esps|registrar|compliance|regulation 74|annual report|investor complaints|"
    r"postal ballot|loss of share|duplicate|reg\. ?74|spurious|clarification - news item",
    re.IGNORECASE,
)

MAX_PER_SYMBOL = 12


@dataclass
class SourceStatus:
    source: str
    ok: bool = True
    requests: int = 0
    failures: int = 0
    seen: int = 0
    new: int = 0
    secs: float = 0.0
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.failures += 1
        if len(self.errors) < 5:
            self.errors.append(msg[:200])

    def as_dict(self) -> dict:
        d = asdict(self)
        d["secs"] = round(self.secs, 1)
        return d


def _safe_xml(body: bytes) -> ET.Element:
    head = body[:4096].upper()
    if b"<!ENTITY" in head or (b"<!DOCTYPE" in head and b"[" in head):
        raise ValueError("refusing XML with inline DTD/entities")
    return ET.fromstring(body)


def parse_rss(body: bytes) -> list[dict]:
    """[{title, link, published_at (aware), publisher}] from an RSS 2.0 body."""
    root = _safe_xml(body)
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src = item.find("source")
        publisher = (src.text or "").strip() if src is not None and src.text else ""
        try:
            published = parsedate_to_datetime(pub) if pub else None
        except (TypeError, ValueError):
            published = None
        if published is not None and timezone.is_naive(published):
            published = timezone.make_aware(published, timezone.get_current_timezone())
        if title and link:
            out.append({"title": title, "link": link, "published_at": published,
                         "publisher": publisher})
    return out


def _timed(fn: Callable[[SourceStatus], None], status: SourceStatus) -> SourceStatus:
    t0 = time.monotonic()
    try:
        fn(status)
    except Exception as e:  # noqa: BLE001 — never let a source abort the scan
        status.ok = False
        status.fail(f"{type(e).__name__}: {e}")
        log.warning("news.source_failed", source=status.source, error=str(e))
    status.secs = time.monotonic() - t0
    if status.requests and status.failures >= status.requests:
        status.ok = False
    return status


# ── Google News (per stock) ───────────────────────────────────────────────

def collect_google_news(symbols: list[str], *, since: datetime,
                        fetch=None) -> SourceStatus:
    fetch = fetch or http.fetch

    def run(st: SourceStatus) -> None:
        for sym in symbols:
            name = constants.company_name(sym)
            url = GOOGLE_NEWS_URL.format(q=quote_plus(f'"{name}" share'))
            st.requests += 1
            try:
                items = parse_rss(fetch(url, accept="application/rss+xml,application/xml"))
            except Exception as e:  # noqa: BLE001
                st.fail(f"{sym}: {e}")
                continue
            fresh = [i for i in items if i["published_at"] and i["published_at"] >= since
                     and constants.mentions(i["title"], sym)]
            fresh.sort(key=lambda i: i["published_at"], reverse=True)
            for i in fresh[:MAX_PER_SYMBOL]:
                st.seen += 1
                _, created = upsert_item(
                    title=i["title"], url=i["link"], source="google_news",
                    publisher=i["publisher"], published_at=i["published_at"], symbols=[sym],
                )
                st.new += int(created)

    return _timed(run, SourceStatus("google_news"))


# ── Market RSS feeds (matched to symbols by company-name mention) ────────

def collect_rss_feeds(universe: list[str], *, since: datetime, fetch=None) -> list[SourceStatus]:
    fetch = fetch or http.fetch
    out = []
    for key, (publisher, url) in RSS_FEEDS.items():
        def run(st: SourceStatus, url=url, publisher=publisher, key=key) -> None:
            st.requests += 1
            items = parse_rss(fetch(url, accept="application/rss+xml,application/xml"))
            for i in items:
                if not i["published_at"] or i["published_at"] < since:
                    continue
                st.seen += 1
                syms = constants.match_symbols(i["title"], universe)
                _, created = upsert_item(
                    title=i["title"], url=i["link"], source=key, publisher=publisher,
                    published_at=i["published_at"], symbols=syms, is_market_wide=not syms,
                )
                st.new += int(created)
        out.append(_timed(run, SourceStatus(key)))
    return out


# ── NSE corporate announcements (per symbol) ─────────────────────────────

def _nse_dt(s: str) -> datetime | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y"):
        try:
            return timezone.make_aware(datetime.strptime(s.strip(), fmt),
                                       timezone.get_current_timezone())
        except (ValueError, AttributeError):
            continue
    return None


def collect_nse_announcements(symbols: list[str], *, since: datetime, until: datetime,
                              fetch=None) -> SourceStatus:
    fetch = fetch or http.fetch
    frm, to = since.strftime("%d-%m-%Y"), until.strftime("%d-%m-%Y")

    def run(st: SourceStatus) -> None:
        for sym in symbols:
            url = NSE_ANNOUNCEMENTS_URL.format(sym=quote_plus(sym), frm=frm, to=to)
            st.requests += 1
            try:
                data = json.loads(fetch(url, accept="application/json", referer=NSE_REFERER))
            except Exception as e:  # noqa: BLE001
                st.fail(f"{sym}: {e}")
                continue
            if isinstance(data, dict):
                data = data.get("data") or []
            for a in data[:50]:
                published = _nse_dt(a.get("sort_date") or a.get("an_dt") or "")
                if published is None or published < since:
                    continue
                desc = (a.get("desc") or "").strip()
                text = re.sub(r"\s+", " ", (a.get("attchmntText") or "")).strip()
                title = f"{desc}: {text[:240]}" if text else desc
                link = a.get("attchmntFile") or (
                    f"https://www.nseindia.com/get-quotes/equity?symbol={quote_plus(sym)}"
                    f"#ann-{a.get('seq_id', '')}")
                noise = bool(NSE_NOISE_RE.search(desc))
                st.seen += 1
                _, created = upsert_item(
                    title=title, url=link, source="nse_announcements", publisher="NSE",
                    kind=NewsItem.Kind.FILING, published_at=published, symbols=[sym],
                    raw={"desc": desc, "seq_id": a.get("seq_id")},
                    materiality=NewsItem.Materiality.NOISE if noise else "",
                )
                st.new += int(created)

    return _timed(run, SourceStatus("nse_announcements"))


# ── NSE event calendar + board meetings ──────────────────────────────────

def _nse_date(s: str) -> date | None:
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime((s or "").strip(), fmt).date()
        except ValueError:
            continue
    return None


def _store_event(st: SourceStatus, *, symbol, company, event_date, purpose, description, source):
    if not symbol or not event_date or not purpose:
        return
    st.seen += 1
    _, created = CorporateEvent.objects.update_or_create(
        symbol=symbol.strip().upper(), event_date=event_date, purpose=purpose.strip()[:200],
        source=source,
        defaults={"company": (company or "")[:200], "description": (description or "")[:2000]},
    )
    st.new += int(created)


def collect_nse_events(*, fetch=None, from_date: date | None = None) -> list[SourceStatus]:
    fetch = fetch or http.fetch
    from_date = from_date or timezone.localdate()

    def run_calendar(st: SourceStatus) -> None:
        st.requests += 1
        data = json.loads(fetch(NSE_EVENT_CALENDAR_URL, accept="application/json",
                                referer=NSE_REFERER))
        for e in data if isinstance(data, list) else []:
            d = _nse_date(e.get("date", ""))
            if d and d >= from_date:
                _store_event(st, symbol=e.get("symbol"), company=e.get("company"), event_date=d,
                             purpose=e.get("purpose") or "", description=e.get("bm_desc") or "",
                             source="nse_event_calendar")

    def run_board(st: SourceStatus) -> None:
        st.requests += 1
        data = json.loads(fetch(NSE_BOARD_MEETINGS_URL, accept="application/json",
                                referer=NSE_REFERER))
        for e in data if isinstance(data, list) else []:
            d = _nse_date(e.get("bm_date", ""))
            if d and d >= from_date:
                _store_event(st, symbol=e.get("bm_symbol"), company=e.get("sm_name"),
                             event_date=d, purpose=e.get("bm_purpose") or "",
                             description=e.get("bm_desc") or "", source="nse_board_meetings")

    return [_timed(run_calendar, SourceStatus("nse_event_calendar")),
            _timed(run_board, SourceStatus("nse_board_meetings"))]


# ── Everything ────────────────────────────────────────────────────────────

def collect_all(symbols: list[str], *, now: datetime | None = None,
                lookback_days: int = 3, fetch=None) -> dict[str, dict]:
    """Run every collector; returns ``{source: status_dict}``."""
    now = now or timezone.now()
    since = now - timedelta(days=lookback_days)
    statuses: list[SourceStatus] = []
    statuses.append(collect_google_news(symbols, since=since, fetch=fetch))
    statuses.extend(collect_rss_feeds(symbols, since=since, fetch=fetch))
    statuses.append(collect_nse_announcements(symbols, since=since, until=now, fetch=fetch))
    statuses.extend(collect_nse_events(fetch=fetch, from_date=timezone.localtime(now).date()))
    return {s.source: s.as_dict() for s in statuses}
