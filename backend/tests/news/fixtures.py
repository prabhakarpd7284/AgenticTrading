"""Canned source payloads for the news tests (no network)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from email.utils import format_datetime

from django.utils import timezone


def rfc822(dt: datetime) -> str:
    return format_datetime(dt)


def rss(items: list[dict]) -> bytes:
    body = "".join(
        f"<item><title>{i['title']}</title><link>{i['link']}</link>"
        f"<pubDate>{rfc822(i['when'])}</pubDate>"
        + (f"<source url='https://x'>{i['publisher']}</source>" if i.get("publisher") else "")
        + "</item>"
        for i in items
    )
    return f"<?xml version='1.0'?><rss><channel>{body}</channel></rss>".encode()


def make_fetch(now: datetime, *, fail: set[str] | None = None, calendar_date=None):
    """A fake ``http.fetch`` routing by URL substring."""
    fail = fail or set()
    recent = now - timedelta(hours=5)
    old = now - timedelta(days=10)
    cal_date = (calendar_date or (timezone.localtime(now).date() + timedelta(days=1))).strftime("%d-%b-%Y")

    def fetch(url, **kw):
        for f in fail:
            if f in url:
                from apps.news.services.http import FetchError
                raise FetchError(f"boom {f}")
        if "news.google.com" in url:
            if "Infosys" in url:
                return rss([
                    {"title": "Infosys Q2 profit beats estimates, raises guidance - Mint",
                     "link": "https://g/infy1", "when": recent, "publisher": "Mint"},
                    {"title": "Infosys shares surge on strong deal wins - ET",
                     "link": "https://g/infy2", "when": recent, "publisher": "ET"},
                    {"title": "Old Infosys story", "link": "https://g/old", "when": old},
                    {"title": "Unrelated TCS story", "link": "https://g/tcs-in-infy", "when": recent},
                ])
            if "HDFC" in url:
                return rss([{"title": "HDFC Bank faces RBI penalty over KYC lapses",
                             "link": "https://g/hdfc1", "when": recent, "publisher": "BS"}])
            return rss([])
        if "economictimes" in url and "1977021501" in url:
            return rss([
                {"title": "Sensex, Nifty end higher as FIIs return", "link": "https://et/m1", "when": recent},
                {"title": "Infosys Q2 profit beats estimates, raises guidance", "link": "https://et/infy",
                 "when": recent},
            ])
        if "economictimes" in url or "livemint" in url or "business-standard" in url:
            return rss([])
        if "corporate-announcements" in url:
            if "symbol=INFY" in url:
                return json.dumps([
                    {"desc": "Outcome of Board Meeting", "attchmntText": "Financial results for Q2",
                     "sort_date": timezone.localtime(recent).strftime("%Y-%m-%d %H:%M:%S"),
                     "attchmntFile": "https://nse/infy.pdf", "seq_id": "1"},
                    {"desc": "Copy of Newspaper Publication", "attchmntText": "clippings",
                     "sort_date": timezone.localtime(recent).strftime("%Y-%m-%d %H:%M:%S"),
                     "attchmntFile": "https://nse/infy-np.pdf", "seq_id": "2"},
                ]).encode()
            return b"[]"
        if "event-calendar" in url:
            return json.dumps([
                {"symbol": "HDFCBANK", "company": "HDFC Bank", "purpose": "Financial Results",
                 "bm_desc": "To consider results", "date": cal_date},
                {"symbol": "TCS", "company": "TCS", "purpose": "Other business matters",
                 "bm_desc": "misc", "date": cal_date},
            ]).encode()
        if "corporate-board-meetings" in url:
            return b"[]"
        raise AssertionError(f"unexpected url {url}")

    return fetch


def fake_runner(score_map: dict[str, float] | None = None):
    """Fake Claude CLI: scores headlines by keyword."""
    score_map = score_map or {"beats": 0.8, "surge": 0.7, "results for q2": 0.6,
                              "penalty": -0.8, "higher": 0.3}
    calls = []

    def runner(prompt, *, timeout, model=""):
        calls.append(prompt)
        out = []
        for line in prompt.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            title = parts[4].lower()
            score = next((v for k, v in score_map.items() if k in title), 0.0)
            mat = "results" if "profit" in title or "result" in title else (
                "regulatory" if "rbi" in title else ("macro" if parts[1] == "MARKET" else "order"))
            out.append({"id": int(parts[0]), "score": score, "materiality": mat, "confidence": 0.9})
        return "```json\n" + json.dumps(out) + "\n```"

    runner.calls = calls
    return runner
