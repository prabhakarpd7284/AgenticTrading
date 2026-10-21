"""Benchmark Total Return Index (TRI) history from niftyindices.com (S2).

Endpoint (verified 2026-10-21 from this machine; it is what the public
"Historical Data -> Total Return Index" page calls):

    POST https://www.niftyindices.com/BackPage/getTotalReturnIndexString
    Content-Type: application/json; charset=utf-8
    {"cinfo": "{'name':'NIFTY 50','startDate':'01-Jan-2024','endDate':'31-Jan-2024','indexName':'NIFTY 50'}"}

Returns a JSON list of ``{"Index Name", "Date": "31 Jan 2024",
"TotalReturnsIndex": "31939.59", "NTR_Value"}``. The page's own script caps a
request at 365 days, so we fetch 1-year chunks, throttled. No cookies or
tokens are needed; nothing here works around any protection. If the endpoint
stops answering, benchmark fields stay null and the screener says so.
Collection code: pure fetch + parse, no ORM.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation

import requests
import structlog

log = structlog.get_logger(__name__)

TRI_URL = "https://www.niftyindices.com/BackPage/getTotalReturnIndexString"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AlphaDesk-MF/1.0 (personal research)"

# The indices the screener's category -> benchmark map uses.
BENCHMARK_INDICES = (
    "NIFTY 50",
    "NIFTY NEXT 50",
    "NIFTY 100",
    "NIFTY 500",
    "NIFTY LARGEMIDCAP 250",
    "NIFTY MIDCAP 150",
    "NIFTY SMALLCAP 250",
)


def parse_tri_json(payload) -> list[tuple[date, Decimal]]:
    if isinstance(payload, dict) and "d" in payload:      # ASP.NET wraps some replies
        payload = json.loads(payload["d"]) if isinstance(payload["d"], str) else payload["d"]
    out: dict[date, Decimal] = {}
    for row in payload or []:
        try:
            d = datetime.strptime(str(row["Date"]).strip(), "%d %b %Y").date()
            v = Decimal(str(row["TotalReturnsIndex"]).replace(",", "").strip())
        except (KeyError, ValueError, InvalidOperation):
            continue
        if v.is_finite() and v > 0:
            out.setdefault(d, v)
    return sorted(out.items())


def fetch_tri(index_name: str, start: date, end: date, timeout: int = 60, session=None) -> list[tuple[date, Decimal]]:
    cinfo = (f"{{'name':'{index_name}','startDate':'{start.strftime('%d-%b-%Y')}',"
             f"'endDate':'{end.strftime('%d-%b-%Y')}','indexName':'{index_name}'}}")
    s = session or requests
    resp = s.post(TRI_URL, data=json.dumps({"cinfo": cinfo}), timeout=timeout, headers={
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "Referer": "https://www.niftyindices.com/reports/historical-data",
    })
    resp.raise_for_status()
    return parse_tri_json(resp.json())


def chunks(start: date, end: date, days: int = 364) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        nxt = min(cur + timedelta(days=days), end)
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


def fetch_tri_range(index_name: str, start: date, end: date, pause_s: float = 1.5,
                    retries: int = 3) -> tuple[list[tuple[date, Decimal]], list[str]]:
    rows: list[tuple[date, Decimal]] = []
    errors: list[str] = []
    for a, b in chunks(start, end):
        for attempt in range(retries + 1):
            try:
                rows.extend(fetch_tri(index_name, a, b))
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == retries:
                    errors.append(f"{index_name} {a}..{b}: {exc}")
                    log.warning("benchmarks.chunk_failed", index=index_name, start=str(a), error=str(exc))
                else:
                    time.sleep(2 ** attempt * 3)
        time.sleep(pause_s)
    return rows, errors
