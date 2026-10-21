"""Historical NAV backfill from AMFI's NAV-history report.

Endpoint: ``DownloadNAVHistoryReport_Po.aspx?mf=<amc id>&frmdt=..&todt=..``
(dates ``DD-Mon-YYYY``). It cannot filter by scheme — the smallest unit is
one AMC, so a scheme backfill downloads that AMC's whole range (about 15 MB
per AMC-year) and keeps only the requested scheme codes. Keep scheme lists
short (a holding's worth), not "everything".

Row format (semicolon-delimited, category/AMC header lines interleaved):

    Scheme Code;NAV Name;Plan;Option;ISIN Div Payout/ISIN Growth;ISIN Div Reinvestment;Net Asset Value;Date

AMC ids are sparse and not published as a list, so they are resolved by
probing ``mf=1..MAX_AMC_ID`` with a one-week window and reading the AMC
header line out of the response. Results are cached per process, and a
persisted map (``data/amfi_amc_ids.json``, probed 2026-10-21) is consulted
first so a normal run makes no probe calls at all.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests
import structlog

log = structlog.get_logger(__name__)

AMFI_HISTORY_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"
MAX_AMC_ID = 150
_AMC_ID_CACHE: dict[str, int] = {}
AMC_ID_FILE = Path(__file__).resolve().parent.parent / "data" / "amfi_amc_ids.json"


@dataclass(frozen=True)
class HistoryNav:
    scheme_code: str
    nav_date: date
    nav: Decimal


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _fmt(d: date) -> str:
    return d.strftime("%d-%b-%Y")


def fetch_history_text(amc_id: int, start: date, end: date, timeout: int = 120) -> str:
    resp = requests.get(
        AMFI_HISTORY_URL,
        params={"mf": amc_id, "frmdt": _fmt(start), "todt": _fmt(end)},
        timeout=timeout,
        headers={"User-Agent": "AlphaDesk/1.0"},
    )
    resp.raise_for_status()
    # text/plain with no charset: decode explicitly (see amfi_nav_collector).
    return resp.content.decode("utf-8", errors="replace")


def parse_history_text(text: str, scheme_codes: set[str] | None = None) -> list[HistoryNav]:
    """Parses history rows, optionally keeping only ``scheme_codes``.
    Unparseable rows (``N.A.`` NAVs, bad dates) are skipped, not fatal."""
    out: list[HistoryNav] = []
    for line in text.splitlines():
        parts = line.strip().split(";")
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        code = parts[0]
        if scheme_codes is not None and code not in scheme_codes:
            continue
        try:
            nav = Decimal(parts[6].strip())
            nav_date = datetime.strptime(parts[7].strip(), "%d-%b-%Y").date()
        except (InvalidOperation, ValueError):
            continue
        out.append(HistoryNav(code, nav_date, nav))
    return out


def known_amc_ids() -> dict[str, int]:
    """Persisted AMC-name -> AMFI id map (normalised names)."""
    try:
        raw = json.loads(AMC_ID_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return {_norm(k): int(v) for k, v in (raw.get("amc_ids") or {}).items()}


def resolve_amc_id(amc_name: str, max_id: int = MAX_AMC_ID, probe: bool = True) -> int | None:
    """AMFI's numeric id for an AMC: the persisted map first, else probe
    ascending until the AMC header line in a one-week report matches."""
    key = _norm(amc_name)
    if key in _AMC_ID_CACHE:
        return _AMC_ID_CACHE[key]
    persisted = known_amc_ids()
    if key in persisted:
        _AMC_ID_CACHE[key] = persisted[key]
        return persisted[key]
    if not probe:
        return None
    end = date.today()
    start = end - timedelta(days=7)
    for amc_id in range(1, max_id + 1):
        try:
            text = fetch_history_text(amc_id, start, end, timeout=40)
        except requests.RequestException as exc:
            log.warning("amfi_history.probe_failed", amc_id=amc_id, error=str(exc))
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.endswith("Mutual Fund") and ";" not in line:
                _AMC_ID_CACHE[_norm(line)] = amc_id
                break
        if key in _AMC_ID_CACHE:
            return _AMC_ID_CACHE[key]
    return None


def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Splits [start, end] into <=365-day windows (keeps each download bounded)."""
    chunks = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=364), end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks
