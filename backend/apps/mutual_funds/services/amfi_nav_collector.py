"""Pulls AMFI's daily NAV file and upserts MFScheme + MFNav rows.

AMFI publishes one semicolon-delimited .txt covering every scheme, every
day, free, no auth: https://portal.amfiindia.com/spages/NAVAll.txt (the
www. host 302-redirects here). Format is NOT a clean CSV — it's a flat
text report with two kinds of header lines interleaved among data rows:

    Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date
    <blank line>
    Open Ended Schemes(Children's Fund - Childrens' Fund)      <- category header (no ';')
    <blank line>
    Axis Mutual Fund                                            <- AMC header (no ';', not a known category)
    135762;INF846K01WO1;-;Axis Children's Fund;Direct Plan;Growth Option;28.9937;12-Oct-2026

A row belongs to whatever category/AMC header most recently preceded it.
'-' marks an empty ISIN. Date is DD-Mon-YYYY.

This module is pure parsing + a thin fetch wrapper — no Django ORM calls
here, so it's unit-testable without a database. The management command
(`pull_amfi_nav`) does the upsert.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import requests

AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
# The www host 302s to portal.amfiindia.com as of Oct 2026; kept as a
# fallback in case that flips back.
AMFI_NAV_URL_FALLBACK = "https://www.amfiindia.com/spages/NAVAll.txt"


@dataclass(frozen=True)
class SchemeNavRow:
    scheme_code: str
    isin_growth: str
    isin_div_reinvestment: str
    scheme_name: str
    plan: str
    option: str
    nav: str  # kept as str; the DB layer decides Decimal conversion
    nav_date: date
    amc_name: str
    category: str


def fetch_nav_text(timeout: int = 30) -> str:
    """Downloads the raw NAV file. Run this where the process has real
    internet (the dev machine / prod worker) — it will fail behind an
    egress allowlist that doesn't permit amfiindia.com."""
    last_exc: Exception | None = None
    for url in (AMFI_NAV_URL, AMFI_NAV_URL_FALLBACK):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "AlphaDesk/1.0"})
            resp.raise_for_status()
            # AMFI sends text/plain with no charset; resp.text would assume
            # latin-1 and mangle non-ASCII category names (e.g. "Children's").
            return resp.content.decode("utf-8", errors="replace")
        except requests.RequestException as exc:  # noqa: PERF203
            last_exc = exc
    raise RuntimeError(f"Could not fetch AMFI NAV file from either URL: {last_exc}")


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw.strip(), "%d-%b-%Y").date()


def parse_nav_text(text: str) -> list[SchemeNavRow]:
    """Parses the AMFI NAV file's text into rows, tracking the most recent
    category and AMC header lines as context for each data row."""
    rows: list[SchemeNavRow] = []
    current_category = ""
    current_amc = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Scheme Code;"):
            continue  # the one column-header line

        if ";" not in line:
            # Either a category header ("Open Ended Schemes(...)") or an
            # AMC name line. AMFI doesn't tag which is which, but category
            # headers always contain a parenthesis; AMC lines don't.
            if "(" in line:
                current_category = line
            else:
                current_amc = line
            continue

        fields = [f.strip() for f in line.split(";")]
        if len(fields) < 8:
            continue  # malformed row — skip rather than crash the whole pull
        scheme_code, isin_g, isin_d, scheme_name, plan, option, nav, nav_date_raw = fields[:8]
        if not scheme_code or not nav or nav == "N.A.":
            continue  # some schemes report no NAV on a given day

        try:
            nav_date = _parse_date(nav_date_raw)
        except ValueError:
            continue

        rows.append(
            SchemeNavRow(
                scheme_code=scheme_code,
                isin_growth="" if isin_g == "-" else isin_g,
                isin_div_reinvestment="" if isin_d == "-" else isin_d,
                scheme_name=scheme_name,
                plan=plan,
                option=option,
                nav=nav,
                nav_date=nav_date,
                amc_name=current_amc,
                category=current_category,
            )
        )
    return rows
