"""mfapi.in client — per-scheme NAV history (S0/S1 primary backfill source).

``GET https://api.mfapi.in/mf/{scheme_code}`` returns
``{"meta": {...}, "data": [{"date": "DD-MM-YYYY", "nav": "123.45600"}, ...],
"status": "SUCCESS"}`` newest first, covering the scheme's whole life (back to
2006 for some dead schemes). Unknown codes return 200 with ``scheme_code: 0``
and empty data. ``GET /mf`` lists ~38k codes, including dead schemes.

The site says "no rate limiting", but we throttle anyway (default 0.35 s
between calls), retry 5xx/timeouts with exponential backoff, back off hard on
429, and send an identifying User-Agent. It is a free community mirror of
AMFI data with no published ToS text, so AMFI stays the daily source of truth
and ``mf_crosscheck_sources`` verifies a sample against AMFI's own history
report. Pure fetch + parse; no ORM here.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests
import structlog

log = structlog.get_logger(__name__)

MFAPI_BASE = "https://api.mfapi.in"
USER_AGENT = "AlphaDesk-MF/1.0 (personal research; single-user, throttled)"
DEFAULT_MIN_INTERVAL_S = 0.35
DEFAULT_RETRIES = 4


@dataclass
class SchemeHistory:
    scheme_code: str
    fund_house: str = ""
    scheme_type: str = ""
    scheme_category: str = ""
    scheme_name: str = ""
    isin_growth: str = ""
    isin_div_reinvestment: str = ""
    navs: list[tuple[date, Decimal]] = field(default_factory=list)   # oldest first
    bad_rows: int = 0
    raw_bytes: int = 0

    @property
    def found(self) -> bool:
        return bool(self.scheme_name)

    @property
    def amfi_category(self) -> str:
        """Same shape as the AMFI daily file's header ("Open Ended Schemes(X)")."""
        if self.scheme_type and self.scheme_category:
            return f"{self.scheme_type}({self.scheme_category})"
        return self.scheme_category or ""


def parse_scheme_json(code: str, payload: dict, raw_bytes: int = 0) -> SchemeHistory:
    meta = payload.get("meta") or {}
    h = SchemeHistory(
        scheme_code=str(code),
        fund_house=(meta.get("fund_house") or "").strip(),
        scheme_type=(meta.get("scheme_type") or "").strip(),
        scheme_category=(meta.get("scheme_category") or "").strip(),
        scheme_name=(meta.get("scheme_name") or "").strip(),
        isin_growth=(meta.get("isin_growth") or "") or "",
        isin_div_reinvestment=(meta.get("isin_div_reinvestment") or "") or "",
        raw_bytes=raw_bytes,
    )
    seen: dict[date, Decimal] = {}
    for row in payload.get("data") or []:
        try:
            d = datetime.strptime(str(row["date"]).strip(), "%d-%m-%Y").date()
            v = Decimal(str(row["nav"]).strip())
        except (KeyError, ValueError, InvalidOperation):
            h.bad_rows += 1
            continue
        if not v.is_finite():
            h.bad_rows += 1
            continue
        seen.setdefault(d, v)   # first occurrence wins on duplicate dates
    h.navs = sorted(seen.items())
    return h


class MfapiClient:
    def __init__(self, min_interval_s: float = DEFAULT_MIN_INTERVAL_S, retries: int = DEFAULT_RETRIES,
                 timeout: int = 45, raw_dir: str | os.PathLike | None = None, session=None):
        self.min_interval_s = min_interval_s
        self.retries = retries
        self.timeout = timeout
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._last = 0.0
        self.bytes_downloaded = 0
        self.calls = 0

    def _throttle(self):
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str, params: dict | None = None) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                self.calls += 1
                if resp.status_code == 429:
                    delay = float(resp.headers.get("Retry-After") or 30 * (attempt + 1))
                    log.warning("mfapi.rate_limited", url=url, sleep_s=delay)
                    time.sleep(min(delay, 300))
                    last_exc = RuntimeError("HTTP 429")
                    continue
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                self.bytes_downloaded += len(resp.content)
                return resp.content
            except requests.RequestException as exc:
                last_exc = exc
                delay = min(2 ** attempt * 2, 60)
                log.warning("mfapi.retry", url=url, attempt=attempt + 1, error=str(exc), sleep_s=delay)
                time.sleep(delay)
        raise RuntimeError(f"mfapi request failed after {self.retries + 1} attempts: {last_exc}")

    def scheme_history(self, code: str, start: date | None = None, end: date | None = None) -> SchemeHistory:
        params = {}
        if start:
            params["startDate"] = start.isoformat()
        if end:
            params["endDate"] = (end or date.today()).isoformat()
        body = self._get(f"{MFAPI_BASE}/mf/{code}", params or None)
        if self.raw_dir is not None:
            try:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                suffix = f"_{start.isoformat()}" if start else ""
                (self.raw_dir / f"{code}{suffix}.json.gz").write_bytes(gzip.compress(body))
            except OSError as exc:   # raw archive is best-effort
                log.warning("mfapi.raw_write_failed", code=code, error=str(exc))
        return parse_scheme_json(code, json.loads(body), raw_bytes=len(body))

    def scheme_list(self) -> list[dict]:
        return json.loads(self._get(f"{MFAPI_BASE}/mf"))
