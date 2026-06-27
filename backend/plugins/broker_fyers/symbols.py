"""Fyers symbol-master resolution for option contracts.

Hand-formatting Fyers *weekly* option strings is bug-prone (the community has
documented `-300 Invalid symbol` errors for the weekly encoding). The robust
path is to resolve the exact tradeable ticker from Fyers' published symbol-master
CSVs, matching by (underlying, expiry, strike, CE/PE).

The CSVs are public (no auth) and refreshed daily; we disk-cache them. Matching
pins the known columns (ticker col9, expiry-epoch col8, strike col15, opt col16)
with a regex heuristic only as a fallback for column drift, and asserts the match
is unique — so a stray cell equal to the strike can never resolve a wrong contract.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import threading
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

from plugins.strategy_scalp.timeutil import IST

from .adapter import logger

# Pinned Fyers FO symbol-master columns (0-indexed) — verified against the live
# NSE_FO.csv / BSE_FO.csv layout. Matching by column (not "any cell equals the
# value") stops a stray cell — lot size, freeze qty, the col14 token — from
# being mistaken for the strike/expiry and resolving the WRONG contract.
_COL_EXPIRY_EPOCH = 8
_COL_TICKER = 9
_COL_STRIKE = 15
_COL_OPT = 16
# Plausible option-expiry epoch window (2020-09 .. 2033-05) so a 15-digit
# fytoken can never be parsed as an expiry date.
_EPOCH_MIN, _EPOCH_MAX = 1_600_000_000, 2_000_000_000
# Bounded download — a stalled public.fyers.in must fail, not hang forever.
_DOWNLOAD_TIMEOUT = 15

# Public Fyers symbol masters (no auth required).
MASTER_URLS = {
    "NSE": "https://public.fyers.in/sym_details/NSE_FO.csv",
    "BSE": "https://public.fyers.in/sym_details/BSE_FO.csv",
}
INDEX_SYMBOL = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX",
}
_EXCHANGE = {"NIFTY": "NSE", "BANKNIFTY": "NSE", "FINNIFTY": "NSE", "SENSEX": "BSE"}
_CACHE_DIR = Path(os.environ.get("FYERS_SYMBOL_MASTER_CACHE", "/tmp/alphadesk_fyers"))
_TICKER_RE = re.compile(r"^(NSE|BSE):[A-Z0-9]+.*(CE|PE)$")
_ROOT_RE = re.compile(r"^(?:NSE|BSE):([A-Z]+)\d")
_MAX_AGE_SECS = 18 * 3600


def _ticker_root(ticker: str) -> str:
    """Underlying root of an option ticker, e.g. NSE:FINNIFTY26JUN... → FINNIFTY.
    Exact-root matching avoids NIFTY matching FINNIFTY/BANKNIFTY (substring bug)."""
    m = _ROOT_RE.match(ticker.upper())
    return m.group(1) if m else ""


def resolve_index_symbol(underlying: str) -> str:
    sym = INDEX_SYMBOL.get(underlying.upper())
    if not sym:
        raise ValueError(f"unknown underlying: {underlying}")
    return sym


def list_expiries(underlying: str) -> list[date]:
    """All listed option expiries for ``underlying`` (sorted). For NIFTY/SENSEX
    this includes weeklies; for BANKNIFTY only monthlies are listed."""
    u = underlying.upper()
    exch = _EXCHANGE.get(u, "NSE")
    exps: set[date] = set()
    for cells in _load_master(exch):
        ticker = _find_ticker(cells)
        if not ticker or _ticker_root(ticker) != u:
            continue
        if _row_opt(cells) not in ("CE", "PE"):  # options only, not futures
            continue
        d = _row_expiry(cells)
        if d is not None:
            exps.add(d)
    return sorted(exps)


def nearest_expiry(underlying: str, on_or_after: date) -> date:
    """Nearest listed expiry on/after ``on_or_after`` — the latest weekly for
    NIFTY/SENSEX, the monthly for BANKNIFTY."""
    future = [e for e in list_expiries(underlying) if e >= on_or_after]
    if not future:
        raise LookupError(f"no listed {underlying} expiry on/after {on_or_after.isoformat()}")
    return future[0]


def resolve_option_symbol(underlying: str, expiry: date, strike: int, opt_type: str) -> str:
    """Return the Fyers tradeable ticker, e.g. ``NSE:NIFTY2570724800CE``.

    Raises ``LookupError`` if no master row matches.
    """
    u = underlying.upper()
    opt = opt_type.upper()
    exch = _EXCHANGE.get(u, "NSE")
    rows = _load_master(exch)
    strike_f = float(strike)
    matches: list[str] = []
    for cells in rows:
        ticker = _find_ticker(cells)
        if not ticker or _ticker_root(ticker) != u:
            continue
        if _row_opt(cells) != opt:
            continue
        s = _row_strike(cells)
        if s is None or abs(s - strike_f) >= 1e-6:
            continue
        if _row_expiry(cells) != expiry:
            continue
        matches.append(ticker)
    uniq = sorted(set(matches))
    if not uniq:
        raise LookupError(
            f"Fyers symbol not found for {u} {strike}{opt} exp {expiry.isoformat()} "
            f"(searched {len(rows)} {exch} rows — is the expiry a real contract?)"
        )
    if len(uniq) > 1:
        # Refuse to guess — never silently fetch/trade the wrong contract.
        raise LookupError(
            f"Ambiguous Fyers match for {u} {strike}{opt} exp {expiry.isoformat()}: "
            f"{uniq[:5]} — refusing to resolve"
        )
    return uniq[0]


# ── internals ──────────────────────────────────────────────────────────
def _find_ticker(cells: list[str]) -> str | None:
    # Pinned column first; heuristic scan only as a fallback for column drift.
    if len(cells) > _COL_TICKER:
        c = cells[_COL_TICKER].strip()
        if _TICKER_RE.match(c.upper()):
            return c
    for c in cells:
        if _TICKER_RE.match(c.strip().upper()):
            return c.strip()
    return None


def _row_strike(cells: list[str]) -> float | None:
    """The contract strike from the pinned column (None if unavailable)."""
    if len(cells) > _COL_STRIKE:
        try:
            return float(cells[_COL_STRIKE])
        except (TypeError, ValueError):
            return None
    return None


def _row_opt(cells: list[str]) -> str:
    """Pinned option type — CE / PE (XX for futures)."""
    return cells[_COL_OPT].strip().upper() if len(cells) > _COL_OPT else ""


def _row_expiry(cells: list[str]) -> date | None:
    """Parsed expiry from the pinned epoch column, bounded to a plausible range."""
    if len(cells) > _COL_EXPIRY_EPOCH:
        c = cells[_COL_EXPIRY_EPOCH].strip()
        if c.isdigit() and _EPOCH_MIN <= int(c) <= _EPOCH_MAX:
            try:
                return datetime.fromtimestamp(int(c), IST).date()
            except (ValueError, OSError, OverflowError):
                return None
    return None


# In-process memo of the parsed CSV rows, keyed by exchange + file mtime so a
# daily refresh (new file) invalidates it but repeat calls don't re-parse.
_MASTER_CACHE: dict[str, tuple[float, list[list[str]]]] = {}


def _load_master(exchange: str) -> list[list[str]]:
    path = _CACHE_DIR / f"{exchange}_FO.csv"
    if not path.exists() or (time.time() - path.stat().st_mtime) > _MAX_AGE_SECS:
        _download(exchange, path)
    mtime = path.stat().st_mtime
    cached = _MASTER_CACHE.get(exchange)
    if cached and cached[0] == mtime:
        return cached[1]
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    _MASTER_CACHE[exchange] = (mtime, rows)
    return rows


_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _download(exchange: str, path: Path) -> None:
    url = MASTER_URLS.get(exchange)
    if not url:
        raise ValueError(f"no symbol master for exchange {exchange}")
    with _LOCKS_GUARD:
        lock = _DOWNLOAD_LOCKS.setdefault(exchange, threading.Lock())
    with lock:
        # Another thread may have refreshed the file while we waited on the lock.
        if path.exists() and (time.time() - path.stat().st_mtime) <= _MAX_AGE_SECS:
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("fyers.symbols.download exchange=%s url=%s", exchange, url)
        # Unique temp per writer (pid + thread id) so concurrent downloads —
        # across threads OR processes — never interleave bytes into one file;
        # os.replace() then installs it atomically. Bounded timeout so a stalled
        # host raises instead of hanging the worker thread forever.
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
                with tmp.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
