"""Fyers symbol-master resolution for option contracts.

Hand-formatting Fyers *weekly* option strings is bug-prone (the community has
documented `-300 Invalid symbol` errors for the weekly encoding). The robust
path is to resolve the exact tradeable ticker from Fyers' published symbol-master
CSVs, matching by (underlying, expiry, strike, CE/PE).

The CSVs are public (no auth) and refreshed daily; we disk-cache them. Column
layout is matched heuristically (ticker via regex, strike/opt-type/expiry by
value) so the resolver survives minor column-order drift in the master file.
"""
from __future__ import annotations

import csv
import os
import re
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

from .adapter import logger
from plugins.strategy_scalp.timeutil import IST

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
        for c in cells:
            c = c.strip()
            # bound to plausible epoch range (2020-09 .. 2033-05) so option
            # expiry (col ~8) is picked but the 15-digit fytoken/scrip cols aren't.
            if c.isdigit() and 1_600_000_000 <= int(c) <= 2_000_000_000:
                try:
                    exps.add(datetime.fromtimestamp(int(c), IST).date())
                except (ValueError, OSError, OverflowError):
                    continue
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
    for cells in rows:
        ticker = _find_ticker(cells)
        if not ticker or not ticker.upper().endswith(opt):
            continue
        if _ticker_root(ticker) != u:
            continue
        if not _row_has_strike(cells, strike_f):
            continue
        if not _row_has_expiry(cells, expiry):
            continue
        return ticker
    raise LookupError(
        f"Fyers symbol not found for {u} {strike}{opt} exp {expiry.isoformat()} "
        f"(searched {len(rows)} {exch} rows — is the expiry a real contract?)"
    )


# ── internals ──────────────────────────────────────────────────────────
def _find_ticker(cells: list[str]) -> str | None:
    for c in cells:
        if _TICKER_RE.match(c.strip().upper()):
            return c.strip()
    return None


def _row_has_strike(cells: list[str], strike: float) -> bool:
    for c in cells:
        try:
            if abs(float(c) - strike) < 1e-6:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _row_has_expiry(cells: list[str], expiry: date) -> bool:
    for c in cells:
        c = c.strip()
        if not c.isdigit() or len(c) < 10:
            continue
        try:
            d = datetime.fromtimestamp(int(c), IST).date()
        except (ValueError, OSError, OverflowError):
            continue
        if d == expiry:
            return True
    return False


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


def _download(exchange: str, path: Path) -> None:
    url = MASTER_URLS.get(exchange)
    if not url:
        raise ValueError(f"no symbol master for exchange {exchange}")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("fyers.symbols.download exchange=%s url=%s", exchange, url)
    tmp = path.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed https Fyers host
    tmp.replace(path)
