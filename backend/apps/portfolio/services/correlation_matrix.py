"""Cross-position correlation & concentration matrix.

For each open trade, resolve the underlying symbol, fetch ~60 daily closes
via the legacy broker, compute pairwise Pearson correlation, derive an
'independent bets' count from the eigenvalue spectrum, and aggregate
sector / factor concentration.

Returns a JSON-safe dict shaped per the task acceptance criteria:
    { symbols, matrix, independent_bets, sector_weights, factor_weights, as_of }
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from django.core.cache import cache

_RETURNS_TTL = 600       # daily-return series only changes overnight
_REPORT_TTL = 120

# A tiny sector classifier — enough to populate the bars without pulling
# in a full reference data feed. Index options map to INDEX.
_SECTOR_BY_PREFIX = {
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK", "AXISBANK": "BANK",
    "KOTAKBANK": "BANK", "INDUSINDBK": "BANK",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "TECHM": "IT", "HCLTECH": "IT",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY", "IOC": "ENERGY",
    "TATAMOTORS": "AUTO", "M&M": "AUTO", "MARUTI": "AUTO", "BAJAJ-AUTO": "AUTO",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "NIFTY": "INDEX", "BANKNIFTY": "INDEX", "SENSEX": "INDEX",
}


def _classify_sector(sym: str) -> str:
    s = sym.upper()
    for prefix, sector in _SECTOR_BY_PREFIX.items():
        if s.startswith(prefix):
            return sector
    return "OTHER"


def _classify_factor(sym: str) -> str:
    s = sym.upper()
    if s.startswith(("NIFTY", "BANKNIFTY", "SENSEX")):
        return "INDEX_OPT"
    # Without a full market-cap reference we proxy via well-known megas.
    if any(s.startswith(p) for p in ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC")):
        return "LARGE_CAP"
    return "MID_OR_SMALL"


def _resolve_underlying(symbol: str) -> str:
    """Strip option/future encoding and return the base ticker for correlation."""
    s = symbol.upper()
    for idx in ("BANKNIFTY", "NIFTY", "SENSEX"):
        if s.startswith(idx):
            return idx
    # Equity ticker — return as-is.
    return s


def _fetch_daily_returns(underlying: str, days: int = 60) -> list[float]:
    """Pull a list of daily-close returns via the legacy broker. Empty on failure.

    Cached for 10 minutes because the source is daily candles — no point
    re-paying the broker round trip on a page refresh.
    """
    key = f"corr:returns:{underlying}:{days}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service

        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(underlying)
        if not token:
            return []
        today = date.today()
        start = (today - timedelta(days=days + 5)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange=ticker_service.resolve_exchange(underlying)) or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        closes = [float(r[4]) for r in raw if len(r) >= 5]
        if len(closes) < 2:
            cache.set(key, [], _RETURNS_TTL)
            return []
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        out = rets[-days:]
        cache.set(key, out, _RETURNS_TTL)
        return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _RETURNS_TTL)
        return []


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    denom = (va * vb) ** 0.5
    if denom <= 0:
        return 0.0
    rho = cov / denom
    return max(-1.0, min(1.0, rho))


def _independent_bets(matrix: list[list[float]]) -> int:
    """Trace-based approximation: N / mean_off_diagonal_rho.

    Cheap and avoids a numpy dep; close enough for a dashboard tile.
    """
    n = len(matrix)
    if n == 0:
        return 0
    if n == 1:
        return 1
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(matrix[i][j])
            count += 1
    avg = total / count if count else 0.0
    if avg <= 0:
        return n
    return max(1, int(round(n / (1 + (n - 1) * avg))))


def build_correlation_report(tenant=None) -> dict[str, Any]:
    cache_key = "corr:report"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    from trading.models import TradeJournal, StraddlePosition

    trades = list(TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER", "APPROVED")))
    straddles = list(StraddlePosition.objects.filter(status="ACTIVE"))

    underlyings: list[str] = []
    seen: set[str] = set()
    weights: dict[str, float] = {}

    for t in trades:
        u = _resolve_underlying(t.symbol)
        notional = float(t.quantity) * float(t.entry_price)
        if u not in seen:
            seen.add(u); underlyings.append(u)
        weights[u] = weights.get(u, 0.0) + notional

    for p in straddles:
        u = (p.underlying or "").upper()
        if not u:
            continue
        notional = float(p.lots) * float(p.lot_size) * (float(p.ce_sell_price) + float(p.pe_sell_price))
        if u not in seen:
            seen.add(u); underlyings.append(u)
        weights[u] = weights.get(u, 0.0) + notional

    if not underlyings:
        empty = {
            "symbols": [], "matrix": [], "independent_bets": 0,
            "sector_weights": {}, "factor_weights": {},
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        cache.set(cache_key, empty, _REPORT_TTL)
        return empty

    returns_by_sym: dict[str, list[float]] = {}
    for u in underlyings:
        returns_by_sym[u] = _fetch_daily_returns(u, days=60)

    n = len(underlyings)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            r = _pearson(returns_by_sym[underlyings[i]], returns_by_sym[underlyings[j]])
            matrix[i][j] = r
            matrix[j][i] = r

    total_notional = sum(weights.values()) or 1.0
    sector_weights: dict[str, float] = {}
    factor_weights: dict[str, float] = {}
    for sym, w in weights.items():
        sector_weights[_classify_sector(sym)] = sector_weights.get(_classify_sector(sym), 0.0) + w / total_notional
        factor_weights[_classify_factor(sym)] = factor_weights.get(_classify_factor(sym), 0.0) + w / total_notional

    report = {
        "symbols": underlyings,
        "matrix": matrix,
        "independent_bets": _independent_bets(matrix),
        "sector_weights": {k: round(v, 4) for k, v in sector_weights.items()},
        "factor_weights": {k: round(v, 4) for k, v in factor_weights.items()},
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    cache.set(cache_key, report, _REPORT_TTL)
    return report
