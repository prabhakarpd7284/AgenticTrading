"""
Oliver Kell Cycle Scanner — batch scan stocks for cycle phases.

Fetches daily candles, aggregates to weekly, runs CycleDetector
on each symbol, and returns ranked results.
"""
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from logzero import logger

from dashboard_utils.candle_cache import _aggregate_to_weekly
from trading.config import OKCycleConfig, config as default_config
from trading.swing.ok_cycles import (
    BULLISH_ACTIONABLE,
    BEARISH_ACTIONABLE,
    CycleDetector,
    CyclePhase,
    CycleResult,
    TrendState,
)


# ══════════════════════════════════════════════
# Disk cache
# ══════════════════════════════════════════════

CACHE_DIR = Path("/tmp/ok_scanner_cache")


def _cache_key(symbol: str, scan_date: str) -> Path:
    """Cache file path for a symbol's daily candles."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{scan_date}.json"


def _load_cache(symbol: str, scan_date: str) -> Optional[List[dict]]:
    """Load cached daily candles if fresh (same trading day)."""
    path = _cache_key(symbol, scan_date)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_cache(symbol: str, scan_date: str, candles: List[dict]):
    """Save daily candles to disk cache."""
    path = _cache_key(symbol, scan_date)
    try:
        path.write_text(json.dumps(candles))
    except IOError as e:
        logger.warning(f"Cache write failed for {symbol}: {e}")


def invalidate_cache(symbol: str = None):
    """Clear cache for a symbol or all symbols."""
    if not CACHE_DIR.exists():
        return
    if symbol:
        for f in CACHE_DIR.glob(f"{symbol}_*.json"):
            f.unlink()
    else:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()


# ══════════════════════════════════════════════
# Scanner
# ══════════════════════════════════════════════

class OKScanner:
    """
    Batch scanner for Oliver Kell cycle phases.

    Usage:
        from trading.services.data_service import DataService
        scanner = OKScanner(DataService())
        results = scanner.scan(["RELIANCE", "TCS", "HDFCBANK"])
        actionable = scanner.get_actionable()
    """

    def __init__(
        self,
        data_svc=None,
        cfg: Optional[OKCycleConfig] = None,
    ):
        self.cfg = cfg or default_config.ok_cycle
        self._data_svc = data_svc
        self._detector = CycleDetector(self.cfg)
        self._results: List[CycleResult] = []

    def _ensure_data_svc(self):
        """Lazy-init DataService to avoid import-time broker auth."""
        if self._data_svc is None:
            from trading.services.data_service import DataService
            self._data_svc = DataService()

    # ──────────────────────────────────────────
    # Data fetching
    # ──────────────────────────────────────────

    def _fetch_daily(self, symbol: str, scan_date: str) -> List[dict]:
        """Fetch daily candles with caching. Returns list of OHLCV dicts."""
        # Check cache first
        cached = _load_cache(symbol, scan_date)
        if cached:
            return cached

        self._ensure_data_svc()

        # Lookback: enough days for EMA50 to stabilize
        to_date = scan_date
        from_dt = datetime.strptime(scan_date, "%Y-%m-%d").date() - timedelta(days=self.cfg.lookback_days)
        from_date = from_dt.strftime("%Y-%m-%d")

        try:
            candles = self._data_svc.fetch_historical(
                symbol, from_date, to_date, interval="ONE_DAY"
            )
        except Exception as e:
            logger.error(f"Failed to fetch daily candles for {symbol}: {e}")
            return []

        if candles:
            _save_cache(symbol, scan_date, candles)

        return candles

    def _daily_to_weekly(self, daily_candles: List[dict]) -> List[dict]:
        """Aggregate daily candles to weekly bars.

        Adapts the daily candle format (uses 'date' key) to match
        what _aggregate_to_weekly expects ('timestamp' key).
        """
        # _aggregate_to_weekly expects 'timestamp' key
        adapted = []
        for c in daily_candles:
            adapted.append({
                "timestamp": c.get("date", c.get("timestamp", "")),
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c["volume"],
            })
        return _aggregate_to_weekly(adapted)

    # ──────────────────────────────────────────
    # Scanning
    # ──────────────────────────────────────────

    def scan(
        self,
        symbols: List[str],
        scan_date: str = None,
    ) -> List[CycleResult]:
        """
        Scan all symbols for Oliver Kell cycle phases.

        Args:
            symbols: List of NSE symbols
            scan_date: Date to scan (default: today). Format: YYYY-MM-DD

        Returns:
            List of CycleResult, sorted: actionable phases first, then by confidence desc
        """
        if scan_date is None:
            scan_date = date.today().strftime("%Y-%m-%d")

        self._results = []
        total = len(symbols)

        logger.info(f"OK Scanner: scanning {total} symbols for date {scan_date}")
        start_time = time.time()

        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{total}")

            result = self._scan_one(symbol, scan_date)
            self._results.append(result)

        elapsed = time.time() - start_time
        logger.info(f"OK Scanner: completed {total} symbols in {elapsed:.1f}s")

        # Sort: actionable first, then by confidence descending
        self._results.sort(
            key=lambda r: (
                r.phase not in BULLISH_ACTIONABLE and r.phase not in BEARISH_ACTIONABLE,
                not r.aligned,
                -r.confidence,
            )
        )

        return self._results

    def _scan_one(self, symbol: str, scan_date: str) -> CycleResult:
        """Scan a single symbol."""
        daily = self._fetch_daily(symbol, scan_date)
        if not daily:
            return CycleResult(
                symbol=symbol,
                phase=CyclePhase.NONE,
                trend_daily=TrendState.NEUTRAL,
                trend_weekly=TrendState.NEUTRAL,
                aligned=False,
                action="—",
                confidence=0.0,
                error="No daily data",
            )

        weekly = self._daily_to_weekly(daily)
        return self._detector.analyze(symbol, daily, weekly)

    # ──────────────────────────────────────────
    # Filtering
    # ──────────────────────────────────────────

    def get_actionable(self) -> List[CycleResult]:
        """Return only actionable results (BUY/SHORT phases with aligned trends)."""
        return [
            r for r in self._results
            if (r.phase in BULLISH_ACTIONABLE or r.phase in BEARISH_ACTIONABLE)
            and r.aligned
        ]

    def get_watchlist(self) -> List[str]:
        """Return symbol names of actionable stocks (for intraday filtering)."""
        return [r.symbol for r in self.get_actionable()]

    def get_by_phase(self, phase: CyclePhase) -> List[CycleResult]:
        """Filter results by specific phase."""
        return [r for r in self._results if r.phase == phase]

    def get_bullish(self) -> List[CycleResult]:
        """All stocks in bullish cycle phases."""
        return [r for r in self._results if r.phase in BULLISH_ACTIONABLE]

    def get_bearish(self) -> List[CycleResult]:
        """All stocks in bearish cycle phases."""
        return [r for r in self._results if r.phase in BEARISH_ACTIONABLE]

    def summary(self) -> Dict[str, int]:
        """Phase distribution summary."""
        counts: Dict[str, int] = {}
        for r in self._results:
            key = r.phase.value
            counts[key] = counts.get(key, 0) + 1
        return counts
