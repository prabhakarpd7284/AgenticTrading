"""
v2 scanner — fetches the tier's candles (+ HTF + NIFTY for RS) and runs the
SwingEngineV2 across a universe. Isolated from the v1 OKScanner; uses its own
disk cache so the two never collide.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from logzero import logger

from plugins.strategy_swing.candle_utils import _aggregate_to_weekly
from plugins.strategy_swing.v2.config import SwingV2Config, SwingTier, get_tier_config
from plugins.strategy_swing.v2.engine import SwingEngineV2, SwingSignalV2, Action


CACHE_DIR = Path("/tmp/ok_scanner_v2_cache")
_GRADE_RANK = {"A": 0, "B": 1, "C": 2, "—": 3}


def _cache_path(symbol: str, interval: str, scan_date: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{interval}_{scan_date}.json"


def invalidate_cache(symbol: Optional[str] = None):
    if not CACHE_DIR.exists():
        return
    pattern = f"{symbol}_*.json" if symbol else "*.json"
    for f in CACHE_DIR.glob(pattern):
        f.unlink()


class SwingScannerV2:
    def __init__(
        self,
        tier: SwingTier | str = SwingTier.MEDIUM,
        cfg: Optional[SwingV2Config] = None,
        data_svc=None,
        capital: float = 500_000.0,
        use_cache: bool = True,
    ):
        self.cfg = cfg or get_tier_config(tier)
        self.engine = SwingEngineV2(self.cfg, capital=capital)
        self._data_svc = data_svc
        self.use_cache = use_cache
        self._benchmark_cache: Optional[List[dict]] = None
        self._results: List[SwingSignalV2] = []

    def _svc(self):
        if self._data_svc is None:
            from trading.services.data_service import DataService
            self._data_svc = DataService()
        return self._data_svc

    # ── fetching ──────────────────────────────────────────────────────

    def _fetch(self, symbol: str, interval: str, scan_date: str) -> List[dict]:
        if self.use_cache:
            p = _cache_path(symbol, interval, scan_date)
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except (json.JSONDecodeError, IOError):
                    pass
        frm = (datetime.strptime(scan_date, "%Y-%m-%d").date()
               - timedelta(days=self.cfg.lookback_days)).strftime("%Y-%m-%d")
        try:
            candles = self._svc().fetch_historical(symbol, frm, scan_date, interval=interval)
        except Exception as e:
            logger.error(f"v2 fetch failed for {symbol} {interval}: {e}")
            return []
        if candles and self.use_cache:
            try:
                _cache_path(symbol, interval, scan_date).write_text(json.dumps(candles))
            except IOError:
                pass
        return candles

    def _htf(self, symbol: str, primary: List[dict], scan_date: str) -> List[dict]:
        if self.cfg.htf_from_weekly_agg:
            adapted = [{"timestamp": c.get("date", c.get("timestamp", "")), **{
                k: c[k] for k in ("open", "high", "low", "close", "volume")}}
                for c in primary]
            return _aggregate_to_weekly(adapted)
        return self._fetch(symbol, self.cfg.htf_interval, scan_date)

    def _benchmark(self, scan_date: str) -> List[dict]:
        if self._benchmark_cache is None:
            self._benchmark_cache = self._fetch(self.cfg.benchmark, self.cfg.interval, scan_date) or []
            if not self._benchmark_cache:
                logger.warning(f"v2: no benchmark ({self.cfg.benchmark}) data — RS factor neutral")
        return self._benchmark_cache

    # ── scanning ──────────────────────────────────────────────────────

    def scan(self, symbols: List[str], scan_date: Optional[str] = None) -> List[SwingSignalV2]:
        scan_date = scan_date or date.today().strftime("%Y-%m-%d")
        benchmark = self._benchmark(scan_date)
        self._results = []
        start = time.time()
        logger.info(f"v2 scan [{self.cfg.tier.value}/{self.cfg.interval}]: {len(symbols)} symbols @ {scan_date}")

        for i, sym in enumerate(symbols):
            if (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{len(symbols)}")
            self._results.append(self._scan_one(sym, scan_date, benchmark))

        logger.info(f"v2 scan done in {time.time() - start:.1f}s")
        self._results.sort(key=lambda r: (
            not r.actionable,
            _GRADE_RANK.get(r.grade, 3),
            -r.score,
        ))
        return self._results

    def _scan_one(self, symbol: str, scan_date: str, benchmark: List[dict]) -> SwingSignalV2:
        primary = self._fetch(symbol, self.cfg.interval, scan_date)
        if not primary:
            sig = SwingSignalV2(symbol=symbol, tier=self.cfg.tier.value)
            sig.error = "no data"
            return sig
        htf = self._htf(symbol, primary, scan_date)
        return self.engine.analyze(symbol, primary, htf, benchmark)

    # ── filters ───────────────────────────────────────────────────────

    def actionable(self) -> List[SwingSignalV2]:
        return [r for r in self._results if r.actionable]

    def longs(self) -> List[SwingSignalV2]:
        return [r for r in self.actionable() if r.action in (Action.BUY, Action.ADD)]

    def shorts(self) -> List[SwingSignalV2]:
        return [r for r in self.actionable() if r.action == Action.SHORT]

    def summary(self) -> dict:
        out: dict = {}
        for r in self._results:
            out[r.phase.value] = out.get(r.phase.value, 0) + 1
        return out
