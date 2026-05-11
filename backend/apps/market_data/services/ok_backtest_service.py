"""OK Backtest service — wraps both daily and intraday backtests for the API.

Returns results synchronously. Daily backtest ~25s, intraday ~90s.
Cached 3600s (results don't change within a day for same params).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from django.core.cache import cache

logger = logging.getLogger(__name__)

CACHE_TTL = 3600


@dataclass
class OKBacktestPayload:
    """API response for OK backtest."""
    as_of: str
    mode: str               # "daily" or "intraday"
    from_date: str
    to_date: str
    capital: float
    symbols_count: int

    # Summary
    total_trades: int = 0
    winners: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_bars_held: float = 0.0

    # Universe info
    scanned_universe: int = 0       # How many stocks the OK scanner checked
    active_phases: int = 0          # How many had active phases (= backtest universe)
    universe_symbols: list[str] = field(default_factory=list)

    # Breakdown
    phase_stats: dict[str, Any] = field(default_factory=dict)
    weekly_pnl: dict[str, float] = field(default_factory=dict)

    # Trades list
    trades: list[dict[str, Any]] = field(default_factory=list)

    # Equity curve (for charting)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    # Intraday multi-TF grid results (only for mode=intraday)
    tf_grid: list[dict[str, Any]] = field(default_factory=list)
    best_config: dict[str, Any] = field(default_factory=dict)

    errors: list[str] = field(default_factory=list)


def _cache_key(mode: str, from_date: str, to_date: str, symbols_hash: str) -> str:
    return f"ok_backtest:{mode}:{from_date}:{to_date}:{symbols_hash}"


UNIVERSE_CACHE_KEY = "ok_backtest:smart_universe:v1"
UNIVERSE_CACHE_TTL = 3600  # 1 hour — phases don't change intra-hour


def _smart_universe(mode: str, scan_date: str | None = None) -> list[str]:
    """Use OK cycle scanner to find stocks in active phases — smart universe.

    Cached 1 hour. Scans full NIFTY 100, returns only stocks with active phases.
    Falls back to NIFTY 50 if scanner fails.
    """
    # Check cache first
    cached = cache.get(UNIVERSE_CACHE_KEY)
    if cached is not None:
        _smart_universe._last_scan = cached["scan"]
        logger.info(f"Smart universe: cached — {cached['scan']['active']} active phases")
        return cached["symbols"]

    try:
        from apps.market_data.constants import SCREENER_UNIVERSE
        from trading.swing.ok_scanner import OKScanner
        from trading.swing.ok_cycles import CyclePhase

        logger.info("Smart universe: scanning NIFTY 100 for active OK phases...")
        scanner = OKScanner()
        results = scanner.scan(list(SCREENER_UNIVERSE), scan_date=scan_date)

        # Stocks with any active cycle phase
        active = [r.symbol for r in results if r.phase != CyclePhase.NONE]

        # Cache the scan stats for the payload
        _smart_universe._last_scan = {
            "scanned": len(SCREENER_UNIVERSE),
            "active": len(active),
            "symbols": active,
        }

        if active:
            logger.info(f"Smart universe: {len(active)} stocks with active phases "
                        f"(from {len(SCREENER_UNIVERSE)} scanned)")
            cache.set(UNIVERSE_CACHE_KEY, {
                "symbols": active,
                "scan": _smart_universe._last_scan,
            }, UNIVERSE_CACHE_TTL)
            return active

        logger.warning("Smart universe: no active phases, falling back to NIFTY 50")
    except Exception as e:
        logger.warning(f"Smart universe scan failed: {e}, falling back to NIFTY 50")

    try:
        from apps.market_data.constants import NIFTY_50_SYMBOLS
        fallback = list(NIFTY_50_SYMBOLS)
        _smart_universe._last_scan = {
            "scanned": 0, "active": 0, "symbols": fallback,
        }
        return fallback
    except Exception:
        return []

_smart_universe._last_scan = {"scanned": 0, "active": 0, "symbols": []}


def build_ok_backtest(
    mode: str = "daily",
    from_date: str = None,
    to_date: str = None,
    symbols: list[str] = None,
    capital: float = None,
    force: bool = False,
) -> OKBacktestPayload:
    """Run OK backtest and return structured payload.

    Args:
        mode: "daily" for swing backtest, "intraday" for multi-TF grid
        from_date: Start date (default: 30 days ago)
        to_date: End date (default: today)
        symbols: NSE symbols (default: NIFTY 50)
        capital: Starting capital (default: 500000)
        force: Bypass cache
    """
    today = date.today()
    from_date = from_date or (today - __import__("datetime").timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = to_date or today.strftime("%Y-%m-%d")
    capital = capital or 500000.0

    if symbols is None:
        symbols = _smart_universe(mode, to_date)

    sym_hash = hashlib.md5(",".join(sorted(symbols)).encode()).hexdigest()[:8]
    ck = _cache_key(mode, from_date, to_date, sym_hash)

    if not force:
        cached = cache.get(ck)
        if cached is not None:
            return cached

    if mode == "intraday":
        payload = _run_intraday(symbols, from_date, to_date, capital)
    elif mode == "basket":
        payload = _run_basket(symbols, from_date, to_date, capital)
    else:
        payload = _run_daily(symbols, from_date, to_date, capital)

    # Attach smart universe stats
    scan_info = getattr(_smart_universe, "_last_scan", {})
    payload.scanned_universe = scan_info.get("scanned", 0)
    payload.active_phases = scan_info.get("active", 0)
    payload.universe_symbols = scan_info.get("symbols", [])

    if not force:
        cache.set(ck, payload, CACHE_TTL)

    return payload


def _stats_to_payload(
    stats, mode: str, from_date: str, to_date: str, capital: float,
    symbols_count: int, tf_grid: list = None, best_config: dict = None,
) -> OKBacktestPayload:
    """Convert BacktestStats to OKBacktestPayload."""
    phase_stats = {k: {"trades": v.trades, "win_rate": v.win_rate, "pnl": v.pnl}
                   for k, v in stats.per_phase.items()}

    trades = []
    for t in getattr(stats, "_trades", []):
        trades.append(t.to_dict())

    return OKBacktestPayload(
        as_of=datetime.now(timezone.utc).isoformat(),
        mode=mode, from_date=from_date, to_date=to_date,
        capital=capital, symbols_count=symbols_count,
        total_trades=stats.total_trades, winners=stats.winners,
        win_rate=stats.win_rate, total_pnl=stats.total_pnl,
        total_pnl_pct=stats.total_pnl_pct,
        profit_factor=stats.profit_factor, avg_rr=stats.avg_rr,
        max_drawdown=stats.max_drawdown,
        max_drawdown_pct=stats.max_drawdown_pct,
        avg_win=stats.avg_win, avg_loss=stats.avg_loss,
        best_trade=stats.best_trade, worst_trade=stats.worst_trade,
        avg_bars_held=stats.avg_bars_held,
        phase_stats=phase_stats,
        weekly_pnl=stats.weekly_pnl,
        equity_curve=stats.equity_curve,
        tf_grid=tf_grid or [],
        best_config=best_config or {},
    )


def _run_daily(
    symbols: list[str], from_date: str, to_date: str, capital: float
) -> OKBacktestPayload:
    """Run daily swing backtest using unified BacktestEngine."""
    try:
        from trading.backtester.compat import run_ok_backtest
    except Exception as e:
        return OKBacktestPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            mode="daily", from_date=from_date, to_date=to_date,
            capital=capital, symbols_count=len(symbols),
            errors=[f"Import error: {e}"],
        )

    try:
        stats = run_ok_backtest(symbols, from_date, to_date, capital=capital)
    except Exception as e:
        logger.exception("Daily backtest failed: %s", e)
        return OKBacktestPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            mode="daily", from_date=from_date, to_date=to_date,
            capital=capital, symbols_count=len(symbols),
            errors=[f"Backtest error: {e}"],
        )

    return _stats_to_payload(stats, "daily", from_date, to_date, capital, len(symbols))


def _run_intraday(
    symbols: list[str], from_date: str, to_date: str, capital: float
) -> OKBacktestPayload:
    """Run intraday multi-TF grid backtest using unified BacktestEngine."""
    try:
        from trading.backtester.compat import run_intraday_backtest
    except Exception as e:
        return OKBacktestPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            mode="intraday", from_date=from_date, to_date=to_date,
            capital=capital, symbols_count=len(symbols),
            errors=[f"Import error: {e}"],
        )

    try:
        results = run_intraday_backtest(symbols, from_date, to_date, capital=capital)
    except Exception as e:
        logger.exception("Intraday backtest failed: %s", e)
        return OKBacktestPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            mode="intraday", from_date=from_date, to_date=to_date,
            capital=capital, symbols_count=len(symbols),
            errors=[f"Backtest error: {e}"],
        )

    # Build grid from new engine results
    tf_grid = []
    for r in results:
        s = r["stats"]
        if s.total_trades == 0:
            continue
        phase_stats_dict = {k: {"trades": v.trades, "win_rate": v.win_rate, "pnl": v.pnl}
                           for k, v in s.per_phase.items()}
        tf_grid.append({
            "tf": r["tf"], "sl_atr": r["sl_atr"], "rr": r["rr"],
            "trades": s.total_trades, "win_rate": s.win_rate,
            "pf": s.profit_factor, "pnl": s.total_pnl,
            "pnl_pct": s.total_pnl_pct, "max_dd": s.max_drawdown,
            "avg_bars": s.avg_bars_held, "avg_win": s.avg_win,
            "avg_loss": s.avg_loss, "phase_stats": phase_stats_dict,
        })
    tf_grid.sort(key=lambda x: -x["pf"])

    # Find best config (min 5 trades)
    valid = [r for r in results if r["stats"].total_trades >= 5]
    best_r = max(valid, key=lambda r: r["stats"].profit_factor) if valid else None

    best_config = {}
    if best_r:
        bs = best_r["stats"]
        best_config = {
            "tf": best_r["tf"], "sl_atr": best_r["sl_atr"], "rr": best_r["rr"],
            "trades": bs.total_trades, "win_rate": bs.win_rate,
            "pf": bs.profit_factor, "pnl": bs.total_pnl,
        }
        return _stats_to_payload(
            bs, "intraday", from_date, to_date, capital, len(symbols),
            tf_grid=tf_grid, best_config=best_config,
        )

    return OKBacktestPayload(
        as_of=datetime.now(timezone.utc).isoformat(),
        mode="intraday", from_date=from_date, to_date=to_date,
        capital=capital, symbols_count=len(symbols),
        tf_grid=tf_grid, best_config=best_config,
    )


def _run_basket(
    symbols: list[str], from_date: str, to_date: str, capital: float
) -> OKBacktestPayload:
    """Run morning basket backtest using unified BacktestEngine."""
    try:
        from trading.backtester.compat import run_basket_backtest
    except Exception as e:
        return OKBacktestPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            mode="basket", from_date=from_date, to_date=to_date,
            capital=capital, symbols_count=len(symbols),
            errors=[f"Import error: {e}"],
        )

    try:
        stats = run_basket_backtest(symbols, from_date, to_date, capital=capital)
    except Exception as e:
        logger.exception("Basket backtest failed: %s", e)
        return OKBacktestPayload(
            as_of=datetime.now(timezone.utc).isoformat(),
            mode="basket", from_date=from_date, to_date=to_date,
            capital=capital, symbols_count=len(symbols),
            errors=[f"Backtest error: {e}"],
        )

    return _stats_to_payload(stats, "basket", from_date, to_date, capital, len(symbols))
