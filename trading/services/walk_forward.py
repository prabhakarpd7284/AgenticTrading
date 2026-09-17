"""
Walk-Forward Backtester — prevents overfitting.

Instead of optimizing on the same data you test on (which overfits),
this splits historical data into rolling train/test windows:

  Window 1: Train on days 1-10, test on days 11-15
  Window 2: Train on days 6-15, test on days 16-20
  Window 3: Train on days 11-20, test on days 21-25

The test results are the TRUE out-of-sample performance.
If train performance >> test performance, you're overfitting.

Usage:
    from trading.services.walk_forward import walk_forward_test
    results = walk_forward_test(symbols, '2026-03-01', '2026-03-21')
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from logzero import logger


@dataclass
class WalkForwardResult:
    """Results from one walk-forward window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_trades: int = 0
    train_win_rate: float = 0
    train_pnl: float = 0
    test_trades: int = 0
    test_win_rate: float = 0
    test_pnl: float = 0
    best_params: dict = field(default_factory=dict)


@dataclass
class WalkForwardSummary:
    """Aggregated walk-forward results."""
    windows: List[WalkForwardResult] = field(default_factory=list)
    total_test_trades: int = 0
    total_test_pnl: float = 0
    avg_test_win_rate: float = 0
    overfit_ratio: float = 0  # train_pnl / test_pnl — >2.0 = overfitting


def walk_forward_test(
    symbols: list,
    start_date: str,
    end_date: str,
    train_days: int = 10,
    test_days: int = 5,
    step_days: int = 5,
) -> WalkForwardSummary:
    """
    Run walk-forward optimization.

    Args:
        symbols: Stock symbols to test
        start_date: Overall start date (YYYY-MM-DD)
        end_date: Overall end date (YYYY-MM-DD)
        train_days: Days in each training window
        test_days: Days in each test window
        step_days: How far to slide the window each iteration

    Returns:
        WalkForwardSummary with per-window and aggregate results.
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate trading days (skip weekends)
    all_days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            all_days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if len(all_days) < train_days + test_days:
        logger.warning(f"Not enough trading days ({len(all_days)}) for walk-forward "
                       f"(need {train_days + test_days})")
        return WalkForwardSummary()

    summary = WalkForwardSummary()

    # Slide the window
    i = 0
    while i + train_days + test_days <= len(all_days):
        train_period = all_days[i:i + train_days]
        test_period = all_days[i + train_days:i + train_days + test_days]

        logger.info(f"Walk-forward window: train {train_period[0]}→{train_period[-1]}, "
                    f"test {test_period[0]}→{test_period[-1]}")

        # Train: find best parameters
        train_result = _run_backtest_period(symbols, train_period[0], train_period[-1])

        # Test: evaluate with those parameters on unseen data
        test_result = _run_backtest_period(symbols, test_period[0], test_period[-1])

        window = WalkForwardResult(
            train_start=train_period[0], train_end=train_period[-1],
            test_start=test_period[0], test_end=test_period[-1],
            train_trades=train_result.get("trades", 0),
            train_win_rate=train_result.get("win_rate", 0),
            train_pnl=train_result.get("pnl", 0),
            test_trades=test_result.get("trades", 0),
            test_win_rate=test_result.get("win_rate", 0),
            test_pnl=test_result.get("pnl", 0),
        )
        summary.windows.append(window)

        i += step_days

    # Aggregate
    if summary.windows:
        summary.total_test_trades = sum(w.test_trades for w in summary.windows)
        summary.total_test_pnl = sum(w.test_pnl for w in summary.windows)
        test_wins = sum(w.test_win_rate * w.test_trades for w in summary.windows)
        if summary.total_test_trades > 0:
            summary.avg_test_win_rate = test_wins / summary.total_test_trades

        total_train_pnl = sum(w.train_pnl for w in summary.windows)
        if summary.total_test_pnl != 0:
            summary.overfit_ratio = abs(total_train_pnl / summary.total_test_pnl)
        else:
            summary.overfit_ratio = float('inf') if total_train_pnl > 0 else 0

    return summary


def _run_backtest_period(symbols: list, start: str, end: str) -> dict:
    """
    Run the level bounce backtest on a date range.
    Returns {trades, wins, pnl, win_rate}.
    """
    try:
        from trading.services.intraday_backtester import run_intraday_backtest, BacktestConfig
        config = BacktestConfig(
            min_confidence=0.55,
            mtf_enabled=False,
            min_rr=1.5,
        )
        result = run_intraday_backtest(symbols, start, end, config)
        s = result.get("summary", {})
        return {
            "trades": s.get("total_trades", 0),
            "wins": s.get("wins", 0),
            "pnl": s.get("total_pnl", 0),
            "win_rate": s.get("win_rate", 0),
        }
    except Exception as e:
        logger.error(f"Backtest failed for {start}→{end}: {e}")
        return {"trades": 0, "wins": 0, "pnl": 0, "win_rate": 0}
