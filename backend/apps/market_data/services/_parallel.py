"""Parallel per-symbol fanout for universe-scoped cockpit panels.

The cockpit services that iterate the watchlist (orb, first-5min,
orb-failure, second-5min, gap-fill, stop-hunt, depth-imbalance,
intraday-sector-heatmap) used to fetch candles serially — 30 symbols ×
1 broker call each = 12-42s cold. With ThreadPoolExecutor at 8 workers
each drops to 2-5s.

Snapshot the intraday as-of override from the calling thread, then
re-apply it inside each worker. We can't use copy_context() + ctx.run()
here because the caller is already inside the IntradayAsOfMiddleware's
context manager — re-entering the same context raises RuntimeError.
Explicit snapshot is simpler and correct.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def parallel_symbols(
    symbols: Iterable[str],
    fn: Callable[[str], T],
    max_workers: int = 8,
) -> list[T]:
    """Map ``fn`` over ``symbols`` concurrently, preserving cockpit
    time-travel into worker threads. Order matches input order."""
    from trading.utils.time_utils import _AS_OF_OVERRIDE, use_session_date

    syms = list(symbols)
    if not syms:
        return []
    as_of = _AS_OF_OVERRIDE.get()  # snapshot before fanout

    def _worker(s: str) -> T:
        with use_session_date(as_of):  # re-apply in this thread
            return fn(s)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_worker, syms))
