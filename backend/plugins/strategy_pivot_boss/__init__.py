"""Pivot Boss toolkit — Frank Ochoa, *Secrets of a Pivot Boss* (2010).

Pure, broker-free building blocks (spec: docs/strategies/PIVOT_BOSS_SPEC.md):

- ``levels``         floor pivots, CPR, Camarilla, developing levels, higher
                     time-frame pivots, Money Zone (volume/TPO profile), virgin levels
- ``relationships``  the 7 two-day band relationships + open-location classification
- ``triggers``       the 4 Appendix-B entry triggers + 1m → N-minute resampler
- ``backtest``       a small deterministic intraday backtester (1m fills, NSE costs)
- ``setups``         the book's setups expressed as per-stock-day signal generators
- ``research``       per-stock-day statistics used by ``manage.py pivot_research``

Nothing here touches the DB, a broker or an LLM. Not registered as a strategy
entry-point yet (research only).
"""
