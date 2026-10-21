"""Strategy-search research harness (docs/strategies/STRATEGY_SEARCH_RESULTS.md).

Research only — never imported by live trading code. Reads candles from the DB
(``market_data.Candle``), never calls a broker.

Modules:
    periods           IS / OOS split + the OOS guard
    data              DB candle source (one symbol at a time), sessions, resample
    costs             NSE intraday (MIS) cost model
    rules             owner rules: breakeven→ATR trail, losing-streak cooldown
    sim               intents → per-trade paths → portfolio simulation + metrics
    screener_adapter  replays the live screener's enabled strategies on DB candles
    runner            glue: strategy → candidates → portfolio sim → files
"""
