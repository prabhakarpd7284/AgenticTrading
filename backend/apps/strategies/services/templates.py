"""Built-in strategy templates for the Studio (UX plan §6: "built-in strategies
appear as templates with a real parameter form").

Two families:
  * rule templates — ready-made StrategySpecs (kind=spec), fully editable in
    the rule builder and backtested by the spec engine;
  * plugin templates — the registered strategy plugins (kind=plugin) with the
    params form rendered from their own ``schema()``. Each says honestly
    whether a deterministic backtest exists:
      ``backtest``           real deterministic backtest available
      ``forward_paper_only`` decisions come from an LLM — a backtest would be
                             fake; it can only be judged by forward paper trading
      ``unavailable``        no data / no adapter (reason given)
"""
from __future__ import annotations

import copy

from apps.strategies.services.spec import default_spec

# Screener strategies that never fire in a daily-bootstrapped engine
# (STRATEGY_SEARCH_RESULTS.md, "Two strategies never fire"): their 15m
# indicators can't warm up within one session.
BROKEN_SCREENER_STRATEGIES = {
    "BB Squeeze Breakout": "needs 20 bars of 15m for Bollinger Bands; the 15m buffer starts empty "
                           "each morning, so it is ready at 14:15 — after its 14:00 window closes",
    "EMA Crossover Trend": "needs 35 bars of 15m for MACD; a session has only 25",
}


def _spec(name, description, timeframe, long=None, short=None, **over):
    s = default_spec(name)
    s["meta"]["description"] = description
    s["timeframe"] = timeframe
    s["entry"]["long"] = long
    s["entry"]["short"] = short
    for path, val in over.items():
        cur = s
        keys = path.split("__")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = val
    return s


def _cmp(left, op, right):
    return {"left": left, "op": op, "right": right}


CLOSE = {"price": "close"}

RULE_TEMPLATES = [
    {
        "key": "rule.orb_breakout",
        "title": "Opening-range breakout (5m)",
        "summary": "Buy a 5-minute close above the first 15 minutes' high; short a close below its low.",
        "spec": _spec(
            "Opening-range breakout", "Classic ORB on the 09:15–09:30 range, 2R target, owner trail.",
            "5m",
            long={"all": [_cmp(CLOSE, "crosses_above", {"ind": "orb_high", "minutes": 15}),
                          _cmp(CLOSE, ">", {"ind": "vwap"})]},
            short={"all": [_cmp(CLOSE, "crosses_below", {"ind": "orb_low", "minutes": 15}),
                           _cmp(CLOSE, "<", {"ind": "vwap"})]},
            entry__time_from="09:35", entry__time_to="13:00",
            limits__max_trades_per_day=6),
    },
    {
        "key": "rule.ema_pullback",
        "title": "EMA trend pullback (15m)",
        "summary": "In an EMA(20) > EMA(50) uptrend, buy when price closes back above EMA(20); mirror for shorts.",
        "spec": _spec(
            "EMA trend pullback", "Trend filter EMA20/EMA50, re-entry on a close back through EMA20.",
            "15m",
            long={"all": [_cmp({"ind": "ema", "source": "close", "period": 20}, ">",
                               {"ind": "ema", "source": "close", "period": 50}),
                          _cmp(CLOSE, "crosses_above", {"ind": "ema", "source": "close", "period": 20})]},
            short={"all": [_cmp({"ind": "ema", "source": "close", "period": 20}, "<",
                                {"ind": "ema", "source": "close", "period": 50}),
                           _cmp(CLOSE, "crosses_below", {"ind": "ema", "source": "close", "period": 20})]},
            entry__time_from="10:00", entry__time_to="14:30"),
    },
    {
        "key": "rule.rsi_reversion",
        "title": "RSI mean reversion (5m)",
        "summary": "Buy when RSI(14) crosses back above 30 below VWAP; short when it crosses below 70 above VWAP.",
        "spec": _spec(
            "RSI mean reversion", "Oversold/overbought snap-back toward VWAP, 1.5R target.",
            "5m",
            long={"all": [_cmp({"ind": "rsi", "source": "close", "period": 14}, "crosses_above", {"value": 30}),
                          _cmp(CLOSE, "<", {"ind": "vwap"})]},
            short={"all": [_cmp({"ind": "rsi", "source": "close", "period": 14}, "crosses_below", {"value": 70}),
                           _cmp(CLOSE, ">", {"ind": "vwap"})]},
            exits__target={"kind": "r", "value": 1.5}, exits__stop={"kind": "atr", "value": 1.0, "atr_period": 14},
            entry__time_from="10:00", entry__time_to="14:30"),
    },
    {
        "key": "rule.daily_breakout",
        "title": "20-day high breakout (daily)",
        "summary": "Buy a daily close above the previous 20 days' high on 1.5× average volume; hold up to 10 days.",
        "spec": _spec(
            "20-day breakout", "Donchian-style breakout with a volume filter, delivery (overnight).",
            "1d",
            long={"all": [_cmp(CLOSE, ">", {"ind": "highest", "source": "high", "period": 20}),
                          _cmp({"price": "volume"}, ">", {"ind": "avg", "source": "volume", "period": 20, "mult": 1.5})]},
            exits__stop={"kind": "atr", "value": 2.0, "atr_period": 14},
            exits__target={"kind": "r", "value": 3.0}, exits__max_hold_bars=10, exits__eod_time=None,
            universe__symbols=["RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS", "SBIN", "LT", "ITC",
                               "BHARTIARTL", "AXISBANK"]),
    },
]

# Plugin metadata the schema() doesn't carry.
PLUGIN_META: dict[str, dict] = {
    "intraday_screener": {
        "title": "Intraday screener (8 setups)",
        "summary": "Replays the live screener's enabled setups on 1m candles (breakouts, VWAP, pivots).",
        "backtest": "backtest", "timeframe": "1m",
        "notes": ["Signals come from the live ScreenerEngine replayed tick-by-tick on DB 1m bars "
                  "(plugins/strategy_search/screener_adapter.py).",
                  "Exits are assumed (the live screener only alerts): the signal's own stop and target, "
                  "owner trail (breakeven at +1R, then 1.5×ATR), flat by 15:14."],
        "broken": BROKEN_SCREENER_STRATEGIES,
        "backtest_params": ["only_strategy", "disabled_strategies", "max_risk_pct", "capital"],
    },
    "swing_scanner": {
        "title": "Oliver Kell swing cycles (daily)",
        "summary": "OK cycle detector on daily bars; long setups held up to 10 days.",
        "backtest": "backtest", "timeframe": "1d",
        "notes": ["Needs ~60 daily bars of history before the first signal — DB daily data starts "
                  "2026-04-20, so signals begin around mid-July 2026.",
                  "Short setups are skipped: cash-delivery shorts can't be held overnight.",
                  "Exits: the setup's stop (EMA mid ∓0.5%) and 2R target, owner trail, 10-day max hold."],
        "backtest_params": [],
    },
    "directional": {
        "title": "Directional (Claude planner)",
        "summary": "The LLM planner proposes BUY/SELL equity trades from intraday structure.",
        "backtest": "forward_paper_only",
        "reason": "Decisions come from the Claude planner (trading/agents/planner.py). An LLM can't be "
                  "replayed deterministically — and it may have seen this history — so a backtest number "
                  "would be fake. Judge it by forward paper trading (Action Board / Paper stage).",
    },
    "swing_scanner_v2": {
        "title": "Swing scanner v2 (multi-tier)", "summary": "15m / 1h / daily swing tiers.",
        "backtest": "unavailable",
        "reason": "No Studio backtest adapter for the v2 tiers yet; signals are tracked by the swing_v2 "
                  "enrichment job instead.",
    },
    "premarket_basket": {
        "title": "Premarket basket", "summary": "Morning mood + OK-actionable stocks as a basket.",
        "backtest": "unavailable",
        "reason": "Depends on the 09:15 mood snapshot (gap, VIX, A/D) and the basket executor; no "
                  "deterministic replay adapter yet. The legacy 'Morning Basket' mode stays on the "
                  "Backtester page.",
    },
    "pyramid": {"title": "Pyramid options", "backtest": "unavailable",
                "reason": "Options: there is no option price history in the database. Use the Pyramid page's "
                          "own backtest (live broker candles)."},
    "scalp": {"title": "Options scalp", "backtest": "unavailable",
              "reason": "Options: no option price history in the database (the scalp simulator replays "
                        "one session from the broker)."},
    "short_straddle": {"title": "Short straddle", "backtest": "unavailable",
                       "reason": "Options: no option price history in the database."},
    "vertical_spread": {"title": "Vertical spread", "backtest": "unavailable",
                        "reason": "Options: no option price history in the database."},
}
HIDDEN_PLUGINS = {"backtest"}   # infrastructure, not a strategy


def plugin_schema(name: str) -> dict | None:
    from apps.agents_core.registry import strategy_registry
    try:
        s = strategy_registry.get(name)
        sch = s.schema()
        return {"name": sch.name, "version": sch.version, "asset_class": sch.asset_class,
                "params": sch.params}
    except Exception:  # noqa: BLE001
        return None


def plugin_live_gate(name: str) -> dict:
    if name == "intraday_screener":
        return {"passes": True, "min_rr": 1.5,
                "message": "Setups target 1.5R–2.0R; the live RiskEngine rejects any below 1.5."}
    if name == "swing_scanner":
        return {"passes": True, "min_rr": 1.5, "message": "Setups target 2R."}
    return {"passes": None, "min_rr": 1.5, "message": "Checked by the RiskEngine per trade."}


def list_templates() -> list[dict]:
    from apps.agents_core.registry import strategy_registry
    from apps.strategies.services.spec import live_gate, validate_spec

    out = []
    for t in RULE_TEMPLATES:
        res = validate_spec(copy.deepcopy(t["spec"]))
        out.append({
            "key": t["key"], "kind": "spec", "strategy_name": "spec", "title": t["title"],
            "summary": t["summary"], "backtest": "backtest", "timeframe": res.spec.get("timeframe"),
            "spec": res.spec, "schema": None, "defaults": None, "notes": [], "broken": {},
            "reason": "", "live_gate": live_gate(res.spec) if res.ok else None,
            "errors": res.errors,
        })
    for name in strategy_registry.names():
        if name in HIDDEN_PLUGINS:
            continue
        sch = plugin_schema(name)
        if sch is None:
            continue
        meta = PLUGIN_META.get(name, {"backtest": "unavailable", "reason": "No backtest adapter."})
        props = (sch["params"] or {}).get("properties") or {}
        defaults = {k: v["default"] for k, v in props.items() if isinstance(v, dict) and "default" in v}
        out.append({
            "key": f"plugin.{name}", "kind": "plugin", "strategy_name": name,
            "title": meta.get("title", name), "summary": meta.get("summary", ""),
            "backtest": meta["backtest"], "timeframe": meta.get("timeframe"),
            "asset_class": sch["asset_class"], "version": sch["version"],
            "schema": sch["params"], "defaults": defaults, "spec": None,
            "notes": meta.get("notes", []), "broken": meta.get("broken", {}),
            "reason": meta.get("reason", ""), "live_gate": plugin_live_gate(name), "errors": [],
        })
    return out


def get_template(key: str) -> dict | None:
    return next((t for t in list_templates() if t["key"] == key), None)


def backtestability(kind: str, strategy_name: str) -> dict:
    if kind == "spec":
        return {"backtest": "backtest", "reason": ""}
    meta = PLUGIN_META.get(strategy_name, {})
    return {"backtest": meta.get("backtest", "unavailable"),
            "reason": meta.get("reason", "No backtest adapter for this plugin.")}


def validate_params(strategy_name: str, params: dict) -> list[dict]:
    """JSON-Schema validation of plugin params against the plugin's schema()."""
    sch = plugin_schema(strategy_name)
    if sch is None:
        return [{"path": "", "message": f"unknown strategy plugin {strategy_name!r}"}]
    if not isinstance(params, dict):
        return [{"path": "", "message": "params must be an object"}]
    import jsonschema
    v = jsonschema.Draft7Validator(sch["params"] or {"type": "object"})
    errs = []
    for e in sorted(v.iter_errors(params), key=lambda e: list(e.path)):
        errs.append({"path": ".".join(str(p) for p in e.path), "message": e.message})
    return errs
