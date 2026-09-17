"""Pine Script v5 generator for the live screener strategies.

Translates each enabled screener Strategy (``plugins.strategy_screener.strategies``)
into a self-contained TradingView Pine v5 indicator that fires the SAME webhook
JSON the screener emits. An operator pastes the link's webhook URL into
TradingView's alert dialog, mirrors a strategy on a chart, and the alert feeds
straight back into AlphaDesk via ``/api/v1/webhooks/tradingview/<secret>/``.

The alert message is built at runtime by string concatenation inside ``alert()``
— deliberately NOT using TradingView ``{{...}}`` placeholders, because ``alert()``
fires server-side once per confirmed bar close. Two constants are baked per
strategy: ``SIDE`` (BUY/SELL) and ``STRAT`` (the screener strategy name, the only
thing that sets ``Signal.strategy`` and is <=40 chars). ``price`` is emitted as a
bare unquoted number via ``str.tostring(close)`` so the parser's ``json.loads``
yields a float.

Pure + deterministic: no broker calls, no DB, no network. ``STRATEGIES`` is
imported lazily to avoid dragging ``trading.utils.indicators`` + agents_core
contracts (and the circular-import risk that comes with them) at module load.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Optional


# ── Strategy resolution (lazy STRATEGIES import) ──────────────────────────

def _enabled_strategies() -> list[Any]:
    """Return the enabled screener strategies. Lazy import keeps the heavy
    plugin package (indicators + contracts) off the module-load path."""
    from plugins.strategy_screener.strategies import STRATEGIES
    return [s for s in STRATEGIES if s.enabled]


def _slug(name: str) -> str:
    """'Breakout Long' -> 'breakout-long'. Lowercased, non-alphanumerics
    collapsed to single hyphens. Deterministic so the frontend select key is
    stable across requests."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s


def find_enabled_strategy(key: str) -> Optional[Any]:
    """Reverse lookup: resolve a slug (or exact name) to an enabled Strategy.
    Returns None for unknown/disabled keys."""
    key = (key or "").strip()
    if not key:
        return None
    target = _slug(key)
    for strat in _enabled_strategies():
        if _slug(strat.name) == target or strat.name == key:
            return strat
    return None


def list_pine_strategies() -> list[dict]:
    """Enumerate enabled strategies as ``[{key, label, description, side}]``."""
    return [
        {
            "key": _slug(s.name),
            "label": s.name,
            "description": s.description,
            "side": s.side,
        }
        for s in _enabled_strategies()
    ]


# ── Pine value formatting ─────────────────────────────────────────────────

def _fmt_num(v: Any) -> str:
    """Render a numeric threshold as a Pine float literal (50 -> '50.0',
    0.15 -> '0.15', 2.0 -> '2.0')."""
    f = float(v)
    if f == int(f):
        return f"{int(f)}.0"
    return repr(f)


def _tf_minutes(tf: str) -> str:
    """Screener timeframe label ('15m') -> Pine resolution string ('15')."""
    return tf.replace("m", "").strip() or "5"


def _hhmm(t) -> str:
    """datetime.time -> 'HHMM' for Pine session strings."""
    return f"{t.hour:02d}{t.minute:02d}"


_SEC = 'request.security(syminfo.tickerid, "{tf}", {expr}, lookahead=barmerge.lookahead_off)'

# Base indicator expressions, evaluated on whatever timeframe they are
# wrapped in (chart tf for 5m, request.security for 15m).
_LEVEL_EXPR = {
    "vwap": "ta.vwap",
    "sma_9": "ta.sma(close, 9)",
    "sma_20": "ta.sma(close, 20)",
    "ema_9": "ta.ema(close, 9)",
    "ema_21": "ta.ema(close, 21)",
    "bb_upper": "ta.bb(close, 20, 2.0)[2]",
}


# ── Generator context ─────────────────────────────────────────────────────

class _Ctx:
    """Accumulates deduped helper blocks + per-condition declaration lines."""

    def __init__(self) -> None:
        self.helpers: "OrderedDict[str, list[str]]" = OrderedDict()
        self.lines: list[str] = []
        self.cond_vars: list[str] = []
        self.vol_var: Optional[str] = None
        self.window_var: Optional[str] = None

    def helper(self, key: str, lines: list[str]) -> None:
        if key not in self.helpers:
            self.helpers[key] = lines

    def pivots(self) -> None:
        self.helper("pivots", [
            "// Prior-day classic pivots (computed once, no repaint via lookahead_on)",
            "[pdh, pdl, pdc] = request.security(syminfo.tickerid, \"D\", "
            "[high[1], low[1], close[1]], lookahead=barmerge.lookahead_on)",
            "pivotP    = (pdh + pdl + pdc) / 3.0",
            "classicS1 = 2.0 * pivotP - pdh",
            "classicR1 = 2.0 * pivotP - pdl",
        ])

    def macd_hist(self, tf: str) -> str:
        """Return the macd-histogram variable name for the timeframe, emitting
        its helper once."""
        if tf == "5m":
            self.helper("macd_5m", [
                "[macdLine, macdSignal, _macdHist] = ta.macd(close, 12, 26, 9)",
                "macdHist = macdLine - macdSignal",
            ])
            return "macdHist"
        minutes = _tf_minutes(tf)
        var = f"macdHist{minutes}"
        self.helper(f"macd_{minutes}", [
            f"[macdLine{minutes}, macdSignal{minutes}, _] = "
            + _SEC.format(tf=minutes, expr="ta.macd(close, 12, 26, 9)"),
            f"{var} = macdLine{minutes} - macdSignal{minutes}",
        ])
        return var


def _indicator_expr(indicator: str, tf: str, ctx: _Ctx) -> str:
    """Pine expression for an INDICATOR_COMPARE indicator on a timeframe."""
    if indicator == "macd_histogram":
        return ctx.macd_hist(tf)
    base = {
        "rsi_14": "ta.rsi(close, 14)",
        "atr_14": "ta.atr(14)",
    }.get(indicator)
    if base is None:
        # Unknown indicator — getattr(None) -> False in python; na in pine.
        return "na"
    if tf == "5m":
        return base
    return _SEC.format(tf=_tf_minutes(tf), expr=base)


def _level_expr(name: str, tf: str, ctx: _Ctx) -> Optional[str]:
    """Pine expression resolving a level name on a timeframe. Mirrors
    conditions._resolve_level for the levels enabled strategies reference."""
    if name in ("classic_s1", "classic_r1"):
        ctx.pivots()
        return "classicS1" if name == "classic_s1" else "classicR1"
    base = _LEVEL_EXPR.get(name)
    if base is None:
        return None
    if tf == "5m":
        return base
    return _SEC.format(tf=_tf_minutes(tf), expr=base)


# ── Per-condition emitters ────────────────────────────────────────────────

def _emit_condition(cond: Any, idx: int, ctx: _Ctx) -> None:
    from plugins.strategy_screener.conditions import ConditionType

    t = cond.type
    tf = cond.timeframe
    p = cond.params or {}
    desc = cond.description or ""

    def comment() -> None:
        if desc:
            ctx.lines.append(f"// {desc}")

    if t == ConditionType.VOLATILITY_FILTER:
        lo = _fmt_num(p.get("min_atr_pct", 0.15))
        hi = _fmt_num(p.get("max_atr_pct", 2.0))
        ctx.lines.append(f"// Volatility filter — ATR {lo}-{hi}% of price ({tf})")
        ctx.lines.append("atrPct = close > 0 ? ta.atr(14) / close * 100.0 : na")
        ctx.lines.append(f"volOk = not na(atrPct) and atrPct >= {lo} and atrPct <= {hi}")
        ctx.vol_var = "volOk"
        return

    if t == ConditionType.TIME_WINDOW:
        after = p.get("after", "00:00").replace(":", "")
        before = p.get("before", "23:59").replace(":", "")
        comment()
        ctx.lines.append(
            f'inWindow = not na(time(timeframe.period, "{after}-{before}", "Asia/Kolkata"))'
        )
        ctx.window_var = "inWindow"
        return

    if t in (ConditionType.PRICE_ABOVE, ConditionType.PRICE_BELOW):
        name = p.get("level", "")
        offset = float(p.get("offset_pct", 0) or 0)
        lvl = _level_expr(name, tf, ctx)
        var = f"cond{idx}"
        lvar = f"lvl{idx}"
        comment()
        if lvl is None:
            ctx.lines.append(f"{var} = false  // unsupported level: {name}")
            ctx.cond_vars.append(var)
            return
        ctx.lines.append(f"{lvar} = {lvl}")
        if t == ConditionType.PRICE_ABOVE:
            adj = f"{lvar} * (1.0 + {_fmt_num(offset)}/100.0)" if offset else lvar
            ctx.lines.append(
                f"{var} = not na({lvar}) and {lvar} != 0 and close > {adj}"
            )
        else:
            adj = f"{lvar} * (1.0 - {_fmt_num(offset)}/100.0)" if offset else lvar
            ctx.lines.append(
                f"{var} = not na({lvar}) and {lvar} != 0 and close < {adj}"
            )
        ctx.cond_vars.append(var)
        return

    if t in (ConditionType.PRICE_CROSSES_ABOVE, ConditionType.PRICE_CROSSES_BELOW):
        name = p.get("level", "")
        lvl = _level_expr(name, tf, ctx)
        var = f"cond{idx}"
        fn = "ta.crossover" if t == ConditionType.PRICE_CROSSES_ABOVE else "ta.crossunder"
        comment()
        if lvl is None:
            ctx.lines.append(f"{var} = false  // unsupported level: {name}")
            ctx.cond_vars.append(var)
            return
        if tf == "5m":
            ctx.lines.append(f"{var} = {fn}(close, {lvl})")
        else:
            # Higher-timeframe cross: both close and level snapshotted on the htf.
            minutes = _tf_minutes(tf)
            cvar = f"close{minutes}_{idx}"
            ctx.lines.append(f"{cvar} = " + _SEC.format(tf=minutes, expr="close"))
            ctx.lines.append(f"{var} = {fn}({cvar}, {lvl})")
        ctx.cond_vars.append(var)
        return

    if t == ConditionType.INDICATOR_COMPARE:
        indicator = p.get("indicator", "")
        op = p.get("op", ">")
        value = _fmt_num(p.get("value", 0))
        expr = _indicator_expr(indicator, tf, ctx)
        var = f"cond{idx}"
        ivar = f"ind{idx}"
        comment()
        ctx.lines.append(f"{ivar} = {expr}")
        ctx.lines.append(f"{var} = not na({ivar}) and {ivar} {op} {value}")
        ctx.cond_vars.append(var)
        return

    if t == ConditionType.BB_POSITION:
        position = p.get("position", "")
        var = f"cond{idx}"
        comment()
        if position == "above_upper":
            minutes = _tf_minutes(tf)
            uvar = f"bbUpper{minutes}_{idx}"
            cvar = f"close{minutes}_{idx}"
            if tf == "5m":
                ctx.lines.append(f"{uvar} = ta.bb(close, 20, 2.0)[2]")
                ctx.lines.append(f"{cvar} = close")
            else:
                ctx.lines.append(
                    f"{uvar} = " + _SEC.format(tf=minutes, expr="ta.bb(close, 20, 2.0)[2]")
                )
                ctx.lines.append(f"{cvar} = " + _SEC.format(tf=minutes, expr="close"))
            ctx.lines.append(
                f"{var} = not na({uvar}) and {uvar} > 0 and {cvar} > {uvar}"
            )
        else:
            ctx.lines.append(f"{var} = false  // unsupported bb_position: {position}")
        ctx.cond_vars.append(var)
        return

    if t == ConditionType.N_BAR_HIGH_BREAK:
        lookback = int(p.get("lookback", 20))
        var = f"cond{idx}"
        comment()
        ctx.lines.append(f"{var} = close > ta.highest(high, {lookback})[1]")
        ctx.cond_vars.append(var)
        return

    if t == ConditionType.N_BAR_LOW_BREAK:
        lookback = int(p.get("lookback", 20))
        var = f"cond{idx}"
        comment()
        ctx.lines.append(f"{var} = close < ta.lowest(low, {lookback})[1]")
        ctx.cond_vars.append(var)
        return

    # Any condition type not exercised by an enabled strategy: emit a false
    # placeholder so the script still compiles and never over-fires.
    var = f"cond{idx}"
    ctx.lines.append(f"// {desc}".rstrip())
    ctx.lines.append(f"{var} = false  // unsupported condition: {t.value}")
    ctx.cond_vars.append(var)


# ── Public generator ──────────────────────────────────────────────────────

def generate_pine(strategy: Any, webhook_url: str = "") -> str:
    """Generate a complete Pine v5 indicator for a screener strategy.

    Accepts a Strategy dataclass, a slug ('breakout-long'), or the exact
    strategy name. ``webhook_url``, when supplied, is emitted as a comment
    header (the script does not need it to function — the URL is configured in
    TradingView's alert dialog)."""
    if isinstance(strategy, str):
        resolved = find_enabled_strategy(strategy)
        if resolved is None:
            raise ValueError(f"unknown or disabled strategy: {strategy!r}")
        strat = resolved
    else:
        strat = strategy

    side = (strat.side or "BUY").upper()
    name = strat.name

    ctx = _Ctx()
    idx = 0
    for cond in strat.conditions:
        # VOLATILITY_FILTER / TIME_WINDOW are gates, not numbered conds — but
        # numbering by position keeps variable names unique regardless.
        idx += 1
        _emit_condition(cond, idx, ctx)

    # Build fireSignal expression.
    gates: list[str] = []
    if ctx.vol_var:
        gates.append(ctx.vol_var)
    gates.extend(ctx.cond_vars)
    # active_window is a Strategy-level gate applied IN ADDITION to any
    # TIME_WINDOW condition.
    aw_start, aw_end = strat.active_window
    gates.append("inActiveWindow")
    if ctx.window_var:
        gates.append(ctx.window_var)

    fire_expr = " and ".join(gates) if gates else "false"

    out: list[str] = []
    out.append("//@version=5")
    out.append(f'indicator("AlphaDesk · {name}", overlay=true)')
    out.append("")
    if webhook_url:
        out.append(f"// Webhook URL (paste into TradingView's alert dialog): {webhook_url}")
        out.append("")
    out.append("// Auto-generated from the AlphaDesk screener strategy.")
    out.append(f"// Strategy : {name}")
    out.append(f"// Side     : {side}")
    if strat.description:
        out.append(f"// Setup    : {strat.description}")
    out.append("// Targets NSE/BSE IST symbols — all time gates are Asia/Kolkata.")
    out.append("// Entry is market on signal; SL/target are managed server-side")
    out.append("// (the alert payload carries only price=close).")
    out.append("")

    # Baked constants.
    out.append(f'SIDE  = "{side}"')
    out.append(f'STRAT = "{name}"')
    out.append("")

    # Helper blocks (pivots, macd, ...).
    for block in ctx.helpers.values():
        out.extend(block)
        out.append("")

    # Per-condition declarations.
    out.extend(ctx.lines)
    out.append("")

    # active_window gate.
    out.append(
        f"// Active window {aw_start.strftime('%H:%M')}-{aw_end.strftime('%H:%M')} IST"
    )
    out.append(
        f'inActiveWindow = not na(time(timeframe.period, "{_hhmm(aw_start)}-{_hhmm(aw_end)}", '
        '"Asia/Kolkata"))'
    )
    out.append("")

    out.append(f"fireSignal = {fire_expr}")
    out.append("")

    # Visual marker.
    if side == "SELL":
        out.append(
            'plotshape(fireSignal, title="Signal", style=shape.triangledown, '
            "location=location.abovebar, color=color.red, size=size.small)"
        )
    else:
        out.append(
            'plotshape(fireSignal, title="Signal", style=shape.triangleup, '
            "location=location.belowbar, color=color.lime, size=size.small)"
        )
    out.append("")

    # Alert — JSON built by concatenation, fires once per confirmed bar close.
    out.append("if fireSignal")
    out.append(
        "    alert('{\"symbol\":\"' + syminfo.ticker + '\",\"action\":\"' + SIDE + "
        "'\",\"price\":' + str.tostring(close) + ',\"strategy\":\"' + STRAT + '\"}', "
        "alert.freq_once_per_bar_close)"
    )
    out.append("")

    return "\n".join(out)
