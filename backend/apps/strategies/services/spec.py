"""StrategySpec v1 — the no-code strategy definition (Strategy Studio plan §2.1).

A spec is plain JSON. ``validate_spec`` normalises it (fills defaults), checks
every reference, operator and limit, and returns errors + warnings with a
JSON path each. ``spec_hash`` is the sha256 of the normalised canonical JSON,
so two equal specs always hash equal. ``explain`` renders the plain-English
summary shown next to the builder. Nothing here runs a strategy.

Scope of v1 (deliberately small, all deterministic):
  instruments  NSE cash equity only (options stay plugins)
  timeframes   1m / 5m / 15m / 60m (from DB 1m candles) and 1d (NSE bhavcopy)
  entry        long and/or short condition tree: all / any / not over
               comparisons (> >= < <= crosses_above crosses_below) and
               rising / falling; operands are a price field, a constant, an
               inline indicator or a named one, each with optional bars_ago
               and a multiplier
  exits        stop (ATR multiple | %), target (R multiple | % | none),
               breakeven, trail (owner rule | ATR | none), EOD time, max hold
  sizing       risk % per trade, notional cap %, max open positions —
               capped at the live RiskEngine limits (settings.ALPHADESK)
  limits       daily loss %, trades per day, owner losing-streak cooldown
  costs        slippage % per side; NSE MIS (intraday) or delivery cost model
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "60m": 60, "1d": 0}
PRICE_FIELDS = {"open", "high", "low", "close", "volume"}
COMPARE_OPS = {">", ">=", "<", "<=", "crosses_above", "crosses_below"}
TREND_OPS = {"rising", "falling"}

# kind → (allowed params with defaults, intraday-only?)
INDICATORS: dict[str, tuple[dict, bool]] = {
    "ema": ({"source": "close", "period": 20}, False),
    "sma": ({"source": "close", "period": 20}, False),
    "rsi": ({"source": "close", "period": 14}, False),
    "atr": ({"period": 14}, False),
    "bb_upper": ({"period": 20, "std": 2.0}, False),
    "bb_middle": ({"period": 20, "std": 2.0}, False),
    "bb_lower": ({"period": 20, "std": 2.0}, False),
    "highest": ({"source": "high", "period": 20}, False),
    "lowest": ({"source": "low", "period": 20}, False),
    "avg": ({"source": "volume", "period": 20}, False),
    "prev_day_high": ({}, False),
    "prev_day_low": ({}, False),
    "prev_day_close": ({}, False),
    "vwap": ({}, True),
    "day_open": ({}, True),
    "gap_pct": ({}, True),
    "orb_high": ({"minutes": 15}, True),
    "orb_low": ({"minutes": 15}, True),
}

INDICATOR_LABELS = {
    "ema": "EMA", "sma": "SMA", "rsi": "RSI", "atr": "ATR", "bb_upper": "upper Bollinger",
    "bb_middle": "middle Bollinger", "bb_lower": "lower Bollinger", "highest": "highest",
    "lowest": "lowest", "avg": "average", "prev_day_high": "previous day's high",
    "prev_day_low": "previous day's low", "prev_day_close": "previous day's close",
    "vwap": "VWAP", "day_open": "today's open", "gap_pct": "gap % vs previous close",
    "orb_high": "opening-range high", "orb_low": "opening-range low",
}

MIN_TRADES_MEANINGFUL = 30   # below this a result is labelled "too few trades"


def live_limits() -> dict:
    from django.conf import settings
    a = getattr(settings, "ALPHADESK", {})
    return {
        "max_risk_pct": float(a.get("MAX_RISK_PER_TRADE_PCT", 1.0)),
        "max_position_pct": float(a.get("MAX_POSITION_SIZE_PCT", 10.0)),
        "max_positions": int(a.get("MAX_OPEN_POSITIONS", 3)),
        "max_daily_loss_pct": float(a.get("MAX_DAILY_LOSS_PCT", 3.0)),
        "min_rr": float(a.get("MIN_RISK_REWARD_RATIO", 1.5)),
        "square_off": str(getattr(settings, "TRADING_RULES", {}).get("AUTO_SQUARE_OFF", "15:14")),
    }


def default_spec(name: str = "New strategy") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {"name": name, "description": "", "tags": []},
        "universe": {"symbols": ["RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS"]},
        "timeframe": "5m",
        "indicators": [],
        "entry": {"long": None, "short": None, "time_from": "09:20", "time_to": "14:30",
                  "weekdays": [0, 1, 2, 3, 4], "cooldown_bars": 0},
        "exits": {"stop": {"kind": "atr", "value": 1.5, "atr_period": 14},
                  "target": {"kind": "r", "value": 2.0},
                  "breakeven_r": None,
                  "trail": {"kind": "owner"},
                  "eod_time": "15:14", "max_hold_bars": None},
        "sizing": {"risk_pct": 0.5, "max_position_pct": 10.0, "max_positions": 3},
        "limits": {"daily_loss_pct": 3.0, "max_trades_per_day": 0, "owner_cooldown": True},
        "costs": {"slippage_pct": 0.02, "model": "auto"},
    }


@dataclass
class SpecResult:
    spec: dict
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"valid": self.ok, "errors": self.errors, "warnings": self.warnings,
                "spec": self.spec, "spec_hash": spec_hash(self.spec) if self.ok else None,
                "explain": explain(self.spec) if self.ok else [],
                "live_gate": live_gate(self.spec) if self.ok else None}


def _merge(defaults: dict, given) -> dict:
    out = copy.deepcopy(defaults)
    if isinstance(given, dict):
        for k, v in given.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
    return out


def _hhmm(s) -> int | None:
    try:
        h, m = str(s).split(":")
        h, m = int(h), int(m)
        if 0 <= h < 24 and 0 <= m < 60:
            return h * 60 + m
    except (ValueError, AttributeError):
        pass
    return None


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def validate_spec(raw) -> SpecResult:
    """Normalise + validate. Never raises on bad input; errors are returned."""
    if not isinstance(raw, dict):
        return SpecResult(spec={}, errors=[{"path": "", "message": "spec must be a JSON object"}])
    spec = _merge(default_spec(), raw)
    spec["schema_version"] = SCHEMA_VERSION
    r = SpecResult(spec=spec)

    def err(path, msg):
        r.errors.append({"path": path, "message": msg})

    def warn(path, msg):
        r.warnings.append({"path": path, "message": msg})

    tf = spec.get("timeframe")
    if tf not in TIMEFRAMES:
        err("timeframe", f"timeframe must be one of {sorted(TIMEFRAMES)}")
        tf = "5m"
    daily = tf == "1d"

    name = str(spec["meta"].get("name") or "").strip()
    if not name:
        err("meta.name", "name is required")

    syms = spec["universe"].get("symbols")
    if not isinstance(syms, list) or not syms or not all(isinstance(s, str) and s.strip() for s in syms):
        err("universe.symbols", "list at least one NSE symbol")
    else:
        spec["universe"]["symbols"] = sorted({s.strip().upper() for s in syms})
        if len(spec["universe"]["symbols"]) > 100:
            err("universe.symbols", "at most 100 symbols")

    # named indicators
    named: dict[str, dict] = {}
    inds = spec.get("indicators") or []
    if not isinstance(inds, list):
        err("indicators", "indicators must be a list")
        inds = []
    for i, ind in enumerate(inds):
        p = f"indicators[{i}]"
        if not isinstance(ind, dict) or not str(ind.get("id") or "").strip():
            err(p, "each indicator needs an id")
            continue
        iid = str(ind["id"]).strip()
        if iid in named:
            err(p + ".id", f"duplicate indicator id {iid!r}")
        norm = _check_indicator(ind, p, daily, err)
        if norm is not None:
            norm["id"] = iid
            named[iid] = norm
            inds[i] = norm
    spec["indicators"] = inds

    # entry
    entry = spec["entry"]
    has_rule = False
    for side in ("long", "short"):
        node = entry.get(side)
        if node in (None, {}, []):
            entry[side] = None
            continue
        has_rule = True
        entry[side] = _check_node(node, f"entry.{side}", named, daily, err, depth=0)
    if not has_rule:
        err("entry", "add at least one long or short entry condition")
    if daily and entry.get("short"):
        err("entry.short", "daily (1d) strategies are long-only: cash-delivery shorts can't be held overnight")
    if not daily:
        tf_from, tf_to = _hhmm(entry.get("time_from")), _hhmm(entry.get("time_to"))
        if tf_from is None or tf_to is None:
            err("entry.time_from", "time window must be HH:MM")
        elif not (9 * 60 + 15 <= tf_from <= tf_to <= 15 * 60 + 14):
            err("entry.time_from", "entry window must sit inside 09:15–15:14 and from ≤ to")
    wd = entry.get("weekdays")
    if not isinstance(wd, list) or not wd or not all(isinstance(d, int) and 0 <= d <= 4 for d in wd):
        err("entry.weekdays", "weekdays must be a non-empty list of 0 (Mon) … 4 (Fri)")
    cd = entry.get("cooldown_bars")
    if not isinstance(cd, int) or isinstance(cd, bool) or cd < 0 or cd > 1000:
        err("entry.cooldown_bars", "cooldown_bars must be an integer 0–1000")

    # exits
    ex = spec["exits"]
    stop = ex.get("stop") or {}
    sv = _num(stop.get("value"))
    if stop.get("kind") not in ("atr", "pct"):
        err("exits.stop.kind", "stop kind must be 'atr' or 'pct'")
    elif sv is None or sv <= 0 or (stop["kind"] == "pct" and sv > 20) or (stop["kind"] == "atr" and sv > 20):
        err("exits.stop.value", "stop value must be > 0 (ATR multiple or % of price, at most 20)")
    ap = stop.get("atr_period", 14)
    if not isinstance(ap, int) or isinstance(ap, bool) or not 2 <= ap <= 200:
        err("exits.stop.atr_period", "atr_period must be an integer 2–200")
    tgt = ex.get("target") or {"kind": "none"}
    ex["target"] = tgt
    if tgt.get("kind") not in ("r", "pct", "none"):
        err("exits.target.kind", "target kind must be 'r', 'pct' or 'none'")
    elif tgt["kind"] != "none":
        tv = _num(tgt.get("value"))
        if tv is None or tv <= 0 or tv > 50:
            err("exits.target.value", "target value must be > 0 (R multiple or %, at most 50)")
    be = ex.get("breakeven_r")
    if be is not None and (_num(be) is None or _num(be) <= 0):
        err("exits.breakeven_r", "breakeven_r must be a positive R multiple or null")
    tr = ex.get("trail") or {"kind": "none"}
    ex["trail"] = tr
    if tr.get("kind") not in ("owner", "atr", "none"):
        err("exits.trail.kind", "trail kind must be 'owner', 'atr' or 'none'")
    elif tr["kind"] == "atr":
        for k, lo, hi in (("trigger_r", 0.1, 20), ("atr_mult", 0.1, 20)):
            v = _num(tr.get(k, 1.0 if k == "trigger_r" else 1.5))
            if v is None or not lo <= v <= hi:
                err(f"exits.trail.{k}", f"{k} must be {lo}–{hi}")
            else:
                tr[k] = v
    mh = ex.get("max_hold_bars")
    if mh is not None and (not isinstance(mh, int) or isinstance(mh, bool) or mh <= 0):
        err("exits.max_hold_bars", "max_hold_bars must be a positive integer or null")
    lim = live_limits()
    if not daily:
        eod = _hhmm(ex.get("eod_time"))
        sq = _hhmm(lim["square_off"]) or 15 * 60 + 14
        if eod is None:
            err("exits.eod_time", "eod_time must be HH:MM")
        elif eod > sq:
            err("exits.eod_time", f"intraday positions must be flat by the {lim['square_off']} auto square-off")
    else:
        ex["eod_time"] = None
        if mh is None:
            warn("exits.max_hold_bars", "no max hold: a daily trade without a target can run to the end of the test")

    # sizing + limits, capped at the live RiskEngine's
    sz = spec["sizing"]
    for key, cap, label in (("risk_pct", lim["max_risk_pct"], "MAX_RISK_PER_TRADE_PCT"),
                            ("max_position_pct", lim["max_position_pct"], "MAX_POSITION_SIZE_PCT")):
        v = _num(sz.get(key))
        if v is None or v <= 0:
            err(f"sizing.{key}", f"{key} must be > 0")
        elif v > cap:
            err(f"sizing.{key}", f"{key} {v} is above the live limit {cap} ({label}) — "
                                 "a backtest may not trade bigger than the RiskEngine allows")
    mp = sz.get("max_positions")
    if not isinstance(mp, int) or isinstance(mp, bool) or mp < 1:
        err("sizing.max_positions", "max_positions must be an integer ≥ 1")
    elif mp > lim["max_positions"]:
        err("sizing.max_positions", f"max_positions {mp} is above the live limit {lim['max_positions']} "
                                    "(MAX_OPEN_POSITIONS)")
    lm = spec["limits"]
    dl = _num(lm.get("daily_loss_pct"))
    if dl is None or dl <= 0:
        err("limits.daily_loss_pct", "daily_loss_pct must be > 0")
    elif dl > lim["max_daily_loss_pct"]:
        err("limits.daily_loss_pct", f"daily_loss_pct {dl} is above the live limit {lim['max_daily_loss_pct']}")
    mt = lm.get("max_trades_per_day")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt < 0:
        err("limits.max_trades_per_day", "max_trades_per_day must be an integer ≥ 0 (0 = no limit)")
    if not isinstance(lm.get("owner_cooldown"), bool):
        err("limits.owner_cooldown", "owner_cooldown must be true or false")
    cs = spec["costs"]
    sl = _num(cs.get("slippage_pct"))
    if sl is None or not 0 <= sl <= 2:
        err("costs.slippage_pct", "slippage_pct must be 0–2 (% per side)")
    if cs.get("model") not in ("auto", "nse_intraday_mis", "nse_delivery"):
        err("costs.model", "cost model must be auto, nse_intraday_mis or nse_delivery")
    elif daily and cs["model"] == "nse_intraday_mis":
        err("costs.model", "daily strategies hold overnight: use the delivery cost model")
    elif not daily and cs["model"] == "nse_delivery":
        err("costs.model", "intraday strategies are MIS: use the intraday cost model")

    if not r.errors:
        gate = live_gate(spec)
        if not gate["passes"]:
            warn("exits.target", gate["message"])
    return r


def _check_indicator(ind: dict, path: str, daily: bool, err) -> dict | None:
    kind = ind.get("kind") or ind.get("ind")
    if kind not in INDICATORS:
        err(path + ".kind", f"unknown indicator {kind!r}")
        return None
    defaults, intraday_only = INDICATORS[kind]
    if intraday_only and daily:
        err(path + ".kind", f"{kind} is intraday-only (not available on 1d bars)")
        return None
    out = {"kind": kind}
    for k, dv in defaults.items():
        v = ind.get(k, dv)
        if k == "source":
            if v not in PRICE_FIELDS:
                err(f"{path}.{k}", f"source must be one of {sorted(PRICE_FIELDS)}")
                return None
        elif k == "std":
            v = _num(v)
            if v is None or not 0.5 <= v <= 5:
                err(f"{path}.{k}", "std must be 0.5–5")
                return None
        else:  # period / minutes
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 400:
                err(f"{path}.{k}", f"{k} must be an integer 1–400")
                return None
        out[k] = v
    return out


def _check_operand(op, path: str, named: dict, daily: bool, err) -> dict | None:
    if isinstance(op, (int, float)) and not isinstance(op, bool):
        return {"value": float(op)}
    if not isinstance(op, dict):
        err(path, "operand must be an object")
        return None
    out: dict
    if "value" in op:
        v = _num(op["value"])
        if v is None:
            err(path + ".value", "value must be a number")
            return None
        return {"value": v}
    if "price" in op:
        if op["price"] not in PRICE_FIELDS:
            err(path + ".price", f"price must be one of {sorted(PRICE_FIELDS)}")
            return None
        out = {"price": op["price"]}
    elif "ref" in op:
        if op["ref"] not in named:
            err(path + ".ref", f"no indicator named {op['ref']!r}")
            return None
        out = {"ref": op["ref"]}
    elif "ind" in op or "kind" in op:
        norm = _check_indicator(op, path, daily, err)
        if norm is None:
            return None
        out = {"ind": norm.pop("kind"), **norm}
    else:
        err(path, "operand needs one of: price, value, ind, ref")
        return None
    ba = op.get("bars_ago", 0)
    if not isinstance(ba, int) or isinstance(ba, bool) or not 0 <= ba <= 200:
        err(path + ".bars_ago", "bars_ago must be an integer 0–200")
        return None
    if ba:
        out["bars_ago"] = ba
    if "mult" in op and op["mult"] not in (None, 1, 1.0):
        m = _num(op["mult"])
        if m is None or m <= 0:
            err(path + ".mult", "mult must be a positive number")
            return None
        out["mult"] = m
    return out


def _check_node(node, path: str, named: dict, daily: bool, err, depth: int):
    if depth > 8:
        err(path, "conditions nested too deep (max 8)")
        return None
    if not isinstance(node, dict):
        err(path, "condition must be an object")
        return None
    for key in ("all", "any"):
        if key in node:
            kids = node[key]
            if not isinstance(kids, list) or not kids:
                err(f"{path}.{key}", f"'{key}' needs a non-empty list")
                return None
            return {key: [_check_node(k, f"{path}.{key}[{i}]", named, daily, err, depth + 1)
                          for i, k in enumerate(kids)]}
    if "not" in node:
        return {"not": _check_node(node["not"], path + ".not", named, daily, err, depth + 1)}
    op = node.get("op")
    if op in TREND_OPS:
        left = _check_operand(node.get("left"), path + ".left", named, daily, err)
        bars = node.get("bars", 1)
        if not isinstance(bars, int) or isinstance(bars, bool) or not 1 <= bars <= 200:
            err(path + ".bars", "bars must be an integer 1–200")
        return {"left": left, "op": op, "bars": bars}
    if op in COMPARE_OPS:
        return {"left": _check_operand(node.get("left"), path + ".left", named, daily, err),
                "op": op,
                "right": _check_operand(node.get("right"), path + ".right", named, daily, err)}
    err(path + ".op", f"unknown operator {op!r} (use {sorted(COMPARE_OPS | TREND_OPS)} or all/any/not)")
    return None


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def params_hash(name: str, params: dict) -> str:
    return spec_hash({"plugin": name, "params": params})


# ── live gate + plain English ───────────────────────────────────────────────

def live_gate(spec: dict) -> dict:
    """Can trades from this spec pass the live RiskEngine's R:R criterion?"""
    min_rr = live_limits()["min_rr"]
    tgt = (spec.get("exits") or {}).get("target") or {"kind": "none"}
    stop = (spec.get("exits") or {}).get("stop") or {}
    if tgt.get("kind") == "none":
        return {"passes": False, "min_rr": min_rr,
                "message": f"No target: the live RiskEngine requires target R:R ≥ {min_rr}, so this "
                           "strategy can be backtested but can't pass the live gate."}
    if tgt["kind"] == "r":
        ok = float(tgt["value"]) >= min_rr
        return {"passes": ok, "min_rr": min_rr,
                "message": "" if ok else f"Target {tgt['value']}R is below the live minimum R:R {min_rr}: "
                                         "the RiskEngine would reject every trade."}
    if stop.get("kind") == "pct":
        rr = float(tgt["value"]) / float(stop["value"])
        ok = round(rr, 2) >= min_rr
        return {"passes": ok, "min_rr": min_rr,
                "message": "" if ok else f"Target/stop = {rr:.2f} R, below the live minimum {min_rr}."}
    return {"passes": True, "min_rr": min_rr,
            "message": f"Target is a % and stop an ATR multiple, so R:R varies per trade; the live "
                       f"RiskEngine rejects any trade below {min_rr}R."}


def _op_text(op: dict | None, named: dict) -> str:
    if not op:
        return "?"
    if "value" in op:
        v = op["value"]
        return f"{v:g}"
    if "price" in op:
        base = op["price"]
    else:
        ind = named.get(op["ref"], {}) if "ref" in op else {**op, "kind": op.get("ind")}
        kind = ind.get("kind")
        label = INDICATOR_LABELS.get(kind, kind or "?")
        if kind in ("ema", "sma", "rsi"):
            src = "" if ind.get("source", "close") == "close" else f" of {ind['source']}"
            base = f"{label}({ind.get('period')}){src}"
        elif kind == "atr":
            base = f"ATR({ind.get('period')})"
        elif kind and kind.startswith("bb_"):
            base = f"{label} band({ind.get('period')}, {ind.get('std')})"
        elif kind in ("highest", "lowest"):
            base = f"{label} {ind.get('source')} of the previous {ind.get('period')} bars"
        elif kind == "avg":
            base = f"average {ind.get('source')} of the previous {ind.get('period')} bars"
        elif kind in ("orb_high", "orb_low"):
            base = f"{label} ({ind.get('minutes')} min)"
        else:
            base = label
        if "ref" in op:
            base = f"{op['ref']} [{base}]"
    if op.get("mult"):
        base = f"{op['mult']:g} × {base}"
    if op.get("bars_ago"):
        base = f"{base} {op['bars_ago']} bar(s) ago"
    return base


def _node_text(node, named) -> str:
    if not node:
        return "?"
    if "all" in node:
        return " AND ".join(f"({_node_text(k, named)})" if ("any" in (k or {})) else _node_text(k, named)
                            for k in node["all"])
    if "any" in node:
        return " OR ".join(_node_text(k, named) for k in node["any"])
    if "not" in node:
        return f"NOT ({_node_text(node['not'], named)})"
    op = node.get("op")
    words = {">": "is above", ">=": "is at or above", "<": "is below", "<=": "is at or below",
             "crosses_above": "crosses above", "crosses_below": "crosses below"}
    if op in TREND_OPS:
        return f"{_op_text(node.get('left'), named)} is {op} over {node.get('bars', 1)} bar(s)"
    return f"{_op_text(node.get('left'), named)} {words.get(op, op)} {_op_text(node.get('right'), named)}"


def explain(spec: dict) -> list[str]:
    """Plain-English summary, one sentence per line."""
    named = {i["id"]: i for i in spec.get("indicators") or [] if isinstance(i, dict) and "id" in i}
    tf = spec["timeframe"]
    daily = tf == "1d"
    syms = spec["universe"]["symbols"]
    lines = [f"Trades {len(syms)} NSE stock(s) ({', '.join(syms[:6])}{'…' if len(syms) > 6 else ''}) "
             f"on {'daily' if daily else tf} bars, "
             f"{'holding overnight (delivery)' if daily else 'intraday (MIS)'}."]
    e = spec["entry"]
    for side, word in (("long", "Buys"), ("short", "Sells short")):
        if e.get(side):
            lines.append(f"{word} when {_node_text(e[side], named)} — decided on the bar's close, "
                         "filled at the next bar's open.")
    if not daily:
        lines.append(f"New entries only between {e['time_from']} and {e['time_to']} IST"
                     + (f", at least {e['cooldown_bars']} bars after the last exit in that stock."
                        if e.get("cooldown_bars") else "."))
    ex = spec["exits"]
    st = ex["stop"]
    lines.append("Stop: " + (f"{st['value']:g} × ATR({st.get('atr_period', 14)}) from entry."
                             if st["kind"] == "atr" else f"{st['value']:g}% from entry."))
    tg = ex["target"]
    lines.append("Target: " + ("none." if tg["kind"] == "none" else
                               f"{tg['value']:g}R." if tg["kind"] == "r" else f"{tg['value']:g}% from entry."))
    if ex.get("breakeven_r"):
        lines.append(f"Stop moves to breakeven at +{ex['breakeven_r']:g}R.")
    tr = ex["trail"]
    if tr["kind"] == "owner":
        lines.append("Trail (owner rule): at +1R the stop moves to entry, then follows the best price "
                     "minus 1.5 × ATR(14) of this timeframe; it only ratchets.")
    elif tr["kind"] == "atr":
        lines.append(f"Trail: from +{tr['trigger_r']:g}R the stop follows the best price minus "
                     f"{tr['atr_mult']:g} × ATR; it only ratchets.")
    if ex.get("max_hold_bars"):
        lines.append(f"Exits after {ex['max_hold_bars']} bars at the latest.")
    if not daily:
        lines.append(f"Flat by the close of the bar containing {ex['eod_time']} IST.")
    sz, lm = spec["sizing"], spec["limits"]
    lines.append(f"Risks {sz['risk_pct']:g}% of capital per trade (notional capped at "
                 f"{sz['max_position_pct']:g}%), at most {sz['max_positions']} open position(s).")
    extra = []
    if lm.get("max_trades_per_day"):
        extra.append(f"at most {lm['max_trades_per_day']} trades a day")
    extra.append(f"no new entries after a {lm['daily_loss_pct']:g}% daily loss")
    if lm.get("owner_cooldown"):
        extra.append("2 days off after 3 losing days, then half risk until a winning day (owner rule)")
    lines.append("Limits: " + "; ".join(extra) + ".")
    cs = spec["costs"]
    model = cs["model"] if cs["model"] != "auto" else ("nse_delivery" if daily else "nse_intraday_mis")
    lines.append(f"Costs: {cs['slippage_pct']:g}% slippage per side plus "
                 f"{'NSE delivery' if model == 'nse_delivery' else 'NSE intraday (MIS)'} charges.")
    return lines
