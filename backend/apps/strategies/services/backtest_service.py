"""Deterministic Studio backtests (Strategy Studio plan §4, P2).

``execute(req, loader)`` is pure apart from the candle loader: it builds a
plan for the strategy kind, loads one symbol at a time, precomputes signals,
runs the ``strategy_backtest`` engine in TIMELINE mode (next-bar-open fills,
portfolio limits, owner cooldown, NSE costs + slippage) and returns trades,
metrics (whole period + in-sample / out-of-sample), an equity curve and a
provenance record. ``run_backtest_job`` is the Django wrapper the Celery task
calls: it persists results on the ``Backtest`` row (whose post_save keeps the
run receipt in sync) and writes ``BacktestTrade`` rows.

No LLM is involved anywhere; every number comes from candles + the engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import structlog

from plugins.strategy_backtest.engine import BacktestEngine
from plugins.strategy_backtest.types import EntrySignal, TradeSide

from apps.strategies.services import costs as cost_models
from apps.strategies.services.adapter import (
    PrecomputedDetector, RatchetStop, engine_config, spec_engine_config, spec_exit_checks, spec_signals,
)
from apps.strategies.services.rules import CompiledSpec, indicator_series
from apps.strategies.services.spec import MIN_TRADES_MEANINGFUL, live_limits

log = structlog.get_logger(__name__)

ENGINE_LABEL = "plugins.strategy_backtest BacktestEngine · TIMELINE (signal on close, fill next bar open)"
MAX_ENGINE_BARS = 600_000       # memory guard (~7 GB laptop): narrow the run beyond this
MAX_SYMBOLS = 60
OOS_FRACTION = 0.3
RESEARCH_OOS = (date(2026, 9, 1), date(2026, 10, 19))


class Cancelled(Exception):
    pass


@dataclass
class RunRequest:
    kind: str                         # spec | plugin
    strategy_name: str                # "spec" or plugin name
    definition: dict                  # spec or plugin params
    from_date: date
    to_date: date
    symbols: list[str] | None = None
    capital: float = 500_000.0
    slippage_pct: float | None = None
    oos_fraction: float = OOS_FRACTION
    version: int | None = None
    definition_hash: str = ""


@dataclass
class Plan:
    timeframe: str
    symbols: list[str]
    warmup_bars: int
    exits: list
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    slippage_pct: float = 0.02
    cost_model: str = "auto"
    live_gate: dict | None = None
    keep_raw: bool = False
    dropped_signals: dict = field(default_factory=dict)

    def signals(self, loaded) -> dict[int, EntrySignal]:  # pragma: no cover - overridden
        raise NotImplementedError

    def config(self, capital, entry_from, entry_to, day_gate):  # pragma: no cover
        raise NotImplementedError


# ── plans ───────────────────────────────────────────────────────────────────

def _max_period(node_or_list) -> int:
    best = 0
    if isinstance(node_or_list, dict):
        for k, v in node_or_list.items():
            if k in ("period", "minutes") and isinstance(v, int):
                best = max(best, v)
            elif k == "bars_ago" and isinstance(v, int):
                best = max(best, v)
            else:
                best = max(best, _max_period(v))
    elif isinstance(node_or_list, list):
        for v in node_or_list:
            best = max(best, _max_period(v))
    return best


class SpecPlan(Plan):
    def __init__(self, spec: dict, req: RunRequest):
        from apps.strategies.services.spec import live_gate
        self.spec = spec
        self.compiled = CompiledSpec(spec)
        super().__init__(
            timeframe=spec["timeframe"],
            symbols=req.symbols or list(spec["universe"]["symbols"]),
            warmup_bars=max(_max_period(spec["indicators"]), _max_period(spec["entry"]),
                            int(spec["exits"]["stop"].get("atr_period", 14)), 14) * 2 + 5,
            exits=spec_exit_checks(spec),
            slippage_pct=float(spec["costs"]["slippage_pct"] if req.slippage_pct is None else req.slippage_pct),
            cost_model=spec["costs"]["model"], live_gate=live_gate(spec),
        )
        if spec["exits"]["trail"]["kind"] == "owner":
            self.notes.append("Owner trail uses the ATR(14) of this timeframe at the signal bar "
                              "(live trails on a rolling 5m ATR).")

    def signals(self, loaded):
        return spec_signals(self.compiled, loaded.frame)

    def config(self, capital, entry_from, entry_to, day_gate):
        spec = {**self.spec, "costs": {**self.spec["costs"], "slippage_pct": self.slippage_pct}}
        gate = day_gate if self.spec["limits"].get("owner_cooldown") else None
        return spec_engine_config(spec, capital=capital, entry_from=entry_from, entry_to=entry_to,
                                  day_gate=gate)


class ScreenerPlan(Plan):
    def __init__(self, params: dict, req: RunRequest):
        from apps.strategies.services.templates import BROKEN_SCREENER_STRATEGIES
        from plugins.strategy_screener.strategies import STRATEGIES
        enabled = [s.name for s in STRATEGIES if s.enabled]
        names = enabled
        if params.get("only_strategy"):
            names = [n for n in enabled if n.lower() == str(params["only_strategy"]).lower()]
            if not names:
                raise ValueError(f"only_strategy {params['only_strategy']!r} is not an enabled screener setup "
                                 f"({', '.join(enabled)})")
        disabled = {str(n).lower() for n in params.get("disabled_strategies") or []}
        names = [n for n in names if n.lower() not in disabled]
        if not names:
            raise ValueError("every screener setup is disabled — nothing to backtest")
        self.names = names
        lim = live_limits()
        self.risk_pct = min(float(params.get("max_risk_pct") or 1.0), lim["max_risk_pct"])
        syms = req.symbols or params.get("symbols") or []
        super().__init__(timeframe="1m", symbols=list(syms), warmup_bars=1,
                         exits=[RatchetStop(1.0, 1.5), _stop(), _target()],
                         slippage_pct=0.02 if req.slippage_pct is None else req.slippage_pct,
                         cost_model="nse_intraday_mis", keep_raw=True,
                         live_gate={"passes": True, "min_rr": lim["min_rr"],
                                    "message": "Setups target 1.5R–2.0R."})
        broken = [n for n in names if n in BROKEN_SCREENER_STRATEGIES]
        for n in broken:
            self.warnings.append(f"{n} never fires: {BROKEN_SCREENER_STRATEGIES[n]}.")
        if broken and len(broken) == len(names):
            raise ValueError("only broken screener setups selected (they never fire): " + ", ".join(broken))
        self.notes += ["Signals: live ScreenerEngine replayed tick-by-tick on DB 1m bars "
                       "(strategy_search.screener_adapter).",
                       "Exits assumed (live screener only alerts): signal stop/target, owner trail with "
                       "5m ATR(14), flat by 15:14; no fills on bars after 14:30.",
                       f"Setups: {', '.join(names)}."]

    def signals(self, loaded):
        from plugins.strategy_search.data import resample
        from plugins.strategy_search.screener_adapter import ScreenerSignalSource

        from apps.strategies.services import indicators as ind
        raw = loaded.raw
        src = ScreenerSignalSource(self.names)
        intents = sorted(src.generate(raw), key=lambda it: (it.signal_t, -it.priority))
        # 5m ATR(14), continuous across sessions, as published at each 5m close.
        m1 = raw.all_bars()
        b5 = resample(m1, 5)
        atr5 = ind.atr([b.h for b in b5], [b.l for b in b5], [b.c for b in b5], 14)
        ends5 = [b.t + 300 for b in b5]
        frame = loaded.frame
        # engine 1m bar epoch starts, recovered from the frame (same order as m1 cut to range)
        from plugins.strategy_search.data import epoch_of_ist
        epochs = [epoch_of_ist(datetime.fromisoformat(t[:19])) for t in frame.timestamps]
        idx = {t: k for k, t in enumerate(epochs)}
        out: dict[int, EntrySignal] = {}
        k5 = -1
        atr_at: list[float] = []
        for t in epochs:
            while k5 + 1 < len(ends5) and ends5[k5 + 1] <= t + 60:
                k5 += 1
            atr_at.append(atr5[k5] if k5 >= 0 and atr5[k5] is not None else 0.0)
        for it in intents:
            j = idx.get(it.signal_t)
            if j is None or j == 0 or frame.session[j - 1] != frame.session[j]:
                self.dropped_signals["no_bar_at_signal_time"] = self.dropped_signals.get(
                    "no_bar_at_signal_time", 0) + 1
                continue
            i = j - 1
            if i in out:
                continue
            side = TradeSide.BUY if it.side > 0 else TradeSide.SHORT
            px = frame.close[i]
            out[i] = EntrySignal(symbol=frame.symbol, timestamp=frame.timestamps[i], side=side,
                                 entry_price=px, stoploss=it.stop, target=it.target,
                                 risk_points=abs(px - it.stop), atr=atr_at[i],
                                 metadata={"levels": "absolute", "trade_type": it.strategy,
                                           "strategy": it.strategy, "engine_rr": it.meta.get("rr")})
        loaded.raw = None
        return out

    def config(self, capital, entry_from, entry_to, day_gate):
        lim = live_limits()
        return engine_config(timeframe="1m", capital=capital, risk_pct=self.risk_pct,
                             max_position_pct=lim["max_position_pct"], max_positions=lim["max_positions"],
                             daily_loss_pct=lim["max_daily_loss_pct"], slippage_pct=self.slippage_pct,
                             cost_model="nse_intraday_mis", entry_from=entry_from, entry_to=entry_to,
                             eod_time=lim["square_off"], last_entry_time="14:30", day_gate=day_gate)


class OKSwingPlan(Plan):
    def __init__(self, params: dict, req: RunRequest):
        lim = live_limits()
        syms = req.symbols or params.get("symbols") or []
        super().__init__(timeframe="1d", symbols=list(syms), warmup_bars=80,
                         exits=[RatchetStop(1.0, 1.5), _stop(), _target(), _maxhold(10)],
                         slippage_pct=0.05 if req.slippage_pct is None else req.slippage_pct,
                         cost_model="nse_delivery",
                         live_gate={"passes": True, "min_rr": lim["min_rr"], "message": "Setups target 2R."})
        self.notes += ["Signals: OK CycleDetector (plugins/strategy_swing) on DB daily bars, walk-forward.",
                       "Needs ~60 daily bars before the first signal; DB daily data starts 2026-04-20.",
                       "Exits: setup stop + 2R target, owner trail (daily ATR14), 10-day max hold."]
        self.start = req.from_date

    def signals(self, loaded):
        from plugins.strategy_backtest.entry import OKCycleAdapter
        adapter = OKCycleAdapter(min_rr=2.0)
        frame = loaded.frame
        atr14 = indicator_series(frame, {"kind": "atr", "period": 14})
        out: dict[int, EntrySignal] = {}
        lo = self.start.isoformat()
        for i in range(len(loaded.bars)):
            if frame.session[i] < lo:
                continue
            for s in adapter.detect(frame.symbol, loaded.bars, i, {}):
                if s.side != TradeSide.BUY:
                    self.dropped_signals["short_setup_skipped"] = self.dropped_signals.get(
                        "short_setup_skipped", 0) + 1
                    continue
                s.metadata = {**s.metadata, "levels": "absolute", "strategy": "ok_cycle"}
                s.atr = atr14[i] or 0.0
                out[i] = s
                break
        return out

    def config(self, capital, entry_from, entry_to, day_gate):
        lim = live_limits()
        return engine_config(timeframe="1d", capital=capital, risk_pct=lim["max_risk_pct"],
                             max_position_pct=lim["max_position_pct"], max_positions=lim["max_positions"],
                             daily_loss_pct=lim["max_daily_loss_pct"], slippage_pct=self.slippage_pct,
                             cost_model="nse_delivery", entry_from=entry_from, entry_to=entry_to,
                             eod_time=None, day_gate=day_gate)


def _stop():
    from plugins.strategy_backtest.exits import StoplossExit
    return StoplossExit(gap_aware=True)


def _target():
    from plugins.strategy_backtest.exits import TargetExit
    return TargetExit()


def _maxhold(n):
    from plugins.strategy_backtest.exits import MaxHoldExit
    return MaxHoldExit(n)


PLUGIN_PLANS = {"intraday_screener": ScreenerPlan, "swing_scanner": OKSwingPlan}


def make_plan(req: RunRequest) -> Plan:
    if req.kind == "spec":
        from apps.strategies.services.spec import validate_spec
        res = validate_spec(req.definition)
        if not res.ok:
            raise ValueError("invalid spec: " + "; ".join(f"{e['path']}: {e['message']}" for e in res.errors))
        return SpecPlan(res.spec, req)
    cls = PLUGIN_PLANS.get(req.strategy_name)
    if cls is None:
        from apps.strategies.services.templates import backtestability
        b = backtestability("plugin", req.strategy_name)
        raise ValueError(f"{req.strategy_name} can't be backtested deterministically: {b['reason']}")
    return cls(req.definition or {}, req)


# ── owner cooldown gate ─────────────────────────────────────────────────────

class OwnerCooldownGate:
    """TRADING_RULES losing-streak cooldown, via the research harness's
    incremental tracker (mirrors apps/trading/services/cooldown.py)."""

    def __init__(self, sessions: set[str]):
        from django.conf import settings

        from plugins.strategy_search.rules import CooldownTracker
        r = getattr(settings, "TRADING_RULES", {})
        known = {date.fromisoformat(s) for s in sessions}
        first, last = (min(known), max(known)) if known else (None, None)

        def is_trading_day(d: date) -> bool:
            if first and first <= d <= last:
                return d in known
            from apps.market_data.services.market_calendar import is_trading_day as cal
            return cal(d)

        self.tracker = CooldownTracker(
            is_trading_day=is_trading_day, loss_days=int(r.get("COOLDOWN_LOSS_DAYS", 3)),
            cooldown_days=int(r.get("COOLDOWN_TRADING_DAYS", 2)),
            reduced_mult=float(r.get("REDUCED_RISK_MULTIPLIER", 0.5)))
        self.cooldown_days: set[str] = set()

    def status(self, day: str):
        st, mult = self.tracker.status(date.fromisoformat(day))
        if mult <= 0:
            self.cooldown_days.add(day)
        return st, mult

    def close_day(self, day: str, net: float):
        self.tracker.close_day(date.fromisoformat(day), net)


# ── metrics ─────────────────────────────────────────────────────────────────

def _r(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))):
        return None
    return round(x, nd)


def compute_metrics(trades: list[dict], capital: float) -> dict:
    n = len(trades)
    out = {"trades": n, "too_few_trades": n < MIN_TRADES_MEANINGFUL,
           "min_trades": MIN_TRADES_MEANINGFUL}
    if not n:
        out.update(win_rate=None, avg_r=None, expectancy_r=None, profit_factor=None, net_pnl=0.0,
                   gross_pnl=0.0, costs=0.0, return_pct=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                   wins=0, losses=0, avg_win=None, avg_loss=None, total_r=0.0, expectancy_inr=None,
                   best=None, worst=None, avg_bars_held=None, long_trades=0, short_trades=0)
        return out
    net = [t["net_pnl"] for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    gw, gl = sum(wins), -sum(losses)
    rs = [t["r_multiple"] for t in trades]
    eq, peak, mdd = capital, capital, 0.0
    for t in sorted(trades, key=lambda t: (t["exit_time"], t["seq"])):
        eq += t["net_pnl"]
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    out.update(
        wins=len(wins), losses=len(losses), win_rate=_r(len(wins) / n * 100, 1),
        avg_r=_r(sum(rs) / n, 3), expectancy_r=_r(sum(rs) / n, 3), total_r=_r(sum(rs), 2),
        expectancy_inr=_r(sum(net) / n), profit_factor=_r(gw / gl, 2) if gl > 0 else None,
        net_pnl=_r(sum(net)), gross_pnl=_r(sum(t["gross_pnl"] for t in trades)),
        costs=_r(sum(t["costs"] for t in trades)), return_pct=_r(sum(net) / capital * 100, 2),
        max_drawdown=_r(mdd), max_drawdown_pct=_r(mdd / capital * 100, 2),
        avg_win=_r(gw / len(wins)) if wins else None, avg_loss=_r(-gl / len(losses)) if losses else None,
        best=_r(max(net)), worst=_r(min(net)),
        avg_bars_held=_r(sum(t["bars_held"] for t in trades) / n, 1),
        long_trades=sum(1 for t in trades if t["side"] == "BUY"),
        short_trades=sum(1 for t in trades if t["side"] != "BUY"),
    )
    if gl == 0:
        out["profit_factor_note"] = "no losing trades"
    return out


def equity_curve(trades: list[dict], capital: float, max_points: int = 1500) -> list[dict]:
    eq, peak, pts = capital, capital, []
    for t in sorted(trades, key=lambda t: (t["exit_time"], t["seq"])):
        eq += t["net_pnl"]
        peak = max(peak, eq)
        pts.append({"t": t["exit_time"], "equity": round(eq, 2), "drawdown": round(eq - peak, 2)})
    if len(pts) > max_points:
        step = math.ceil(len(pts) / max_points)
        pts = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
    return pts


def _trade_row(seq: int, t, oos_start: str | None) -> dict:
    meta = t.metadata or {}
    tgt = t.target if t.target not in (float("inf"), float("-inf")) and t.target else None
    seg = "oos" if (oos_start and t.entry_time[:10] >= oos_start) else "is"
    return {
        "seq": seq, "symbol": t.symbol, "side": t.side.value, "segment": seg,
        "entry_time": t.entry_time, "exit_time": t.exit_time,
        "entry_price": round(t.entry_price, 2), "exit_price": round(t.exit_price, 2),
        "stop": round(float(meta.get("initial_stop", t.stoploss)), 2),
        "target": round(tgt, 2) if tgt else None, "qty": int(t.quantity),
        "gross_pnl": round(float(meta.get("gross_pnl", t.pnl)), 2),
        "costs": round(float(meta.get("costs", 0.0)), 2), "net_pnl": round(t.pnl, 2),
        "r_multiple": round(t.rr_achieved, 3), "exit_reason": t.exit_reason[:40],
        "bars_held": int(t.bars_held),
        "meta": {k: v for k, v in {
            "setup": meta.get("strategy") or meta.get("trade_type"),
            "signal_time": meta.get("signal_time"), "risk_mult": meta.get("risk_mult"),
            "mfe": round(t.max_favorable, 2), "mae": round(t.max_adverse, 2),
        }.items() if v is not None},
    }


# ── the run ─────────────────────────────────────────────────────────────────

def execute(req: RunRequest, loader, progress=lambda pct, note: None, should_cancel=lambda: False) -> dict:
    if req.to_date < req.from_date:
        raise ValueError("to_date is before from_date")
    plan = make_plan(req)
    symbols = [s.strip().upper() for s in plan.symbols if s and s.strip()]
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise ValueError("no symbols to backtest — pick a universe")
    if len(symbols) > MAX_SYMBOLS:
        raise ValueError(f"{len(symbols)} symbols is above the {MAX_SYMBOLS}-symbol limit per run")

    detector = PrecomputedDetector()
    data, stats, missing = {}, {}, []
    total_bars, n_signals = 0, 0
    sessions_in: set[str] = set()
    sessions_all: set[str] = set()
    lo, hi = req.from_date.isoformat(), req.to_date.isoformat()
    progress(2, f"loading {len(symbols)} symbol(s)")
    for k, sym in enumerate(symbols):
        if should_cancel():
            raise Cancelled()
        loaded = loader.load(sym, plan.timeframe, req.from_date, req.to_date, plan.warmup_bars,
                             keep_raw=plan.keep_raw)
        if loaded is None or not loaded.bars or not loaded.stats.get("sessions"):
            missing.append(sym)
            progress(2 + int(80 * (k + 1) / len(symbols)), f"{sym}: no data")
            continue
        total_bars += len(loaded.bars)
        if total_bars > MAX_ENGINE_BARS:
            raise ValueError(f"run too large: over {MAX_ENGINE_BARS:,} {plan.timeframe} bars after {sym} — "
                             "use fewer symbols, a shorter period or a longer timeframe")
        sigs = plan.signals(loaded)
        n_signals += len(sigs)
        detector.set(sym, sigs)
        data[sym] = loaded.bars
        stats[sym] = loaded.stats
        sessions_all.update(loaded.frame.session)
        sessions_in.update(s for s in loaded.frame.session if lo <= s <= hi)
        loaded.frame._cache.clear()
        progress(2 + int(80 * (k + 1) / len(symbols)), f"{sym}: {len(sigs)} signal(s)")
    if not data:
        raise ValueError("no candle data for any requested symbol in this period")
    if should_cancel():
        raise Cancelled()

    sess = sorted(sessions_in)
    oos_start = None
    if len(sess) >= 5 and req.oos_fraction > 0:
        oos_start = sess[int(len(sess) * (1 - req.oos_fraction))]
    gate = OwnerCooldownGate(sessions_all)
    cfg = plan.config(req.capital, lo, hi, gate)
    progress(85, "running engine")
    engine = BacktestEngine(config=cfg, entry=detector, exits=plan.exits)
    engine.run(data)
    closed = sorted(engine._closed, key=lambda t: (t.entry_time, t.symbol))
    rows = [_trade_row(i + 1, t, oos_start) for i, t in enumerate(closed)]

    metrics = compute_metrics(rows, req.capital)
    split = {
        "is": compute_metrics([r for r in rows if r["segment"] == "is"], req.capital),
        "oos": compute_metrics([r for r in rows if r["segment"] == "oos"], req.capital),
        "oos_start": oos_start, "oos_fraction": req.oos_fraction if oos_start else 0,
    }
    curve = equity_curve(rows, req.capital)
    daily = plan.timeframe == "1d"
    model = cost_models.resolve(plan.cost_model, daily)
    firsts = [s["first"] for s in stats.values() if s.get("first")]
    lasts = [s["last"] for s in stats.values() if s.get("last")]
    gaps = {s: v.get("gap_sessions", 0) for s, v in stats.items() if v.get("gap_sessions")}
    warnings = list(plan.warnings)
    if metrics["too_few_trades"]:
        warnings.append(f"Only {metrics['trades']} trade(s) — under {MIN_TRADES_MEANINGFUL}, the metrics "
                        "are anecdotes, not evidence.")
    if missing:
        warnings.append(f"No data for {len(missing)} symbol(s): {', '.join(missing[:10])}.")
    if gaps:
        warnings.append(f"{sum(gaps.values())} symbol-session(s) with gaps (<90% of expected bars).")
    if plan.live_gate and plan.live_gate.get("passes") is False:
        warnings.append(plan.live_gate["message"])
    overlap = req.from_date <= RESEARCH_OOS[1] and req.to_date >= RESEARCH_OOS[0]
    in_from, in_to = (sess[0], sess[-1]) if sess else (lo, hi)
    split_desc = ({"method": f"last {int(req.oos_fraction * 100)}% of sessions held out",
                   "is": [in_from, _prev(sess, oos_start)], "oos": [oos_start, in_to]}
                  if oos_start else {"method": "no split (fewer than 5 sessions)", "is": [in_from, in_to],
                                     "oos": None})
    provenance = {
        "data_source": loader.source_label(plan.timeframe), "synthetic": False,
        "timeframe": plan.timeframe,
        "requested": {"from": lo, "to": hi},
        "actual": {"first": min(firsts) if firsts else None, "last": max(lasts) if lasts else None},
        "sessions": len(sess), "symbols_requested": len(symbols), "symbols_loaded": sorted(data),
        "symbols_missing": missing, "bars": total_bars, "gap_sessions": sum(gaps.values()),
        "gaps_by_symbol": gaps,
        "split": split_desc,
        "label": "IS + OOS" if oos_start else "in-sample only",
        "research_oos_overlap": overlap,
        "research_note": ("Overlaps 2026-09-01→10-19, the strategy-search out-of-sample window that is "
                          "already spent for research claims; treat this period as in-sample for any "
                          "claim about the research strategies.") if overlap else "",
        "sample": {"trades": metrics["trades"], "oos_trades": split["oos"]["trades"],
                   "threshold": MIN_TRADES_MEANINGFUL, "too_few": metrics["too_few_trades"]},
        "costs": cost_models.describe(model, plan.slippage_pct),
        "engine": ENGINE_LABEL,
        "definition": {"kind": req.kind, "strategy": req.strategy_name, "version": req.version,
                       "hash": req.definition_hash},
        "signals": {"detected": n_signals, "taken_to_fill": engine._total_signals - engine._skipped,
                    "rejected": dict(engine.rejections), "dropped": dict(plan.dropped_signals)},
        "owner_cooldown_days": sorted(gate.cooldown_days),
        "live_gate": plan.live_gate,
        "notes": plan.notes, "warnings": warnings,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    progress(97, "saving")
    return {"trades": rows, "metrics": metrics, "split": split, "equity_curve": curve,
            "provenance": provenance}


def _prev(sess: list[str], d: str | None) -> str | None:
    if not d or d not in sess:
        return None
    i = sess.index(d)
    return sess[i - 1] if i else None


def default_loader(end: date):
    from apps.strategies.services.candles import DbCandleLoader
    return DbCandleLoader(end=end)


# ── Django wrapper ──────────────────────────────────────────────────────────

def request_for(bt) -> RunRequest:
    inst = bt.instance
    ver = bt.version
    cfg = bt.config or {}
    if inst.kind == "spec":
        definition = ver.spec if ver else inst.params
    else:
        definition = ver.params if ver else inst.params
    return RunRequest(
        kind=inst.kind, strategy_name=inst.strategy_name, definition=definition or {},
        from_date=bt.from_date, to_date=bt.to_date, symbols=cfg.get("symbols") or None,
        capital=float(cfg.get("capital") or 500_000), slippage_pct=cfg.get("slippage_pct"),
        oos_fraction=float(cfg.get("oos_fraction", OOS_FRACTION)),
        version=ver.version if ver else None, definition_hash=ver.spec_hash if ver else "")


def run_backtest_job(backtest_id: str, loader=None) -> str:
    """Execute one Backtest row end to end. Returns the final status."""
    from django.db import transaction
    from django.utils import timezone as djtz

    from apps.strategies.models import Backtest, BacktestTrade, StrategyInstance

    bt = Backtest.objects.select_related("instance", "version").get(id=backtest_id)
    if bt.status not in (Backtest.Status.QUEUED, Backtest.Status.RUNNING):
        return bt.status
    if bt.cancel_requested:
        bt.status = Backtest.Status.CANCELLED
        bt.finished_at = djtz.now()
        bt.error = "cancelled before it started"
        bt.save()
        return bt.status
    bt.status = Backtest.Status.RUNNING
    bt.started_at = djtz.now()
    bt.progress, bt.progress_note = 1, "starting"
    bt.save()

    def progress(pct, note):
        Backtest.objects.filter(pk=bt.pk).update(progress=max(0, min(99, int(pct))), progress_note=note[:200])

    def should_cancel():
        return Backtest.objects.filter(pk=bt.pk, cancel_requested=True).exists()

    try:
        req = request_for(bt)
        loader = loader or default_loader(req.to_date)
        result = execute(req, loader, progress, should_cancel)
    except Cancelled:
        bt.refresh_from_db(fields=["progress", "progress_note"])
        bt.status = Backtest.Status.CANCELLED
        bt.error = "cancelled by the user"
        bt.finished_at = djtz.now()
        bt.save()
        return bt.status
    except Exception as e:  # noqa: BLE001 — the row records the failure
        log.warning("studio_backtest.failed", backtest_id=str(bt.id), error=str(e))
        bt.refresh_from_db(fields=["progress", "progress_note"])
        bt.status = Backtest.Status.FAILED
        bt.error = str(e)[:4000]
        bt.finished_at = djtz.now()
        bt.save()
        _emit(bt, "failed")
        return bt.status

    with transaction.atomic():
        BacktestTrade.objects.filter(backtest=bt).delete()
        BacktestTrade.objects.bulk_create(
            [BacktestTrade(tenant_id=bt.tenant_id, backtest=bt, **row) for row in result["trades"]],
            batch_size=1000)
        bt.metrics = result["metrics"]
        bt.split_metrics = result["split"]
        bt.equity_curve = result["equity_curve"]
        bt.data_provenance = result["provenance"]
        bt.status = Backtest.Status.DONE
        bt.progress, bt.progress_note = 100, "done"
        bt.finished_at = djtz.now()
        bt.save()
        inst = bt.instance
        if (inst.stage == StrategyInstance.Stage.DRAFT and bt.version_id
                and bt.version and bt.version.version == inst.current_version):
            StrategyInstance.objects.filter(pk=inst.pk).update(stage=StrategyInstance.Stage.BACKTESTED)
    _emit(bt, "done")
    return bt.status


def _emit(bt, outcome: str) -> None:
    try:
        from apps.events.services.event_writer import emit
        m = bt.metrics or {}
        emit(tenant=bt.instance.tenant, type="workflow.step.completed",
             step_name="strategy_studio.backtest",
             severity="info" if outcome == "done" else "warning",
             text=(f"Backtest {outcome}: {bt.instance.name} — {m.get('trades', 0)} trades, "
                   f"net ₹{m.get('net_pnl', 0)}" if outcome == "done"
                   else f"Backtest {outcome}: {bt.instance.name} — {bt.error[:200]}"),
             payload={"backtest_id": str(bt.id), "instance_id": str(bt.instance_id), "outcome": outcome,
                      "trades": m.get("trades"), "net_pnl": m.get("net_pnl")},
             broadcast=False)
    except Exception:  # noqa: BLE001 — never blocks
        pass
