"""Glue: signal source → per-symbol candidates → portfolio sim(s) → files.

A *signal source* is any object with ``name`` and
``generate(data: SymbolData) -> Iterable[Intent]`` (or a plain callable
``f(data) -> Iterable[Intent]``). It sees one symbol at a time.

    from plugins.strategy_search.periods import resolve_period
    from plugins.strategy_search.runner import run_search
    from plugins.strategy_search.sim import SimConfig

    res = run_search(my_source, symbols, resolve_period("is"),
                     [SimConfig.from_settings(label="base"),
                      SimConfig.from_settings(label="mis5x", max_position_pct=500)],
                     out_dir="research_output/strategy_search/my_strategy_is")

Phase 1 (candles + path resolution) runs once; every SimConfig re-sizes the
same candidates, so a sensitivity run costs almost nothing. ``SimConfig``
fields that change paths (slippage, trail, last entry) must be equal across
configs — checked.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .data import CandleSource
from .periods import Period
from .sim import Candidate, Intent, PortfolioSimulator, SimConfig, SimResult, resolve_symbol

PATH_FIELDS = ("slippage_pct", "trail", "trail_breakeven_r", "trail_atr_mult",
               "trail_atr_period", "last_entry_min")


def collect_candidates(source, symbols: Sequence[str], period: Period, cfg: SimConfig,
                       warmup_sessions: int = 3, log: Callable[[str], None] = print,
                       src: CandleSource | None = None
                       ) -> tuple[list[Candidate], Counter, CandleSource]:
    src = src or CandleSource(max_date=period.end, allow_oos=period.allow_oos)
    gen = source.generate if hasattr(source, "generate") else source
    cands: list[Candidate] = []
    rej: Counter = Counter()
    for sym in symbols:
        t0 = time.time()
        data = src.load_symbol(sym, period.start, period.end, warmup_sessions)
        if not data.sessions:
            log(f"{sym}: no 1m data in range")
            continue
        intents: list[Intent] = []
        for it in gen(data):
            if not period.contains(it.day) or it.symbol != sym:
                raise ValueError(f"intent outside period/symbol: {it.symbol} {it.day}")
            intents.append(it)
        c, r = resolve_symbol(data, intents, cfg)
        cands.extend(c)
        rej.update(r)
        log(f"{sym}: {len(data.period_sessions)} sessions, {len(intents)} intents, "
            f"{len(c)} candidates ({time.time() - t0:.1f}s)")
        del data, intents
        gc.collect()
    return cands, rej, src


def run_search(source, symbols: Sequence[str], period: Period, configs: Sequence[SimConfig],
               out_dir: str | Path | None = None, warmup_sessions: int = 3,
               log: Callable[[str], None] = print) -> list[SimResult]:
    configs = list(configs)
    for c in configs[1:]:
        for f in PATH_FIELDS:
            if getattr(c, f) != getattr(configs[0], f):
                raise ValueError(f"configs differ in path field {f}; run them separately")
    cands, rej, src = collect_candidates(source, symbols, period, configs[0],
                                         warmup_sessions, log)
    sessions = src.sessions(period.start, period.end)
    is_td = src.calendar()
    results = []
    for cfg in configs:
        res = PortfolioSimulator(cfg, is_td).run(cands, sessions, dict(rej))
        results.append(res)
        if out_dir:
            write_outputs(res, Path(out_dir) / cfg.label, period, symbols,
                          getattr(source, "name", getattr(source, "__name__", "source")))
    return results


def write_outputs(res: SimResult, out: Path, period: Period, symbols: Sequence[str],
                  source_name: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in res.rows]
    _write_csv(out / "trades.csv", rows)
    _write_csv(out / "daily.csv", res.daily)
    summ = res.summary()
    meta = {"source": source_name, "period": period.name, "start": period.start.isoformat(),
            "end": period.end.isoformat(), "symbols": list(symbols),
            "sim_config": res.config.as_dict()}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summ},
                                                 indent=2, default=_json_default))
    (out / "summary.md").write_text(summary_md(summ, f"{source_name} — {res.config.label} "
                                                     f"({period.name} {period.start}→{period.end})"))
    return summ


def _json_default(o):
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    return str(o)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _f(v, nd=2, pct=False):
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:,.{nd}f}" if isinstance(v, float) else str(v)


METRIC_COLS = ["trades", "win_rate", "avg_win", "avg_loss", "profit_factor", "gross", "costs",
               "net", "r_per_trade", "gross_r_per_trade", "costs_r_per_trade", "max_drawdown",
               "trades_per_day"]


def metric_row(label: str, b: dict) -> list[str]:
    if not b.get("trades"):
        return [label, "0"] + ["—"] * (len(METRIC_COLS) - 1)
    return [label, str(b["trades"]), _f(b["win_rate"], pct=True), _f(b["avg_win"], 0),
            _f(b["avg_loss"], 0), _f(b["profit_factor"]), _f(b["gross"], 0), _f(b["costs"], 0),
            _f(b["net"], 0), _f(b["r_per_trade"], 3), _f(b["gross_r_per_trade"], 3),
            _f(b["costs_r_per_trade"], 3), _f(b["max_drawdown"], 0), _f(b["trades_per_day"])]


def summary_md(s: dict, title: str) -> str:
    head = ["", "n", "win%", "avg win ₹", "avg loss ₹", "PF", "gross ₹", "costs ₹", "net ₹",
            "R/trade", "gross R", "cost R", "max DD ₹", "trades/day"]
    lines = [f"# {title}", "", "| " + " | ".join(head) + " |",
             "|" + "---|" * len(head), "| " + " | ".join(metric_row("ALL", s)) + " |"]
    for name, b in s.get("per_strategy", {}).items():
        lines.append("| " + " | ".join(metric_row(name, b)) + " |")
    lines += ["", f"- sessions {s['sessions']}, days traded {s.get('days_traded')}, "
                  f"positive days {s.get('days_positive')}",
              f"- return on capital {_f(s.get('return_pct_of_capital'))}%, daily max DD "
              f"₹{_f(s.get('max_drawdown_daily'), 0)}",
              f"- cooldown days {s.get('cooldown_days')}, reduced-risk days "
              f"{s.get('reduced_risk_days')}, shadow trades {s.get('shadow_trades')} "
              f"(shadow net ₹{_f(s.get('shadow_net'), 0)})",
              f"- max concurrent {s.get('max_concurrent')}, notional-capped "
              f"{_f(s.get('notional_capped_pct'), pct=True)}, cost share of gross wins "
              f"{_f(s.get('cost_share_of_gross_wins'), pct=True)}",
              f"- exit reasons {s.get('exit_reasons')}",
              f"- skipped {s.get('skipped')}", f"- phase-1 rejects {s.get('phase1_rejects')}", ""]
    return "\n".join(lines)


def as_source(fn: Callable[[object], Iterable[Intent]], name: str):
    """Wrap a plain function as a named signal source."""
    class _S:
        def __init__(self):
            self.name = name

        def generate(self, data):
            return fn(data)
    return _S()
