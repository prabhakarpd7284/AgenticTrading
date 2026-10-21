"""Strategy-search research harness CLI (docs/strategies/STRATEGY_SEARCH_RESULTS.md).

    nice -n 10 .venv/bin/python manage.py strategy_search info
    nice -n 10 .venv/bin/python manage.py strategy_search screener --period is \\
        [--symbols A,B] [--start 2026-05-04 --end 2026-05-15] [--strategies "Breakout Long,..."]
    nice -n 10 .venv/bin/python manage.py strategy_search run \\
        --source plugins.strategy_search.examples:OpeningRangeBreakout --period is \\
        [--source-kwargs '{"minutes": 15}']

Common sim flags: --risk-pct 0.5  --max-open N  --max-position-pct P
  --sensitivity-cap 100,500 (extra runs with larger notional caps, e.g. MIS)
  --sensitivity-max-open N (max open for the sensitivity runs)
  --no-trail  --no-cooldown  --min-rr 1.5  --live-rr-gate  --slippage-pct 0.02  --label NAME

OOS (2026-09-01 → 2026-10-19) is REFUSED unless --allow-oos is passed. Only
the OOS-validator role passes it, once. Reads the DB only; no broker calls.
Outputs: research_output/strategy_search/<label>/<config>/{trades,daily}.csv,
summary.json, summary.md.
"""
from __future__ import annotations

import importlib
import json
from datetime import date

from django.core.management.base import BaseCommand, CommandError

from apps.market_data.constants import NIFTY_50_SYMBOLS
from plugins.strategy_search.periods import (
    IS_END,
    IS_START,
    OOS_END,
    OOS_START,
    OOSAccessError,
    resolve_period,
)
from plugins.strategy_search.runner import run_search
from plugins.strategy_search.sim import SimConfig


def _d(s):
    return date.fromisoformat(s) if s else None


class Command(BaseCommand):
    help = "Honest intraday strategy search harness (IS by default; OOS needs --allow-oos)."

    def add_arguments(self, parser):
        parser.add_argument("subcommand", choices=["info", "screener", "run"])
        parser.add_argument("--period", default="is", choices=["is", "oos", "all"])
        parser.add_argument("--start", default=None)
        parser.add_argument("--end", default=None)
        parser.add_argument("--allow-oos", action="store_true",
                            help="ONLY for the OOS-validator role, once.")
        parser.add_argument("--symbols", default="", help="comma list (default NIFTY 50)")
        parser.add_argument("--strategies", default="", help="screener: comma list of names")
        parser.add_argument("--source", default="", help="run: module.path:Factory")
        parser.add_argument("--source-kwargs", default="{}", help="run: JSON kwargs for Factory")
        parser.add_argument("--label", default="")
        parser.add_argument("--out-dir", default="research_output/strategy_search")
        parser.add_argument("--warmup", type=int, default=3, help="warm-up sessions before start")
        parser.add_argument("--risk-pct", type=float, default=0.5)
        parser.add_argument("--max-open", type=int, default=None)
        parser.add_argument("--max-position-pct", type=float, default=None)
        parser.add_argument("--sensitivity-cap", default="",
                            help="comma list of larger notional caps in %% (e.g. 100,500)")
        parser.add_argument("--sensitivity-max-open", type=int, default=None)
        parser.add_argument("--min-rr", type=float, default=None)
        parser.add_argument("--live-rr-gate", action="store_true",
                            help="apply live RiskEngine R:R + target-required check "
                                 "(target-less intents rejected; min R:R 1.5 unless --min-rr)")
        parser.add_argument("--slippage-pct", type=float, default=0.02)
        parser.add_argument("--no-trail", action="store_true")
        parser.add_argument("--no-cooldown", action="store_true")

    def handle(self, *args, **o):
        sub = o["subcommand"]
        if sub == "info":
            return self._info()
        try:
            period = resolve_period(o["period"], _d(o["start"]), _d(o["end"]), o["allow_oos"])
        except OOSAccessError as e:
            raise CommandError(f"REFUSED: {e}") from e
        if period.touches_oos:
            self.stderr.write(self.style.WARNING(
                f"!! OUT-OF-SAMPLE run {period.start} → {period.end} (--allow-oos). "
                "OOS is to be touched once."))

        symbols = [s.strip() for s in o["symbols"].split(",") if s.strip()] or list(NIFTY_50_SYMBOLS)
        if sub == "screener":
            from plugins.strategy_search.screener_adapter import ScreenerSignalSource
            names = [s.strip() for s in o["strategies"].split(",") if s.strip()] or None
            source = ScreenerSignalSource(names)
        else:
            if ":" not in o["source"]:
                raise CommandError("run needs --source module.path:Factory")
            mod, attr = o["source"].split(":", 1)
            factory = getattr(importlib.import_module(mod), attr)
            source = factory(**json.loads(o["source_kwargs"]))
            if not hasattr(source, "name"):
                source.name = attr

        over = dict(risk_pct=o["risk_pct"], slippage_pct=o["slippage_pct"],
                    trail=not o["no_trail"], cooldown=not o["no_cooldown"], min_rr=o["min_rr"],
                    live_rr_gate=o["live_rr_gate"])
        if o["max_open"] is not None:
            over["max_open_positions"] = o["max_open"]
        if o["max_position_pct"] is not None:
            over["max_position_pct"] = o["max_position_pct"]
        base = SimConfig.from_settings(label="base", **over)
        configs = [base]
        for cap in [x for x in o["sensitivity_cap"].split(",") if x.strip()]:
            kw = dict(over, max_position_pct=float(cap))
            if o["sensitivity_max_open"] is not None:
                kw["max_open_positions"] = o["sensitivity_max_open"]
            configs.append(SimConfig.from_settings(label=f"cap{float(cap):g}", **kw))

        label = o["label"] or f"{getattr(source, 'name', 'source')}_{period.name}"
        out = f"{o['out_dir'].rstrip('/')}/{label}"
        self.stdout.write(f"{label}: {len(symbols)} symbols, {period.name} "
                          f"{period.start} → {period.end}, configs "
                          f"{[c.label for c in configs]} → {out}")
        results = run_search(source, symbols, period, configs, out_dir=out,
                             warmup_sessions=o["warmup"], log=self.stdout.write)
        for res in results:
            s = res.summary()
            self.stdout.write(
                f"[{res.config.label}] n={s.get('trades', 0)} "
                f"win={s.get('win_rate', 0) * 100:.1f}% PF={s.get('profit_factor', 0):.2f} "
                f"net=₹{s.get('net', 0):,.0f} R/trade={s.get('r_per_trade', 0):+.3f} "
                f"(gross {s.get('gross_r_per_trade', 0):+.3f}, costs "
                f"{s.get('costs_r_per_trade', 0):.3f}) maxDD=₹{s.get('max_drawdown', 0):,.0f} "
                f"trades/day={s.get('trades_per_day', 0):.2f} skipped={s.get('skipped')} "
                f"cooldown_days={s.get('cooldown_days')}")
        self.stdout.write(f"outputs: {out}/<config>/")

    def _info(self):
        from plugins.strategy_search.screener_adapter import enabled_strategies
        cfg = SimConfig.from_settings()
        self.stdout.write(f"IS  {IS_START} → {IS_END}\nOOS {OOS_START} → {OOS_END} "
                          "(needs --allow-oos)")
        self.stdout.write("SimConfig.from_settings(): " + json.dumps(cfg.as_dict(), default=str))
        self.stdout.write("Enabled screener strategies: "
                          + ", ".join(s.name for s in enabled_strategies()))
