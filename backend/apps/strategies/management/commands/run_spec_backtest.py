"""Backtest a StrategySpec (or a built-in template) from the CLI — no DB writes.

    nice -n 10 .venv/bin/python manage.py run_spec_backtest --template rule.orb_breakout \
        --from 2026-05-04 --to 2026-05-29 --symbols RELIANCE,HDFCBANK,INFY,ICICIBANK,TCS

    .venv/bin/python manage.py run_spec_backtest --spec my_spec.json --from ... --to ...
    .venv/bin/python manage.py run_spec_backtest --plugin intraday_screener --params '{}' ...

Reads candles one symbol at a time; prints metrics, the IS/OOS split and the
provenance record. ``--trades`` also prints every trade, ``--json`` the raw result.
"""
from __future__ import annotations

import json
from datetime import date

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Deterministic Studio backtest of a spec / template from the CLI (read-only)"

    def add_arguments(self, p):
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--spec", help="path to a StrategySpec JSON file")
        src.add_argument("--template", help="built-in template key, e.g. rule.orb_breakout")
        src.add_argument("--plugin", help="plugin name, e.g. intraday_screener or swing_scanner")
        p.add_argument("--params", default="{}", help="plugin params JSON (with --plugin)")
        p.add_argument("--from", dest="start", required=True)
        p.add_argument("--to", dest="end", required=True)
        p.add_argument("--symbols", default="", help="comma-separated override of the universe")
        p.add_argument("--capital", type=float, default=500_000)
        p.add_argument("--slippage", type=float, default=None, help="% per side override")
        p.add_argument("--trades", action="store_true")
        p.add_argument("--json", action="store_true")

    def handle(self, *a, **o):
        from apps.strategies.services.backtest_service import RunRequest, execute
        from apps.strategies.services.candles import DbCandleLoader
        from apps.strategies.services.templates import get_template

        if o["spec"]:
            with open(o["spec"]) as f:
                kind, name, definition = "spec", "spec", json.load(f)
        elif o["template"]:
            t = get_template(o["template"])
            if t is None:
                raise CommandError(f"unknown template {o['template']}")
            kind, name = t["kind"], t["strategy_name"]
            definition = t["spec"] if kind == "spec" else (t["defaults"] or {})
        else:
            kind, name, definition = "plugin", o["plugin"], json.loads(o["params"])
        start, end = date.fromisoformat(o["start"]), date.fromisoformat(o["end"])
        syms = [s.strip().upper() for s in o["symbols"].split(",") if s.strip()] or None
        req = RunRequest(kind=kind, strategy_name=name, definition=definition, from_date=start,
                         to_date=end, symbols=syms, capital=o["capital"], slippage_pct=o["slippage"])

        def progress(pct, note):
            self.stderr.write(f"  [{pct:3d}%] {note}")

        try:
            res = execute(req, DbCandleLoader(end=end), progress)
        except ValueError as e:
            raise CommandError(str(e)) from e
        if o["json"]:
            self.stdout.write(json.dumps(res, indent=2, default=str))
            return
        m, pv = res["metrics"], res["provenance"]
        self.stdout.write(f"\n{name} {start} → {end} · {pv['timeframe']} · {len(pv['symbols_loaded'])} symbols "
                          f"· {pv['sessions']} sessions · {pv['bars']:,} bars")
        for label, mm in (("ALL", m), ("IS", res["split"]["is"]), ("OOS", res["split"]["oos"])):
            self.stdout.write(
                f"  {label:<3} n={mm['trades']:<4} win%={mm['win_rate']} avgR={mm['avg_r']} "
                f"PF={mm['profit_factor']} net=₹{mm['net_pnl']} (gross ₹{mm['gross_pnl']}, costs ₹{mm['costs']}) "
                f"maxDD=₹{mm['max_drawdown']} totalR={mm['total_r']}")
        self.stdout.write(f"  split: {pv['split']}")
        self.stdout.write(f"  signals: {pv['signals']}")
        for w in pv["warnings"]:
            self.stdout.write(self.style.WARNING(f"  ! {w}"))
        for n in pv["notes"]:
            self.stdout.write(f"  · {n}")
        if o["trades"]:
            for t in res["trades"]:
                self.stdout.write(f"  {t['seq']:>4} {t['symbol']:<12} {t['side']:<5} {t['entry_time']} → "
                                  f"{t['exit_time']} {t['entry_price']}→{t['exit_price']} qty={t['qty']} "
                                  f"net=₹{t['net_pnl']} R={t['r_multiple']} {t['exit_reason']} [{t['segment']}]")
