"""Point-in-time backtest of the screener score (S5).

Usage:
    nice -n 10 .venv/bin/python manage.py mf_backtest_screen                       # equity categories, 12 + 36 months
    nice -n 10 .venv/bin/python manage.py mf_backtest_screen --category "Equity Scheme - Mid Cap Fund" --horizon 12
    ... --out /path/results.json

One category in memory at a time. See services/backtest.py for the method
(dead schemes included, chained with the category median after death).
"""
from __future__ import annotations

import json
import time
from datetime import date

from django.core.management.base import BaseCommand

from apps.mutual_funds.services.backtest import VARIANTS, backtest_category, summarise
from apps.mutual_funds.services.screener_config import CATEGORY_BENCHMARKS


def _p(x):
    return "  n/a" if x is None else f"{x * 100:5.1f}"


class Command(BaseCommand):
    help = "Quarterly point-in-time backtest of the MF screener score."

    def add_arguments(self, parser):
        parser.add_argument("--category", action="append", default=None)
        parser.add_argument("--horizon", type=int, action="append", default=None, help="months (repeatable)")
        parser.add_argument("--start", default="2016-03-31")
        parser.add_argument("--end", default=None)
        parser.add_argument("--out", default=None, help="write full JSON results here")

    def handle(self, *args, **o):
        cats = o["category"] or list(CATEGORY_BENCHMARKS)
        horizons = o["horizon"] or [12, 36]
        end = date.fromisoformat(o["end"]) if o["end"] else date.today()
        out = {"generated": date.today().isoformat(), "start": o["start"], "categories": {}}
        t0 = time.monotonic()
        for cat in cats:
            out["categories"][cat] = {}
            for h in horizons:
                res = backtest_category(cat, h, date.fromisoformat(o["start"]), end)
                s_all = summarise(res, 1)
                s_ind = summarise(res, max(1, h // 3))   # non-overlapping forward windows
                out["categories"][cat][str(h)] = {
                    "overlapping": s_all, "non_overlapping": s_ind,
                    "per_date": [{"t": r.t.isoformat(), "n": r.n, "passive": r.passive, "basis": r.passive_basis,
                                  "dead_chained": r.dead_chained, "groups": r.groups} for r in res],
                }
                self.stdout.write(f"\n{cat} — {h}m forward, {s_all['dates']} quarterly dates "
                                  f"({s_all.get('first')}..{s_all.get('last')}), ~{s_all.get('avg_funds')} funds/date, "
                                  f"passive mean {_p(s_all.get('passive_mean'))}% ({s_all.get('passive_basis')}) "
                                  f"[{time.monotonic() - t0:.0f}s]")
                self.stdout.write(f"  {'variant':20} {'top':>5} {'mid':>5} {'bot':>5} {'T-B':>5} {'T>B%':>5} "
                                  f"{'IC':>6} {'IC t*':>6} {'T>pas%':>6}")
                for name in VARIANTS:
                    v = s_all["variants"].get(name)
                    vi = s_ind["variants"].get(name) or {}
                    if not v:
                        continue
                    ict = vi.get("ic_t")
                    self.stdout.write(
                        f"  {name:20} {_p(v['top_mean'])} {_p(v['mid_mean'])} {_p(v['bottom_mean'])} "
                        f"{_p(v['top_minus_bottom'])} {v['top_beats_bottom_pct']:5.0f} "
                        f"{(v['ic_mean'] if v['ic_mean'] is not None else float('nan')):6.3f} "
                        f"{(ict if ict is not None else float('nan')):6.2f} "
                        f"{(v['top_beats_passive_pct'] if v['top_beats_passive_pct'] is not None else float('nan')):6.0f}")
                self.stdout.flush()
        if o["out"]:
            with open(o["out"], "w") as fh:
                json.dump(out, fh, indent=1, default=str)
            self.stdout.write(f"\nWrote {o['out']}")
        self.stdout.write(f"(* IC t-stat computed on non-overlapping dates only.) Done in {time.monotonic() - t0:.0f}s")
