"""Step 1 — bias verification, IN-SAMPLE ONLY (docs/strategies/STRATEGY_SEARCH_RESULTS.md).

    nice -n 10 .venv/bin/python manage.py strategy_bias [--symbols A,B] [--start/--end]

No --allow-oos: this command can never read OOS dates. One symbol at a time,
DB only, no broker calls. Outputs in research_output/strategy_search/bias_is/:
days.csv (one row per stock-day), tables.md (descriptive + hypothetical
trades), and <strategy>/<config>/ harness outputs for every tradeable variant,
plus harness_summary.md.
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from django.core.management.base import BaseCommand

from apps.market_data.constants import NIFTY_50_SYMBOLS
from plugins.strategy_search.bias import STRATEGY_NOTES, BiasSources, bhav_loader, bias_tables
from plugins.strategy_search.periods import resolve_period
from plugins.strategy_search.runner import _write_csv, collect_candidates, write_outputs
from plugins.strategy_search.sim import PortfolioSimulator, SimConfig


def configs() -> list[SimConfig]:
    base = SimConfig.from_settings
    return [
        base(label="base"),                                  # RiskEngine: 10 % cap, 3 open
        base(label="cap100", max_position_pct=100.0),
        base(label="cap500", max_position_pct=500.0),        # ~MIS 5x
        # every signal taken (per-trade edge, no slot selection / loss gates / cooldown)
        base(label="all_cap10", max_open_positions=1000, max_daily_loss_pct=1000.0,
             cooldown=False),
        base(label="all_cap500", max_open_positions=1000, max_daily_loss_pct=1000.0,
             cooldown=False, max_position_pct=500.0),
    ]


class Command(BaseCommand):
    help = "Bias verification (open vs PP, PP untouched, CPR side, H3/L3 touch, gap fill) on IS."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default="")
        parser.add_argument("--start", default=None)
        parser.add_argument("--end", default=None)
        parser.add_argument("--out-dir", default="research_output/strategy_search/bias_is")

    def handle(self, *a, **o):
        period = resolve_period("is", date.fromisoformat(o["start"]) if o["start"] else None,
                                date.fromisoformat(o["end"]) if o["end"] else None)
        syms = [s.strip() for s in o["symbols"].split(",") if s.strip()] or list(NIFTY_50_SYMBOLS)
        out = Path(o["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        cfgs = configs()
        from plugins.strategy_search.data import CandleSource
        src = CandleSource(max_date=period.end)
        source = BiasSources(bhav_loader(src))
        t0 = time.time()
        cands, rej, src = collect_candidates(source, syms, period, cfgs[0], 3,
                                             lambda m: self.stdout.write(m), src)
        self.stdout.write(f"phase 1 done: {len(cands)} candidates, {len(source.days)} stock-days "
                          f"({time.time() - t0:.0f}s)")
        _write_csv(out / "days.csv", source.days)
        (out / "tables.md").write_text(bias_tables(source.days))

        sessions = src.sessions(period.start, period.end)
        is_td = src.calendar()
        by_strat = defaultdict(list)
        for c in cands:
            by_strat[c.intent.strategy].append(c)
        lines = ["| strategy | config | n | win % | PF | net ₹ | R/trade net | gross R | cost R | "
                 "max DD ₹ | % capped | exits |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in STRATEGY_NOTES:
            cs = by_strat.get(name, [])
            for cfg in cfgs:
                res = PortfolioSimulator(cfg, is_td).run(cs, sessions, {})
                s = write_outputs(res, out / name / cfg.label, period, syms, name)
                if not s.get("trades"):
                    lines.append(f"| {name} | {cfg.label} | 0 | | | | | | | | | |")
                    continue
                pf = s["profit_factor"]
                lines.append(
                    f"| {name} | {cfg.label} | {s['trades']} | {s['win_rate'] * 100:.1f}% | "
                    f"{pf:.2f} | {s['net']:,.0f} | {s['r_per_trade']:+.3f} | "
                    f"{s['gross_r_per_trade']:+.3f} | {s['costs_r_per_trade']:.3f} | "
                    f"{s['max_drawdown']:,.0f} | {s['notional_capped_pct'] * 100:.0f}% | "
                    f"{s['exit_reasons']} |")
        rej_line = f"\nphase-1 rejects (all strategies): {dict(rej)}\n"
        (out / "harness_summary.md").write_text("\n".join(lines) + rej_line)
        self.stdout.write("\n".join(lines) + rej_line)
        self.stdout.write(f"done in {time.time() - t0:.0f}s → {out}")
