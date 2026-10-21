"""Rank funds within a category (S4). PERSONAL USE ONLY.

Usage:
    .venv/bin/python manage.py screen_mf --category "Equity Scheme - Flexi Cap Fund" --top 10
    .venv/bin/python manage.py screen_mf --category "Equity Scheme - Mid Cap Fund" --min-years 5 --json
    .venv/bin/python manage.py screen_mf --list-categories

Reads MFMetricSnapshot (run compute_mf_metrics first). Component columns are
0-100 within-category percentiles: CONS consistency, RISK risk-adjusted,
DD drawdown, COST (n/a until TER is sourced), AGE size/age.
"""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from apps.mutual_funds.models import MFMetricSnapshot
from apps.mutual_funds.services import screener_config as cfg
from apps.mutual_funds.services.screener import ScreenCriteria, screen
from apps.mutual_funds.services.universe import canonical_category


def _pct(x, nd=1):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def _n(x, nd=0):
    return "n/a" if x is None else f"{x:.{nd}f}"


class Command(BaseCommand):
    help = "Screen + rank mutual funds within a category (deterministic, personal use)."

    def add_arguments(self, parser):
        parser.add_argument("--category", action="append", default=None)
        parser.add_argument("--top", type=int, default=10)
        parser.add_argument("--min-years", type=float, default=3.0)
        parser.add_argument("--min-sharpe", type=float, default=None)
        parser.add_argument("--max-drawdown", type=float, default=None, help="e.g. -0.35")
        parser.add_argument("--exclude-flag", action="append", default=[])
        parser.add_argument("--weights", default=None, help='JSON, e.g. {"consistency":40,"cost":0}')
        parser.add_argument("--no-index-check", action="store_true")
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--list-categories", action="store_true")

    def handle(self, *args, **o):
        if o["list_categories"]:
            for c in sorted(set(MFMetricSnapshot.objects.values_list("category", flat=True))):
                self.stdout.write(c)
            return
        if not o["category"]:
            raise CommandError("Pass --category (see --list-categories).")
        cats = [canonical_category(f"({c})" if "(" not in c else c) for c in o["category"]]
        weights = dict(cfg.DEFAULT_WEIGHTS)
        if o["weights"]:
            weights.update(json.loads(o["weights"]))
        crit = ScreenCriteria(categories=cats, min_history_years=o["min_years"], min_sharpe=o["min_sharpe"],
                              max_drawdown_floor=o["max_drawdown"], exclude_flags=o["exclude_flag"],
                              weights=weights, top=o["top"])
        result = screen(crit, with_index_check=not o["no_index_check"])
        if o["json"]:
            self.stdout.write(json.dumps(result, indent=2, default=str))
            return
        w = self.stdout.write
        w(result["personal_use_note"])
        for c in result["categories"]:
            w(f"\n== {c['category']} (as of {c['as_of']}) — {c['eligible_count']} eligible of "
              f"{c['universe_count']}; excluded {c['excluded'] or 'none'}")
            w(f"   weights used: {c['weights_used']}")
            for note in c["notes"]:
                w(f"   note: {note}")
            w(f"{'#':>3} {'code':>7} {'scheme':44} {'score':>5} {'CONS':>5} {'RISK':>5} {'DD':>5} {'COST':>5} "
              f"{'AGE':>5} {'3y CAGR':>8} {'roll3y med':>10} {'>cat%':>6} {'sortino':>7} {'maxDD':>7} flags")
            for r in c["rows"]:
                comp = r["components"]
                w(f"{r['rank']:>3} {r['scheme_code']:>7} {r['scheme_name'][:44]:44} {_n(r['score']):>5} "
                  f"{_n(comp['consistency']):>5} {_n(comp['risk_adjusted']):>5} {_n(comp['drawdown']):>5} "
                  f"{_n(comp['cost']):>5} {_n(comp['size_age']):>5} {_pct(r['cagr_3y']):>8} "
                  f"{_pct(r['rolling_3y_median']):>10} {_n(r['rolling_3y_pct_above_cat_median']):>6} "
                  f"{_n(r['sortino'], 2):>7} {_pct(r['max_drawdown']):>7} {','.join(r['quality_flags'] or [])}")
            ic = c.get("index_check")
            if ic:
                w(f"   index-vs-active ({ic['basis']}): {ic.get('text')}")
                if ic.get("windows"):
                    w(f"     reference: {ic['reference']}; median active fund beat it in "
                      f"{ic.get('pct_windows_median_active_beat')}% of {ic['windows']} rolling 3y windows "
                      f"({ic.get('first_window_end')}..{ic.get('last_window_end')}; last 3y: "
                      f"{ic.get('pct_windows_recent_3y')}%), median gap {ic.get('median_gap_pp')} pp/yr; "
                      f"funds beating it in most of their windows: {ic.get('funds_beating_reference_majority')}")
