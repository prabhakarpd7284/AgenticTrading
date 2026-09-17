"""Show the StockEdge momentum shortlist for a score universe (e.g. Nifty 500).

Reads the latest ingested ``momentum_scores_<index>`` snapshot and prints a
ranked, filterable momentum shortlist with composite score, acceleration, and
classification (sustained / emerging / fading).

Examples
--------
    ./manage.py stockedge_momentum                       # top 15 by composite
    ./manage.py stockedge_momentum --class sustained --top 20
    ./manage.py stockedge_momentum --class emerging --min-accel 5
    ./manage.py stockedge_momentum --min-mcap 50000 --sector Banking
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Print the StockEdge momentum shortlist (composite/acceleration/classification)."

    def add_arguments(self, parser):
        parser.add_argument("--index", default="nifty_500", help="Universe slug (default: nifty_500).")
        parser.add_argument("--top", type=int, default=15, help="Rows to show (default: 15).")
        parser.add_argument("--class", dest="klass", default="",
                            choices=["", "sustained", "emerging", "fading", "neutral"],
                            help="Filter by classification.")
        parser.add_argument("--min-composite", type=float, default=None)
        parser.add_argument("--min-accel", type=float, default=None,
                            help="Minimum acceleration (1M-3M score).")
        parser.add_argument("--min-mcap", type=float, default=None, help="Min market cap (Rs. Cr.).")
        parser.add_argument("--sector", default="", help="Exact sector filter.")

    def handle(self, *args, **o):
        from apps.market_data.integrations.stockedge import momentum as M

        snap = M.latest_momentum_snapshot(o["index"])
        if snap is None:
            raise CommandError(
                f"No momentum_scores_{o['index']} snapshot found. Ingest one first: "
                "manage.py pull_stockedge_csv --file <…_MomentumScores_….csv>"
            )

        universe = M.momentum_universe(snap)
        summ = M.universe_summary(universe)

        self.stdout.write(self.style.SUCCESS(
            f"\nStockEdge momentum · {o['index']} · as_of {snap.as_of_date} · "
            f"{summ['count']} stocks"
        ))
        cls = summ["classes"]
        self.stdout.write(
            f"  classes: sustained {cls.get('sustained',0)} · emerging {cls.get('emerging',0)} "
            f"· fading {cls.get('fading',0)} · neutral {cls.get('neutral',0)}  |  "
            f"1M bullish {summ['bullish_1m']} ({summ['bullish_1m_pct']}%)  "
            f"avg composite {summ['avg_composite']}"
        )

        rows = M.shortlist(
            universe, top=o["top"],
            classification=o["klass"] or None,
            min_composite=o["min_composite"],
            min_acceleration=o["min_accel"],
            min_mcap_cr=o["min_mcap"],
            sector=o["sector"] or None,
        )
        if not rows:
            self.stdout.write(self.style.WARNING("  (no rows match the filters)"))
            return

        self._print_table(rows)

    def _print_table(self, rows: list[dict]) -> None:
        def n(v, w=6, dash="-"):
            return dash.rjust(w) if v is None else f"{v:>{w}}"

        header = (f"{'Symbol':<12}{'Sector':<22}{'LTP':>9}{'Chg%':>7}"
                  f"{'1M':>5}{'3M':>5}{'6M':>5}{'Comp':>7}{'Accel':>7}  Class")
        self.stdout.write("")
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        tag = {"sustained": self.style.SUCCESS, "emerging": self.style.HTTP_INFO,
               "fading": self.style.ERROR, "neutral": lambda s: s}
        for r in rows:
            accel = r["acceleration"]
            accel_s = "-" if accel is None else f"{accel:+.0f}"
            line = (
                f"{r['symbol']:<12}{(r['sector'] or '')[:21]:<22}"
                f"{(r['ltp'] or 0):>9.2f}{(r['change_pct'] or 0):>7.2f}"
                f"{n(r['score_1m'],5)}{n(r['score_3m'],5)}{n(r['score_6m'],5)}"
                f"{(r['composite'] or 0):>7.1f}{accel_s:>7}  "
            )
            styler = tag.get(r["classification"], lambda s: s)
            self.stdout.write(line + styler(r["classification"]))
