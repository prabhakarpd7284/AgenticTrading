"""Fire StockEdge composite-momentum signals into the AlphaDesk pipeline.

Reads the latest ingested ``momentum_scores_<index>`` snapshot, selects the
strongest momentum names, and writes first-class ``apps.strategies.Signal`` rows
(``source=STOCKEDGE``, ``strategy="StockEdge Composite Momentum"``) + emits one
``signal.fired`` event each. The EOD ``enrich_signals`` job fills outcomes and
the monthly report scores them — same path as the screener.

Examples
--------
    ./manage.py stockedge_momentum_signals --dry-run          # preview, no writes
    ./manage.py stockedge_momentum_signals                    # fire (idempotent per day)
    ./manage.py stockedge_momentum_signals --force            # re-fire today's set
    ./manage.py stockedge_momentum_signals --top 25 --min-composite 75 --classes sustained
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Fire StockEdge composite-momentum signals (persisted Signals + signal.fired events)."

    def add_arguments(self, parser):
        parser.add_argument("--index", default="nifty_500")
        parser.add_argument("--top", type=int, default=20)
        parser.add_argument("--min-composite", type=float, default=70.0)
        parser.add_argument("--min-mcap", type=float, default=5000.0, help="Min market cap (Rs. Cr.).")
        parser.add_argument("--classes", default="sustained,emerging",
                            help="Comma list of classifications to include.")
        parser.add_argument("--stop-pct", type=float, default=0.04, help="Swing stop (fraction).")
        parser.add_argument("--rr", type=float, default=2.0, help="Reward:risk multiple for target.")
        parser.add_argument("--dry-run", action="store_true", help="Preview only; no DB writes.")
        parser.add_argument("--force", action="store_true", help="Replace today's momentum signals.")

    def handle(self, *args, **o):
        from apps.market_data.integrations.stockedge import momentum as M
        from apps.market_data.integrations.stockedge import momentum_signals as MS

        snap = M.latest_momentum_snapshot(o["index"])
        if snap is None:
            raise CommandError(
                f"No momentum_scores_{o['index']} snapshot. Ingest one first: "
                "manage.py pull_stockedge_csv --file <…MomentumScores….csv>"
            )

        params = dict(
            top=o["top"], min_composite=o["min_composite"], min_mcap_cr=o["min_mcap"],
            classes=tuple(c.strip() for c in o["classes"].split(",") if c.strip()),
            stop_pct=o["stop_pct"], rr=o["rr"],
        )
        result = MS.fire_momentum_signals(
            snap, force=o["force"], dry_run=o["dry_run"], **params,
        )

        self.stdout.write(self.style.SUCCESS(
            f"\nStockEdge momentum signals · {o['index']} · as_of {snap.as_of_date} "
            f"· strategy '{MS.STRATEGY_NAME}'"
        ))
        self._print_rows(result["rows"])

        if result.get("dry_run"):
            self.stdout.write(self.style.WARNING(
                f"  DRY RUN — would fire {result['would_fire']} signals, nothing written."
            ))
        elif result.get("reason") == "no_tenant":
            self.stdout.write(self.style.ERROR(
                "  No tenant in DB — signals not written (non-blocking)."
            ))
        elif result.get("already"):
            self.stdout.write(self.style.WARNING(
                f"  Already fired {result['skipped']} momentum signals for "
                f"{snap.as_of_date}. Use --force to replace."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"  Fired {result['fired']} signals (tenant={result['tenant']}). "
                "EOD enrich_signals will fill outcomes; they appear in the monthly "
                f"report under strategy '{MS.STRATEGY_NAME}'."
            ))

    def _print_rows(self, rows: list[dict]) -> None:
        if not rows:
            self.stdout.write(self.style.WARNING("  (no names match the filters)"))
            return
        header = (f"{'Symbol':<12}{'Side':>5}{'Entry':>10}{'Stop':>10}{'Target':>10}"
                  f"{'Conf':>6}  Class / momentum")
        self.stdout.write("")
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for row in rows:
            rec = row["record"]
            self.stdout.write(
                f"{row['symbol']:<12}{row['side']:>5}{row['entry']:>10.2f}"
                f"{row['stop']:>10.2f}{row['target']:>10.2f}{row['confidence']:>6.2f}  "
                f"{rec['classification']} "
                f"({rec['score_1m']}/{rec['score_3m']}/{rec['score_6m']}, "
                f"comp {rec['composite']}, accel {rec['acceleration']:+g})"
            )
