"""Ingest an official StockEdge CSV export (the Download button on any page).

This is the cleanest, ToS-defensible data path: StockEdge's own export, produced
inside a logged-in session and downloaded by the user (or, later, a CDP-attached
download harness). Handles any stock-list export — strategy scans, momentum
scores, etc. — via a generic parser + the ``StockEdgeScanRow`` model.

Examples
--------
    ./manage.py pull_stockedge_csv --file ~/Downloads/cannonmomentum__25-Jun-2026.csv
    ./manage.py pull_stockedge_csv --file ~/Downloads/nifty500_StockWise_MomentumScores_25-Jun-2026.csv
    ./manage.py pull_stockedge_csv --file a.csv --file b.csv          # several at once
    ./manage.py pull_stockedge_csv --file x.csv --dataset my_scan --no-persist
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Ingest official StockEdge CSV export(s) into StockEdgeSnapshot + StockEdgeScanRow."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file", dest="files", action="append", default=[], required=True,
            help="Path to a StockEdge CSV export. Repeatable for several files.",
        )
        parser.add_argument(
            "--dataset", default="",
            help="Override the auto-detected dataset slug (applies to all files).",
        )
        parser.add_argument("--exchange", default="NSE")
        parser.add_argument(
            "--no-persist", action="store_true",
            help="Parse + print only; do not write to the database.",
        )

    def handle(self, *args, **o):
        from apps.market_data.integrations.stockedge.csv_ingest import (
            parse_stockedge_csv, summarize_csv,
        )

        for raw_path in o["files"]:
            path = Path(raw_path).expanduser()
            if not path.exists():
                raise CommandError(f"File not found: {path}")
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError as exc:
                raise CommandError(f"Could not read {path}: {exc}")

            try:
                parsed = parse_stockedge_csv(text)
            except ValueError as exc:
                raise CommandError(f"Malformed StockEdge CSV {path.name}: {exc}")

            dataset = o["dataset"] or parsed["dataset"]
            summary = summarize_csv(parsed)
            as_of = parsed["as_of_date"] or date.today().isoformat()

            self.stdout.write(self.style.SUCCESS(
                f"\n{path.name}\n  dataset={dataset} kind={parsed['kind']} "
                f"as_of={as_of} rows={summary['row_count']}"
            ))
            self._print_summary(parsed, summary)

            if o["no_persist"]:
                self.stdout.write(self.style.WARNING("  --no-persist — nothing written."))
                continue

            snap = self._persist(dataset, as_of, o["exchange"], parsed)
            self.stdout.write(self.style.SUCCESS(
                f"  persisted snapshot #{snap.id} with {snap.scan_rows.count()} rows."
            ))

    # ── persistence ───────────────────────────────────────────────────
    def _persist(self, dataset: str, as_of_iso: str, exchange: str, parsed: dict):
        from django.db import transaction

        from apps.market_data.models import StockEdgeScanRow, StockEdgeSnapshot

        try:
            as_of = datetime.strptime(as_of_iso, "%Y-%m-%d").date()
        except ValueError as exc:
            raise CommandError(f"Bad as_of date {as_of_iso!r}: {exc}")

        with transaction.atomic():
            snapshot, _ = StockEdgeSnapshot.objects.update_or_create(
                dataset=dataset, as_of_date=as_of, exchange=exchange,
                defaults={
                    "source_url": "https://web.stockedge.com (CSV export)",
                    "raw": {"meta_lines": parsed["meta_lines"], "columns": parsed["columns"]},
                    "meta": {
                        "kind": parsed["kind"],
                        "title": parsed["title"],
                        "generated_for": parsed["generated_for"],
                        "row_count": len(parsed["rows"]),
                    },
                },
            )
            snapshot.scan_rows.all().delete()
            StockEdgeScanRow.objects.bulk_create([
                StockEdgeScanRow(
                    snapshot=snapshot, symbol=r["symbol"], name=r["name"],
                    sector=r["sector"], industry=r["industry"], ltp=r["ltp"],
                    change_pct=r["change_pct"], market_cap_cr=r["market_cap_cr"],
                    attrs=r["attrs"], as_of_date=as_of,
                    exchange=(r["exchange"] or exchange),
                )
                for r in parsed["rows"]
            ])
        return snapshot

    # ── output ────────────────────────────────────────────────────────
    def _print_summary(self, parsed: dict, summary: dict) -> None:
        kind = parsed["kind"]
        if kind == "momentum_scores":
            for horizon in ("1m", "3m", "6m"):
                z = summary.get(f"zones_{horizon}", {})
                self.stdout.write(
                    f"  {horizon.upper()} zones: "
                    f"Bullish {z.get('Bullish', 0)} · Neutral {z.get('Neutral', 0)} · "
                    f"Bearish {z.get('Bearish', 0)}"
                )
            top = sorted(
                (r for r in parsed["rows"] if isinstance(r["attrs"].get("1m_score"), int)),
                key=lambda r: r["attrs"]["1m_score"], reverse=True,
            )[:5]
            self.stdout.write("  Top 1M momentum: " + ", ".join(
                f"{r['symbol']}({r['attrs']['1m_score']})" for r in top
            ))
        elif kind == "strategy":
            self.stdout.write("  By type: " + ", ".join(
                f"{k} {v}" for k, v in summary.get("by_type", {}).items()
            ))
            pm = [r["symbol"] for r in parsed["rows"]
                  if r["attrs"].get("type") == "PerfectMatch"]
            if pm:
                self.stdout.write("  PerfectMatch: " + ", ".join(pm))
        else:
            self.stdout.write("  Symbols: " + ", ".join(
                r["symbol"] for r in parsed["rows"][:10]
            ) + (" …" if len(parsed["rows"]) > 10 else ""))
