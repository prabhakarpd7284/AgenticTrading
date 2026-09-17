"""Ingest a StockEdge Market-Breadth snapshot (advisory overlay).

The MVP path ingests a captured JSON payload; production capture rides a
logged-in StockEdge session via the Playwright harness. Either way the payload
is normalized, upserted into ``StockEdgeSnapshot`` + ``StockEdgeBreadthRow``
(shared, non-tenant reference data), and printed as an aligned table.

Examples
--------
    # MVP: ingest the bundled sample (default when no source flag is given)
    ./manage.py pull_stockedge_breadth

    # MVP: ingest a payload you captured yourself
    ./manage.py pull_stockedge_breadth --from-json /path/to/breadth.json

    # Parse + print only, persist nothing
    ./manage.py pull_stockedge_breadth --no-persist

    # Production: capture live from a logged-in session (needs Playwright)
    ./manage.py pull_stockedge_breadth --live --storage-state /secrets/se_state.json
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

# Bundled sample lives next to the harness/parser in the integration package.
_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "integrations" / "stockedge" / "sample_breadth.json"
)


class Command(BaseCommand):
    help = "Ingest a StockEdge Market-Breadth snapshot (sample JSON or live Playwright capture)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--from-json", dest="from_json", default="",
            help="Path to a captured breadth JSON payload. Defaults to the bundled "
                 "sample when neither --from-json nor --live is given.",
        )
        parser.add_argument(
            "--live", action="store_true",
            help="Capture live via the Playwright harness (requires --storage-state).",
        )
        parser.add_argument(
            "--storage-state", dest="storage_state", default="",
            help="Path to a Playwright storage_state.json holding a logged-in "
                 "StockEdge session (required with --live).",
        )
        parser.add_argument(
            "--exchange", default="NSE",
            help="Exchange tag for the snapshot (default: NSE).",
        )
        parser.add_argument(
            "--no-persist", action="store_true",
            help="Parse + print only; do not write to the database.",
        )

    # ── orchestration ─────────────────────────────────────────────────
    def handle(self, *args, **o):
        from apps.market_data.integrations.stockedge.parser import (
            parse_breadth_payload, summarize_breadth,
        )

        payload = self._load_payload(o)

        try:
            parsed = parse_breadth_payload(payload)
        except ValueError as exc:
            raise CommandError(f"Malformed breadth payload: {exc}")

        rows = parsed["rows"]
        summary = summarize_breadth(rows)

        self.stdout.write(self.style.SUCCESS(
            f"StockEdge breadth · {parsed['exchange']} · as_of {parsed['as_of_date']} · "
            f"{len(rows)} index universes"
        ))

        if o["no_persist"]:
            self.stdout.write(self.style.WARNING("--no-persist — parsed only, nothing written."))
            snapshot = None
        else:
            snapshot = self._persist(parsed, payload)
            self.stdout.write(self.style.SUCCESS(
                f"Persisted snapshot #{snapshot.id} with {snapshot.breadth_rows.count()} rows "
                f"(captured {snapshot.captured_at:%Y-%m-%d %H:%M}, "
                f"refreshed {snapshot.refreshed_at:%Y-%m-%d %H:%M})."
            ))
            self._emit_event(snapshot, parsed, summary)

        self._print_table(rows)
        self._print_regime(summary)

    # ── payload loading ───────────────────────────────────────────────
    def _load_payload(self, o) -> dict:
        if o["live"]:
            return self._capture_live(o)

        path = o["from_json"] or str(_SAMPLE_PATH)
        src = Path(path)
        if not src.exists():
            raise CommandError(f"Payload file not found: {src}")
        try:
            with src.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise CommandError(f"Could not read JSON payload {src}: {exc}")
        if not o["from_json"]:
            self.stdout.write(self.style.WARNING(f"No source given — using bundled sample {src}"))
        return payload

    def _capture_live(self, o) -> dict:
        if not o["storage_state"]:
            raise CommandError("--live requires --storage-state /path/to/storage_state.json")
        try:
            from apps.market_data.integrations.stockedge.harness import capture_breadth
        except ImportError as exc:  # pragma: no cover
            raise CommandError(f"StockEdge harness unavailable: {exc}")
        self.stdout.write(self.style.WARNING(
            "LIVE capture — launching headless Chromium against the StockEdge session…"
        ))
        try:
            return capture_breadth(
                storage_state_path=o["storage_state"],
                exchange=o["exchange"],
            )
        except RuntimeError as exc:
            raise CommandError(str(exc))
        except Exception as exc:  # noqa: BLE001
            raise CommandError(f"Live capture failed: {exc}")

    # ── persistence ───────────────────────────────────────────────────
    def _persist(self, parsed: dict, raw_payload: dict):
        from django.db import transaction

        from apps.market_data.models import StockEdgeBreadthRow, StockEdgeSnapshot

        as_of = self._parse_date(parsed["as_of_date"])
        exchange = parsed["exchange"]

        with transaction.atomic():
            snapshot, _created = StockEdgeSnapshot.objects.update_or_create(
                dataset=parsed["dataset"],
                as_of_date=as_of,
                exchange=exchange,
                defaults={
                    "source_url": parsed["source_url"],
                    "raw": raw_payload,
                    "meta": {
                        "columns": raw_payload.get("columns", []),
                        "unit": raw_payload.get("unit", "percent"),
                        "row_count": len(parsed["rows"]),
                    },
                },
            )
            # Replace children so re-pulls don't accumulate stale rows.
            snapshot.breadth_rows.all().delete()
            StockEdgeBreadthRow.objects.bulk_create([
                StockEdgeBreadthRow(
                    snapshot=snapshot,
                    index_name=r["index_name"],
                    constituent_count=r["constituent_count"],
                    rs_pos=r["rs_pos"],
                    sma20=r["sma20"],
                    sma50=r["sma50"],
                    sma100=r["sma100"],
                    sma200=r["sma200"],
                    as_of_date=as_of,
                    exchange=exchange,
                )
                for r in parsed["rows"]
            ])
        return snapshot

    @staticmethod
    def _parse_date(value: str) -> date:
        return datetime.strptime(value, "%Y-%m-%d").date()

    # ── events (non-blocking, best-effort) ────────────────────────────
    def _emit_event(self, snapshot, parsed: dict, summary: dict) -> None:
        """Emit a non-blocking events.Event. StockEdge data is non-tenant, so we
        attach it to any available tenant; failure here never breaks the flow."""
        try:
            from apps.events.services.event_writer import emit
            from apps.tenants.models import Tenant

            tenant = Tenant.objects.order_by("created_at").first()
            if tenant is None:
                return
            emit(
                tenant=tenant,
                type="stockedge.breadth.ingested",
                text=(f"StockEdge market breadth {parsed['exchange']} "
                      f"{parsed['as_of_date']}: {summary['broad_regime']}"),
                actor_kind="workflow",
                payload={
                    "snapshot_id": snapshot.id,
                    "dataset": parsed["dataset"],
                    "exchange": parsed["exchange"],
                    "as_of_date": parsed["as_of_date"],
                    "rows": len(parsed["rows"]),
                    **summary,
                },
                broadcast=False,
            )
        except Exception:  # noqa: BLE001 - events must never break ingestion
            pass

    # ── output ────────────────────────────────────────────────────────
    def _print_table(self, rows: list[dict]) -> None:
        from apps.market_data.integrations.stockedge.parser import BREADTH_COLUMNS

        def _score(r: dict):
            vals = [r.get(c) for c in BREADTH_COLUMNS if r.get(c) is not None]
            return sum(vals) / len(vals) if vals else float("-inf")

        ordered = sorted(rows, key=_score, reverse=True)

        def cell(v) -> str:
            return "-" if v is None else f"{v:g}"

        header = (f"{'Index':<26}{'N':>6}{'RS':>7}{'SMA20':>8}{'SMA50':>8}"
                  f"{'SMA100':>8}{'SMA200':>8}{'Score':>8}")
        self.stdout.write("")
        self.stdout.write(header)
        self.stdout.write("-" * len(header))
        for r in ordered:
            score = _score(r)
            score_s = "-" if score == float("-inf") else f"{score:.1f}"
            self.stdout.write(
                f"{r['index_name']:<26}"
                f"{cell(r['constituent_count']):>6}"
                f"{cell(r['rs_pos']):>7}"
                f"{cell(r['sma20']):>8}"
                f"{cell(r['sma50']):>8}"
                f"{cell(r['sma100']):>8}"
                f"{cell(r['sma200']):>8}"
                f"{score_s:>8}"
            )

    def _print_regime(self, summary: dict) -> None:
        self.stdout.write("")
        line = (
            f"Regime: {summary['broad_regime'].upper()}  "
            f"(avg SMA20={summary['avg_sma20']}, SMA50={summary['avg_sma50']}, "
            f"SMA200={summary['avg_sma200']})  "
            f"strongest={summary['strongest_index']}  weakest={summary['weakest_index']}"
        )
        styler = {
            "risk-on": self.style.SUCCESS,
            "neutral": self.style.WARNING,
            "risk-off": self.style.ERROR,
        }.get(summary["broad_regime"], self.style.WARNING)
        self.stdout.write(styler(line))
