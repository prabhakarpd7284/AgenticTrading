"""
End-of-Day Signal Enrichment — compute post-hoc outcomes for v2 Signal entries.

For every signal fired today (or a given date), fetches intraday candles and
computes:
  - eod_price: closing price on signal day
  - max_favorable_move: best price move in signal direction after entry
  - max_adverse_move: worst move against signal direction after entry
  - outcome: EXPIRED if still PENDING after market close

Also links signals to ``apps.trading.Trade`` rows where possible
(same tenant + symbol + side + trade_date).

Reads/writes against the v2 multi-tenant tables ``apps.strategies.Signal``
and ``apps.trading.Trade``. The legacy SQLite ``trading.models.SignalLog`` /
``TradeJournal`` were retired in Phase 6.

Usage:
    python manage.py enrich_signals
    python manage.py enrich_signals --date 2026-05-02
    python manage.py enrich_signals --all
    python manage.py enrich_signals --tenant personal
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

from django.core.management.base import BaseCommand, CommandError
from logzero import logger


EPILOG = """\
Examples
--------
  Enrich today's signals across every tenant:
      python manage.py enrich_signals

  Enrich a specific date:
      python manage.py enrich_signals --date 2026-05-02

  Backfill all un-enriched signals (large run):
      python manage.py enrich_signals --all

  Restrict to one tenant (slug or name):
      python manage.py enrich_signals --tenant personal
"""


class Command(BaseCommand):
    help = "Enrich apps.strategies.Signal entries with EOD price + outcome linkage"

    def create_parser(self, prog_name, subcommand, **kwargs):
        parser = super().create_parser(prog_name, subcommand, **kwargs)
        parser.epilog = EPILOG
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        return parser

    def add_arguments(self, parser):
        parser.add_argument("--date", help="Date to enrich (YYYY-MM-DD, default: today)")
        parser.add_argument(
            "--all", action="store_true",
            help="Enrich every un-enriched signal regardless of date",
        )
        parser.add_argument(
            "--tenant", default=None,
            help="Restrict to one tenant (slug or name). Default: all tenants.",
        )

    def handle(self, *args, **options):
        # Imports are deferred so the CLI can be discovered without a Django
        # connection (--help works even when Postgres is down).
        from apps.strategies.models import Signal
        from apps.tenants.models import Tenant
        from apps.trading.models import Trade
        from trading.services.data_service import DataService
        from trading.services.ticker_service import ticker_service

        target_date = self._resolve_date(options)
        tenants = self._resolve_tenants(options["tenant"])
        scope = "ALL DATES" if options["all"] else target_date.isoformat()
        self.stdout.write(self._banner(scope, tenants))

        ds = DataService()
        total_enriched = 0
        total_errors = 0
        total_pending = 0

        for tenant in tenants:
            pending_qs = Signal.objects.filter(
                tenant=tenant, eod_price__isnull=True,
            )
            if not options["all"]:
                pending_qs = pending_qs.filter(signal_date=target_date)

            pending = list(pending_qs.select_related("trade"))
            if not pending:
                continue
            total_pending += len(pending)
            self.stdout.write(
                f"\n[{tenant.slug or tenant.name}] {len(pending)} signals pending"
            )

            grouped: dict[tuple[str, date], list[Signal]] = {}
            for sig in pending:
                grouped.setdefault((sig.symbol, sig.signal_date), []).append(sig)

            for (symbol, sig_date), batch in grouped.items():
                try:
                    enriched = self._enrich_batch(
                        ds, ticker_service, symbol, sig_date, batch,
                    )
                    total_enriched += enriched
                    total_errors += len(batch) - enriched
                except Exception as exc:
                    logger.error(f"Error enriching {symbol} on {sig_date}: {exc}")
                    total_errors += len(batch)

                self._link_trades(tenant, batch, sig_date, Trade, Signal)

        self.stdout.write(
            f"\nDone: {total_enriched} enriched, {total_errors} errors "
            f"out of {total_pending} total.\n"
        )
        self._print_next_steps(total_enriched, target_date)

    # ── Internals ──────────────────────────────────────────────────────

    def _resolve_date(self, options) -> date:
        raw = options.get("date")
        if raw:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        return date.today()

    def _resolve_tenants(self, slug_or_name: str | None) -> list:
        from apps.tenants.models import Tenant
        if slug_or_name:
            t = (
                Tenant.objects.filter(slug=slug_or_name).first()
                or Tenant.objects.filter(name=slug_or_name).first()
            )
            if t is None:
                raise CommandError(f"Tenant {slug_or_name!r} not found.")
            return [t]
        tenants = list(Tenant.objects.all())
        if not tenants:
            raise CommandError(
                "No tenants found. Create one (admin panel or fixtures) before enrichment."
            )
        return tenants

    def _enrich_batch(self, ds, ticker_service, symbol, sig_date, signals) -> int:
        token = ticker_service.get_token(symbol)
        if not token:
            logger.warning(f"No token for {symbol}, skipping")
            return 0

        date_str = sig_date.strftime("%Y-%m-%d")
        ds._ensure_broker()
        raw_candles = ds._broker.fetch_candles(
            token,
            f"{date_str} 09:15",
            f"{date_str} 15:30",
            interval="FIVE_MINUTE",
        )
        if not raw_candles:
            logger.warning(f"No candles for {symbol} on {date_str}")
            return 0

        candles = self._normalise_candles(raw_candles)
        if not candles:
            return 0

        eod_price = candles[-1]["c"]
        enriched = 0

        for sig in signals:
            sig_time = sig.signal_time.replace(tzinfo=None)
            after = [c for c in candles if c["ts"] >= sig_time] or candles

            if sig.side == "BUY":
                max_fav = max(c["h"] for c in after) - sig.entry_price
                max_adv = sig.entry_price - min(c["l"] for c in after)
            else:
                max_fav = sig.entry_price - min(c["l"] for c in after)
                max_adv = max(c["h"] for c in after) - sig.entry_price

            sig.eod_price = eod_price
            sig.max_favorable_move = round(max(0, max_fav), 2)
            sig.max_adverse_move = round(max(0, max_adv), 2)

            if sig.outcome == sig.__class__.Outcome.PENDING:
                sig.outcome = sig.__class__.Outcome.EXPIRED

            sig.save(update_fields=[
                "eod_price", "max_favorable_move", "max_adverse_move", "outcome",
            ])
            enriched += 1

        return enriched

    def _normalise_candles(self, raw):
        out = []
        for c in raw:
            if isinstance(c, list):
                ts_str = c[0]
                o, h, lo, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
            else:
                ts_str = c.get("timestamp") or c.get("time", "")
                o = float(c.get("open", 0))
                h = float(c.get("high", 0))
                lo = float(c.get("low", 0))
                cl = float(c.get("close", 0))

            if isinstance(ts_str, str):
                ts = datetime.fromisoformat(ts_str.replace("+05:30", ""))
            elif isinstance(ts_str, (int, float)):
                ts = datetime.fromtimestamp(ts_str)
            elif isinstance(ts_str, datetime):
                ts = ts_str
            else:
                ts = datetime.now()
            out.append({"ts": ts, "o": o, "h": h, "l": lo, "c": cl})
        return out

    def _link_trades(self, tenant, signals, sig_date, Trade, Signal):
        """Link signals to apps.trading.Trade rows (filled or rejected) for the day."""
        filled = Trade.objects.filter(
            tenant=tenant, trade_date=sig_date,
            status__in=[
                Trade.Status.FILLED, Trade.Status.PARTIAL, Trade.Status.CLOSED,
            ],
        )
        rejected = Trade.objects.filter(
            tenant=tenant, trade_date=sig_date, status=Trade.Status.REJECTED,
        )

        fill_map: dict[tuple[str, str], list] = {}
        for t in filled:
            fill_map.setdefault((t.symbol, t.side), []).append(t)
        reject_map: dict[tuple[str, str], list] = {}
        for t in rejected:
            reject_map.setdefault((t.symbol, t.side), []).append(t)

        for sig in signals:
            if sig.trade_id is not None:
                continue
            key = (sig.symbol, sig.side)

            if key in fill_map:
                sig.trade = fill_map[key][0]
                sig.outcome = Signal.Outcome.TRADED
                sig.save(update_fields=["trade", "outcome"])
                continue

            if sig.outcome == Signal.Outcome.EXPIRED and key in reject_map:
                rej = reject_map[key][0]
                sig.trade = rej
                sig.outcome = Signal.Outcome.REJECTED
                sig.outcome_reason = rej.risk_reason or ""
                sig.save(update_fields=["trade", "outcome", "outcome_reason"])

    # ── UX helpers ─────────────────────────────────────────────────────

    def _banner(self, scope: str, tenants: list) -> str:
        names = ", ".join(t.slug or t.name for t in tenants)
        return (
            "\n[SIGNAL ENRICHMENT]\n"
            f"  scope   : {scope}\n"
            f"  tenants : {names}\n"
        )

    def _print_next_steps(self, enriched: int, target_date: date):
        self.stdout.write("\nSuggested next:")
        if enriched:
            self.stdout.write(
                f"  · Review the monthly report:  GET /api/v1/portfolios/monthly/"
                f"?month={target_date.strftime('%Y-%m')}"
            )
            self.stdout.write(
                "  · Open the React UI:          http://localhost:5173/monthly"
            )
        else:
            self.stdout.write(
                "  · No new enrichment. Run with --all to backfill unenriched history."
            )
