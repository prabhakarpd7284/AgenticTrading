"""
Derive (replay) intraday paper trades for historical day(s).

The live intraday agent only creates trades during market hours; this replays
past sessions through the same monitor so the Monthly report's trade-driven
sections (P&L, equity curve, analytics, benchmark) can be back-filled.

Usage:
    python manage.py derive_trades --date 2026-05-29
    python manage.py derive_trades --from 2026-05-01 --to 2026-05-31
    python manage.py derive_trades --from 2026-05-01 --to 2026-05-31 --max-positions 5

Paper only — reuses BrokerService paper fills (TRADING_MODE=paper). Weekends are
skipped; holidays no-op (broker returns no candles → empty watchlist).
"""
from __future__ import annotations

from datetime import date, timedelta

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Replay historical intraday session(s) to derive paper trades for the Monthly report"

    def add_arguments(self, parser):
        parser.add_argument("--date", help="Single day YYYY-MM-DD")
        parser.add_argument("--from", dest="from_date", help="Range start YYYY-MM-DD")
        parser.add_argument("--to", dest="to_date", help="Range end YYYY-MM-DD")
        parser.add_argument("--capital", type=float, default=500000.0)
        parser.add_argument("--max-positions", type=int, default=3,
                            help="Max concurrent positions per day (default: 3)")
        parser.add_argument(
            "--replace", action="store_true",
            help="Re-derive days that already have trades (deletes their prior "
                 "workflow trades first). Default: skip days that already have data.",
        )

    def handle(self, *args, **opts):
        from apps.trading.models import Trade
        from trading.intraday.replay import IntradayReplay
        from trading.intraday.state import IntradayState

        days = self._resolve_days(opts)
        if not days:
            raise CommandError("Provide --date, or --from and --to.")

        # Idempotency: by default skip days that already carry executed trades
        # (gap-fill), so re-runs + the daily beat task never double-count.
        replace = opts.get("replace")
        executed = {Trade.Status.FILLED, Trade.Status.PARTIAL, Trade.Status.CLOSED}
        existing_dates = set(
            Trade.objects.filter(
                trade_date__gte=days[0], trade_date__lte=days[-1], status__in=executed,
            ).values_list("trade_date", flat=True)
        )

        self.stdout.write(f"Deriving trades for {len(days)} trading day(s)…")
        totals = {"entries": 0, "exits": 0, "skipped": 0}
        for d in days:
            if d in existing_dates:
                if not replace:
                    self.stdout.write(f"  {d} ({d:%a}): already has trades — skipped")
                    totals["skipped"] += 1
                    continue
                # --replace: clear this day's prior *intraday* workflow trades.
                # Exclude swing rows ([Swing …]) — they're owned by
                # derive_swing_trades and span multiple days.
                Trade.objects.filter(
                    trade_date=d, origin=Trade.Origin.WORKFLOW, status__in=executed,
                ).exclude(reasoning__startswith="[Swing").delete()
            state = IntradayState(
                trading_date=d.isoformat(),
                capital=opts["capital"],
                max_positions=opts["max_positions"],
            )
            try:
                r = IntradayReplay(state).replay_day(d)
            except Exception as e:
                self.stderr.write(self.style.WARNING(f"  {d}: FAILED — {e}"))
                totals["skipped"] += 1
                continue
            totals["entries"] += r["entries"]
            totals["exits"] += r["exits"]
            if r["watchlist"] == 0:
                totals["skipped"] += 1
            self.stdout.write(
                f"  {d} ({d:%a}): watchlist={r['watchlist']:2d} "
                f"entries={r['entries']} exits={r['exits']}"
            )

        self.stdout.write(self.style.SUCCESS(
            f"Done — {totals['entries']} entries, {totals['exits']} closed, "
            f"{totals['skipped']} empty/skipped days."
        ))

    def _resolve_days(self, opts) -> list[date]:
        if opts.get("date"):
            d = date.fromisoformat(opts["date"])
            return [d] if d.weekday() < 5 else []
        if opts.get("from_date") and opts.get("to_date"):
            d0 = date.fromisoformat(opts["from_date"])
            d1 = date.fromisoformat(opts["to_date"])
            if d1 < d0:
                raise CommandError("--to is before --from")
            return [
                d0 + timedelta(n)
                for n in range((d1 - d0).days + 1)
                if (d0 + timedelta(n)).weekday() < 5
            ]
        return []
