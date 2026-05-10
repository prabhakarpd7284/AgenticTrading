"""
End-of-Day Signal Enrichment — compute post-hoc outcomes for SignalLog entries.

For every signal fired today (or a given date), fetches intraday candles and
computes:
  - eod_price: closing price on signal day
  - max_favorable_move: best price move in signal direction after entry
  - max_adverse_move: worst move against signal direction after entry
  - outcome: EXPIRED if still PENDING after market close

Also links signals to TradeJournal entries where possible (same symbol+date+side).

Usage:
    # Enrich today's signals
    python manage.py enrich_signals

    # Enrich a specific date
    python manage.py enrich_signals --date 2026-05-02

    # Enrich all un-enriched signals
    python manage.py enrich_signals --all
"""
from datetime import date, datetime

from django.core.management.base import BaseCommand
from logzero import logger


class Command(BaseCommand):
    help = "Enrich SignalLog entries with EOD price data and outcome linkage"

    def add_arguments(self, parser):
        parser.add_argument(
            "--date", help="Date to enrich (YYYY-MM-DD, default: today)"
        )
        parser.add_argument(
            "--all", action="store_true",
            help="Enrich all un-enriched signals (not just today)"
        )

    def handle(self, *args, **options):
        from trading.models import SignalLog, TradeJournal
        from trading.services.data_service import DataService
        from trading.services.ticker_service import ticker_service

        ds = DataService()

        # Determine which signals to enrich
        if options["all"]:
            pending = SignalLog.objects.filter(eod_price__isnull=True)
        else:
            target_date = options.get("date")
            if target_date:
                target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
            else:
                target_date = date.today()
            pending = SignalLog.objects.filter(
                signal_date=target_date, eod_price__isnull=True
            )

        total = pending.count()
        if total == 0:
            self.stdout.write("No signals to enrich.")
            return

        self.stdout.write(f"\nEnriching {total} signals...\n")

        # Group by (symbol, date) to minimize API calls
        signal_groups = {}
        for sig in pending.select_related("trade_journal"):
            key = (sig.symbol, sig.signal_date)
            signal_groups.setdefault(key, []).append(sig)

        enriched = 0
        errors = 0

        for (symbol, sig_date), signals in signal_groups.items():
            try:
                token = ticker_service.get_token(symbol)
                if not token:
                    logger.warning(f"No token for {symbol}, skipping")
                    errors += len(signals)
                    continue

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
                    errors += len(signals)
                    continue

                # Parse candles: each is [timestamp, open, high, low, close, volume]
                candles = []
                for c in raw_candles:
                    if isinstance(c, list):
                        ts_str, o, h, lo, cl = c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4])
                    else:
                        ts_str = c.get("timestamp", c.get("time", ""))
                        o, h, lo, cl = float(c.get("open", 0)), float(c.get("high", 0)), float(c.get("low", 0)), float(c.get("close", 0))

                    if isinstance(ts_str, str):
                        ts = datetime.fromisoformat(ts_str.replace("+05:30", ""))
                    elif isinstance(ts_str, (int, float)):
                        ts = datetime.fromtimestamp(ts_str)
                    elif isinstance(ts_str, datetime):
                        ts = ts_str
                    else:
                        ts = datetime.now()
                    candles.append({"ts": ts, "o": o, "h": h, "l": lo, "c": cl})

                if not candles:
                    errors += len(signals)
                    continue

                eod_price = candles[-1]["c"]

                for sig in signals:
                    # Find candles after signal time
                    sig_time = sig.signal_time.replace(tzinfo=None)
                    after = [c for c in candles if c["ts"] >= sig_time]
                    if not after:
                        after = candles  # fallback: use all candles

                    if sig.side == "BUY":
                        max_fav = max(c["h"] for c in after) - sig.entry_price
                        max_adv = sig.entry_price - min(c["l"] for c in after)
                    else:
                        max_fav = sig.entry_price - min(c["l"] for c in after)
                        max_adv = max(c["h"] for c in after) - sig.entry_price

                    sig.eod_price = eod_price
                    sig.max_favorable_move = round(max(0, max_fav), 2)
                    sig.max_adverse_move = round(max(0, max_adv), 2)

                    # Expire PENDING signals past market close
                    if sig.outcome == SignalLog.Outcome.PENDING:
                        sig.outcome = SignalLog.Outcome.EXPIRED

                    sig.save(update_fields=[
                        "eod_price", "max_favorable_move", "max_adverse_move", "outcome"
                    ])
                    enriched += 1

                # Link to TradeJournal entries
                self._link_to_trades(signals, sig_date)

            except Exception as e:
                logger.error(f"Error enriching {symbol} on {sig_date}: {e}")
                errors += len(signals)

        self.stdout.write(
            f"\nDone: {enriched} enriched, {errors} errors out of {total} total.\n"
        )

    def _link_to_trades(self, signals, sig_date):
        """Link signals to TradeJournal entries by symbol+date+side."""
        from trading.models import TradeJournal

        trades = TradeJournal.objects.filter(
            trade_date=sig_date,
            status__in=["FILLED", "PAPER", "EXECUTED"],
        )
        trade_map = {}
        for t in trades:
            key = (t.symbol, t.side)
            trade_map.setdefault(key, []).append(t)

        for sig in signals:
            if sig.trade_journal_id is not None:
                continue
            matching = trade_map.get((sig.symbol, sig.side), [])
            if matching:
                # Link to closest trade by time
                sig.trade_journal = matching[0]
                sig.outcome = sig.__class__.Outcome.TRADED
                sig.save(update_fields=["trade_journal", "outcome"])

        # Also find rejected trades
        rejected = TradeJournal.objects.filter(
            trade_date=sig_date, status="REJECTED"
        )
        reject_map = {}
        for t in rejected:
            key = (t.symbol, t.side)
            reject_map.setdefault(key, []).append(t)

        for sig in signals:
            if sig.outcome != sig.__class__.Outcome.EXPIRED:
                continue
            if (sig.symbol, sig.side) in reject_map:
                sig.outcome = sig.__class__.Outcome.REJECTED
                sig.outcome_reason = reject_map[(sig.symbol, sig.side)][0].risk_reason
                sig.trade_journal = reject_map[(sig.symbol, sig.side)][0]
                sig.save(update_fields=["outcome", "outcome_reason", "trade_journal"])
