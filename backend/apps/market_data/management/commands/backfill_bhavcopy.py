"""Backfill daily candles for every NSE equity from NSE's bhavcopy archive.

One file per trading day covers every listed stock, so 6 months is ~125
downloads instead of thousands of per-symbol Angel calls — and it uses none
of Angel's rate budget. Rows land in ``Candle`` with ``interval="1d"``.

NSE's ``FinInstrmId`` is the same scrip token Angel uses for NSE equities
(RELIANCE = 2885), so ``Symbol`` rows line up with the ones the 1-minute
backfill and the live ingestor create.

Resumable: a date that already has daily candles is skipped. Holidays and
weekends simply 404 and are skipped.

    ./manage.py backfill_bhavcopy --months 6
    ./manage.py backfill_bhavcopy --from 2026-04-01 --to 2026-04-30
"""
from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from django.core.management.base import BaseCommand, CommandError

IST = ZoneInfo("Asia/Kolkata")
URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
# NSE's archive rejects requests without a browser-like User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
SERIES = {"EQ", "BE"}   # rolling-settlement + trade-for-trade equities
_PAUSE_SECS = 1.0       # be polite to the archive


def download(d: date) -> bytes | None:
    """Zip bytes for ``d``, or None when NSE has no file (holiday/weekend)."""
    req = urllib.request.Request(URL.format(d=d), headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — fixed NSE host
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def parse(zip_bytes: bytes) -> list[dict]:
    """Equity rows from a UDiFF CM bhavcopy zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open(z.namelist()[0]) as f:
            rows = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            return [r for r in rows if r.get("SctySrs") in SERIES]


class Command(BaseCommand):
    help = "Backfill daily candles for all NSE equities from the NSE bhavcopy archive."

    def add_arguments(self, parser):
        parser.add_argument("--months", type=int, default=6)
        parser.add_argument("--from", dest="date_from", default="")
        parser.add_argument("--to", dest="date_to", default="")

    def handle(self, *args, **opts):
        end = date.fromisoformat(opts["date_to"]) if opts["date_to"] else date.today()
        start = (date.fromisoformat(opts["date_from"]) if opts["date_from"]
                 else end - timedelta(days=round(opts["months"] * 30.5)))
        if start > end:
            raise CommandError("--from is after --to")

        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        days = [d for d in days if d.weekday() < 5]
        self.stdout.write(f"Bhavcopy {start}..{end}: {len(days)} weekdays to check")

        loaded = present = missing = candles = 0
        for d in days:
            if self._has_day(d):
                present += 1
                continue
            try:
                blob = download(d)
            except Exception as e:  # noqa: BLE001 — report and keep going
                self.stderr.write(f"  {d}: download failed — {e}")
                time.sleep(_PAUSE_SECS)
                continue
            time.sleep(_PAUSE_SECS)
            if blob is None:
                missing += 1
                continue
            n = self._write(d, parse(blob))
            candles += n
            loaded += 1
            self.stdout.write(f"  {d}: {n} stocks")

        self.stdout.write(self.style.SUCCESS(
            f"Done: {loaded} days loaded ({candles} candles), {present} already present, "
            f"{missing} holidays/no file."))

    @staticmethod
    def _day_ts(d: date) -> datetime:
        # Same stamp Angel's ONE_DAY rows carry: midnight IST of the session.
        return datetime(d.year, d.month, d.day, tzinfo=IST)

    def _has_day(self, d: date) -> bool:
        from apps.market_data.models import Candle
        return Candle.objects.filter(interval="1d", t=self._day_ts(d)).exists()

    def _write(self, d: date, rows: list[dict]) -> int:
        from apps.market_data.models import Candle, Symbol

        by_token = {r["FinInstrmId"]: r for r in rows if r.get("FinInstrmId")}
        existing = dict(Symbol.objects.filter(exchange="NSE", token__in=list(by_token))
                        .values_list("token", "id"))
        new = [Symbol(exchange="NSE", token=tok, tradingsymbol=r["TckrSymb"],
                      name=r.get("FinInstrmNm") or r["TckrSymb"], segment="EQ")
               for tok, r in by_token.items() if tok not in existing]
        if new:
            Symbol.objects.bulk_create(new, ignore_conflicts=True)
            existing = dict(Symbol.objects.filter(exchange="NSE", token__in=list(by_token))
                            .values_list("token", "id"))

        t = self._day_ts(d)
        objs = [
            Candle(symbol_id=existing[tok], interval="1d", t=t,
                   o=r["OpnPric"], h=r["HghPric"], l=r["LwPric"], c=r["ClsPric"],
                   v=int(float(r.get("TtlTradgVol") or 0)))
            for tok, r in by_token.items() if tok in existing
        ]
        Candle.objects.bulk_create(objs, batch_size=5000, ignore_conflicts=True)
        return len(objs)
