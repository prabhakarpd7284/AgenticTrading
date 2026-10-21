"""backfill_candles / backfill_bhavcopy, with the broker and NSE download faked."""
from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from io import StringIO
from zoneinfo import ZoneInfo

import pytest
from django.core.management import call_command

from apps.market_data.management.commands import backfill_bhavcopy as bhav
from apps.market_data.management.commands import backfill_candles as bc
from apps.market_data.models import Candle, Symbol

IST = ZoneInfo("Asia/Kolkata")


def test_windows_cover_range_without_overlap():
    w = bc.windows(date(2026, 1, 1), date(2026, 3, 1), 30)
    assert w[0] == (date(2026, 1, 1), date(2026, 1, 30))
    assert w[-1][1] == date(2026, 3, 1)
    assert all(a[1] < b[0] for a, b in zip(w, w[1:]))


def test_fno_stock_names_uses_stock_futures_only():
    ins = [
        {"exch_seg": "NFO", "instrumenttype": "FUTSTK", "name": "SBIN"},
        {"exch_seg": "NFO", "instrumenttype": "FUTSTK", "name": "SBIN"},
        {"exch_seg": "NFO", "instrumenttype": "FUTIDX", "name": "NIFTY"},
        {"exch_seg": "NSE", "instrumenttype": "", "name": "TCS"},
        {"exch_seg": "NFO", "instrumenttype": "FUTSTK", "name": "011NSETEST"},
    ]
    assert bc.fno_stock_names(ins) == ["SBIN"]


def test_missing_ranges_head_tail_and_slack():
    s, e = date(2026, 4, 18), date(2026, 10, 21)
    assert bc.missing_ranges(s, e, None, None) == [(s, e)]
    # first candle on Mon 20 Apr (start was a Sat) → head is complete; tail refetches last day
    assert bc.missing_ranges(s, e, date(2026, 4, 20), date(2026, 10, 16)) == [(date(2026, 10, 16), e)]
    # interrupted mid-history: real head gap
    assert bc.missing_ranges(s, e, date(2026, 6, 1), e) == [(s, date(2026, 5, 31))]


def test_parse_rows_keeps_ist_timestamps():
    (t, o, h, l, c, v), = bc.parse_rows([["2026-10-16T09:15:00+05:30", 1, 2, 0.5, 1.5, "10"]])
    assert t == datetime(2026, 10, 16, 9, 15, tzinfo=IST) and v == 10


class _FakeTicker:
    _instruments: list = []

    def _ensure_loaded(self):
        pass

    def get_token(self, name):
        return {"SBIN": "3045"}.get(name)


class _FakeBroker:
    def __init__(self):
        self.calls = 0

    def fetch_candles(self, token, start, end, interval, exchange=None, wait_on_breaker=False):
        self.calls += 1
        first, last = start[:10], end[:10]
        return [[f"{first}T09:15:00+05:30", 1, 2, 0.5, 1.5, 100],
                [f"{last}T09:16:00+05:30", 1.5, 2, 1, 1.8, 50]]


@pytest.mark.django_db
def test_backfill_candles_writes_and_resumes(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr("trading.services.ticker_service.TickerService", _FakeTicker)
    monkeypatch.setattr("trading.services.data_service.BrokerClient.get_instance",
                        classmethod(lambda cls: broker))
    monkeypatch.setattr("apps.market_data.services.candle_ingestor._resolve_symbol_meta",
                        lambda s: {"exchange": "NSE", "token": "3045", "name": s,
                                   "segment": "", "lot_size": 1, "tick_size": 0.05})

    args = ("--universe", "none", "--symbols", "SBIN", "--months", "2")
    call_command("backfill_candles", *args, stdout=StringIO())
    first_calls = broker.calls
    assert first_calls >= 2
    sym = Symbol.objects.get(tradingsymbol="SBIN")
    assert Candle.objects.filter(symbol=sym, interval="1m").count() == 2 * first_calls

    call_command("backfill_candles", *args, stdout=StringIO())   # everything present now
    assert broker.calls == first_calls


def _bhav_zip(rows: list[dict]) -> bytes:
    cols = ["TradDt", "FinInstrmId", "TckrSymb", "SctySrs", "FinInstrmNm",
            "OpnPric", "HghPric", "LwPric", "ClsPric", "TtlTradgVol"]
    lines = [",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BhavCopy.csv", "\n".join(lines))
    return buf.getvalue()


@pytest.mark.django_db
def test_bhavcopy_loads_equities_skips_holidays_and_resumes(monkeypatch):
    blob = _bhav_zip([
        {"FinInstrmId": "2885", "TckrSymb": "RELIANCE", "SctySrs": "EQ", "OpnPric": "1215",
         "HghPric": "1215.4", "LwPric": "1193", "ClsPric": "1211.2", "TtlTradgVol": "17225283"},
        {"FinInstrmId": "999", "TckrSymb": "SOMEGSEC", "SctySrs": "GS", "OpnPric": "1",
         "HghPric": "1", "LwPric": "1", "ClsPric": "1", "TtlTradgVol": "1"},
    ])
    fetched = []

    def fake_download(d):
        fetched.append(d)
        return blob if d == date(2026, 10, 16) else None   # 15th = "holiday"

    monkeypatch.setattr(bhav, "download", fake_download)
    monkeypatch.setattr(bhav, "_PAUSE_SECS", 0)

    out = StringIO()
    call_command("backfill_bhavcopy", "--from", "2026-10-15", "--to", "2026-10-16", stdout=out)
    rel = Symbol.objects.get(exchange="NSE", token="2885")
    c = Candle.objects.get(symbol=rel, interval="1d")
    assert float(c.c) == 1211.2 and c.v == 17225283
    assert not Symbol.objects.filter(token="999").exists()   # non-equity series dropped
    assert "1 holidays" in out.getvalue()

    fetched.clear()
    call_command("backfill_bhavcopy", "--from", "2026-10-16", "--to", "2026-10-16", stdout=StringIO())
    assert fetched == []   # day already present
