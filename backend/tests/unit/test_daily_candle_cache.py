"""Per-day Redis cache for completed daily candles (fetch_historical ONE_DAY).

Completed daily candles are immutable, so a repeat/overlapping backtest should
serve them from Redis and hit the broker only for today's still-forming bar —
the fix for "Run again" re-fetching the whole 120-day lookback every time.
"""
from datetime import date, timedelta

import pytest

import trading.services.data_service as ds_mod
from trading.services.data_service import DataService


class _FakePipe:
    def __init__(self, store):
        self.store = store
        self._ops = []

    def set(self, k, v, ex=None):
        self._ops.append((k, v))
        return self

    def execute(self):
        for k, v in self._ops:
            self.store[k] = v
        self._ops = []


class _FakeRedis:
    """Minimal Redis stand-in: get/set/mget + pipeline(set).execute()."""
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v

    def mget(self, keys):
        return [self.store.get(k) for k in keys]

    def pipeline(self):
        return _FakePipe(self.store)


def _raw_rows_for(span_start, span_end):
    """Synthetic ONE_DAY broker rows for every weekday in the span (weekends
    get no row, like the real broker)."""
    rows = []
    d = span_start
    while d <= span_end:
        if d.weekday() < 5:
            rows.append([f"{d.isoformat()} 00:00", 100.0, 101.0, 99.0, 100.5, 1000])
        d += timedelta(days=1)
    return rows


@pytest.fixture
def fake_redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(ds_mod, "_get_redis_for_throttle", lambda: r)
    return r


def _instrumented_service():
    """DataService whose raw broker fetch is replaced with a call-counting stub."""
    ds = DataService()
    calls = []

    def fake_raw(token, start_dt, end_dt, interval):
        calls.append((start_dt, end_dt))
        return _raw_rows_for(start_dt, end_dt)

    ds._fetch_raw_rows = fake_raw  # type: ignore[assignment]
    return ds, calls


def test_completed_days_cached_then_served_without_refetch(fake_redis):
    """Second run over the same past window does ZERO broker fetches."""
    # Fully in the past (today is 2026-06-26 in this env): Mon–Fri 2026-06-01..05.
    start, end = date(2026, 6, 1), date(2026, 6, 5)
    ds, calls = _instrumented_service()

    first = ds._fetch_daily_cached("TOK", "TEST", start, end, "ONE_DAY")
    assert len(first) == 5            # 5 weekdays
    assert len(calls) == 1            # one broker fetch on the cold cache

    second = ds._fetch_daily_cached("TOK", "TEST", start, end, "ONE_DAY")
    assert len(second) == 5
    assert len(calls) == 1            # warm cache → NO further broker fetch
    assert [c["date"] for c in second] == [c["date"] for c in first]


def test_weekends_cached_as_null_markers_not_refetched(fake_redis):
    """Non-trading days are remembered as empty so they aren't re-requested."""
    start, end = date(2026, 6, 5), date(2026, 6, 8)   # Fri, Sat, Sun, Mon
    ds, calls = _instrumented_service()

    out = ds._fetch_daily_cached("TOK", "TEST", start, end, "ONE_DAY")
    assert {c["date"] for c in out} == {"2026-06-05", "2026-06-08"}  # weekdays only
    # Sat/Sun stored as null markers.
    assert fake_redis.store["alphadesk:candles:ONE_DAY:TOK:2026-06-06"] == "null"
    assert fake_redis.store["alphadesk:candles:ONE_DAY:TOK:2026-06-07"] == "null"

    ds._fetch_daily_cached("TOK", "TEST", start, end, "ONE_DAY")
    assert len(calls) == 1            # everything cached → no refetch


def test_todays_bar_is_never_cached(fake_redis):
    """Today's still-forming session must always be fetched fresh."""
    today = date.today()
    ds, calls = _instrumented_service()

    ds._fetch_daily_cached("TOK", "TEST", today, today, "ONE_DAY")
    key = f"alphadesk:candles:ONE_DAY:TOK:{today.isoformat()}"
    assert key not in fake_redis.store          # not cached
    assert len(calls) == 1

    ds._fetch_daily_cached("TOK", "TEST", today, today, "ONE_DAY")
    assert len(calls) == 2                       # fetched again, every time


def test_redis_down_falls_back_to_plain_fetch(monkeypatch):
    """No Redis → original uncached behaviour, still returns candles."""
    monkeypatch.setattr(ds_mod, "_get_redis_for_throttle", lambda: None)
    start, end = date(2026, 6, 1), date(2026, 6, 5)
    ds, calls = _instrumented_service()

    out = ds._fetch_daily_cached("TOK", "TEST", start, end, "ONE_DAY")
    assert len(out) == 5
    assert len(calls) == 1


def test_empty_span_is_not_cached_no_poisoning(fake_redis):
    """A fetch that returns nothing (likely transient failure) must NOT cache
    null for real trading days — the next run retries instead."""
    start, end = date(2026, 6, 1), date(2026, 6, 5)
    ds = DataService()
    calls = []

    def empty_raw(token, s, e, interval):
        calls.append((s, e))
        return []                      # broker returned nothing

    ds._fetch_raw_rows = empty_raw  # type: ignore[assignment]

    out1 = ds._fetch_daily_cached("TOK", "TEST", start, end, "ONE_DAY")
    assert out1 == []
    assert fake_redis.store == {}      # nothing poisoned the cache
    ds._fetch_daily_cached("TOK", "TEST", start, end, "ONE_DAY")
    assert len(calls) == 2             # retried, not served from a poisoned null


# ── Intraday cache (fetch_intraday_candles) ────────────────────────────

def _intraday_service(monkeypatch, rows_for_day):
    """DataService whose broker returns rows for EVERY weekday in the requested
    span in one call (the fetch is now chunked, not day-by-day). `calls` records
    each (start, end) span so we can assert call counts."""
    ds = DataService()
    calls = []

    class _FakeBroker:
        def ensure_login(self):
            pass

        def fetch_candles(self, token, start, end, interval, wait_on_breaker=False):
            calls.append((start[:10], end[:10]))
            rows = []
            d = date.fromisoformat(start[:10])
            e = date.fromisoformat(end[:10])
            while d <= e:
                rows.extend(rows_for_day(d.isoformat()))
                d += timedelta(days=1)
            return rows

    ds._broker = _FakeBroker()
    import trading.services.ticker_service as ts_mod
    monkeypatch.setattr(ts_mod.ticker_service, "get_token", lambda sym, *a, **k: "TOK")
    return ds, calls


def _intraday_rows(day):
    """3 synthetic 5m rows for a weekday, none for weekends/holidays."""
    d = date.fromisoformat(day)
    if d.weekday() >= 5:
        return []
    return [[f"{day} 09:{m:02d}:00", 100.0, 101.0, 99.0, 100.5, 500] for m in (15, 20, 25)]


def test_intraday_chunks_span_and_serves_repeat_from_redis(fake_redis, monkeypatch):
    """A multi-day fetch is ONE chunked broker call (not one per day); the
    repeat run does ZERO broker calls."""
    ds, calls = _intraday_service(monkeypatch, _intraday_rows)

    first = ds.fetch_intraday_candles("TEST", "2026-06-01", "2026-06-05", "FIVE_MINUTE")
    assert len(first) == 15            # 5 weekdays × 3 rows
    assert len(calls) == 1             # single span fetch, NOT 5 day-by-day calls

    second = ds.fetch_intraday_candles("TEST", "2026-06-01", "2026-06-05", "FIVE_MINUTE")
    assert len(second) == 15
    assert len(calls) == 1             # all served from Redis


def test_intraday_holidays_cached_as_empty_warm_run_is_full_hit(fake_redis, monkeypatch):
    """A weekday holiday (no rows) within a span that returned data is cached as
    empty so it isn't re-requested — a warm run makes ZERO broker calls."""
    # 2026-06-04 returns no rows (simulated holiday); other weekdays have data.
    def rows(day):
        return [] if day == "2026-06-04" else _intraday_rows(day)
    ds, calls = _intraday_service(monkeypatch, rows)

    ds.fetch_intraday_candles("TEST", "2026-06-01", "2026-06-05", "FIVE_MINUTE")
    assert fake_redis.store["alphadesk:candles:FIVE_MINUTE:TOK:2026-06-04"] == "[]"

    ds.fetch_intraday_candles("TEST", "2026-06-01", "2026-06-05", "FIVE_MINUTE")
    assert len(calls) == 1             # holiday cached as [] → no refetch


def test_intraday_empty_span_not_cached(fake_redis, monkeypatch):
    """A span that returns nothing (transient failure) isn't cached — retried."""
    ds, calls = _intraday_service(monkeypatch, lambda day: [])

    ds.fetch_intraday_candles("TEST", "2026-06-01", "2026-06-02", "FIVE_MINUTE")
    assert fake_redis.store == {}      # nothing cached
    ds.fetch_intraday_candles("TEST", "2026-06-01", "2026-06-02", "FIVE_MINUTE")
    assert len(calls) == 2             # retried, not served from a poisoned cache


def test_fetch_historical_intraday_aggregates_to_daily_and_caches(fake_redis, monkeypatch):
    """fetch_historical(FIVE_MINUTE) reuses the cached intraday fetch and folds
    each session into one daily OHLCV row — open=first, close=last, high=max,
    low=min, volume=sum — then serves from Redis on repeat."""
    def rows_for_day(day):
        return [
            [f"{day} 09:15:00", 100.0, 105.0, 98.0, 101.0, 500],   # first → open 100
            [f"{day} 09:20:00", 101.0, 110.0, 100.0, 103.0, 600],  # high 110
            [f"{day} 09:25:00", 103.0, 104.0, 95.0, 99.0, 700],    # low 95, last → close 99
        ]
    ds, calls = _intraday_service(monkeypatch, rows_for_day)

    out = ds.fetch_historical("TEST", "2026-06-01", "2026-06-01", "FIVE_MINUTE")
    assert out == [{
        "date": "2026-06-01", "open": 100.0, "high": 110.0,
        "low": 95.0, "close": 99.0, "volume": 1800,
    }]
    assert len(calls) == 1

    out2 = ds.fetch_historical("TEST", "2026-06-01", "2026-06-01", "FIVE_MINUTE")
    assert out2 == out
    assert len(calls) == 1             # aggregated from the cached intraday day
