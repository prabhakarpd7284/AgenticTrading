"""Backfill historical candles from Angel One into ``Symbol`` / ``Candle``.

The live tick ingestor only fills ``Candle`` while the market is open, so a
fresh database has no history for backtests or a scalping agent to learn on.
This pulls it from Angel's getCandleData through the shared ``BrokerClient``
(cross-process throttle + rate-limit breaker), so it can run alongside the
dev stack without tripping Angel's limits.

Resumable: per symbol it fetches only what lies outside the stored range —
the head before the first candle and the tail from the last stored day
(re-fetched, since it may be partial) to today — so re-running after an
interruption, or days later, only fetches what is missing.
Windows that come back empty are listed at the end — Angel returns [] both
for "no trading" and for a swallowed broker error, so check them.

Examples
--------
    # 6 months of 1-minute candles for NIFTY 50 + the main indices (default)
    ./manage.py backfill_candles --interval 1m --months 6

    # Every F&O stock instead (~230 names)
    ./manage.py backfill_candles --universe fno

    # Just a few symbols
    ./manage.py backfill_candles --symbols RELIANCE,HDFCBANK --interval 5m

    # Show the plan without calling the broker
    ./manage.py backfill_candles --universe fno --dry-run
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from django.core.management.base import BaseCommand, CommandError

IST = ZoneInfo("Asia/Kolkata")

# Our interval code → Angel interval name (same mapping as data_port).
_ANGEL_INTERVAL = {
    "1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE", "1h": "ONE_HOUR", "1d": "ONE_DAY",
}

# Index tradingsymbol → (exchange, Angel token). Tokens from Angel's scrip
# master (instrumenttype AMXIDX); names follow the underlying codes used
# elsewhere (NIFTY / BANKNIFTY / SENSEX).
INDICES = {
    "NIFTY": ("NSE", "99926000"),
    "BANKNIFTY": ("NSE", "99926009"),
    "FINNIFTY": ("NSE", "99926037"),
    "MIDCPNIFTY": ("NSE", "99926074"),
    "NIFTYNXT50": ("NSE", "99926013"),
    "INDIAVIX": ("NSE", "99926017"),
    "SENSEX": ("BSE", "99919000"),
    "BANKEX": ("BSE", "99919012"),
}


def fno_stock_names(instruments: list[dict]) -> list[str]:
    """Underlyings with stock futures listed on NFO = the F&O stock list.

    Angel's master also lists exchange test scrips (011NSETEST, …) as FUTSTK;
    they have no history, so drop them."""
    return sorted({
        i["name"] for i in instruments
        if i.get("exch_seg") == "NFO" and i.get("instrumenttype") == "FUTSTK"
        and i.get("name") and "NSETEST" not in i["name"]
    })


def windows(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    """Split [start, end] into inclusive chunks of at most ``max_days`` days."""
    out = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=max_days - 1), end)
        out.append((cur, stop))
        cur = stop + timedelta(days=1)
    return out


# A stored range that starts within this many days of the requested start is
# treated as complete at the head — the gap is a weekend/holiday, not a hole.
_HEAD_SLACK_DAYS = 5


def missing_ranges(start: date, end: date, first: date | None,
                   last: date | None) -> list[tuple[date, date]]:
    """Date ranges still to fetch given the stored [first, last] span."""
    if first is None or last is None:
        return [(start, end)]
    out = []
    if (first - start).days > _HEAD_SLACK_DAYS:
        out.append((start, first - timedelta(days=1)))
    if last < end:
        out.append((last, end))   # last stored day may be partial — refetch it
    return out


def parse_rows(rows: list[list]) -> list[tuple]:
    """Angel rows ``[iso_ts, o, h, l, c, v]`` → ``(aware_dt, o, h, l, c, v)``."""
    out = []
    for r in rows:
        t = datetime.fromisoformat(r[0])
        if t.tzinfo is None:
            t = t.replace(tzinfo=IST)
        out.append((t, r[1], r[2], r[3], r[4], int(r[5] or 0)))
    return out


class Command(BaseCommand):
    help = "Backfill historical candles from Angel One into the Candle table (resumable)."

    def add_arguments(self, parser):
        parser.add_argument("--universe", choices=["nifty50", "fno", "indices", "none"],
                            default="nifty50",
                            help="nifty50 = NIFTY 50 + indices (default); fno = F&O stocks "
                                 "+ indices; indices = indices only.")
        parser.add_argument("--symbols", default="",
                            help="Extra comma-separated NSE equity symbols to include.")
        parser.add_argument("--interval", choices=sorted(_ANGEL_INTERVAL), default="1m")
        parser.add_argument("--months", type=int, default=6,
                            help="How far back to fetch (default 6).")
        parser.add_argument("--dry-run", action="store_true",
                            help="Print the plan; no broker calls, no writes.")

    def handle(self, *args, **opts):
        from trading.services.data_service import BrokerClient, DataService

        interval = opts["interval"]
        angel_interval = _ANGEL_INTERVAL[interval]
        end = date.today()
        start = end - timedelta(days=round(opts["months"] * 30.5))
        span = DataService._INTERVAL_MAX_DAYS.get(angel_interval, 30)
        wins = windows(start, end, span)

        targets, missing = resolve_targets(
            opts["universe"], [s.strip().upper() for s in opts["symbols"].split(",") if s.strip()])
        for name in missing:
            self.stderr.write(f"  skip {name}: no NSE token in the scrip master")
        if not targets:
            raise CommandError("Nothing to backfill — check --universe / --symbols.")

        self.stdout.write(
            f"Backfill {interval} {start}..{end}: {len(targets)} symbols × {len(wins)} "
            f"windows of ≤{span}d = up to {len(targets) * len(wins)} Angel calls")
        if opts["dry_run"]:
            self.stdout.write("Dry run — nothing fetched.")
            return

        broker = BrokerClient.get_instance()
        written = calls = skipped = 0
        empty: list[str] = []
        t0 = time.monotonic()
        for n, (name, exchange, token) in enumerate(targets, 1):
            sym = self._symbol(name, exchange, token)
            sym_rows = 0
            first, last = self._stored_span(sym, interval)
            todo = [w for lo, hi in missing_ranges(start, end, first, last)
                    for w in windows(lo, hi, span)]
            skipped += max(0, len(wins) - len(todo))
            for ws, we in todo:
                rows = broker.fetch_candles(
                    token, f"{ws} 09:15", f"{we} 15:30", angel_interval,
                    exchange=exchange, wait_on_breaker=True)
                calls += 1
                if not rows:
                    empty.append(f"{name} {ws}..{we}")
                    continue
                sym_rows += self._write(sym, interval, parse_rows(rows))
            written += sym_rows
            mins = (time.monotonic() - t0) / 60
            self.stdout.write(f"  [{n}/{len(targets)}] {name}: +{sym_rows} rows "
                              f"({calls} calls, {mins:.1f} min)")

        self.stdout.write(self.style.SUCCESS(
            f"Done: {written} candles written, {calls} calls, ~{skipped} windows already present."))
        if empty:
            self.stdout.write(self.style.WARNING(
                f"{len(empty)} window(s) returned no data (holiday-only window, new listing, "
                f"or a swallowed broker error — re-run to retry):"))
            for e in empty[:50]:
                self.stdout.write(f"    {e}")

    def _symbol(self, name: str, exchange: str, token: str):
        return ensure_symbol_row(name, exchange, token)

    @staticmethod
    def _stored_span(sym, interval: str) -> tuple[date | None, date | None]:
        from django.db.models import Max, Min

        from apps.market_data.models import Candle
        agg = Candle.objects.filter(symbol=sym, interval=interval).aggregate(a=Min("t"), b=Max("t"))
        if agg["a"] is None:
            return None, None
        return agg["a"].astimezone(IST).date(), agg["b"].astimezone(IST).date()

    @staticmethod
    def _write(sym, interval: str, rows: list[tuple]) -> int:
        return write_rows(sym, interval, rows)


# ─── Shared with the after-close capture job (apps.market_data.services.session_capture) ──

def resolve_targets(universe: str, extra_symbols: list[str] | None = None
                    ) -> tuple[list[tuple[str, str, str]], list[str]]:
    """``(targets, missing)``: targets are ``(tradingsymbol, exchange, token)``
    for the universe's indices + stocks; missing are stocks with no NSE token."""
    from trading.services.ticker_service import TickerService

    ts = TickerService()
    ts._ensure_loaded()
    targets: list[tuple[str, str, str]] = []
    if universe != "none":
        targets += [(name, ex, tok) for name, (ex, tok) in INDICES.items()]
    if universe == "fno":
        stocks = fno_stock_names(ts._instruments or [])
    elif universe == "nifty50":
        from apps.market_data.constants import NIFTY_50_SYMBOLS
        stocks = list(NIFTY_50_SYMBOLS)
    else:
        stocks = []
    stocks += list(extra_symbols or [])
    missing = []
    for name in dict.fromkeys(stocks):
        tok = ts.get_token(name)
        if tok:
            targets.append((name, "NSE", tok))
        else:
            missing.append(name)
    return targets, missing


def ensure_symbol_row(name: str, exchange: str, token: str):
    from apps.market_data.models import Symbol
    from apps.market_data.services.candle_ingestor import _ensure_symbol

    if name in INDICES:
        row, _ = Symbol.objects.get_or_create(
            exchange=exchange, token=token,
            defaults={"tradingsymbol": name, "name": name, "segment": "INDEX"})
        return row
    return _ensure_symbol(Symbol, name)


def write_rows(sym, interval: str, rows: list[tuple]) -> int:
    """Insert parsed rows (duplicates ignored); returns the rows offered."""
    from apps.market_data.models import Candle
    objs = [Candle(symbol=sym, interval=interval, t=t, o=o, h=h, l=lo, c=c, v=v)
            for t, o, h, lo, c, v in rows]
    Candle.objects.bulk_create(objs, batch_size=5000, ignore_conflicts=True)
    return len(objs)
