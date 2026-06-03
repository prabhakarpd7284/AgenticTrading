"""
Horizon-aware EOD enrichment for v2 swing signals (``swing_v2_*``).

v1 ``enrich_signals`` measures the *signal day only* (intraday 5-min candles)
and marks anything still PENDING as EXPIRED that same evening. That is correct
for intraday signals but wrong for swings, which hold for days–weeks.

This command walks each *open* v2 swing signal across its holding window —
read straight off the signal itself (``indicators.time_stop_bars`` × the
tier's candle interval) — and fills:

  • max_favorable_move / max_adverse_move over the whole window (points)
  • eod_price (latest close seen)
  • indicators["swing"] = {result, hold_bars, realized_r, window_closed, …}
  • outcome: stays PENDING while the trade is still live inside its window;
    becomes EXPIRED once the target/stop is hit OR the window elapses.
    (v1 ``enrich_signals`` and its trade-linking are left untouched; a real
    Trade link still upgrades EXPIRED → TRADED/REJECTED.)

This is what makes the monthly report's capture-rate and per-strategy
win-rate correct for swings — both read these Signal fields.

Usage:
    python manage.py enrich_signals_v2                 # all open swings, all tenants
    python manage.py enrich_signals_v2 --tier medium
    python manage.py enrich_signals_v2 --tenant personal
"""
from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from logzero import logger


# Approx confirmed bars per trading session, per Angel One interval.
BARS_PER_DAY = {"FIFTEEN_MINUTE": 25, "ONE_HOUR": 6, "ONE_DAY": 1}


class Command(BaseCommand):
    help = "Horizon-aware enrichment for v2 swing signals (swing_v2_*)"

    def create_parser(self, prog_name, subcommand, **kwargs):
        parser = super().create_parser(prog_name, subcommand, **kwargs)
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        return parser

    def add_arguments(self, parser):
        parser.add_argument("--tenant", default=None, help="Restrict to one tenant (slug or name)")
        parser.add_argument("--tier", default=None, choices=["small", "medium", "long"])
        parser.add_argument("--all", action="store_true",
                            help="Re-walk every swing signal, not just open (PENDING) ones")

    def handle(self, *args, **opts):
        from apps.strategies.models import Signal
        from trading.services.data_service import DataService
        from plugins.strategy_swing.v2.config import get_tier_config

        tenants = self._resolve_tenants(opts["tenant"])
        ds = DataService()
        walked = resolved = errors = 0

        for tenant in tenants:
            qs = Signal.objects.filter(
                tenant=tenant,
                source=Signal.Source.OK_SCANNER,
                strategy__startswith="swing_v2_",
            )
            if not opts["all"]:
                qs = qs.filter(outcome=Signal.Outcome.PENDING)
            if opts["tier"]:
                qs = qs.filter(strategy__startswith=f"swing_v2_{opts['tier']}_")

            signals = list(qs)
            if not signals:
                continue
            self.stdout.write(f"[{tenant.slug or tenant.name}] {len(signals)} swing signals")

            for sig in signals:
                try:
                    if self._enrich_one(ds, get_tier_config, sig, Signal):
                        resolved += 1
                    walked += 1
                except Exception as exc:  # never let one symbol abort the run
                    logger.error(f"enrich_v2 {sig.symbol} #{sig.id}: {exc}")
                    errors += 1

        self.stdout.write(
            f"\nDone: walked {walked}, resolved {resolved}, errors {errors}.\n"
        )

    # ── internals ──────────────────────────────────────────────────────

    def _resolve_tenants(self, slug_or_name):
        from apps.tenants.models import Tenant
        if slug_or_name:
            t = (Tenant.objects.filter(slug=slug_or_name).first()
                 or Tenant.objects.filter(name=slug_or_name).first())
            if t is None:
                raise CommandError(f"Tenant {slug_or_name!r} not found.")
            return [t]
        tenants = list(Tenant.objects.all())
        if not tenants:
            raise CommandError("No tenants found.")
        return tenants

    def _enrich_one(self, ds, get_tier_config, sig, Signal) -> bool:
        """Enrich one swing signal. Returns True if it became resolved (EXPIRED)."""
        tier = (sig.indicators or {}).get("tier") or self._tier_from_strategy(sig.strategy)
        cfg = get_tier_config(tier)
        interval = cfg.interval
        window_bars = int((sig.indicators or {}).get("time_stop_bars") or cfg.max_hold_bars)

        # bars → trading days → calendar days (+ weekend slack + buffer)
        bpd = BARS_PER_DAY.get(interval, 1)
        trading_days = max(1, math.ceil(window_bars / bpd))
        cal_days = math.ceil(trading_days * 7 / 5) + 3

        start = sig.signal_date
        window_end = start + timedelta(days=cal_days)
        today = date.today()
        end = min(today, window_end)
        window_closed = today >= window_end

        candles = self._fetch(ds, sig.symbol, start, end, interval)
        entry_ts = sig.signal_time.replace(tzinfo=None)
        after = [c for c in candles if c["dt"] > entry_ts]

        eod_price = candles[-1]["c"] if candles else sig.entry_price
        if not after:
            # too early — nothing has happened past entry yet; keep it open
            self._save(sig, Signal, eod_price, 0.0, 0.0, None, 0, window_closed, end)
            return False

        long = sig.side == "BUY"
        entry, target, stop = sig.entry_price, sig.target, sig.stoploss
        max_fav = max_adv = 0.0
        hit = None
        hold_bars = len(after)

        for i, c in enumerate(after):
            if long:
                max_fav = max(max_fav, c["h"] - entry)
                max_adv = max(max_adv, entry - c["l"])
                tgt, stp = c["h"] >= target, c["l"] <= stop
            else:
                max_fav = max(max_fav, entry - c["l"])
                max_adv = max(max_adv, c["h"] - entry)
                tgt, stp = c["l"] <= target, c["h"] >= stop
            if hit is None and (tgt or stp):
                hit = "stop" if (tgt and stp) else ("target" if tgt else "stop")
                hold_bars = i + 1
                break

        eod_price = after[-1]["c"]
        self._save(sig, Signal, eod_price, max(0.0, max_fav), max(0.0, max_adv),
                   hit, hold_bars, window_closed, end)
        return hit is not None or window_closed

    def _save(self, sig, Signal, eod_price, max_fav, max_adv, hit, hold_bars,
              window_closed, through):
        risk = abs(sig.entry_price - sig.stoploss) or 1e-9
        long = sig.side == "BUY"
        if hit == "target":
            realized_r = abs(sig.target - sig.entry_price) / risk
            result = "target"
        elif hit == "stop":
            realized_r = -1.0
            result = "stop"
        else:
            mtm = (eod_price - sig.entry_price) if long else (sig.entry_price - eod_price)
            realized_r = mtm / risk
            result = "timeout" if window_closed else "open"

        resolved = hit is not None or window_closed
        ind = dict(sig.indicators or {})
        ind["swing"] = {
            "result": result,
            "hold_bars": hold_bars,
            "realized_r": round(realized_r, 2),
            "window_closed": window_closed,
            "enriched_through": through.isoformat(),
        }

        sig.eod_price = round(float(eod_price), 2)
        sig.max_favorable_move = round(float(max_fav), 2)
        sig.max_adverse_move = round(float(max_adv), 2)
        sig.indicators = ind
        if resolved and sig.outcome == Signal.Outcome.PENDING:
            sig.outcome = Signal.Outcome.EXPIRED
        sig.save(update_fields=[
            "eod_price", "max_favorable_move", "max_adverse_move",
            "indicators", "outcome",
        ])

    def _fetch(self, ds, symbol, start, end, interval):
        try:
            raw = ds.fetch_historical(
                symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                interval=interval,
            )
        except Exception as exc:
            logger.warning(f"enrich_v2 fetch {symbol}: {exc}")
            return []
        out = []
        for c in raw or []:
            dt = self._parse_ts(c.get("timestamp") or c.get("date") or c.get("time"))
            if dt is None:
                continue
            out.append({
                "dt": dt,
                "h": float(c["high"]), "l": float(c["low"]), "c": float(c["close"]),
            })
        out.sort(key=lambda r: r["dt"])
        return out

    @staticmethod
    def _parse_ts(raw):
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw.replace(tzinfo=None)
        if isinstance(raw, date):
            return datetime(raw.year, raw.month, raw.day)
        if isinstance(raw, str):
            s = raw.replace("Z", "").replace("+05:30", "").strip()
            try:
                return datetime.fromisoformat(s).replace(tzinfo=None)
            except ValueError:
                try:
                    return datetime.strptime(s[:10], "%Y-%m-%d")
                except ValueError:
                    return None
        return None

    @staticmethod
    def _tier_from_strategy(strategy: str) -> str:
        # "swing_v2_<tier>_<phase>"
        parts = strategy.split("_")
        return parts[2] if len(parts) >= 3 else "medium"
