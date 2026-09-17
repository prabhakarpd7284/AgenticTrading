"""
CLI: Oliver Kell v2 timeframe-tiered swing scanner.

    python manage.py run_ok_scanner_v2 --tier medium --actionable-only
    python manage.py run_ok_scanner_v2 --tier small --symbols RELIANCE,TCS
    python manage.py run_ok_scanner_v2 --tier long --universe nifty50 --json

Tiers:  small=15m (days) · medium=1h (~1–2 wk) · long=daily (wk–mo)

Isolated from v1 `run_ok_scanner`. Persists graded signals to
apps.strategies.Signal (source=OK_SCANNER, strategy tagged `swing_v2_*`) and
emits a SIGNAL_FIRED event per signal so they land in the monthly ledger.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand
from logzero import logger


_ACTION_COLOR = {"BUY": "\033[92m", "ADD": "\033[96m", "SHORT": "\033[91m",
                 "TRIM": "\033[93m", "WATCH": "\033[90m", "AVOID": "\033[90m"}
_RESET = "\033[0m"


class Command(BaseCommand):
    help = "Oliver Kell v2 swing scanner (timeframe-tiered, robust)"

    def add_arguments(self, parser):
        parser.add_argument("--tier", default="medium", choices=["small", "medium", "long"])
        parser.add_argument("--symbols", default="", help="Comma-separated NSE symbols")
        parser.add_argument("--universe", default="all", choices=["nifty50", "nifty100", "all"])
        parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
        parser.add_argument("--capital", type=float, default=500_000.0)
        parser.add_argument("--actionable-only", action="store_true")
        parser.add_argument("--json", action="store_true", help="Emit JSON, skip persistence")
        parser.add_argument("--no-persist", action="store_true")
        parser.add_argument("--clear-cache", action="store_true")

    def handle(self, *args, **opts):
        from plugins.strategy_swing.v2.scanner import SwingScannerV2, invalidate_cache

        if opts["clear_cache"]:
            invalidate_cache()
            logger.info("v2 cache cleared")

        symbols = self._resolve_symbols(opts)
        scanner = SwingScannerV2(tier=opts["tier"], capital=opts["capital"])
        results = scanner.scan(symbols, scan_date=opts["date"])

        if opts["actionable_only"]:
            results = [r for r in results if r.actionable]

        if opts["json"]:
            import json
            self.stdout.write(json.dumps([r.to_dict() for r in results], indent=2, default=str))
            return

        self._print_table(results, opts["tier"])

        if not opts["no_persist"]:
            n = self._persist(scanner.actionable(), opts["tier"])
            logger.info(f"v2: persisted {n} signals")

    # ──────────────────────────────────────────────────────────────────

    def _resolve_symbols(self, opts):
        if opts["symbols"]:
            return [s.strip().upper() for s in opts["symbols"].split(",") if s.strip()]
        from apps.market_data.constants import NIFTY_50_SYMBOLS, SCREENER_UNIVERSE
        return NIFTY_50_SYMBOLS if opts["universe"] == "nifty50" else SCREENER_UNIVERSE

    def _print_table(self, results, tier):
        if not results:
            self.stdout.write("No results.")
            return
        self.stdout.write(f"\n  Oliver Kell v2 — tier={tier}  ({len(results)} rows)\n")
        hdr = f"  {'Symbol':<14}{'Phase':<6}{'Action':<7}{'Grade':<6}{'Score':<7}{'Regime(P/H)':<16}{'Entry':>9}{'Stop':>9}{'Target':>9}{'R:R':>6}{'Qty':>7}"
        self.stdout.write(hdr)
        self.stdout.write("  " + "─" * (len(hdr)))
        for r in results:
            col = _ACTION_COLOR.get(r.action.value, "")
            reg = f"{r.regime_primary.value[:4]}/{r.regime_htf.value[:4]}"
            self.stdout.write(
                f"  {r.symbol:<14}{r.phase.value:<6}{col}{r.action.value:<7}{_RESET}"
                f"{r.grade:<6}{r.score:<7}{reg:<16}"
                f"{r.entry:>9.2f}{r.stop:>9.2f}{r.target:>9.2f}{r.rr:>6.2f}{r.qty:>7}"
            )
        self.stdout.write("")

    def _persist(self, actionable, tier) -> int:
        """Persist graded signals to the ledger.

        DAO-optimised: instead of one get_or_create (SELECT+INSERT) per row,
        we (1) pull the day's existing v2 keys for this tier in a single
        indexed query, (2) bulk_create only the new rows in one INSERT, then
        (3) emit one SIGNAL_FIRED event per *new* row (kept individual so the
        live firehose broadcast still fires; the actionable set is small).
        """
        if not actionable:
            return 0
        try:
            from apps.strategies.models import Signal
            from apps.tenants.models import Membership
            from apps.events.services import event_writer
            from django.utils import timezone
        except Exception as e:
            logger.warning(f"v2 persist skipped (imports): {e}")
            return 0

        owner = (
            Membership.objects.filter(role="owner")
            .select_related("tenant")
            .only("id", "tenant__id")
            .first()
        )
        if not owner:
            logger.warning("v2 persist: no owner tenant")
            return 0
        tenant = owner.tenant
        now = timezone.now()
        today = now.date()
        strat_of = lambda r: f"swing_v2_{tier}_{r.phase.value}"[:48]

        # (1) one indexed query (tenant, source, signal_date) → existing keys
        existing = set(
            Signal.objects.filter(
                tenant=tenant,
                source=Signal.Source.OK_SCANNER,
                signal_date=today,
                strategy__startswith=f"swing_v2_{tier}_",
            ).values_list("symbol", "strategy")
        )

        # (2) build only-new rows, de-duped within the batch too
        to_create, seen = [], set()
        for r in actionable:
            key = (r.symbol, strat_of(r))
            if key in existing or key in seen:
                continue
            seen.add(key)
            to_create.append(Signal(
                tenant=tenant,
                symbol=r.symbol,
                signal_date=today,
                signal_time=now,
                source=Signal.Source.OK_SCANNER,
                strategy=key[1],
                side="BUY" if r.action.value in ("BUY", "ADD") else "SELL",
                entry_price=r.entry,
                stoploss=r.stop,
                target=r.target,
                confidence=r.score,
                risk_reward=r.rr,
                reasons=r.reasons,
                indicators={
                    "version": "v2", "tier": tier, "phase": r.phase.value,
                    "direction": r.direction, "grade": r.grade,
                    "regime_primary": r.regime_primary.value,
                    "regime_htf": r.regime_htf.value, "aligned": r.aligned,
                    "ema_fast": r.ema_fast, "ema_mid": r.ema_mid, "ema_slow": r.ema_slow,
                    "atr": r.atr, "ext_atr": r.ext_atr, "rsi": r.rsi, "roc": r.roc,
                    "rs_slope": r.rs_slope, "base_len": r.base_len,
                    "contraction_ratio": r.contraction_ratio, "vol_ratio": r.vol_ratio,
                    "bars_since_phase": r.bars_since_phase,
                    "target_measured": r.target_measured, "qty": r.qty,
                    "time_stop_bars": r.time_stop_bars, "factors": r.factors,
                },
            ))

        if not to_create:
            return 0

        # (3) single INSERT; Postgres backfills PKs onto the objects
        created = Signal.objects.bulk_create(to_create)

        for sig in created:
            ind = sig.indicators
            event_writer.emit(
                tenant=tenant,
                type="signal.fired",   # apps.events.models EventType.SIGNAL_FIRED value
                text=f"[swing v2/{tier}] {sig.side} {sig.symbol} {ind['phase']} "
                     f"grade {ind['grade']} @ {sig.entry_price} (R:R {sig.risk_reward})",
                severity="info",
                actor_kind="system",
                signal_id=sig.id,
                payload={"source": "swing_v2", "tier": tier, "grade": ind["grade"],
                         "score": sig.confidence, "direction": ind["direction"]},
            )
        return len(created)
