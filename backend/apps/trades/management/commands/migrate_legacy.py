"""One-shot data lift from legacy SQLite (`trading.*`) to Postgres (v2 apps).

Reads from the `legacy` DB alias (configured in settings/dev.py to point at
the repo-root db.sqlite3) and writes to the `default` Postgres alias. The
script is idempotent — re-running won't duplicate rows because each legacy
row carries its primary key into a `legacy_*_id` shim column on the v2 row,
and the upsert is keyed on that shim.

Phases (each in its own transaction):
  1. Resolve --tenant and ensure a Portfolio exists for it
  2. TradeJournal      → trades.Trade
  3. AuditLog          → events.Event
  4. StraddlePosition  → trades.OptionsPosition + OptionsLeg rows
                         (management_log[] entries → events.Event rows)
  5. PortfolioSnapshot → portfolio.PortfolioSnapshot

Run with `--dry-run` to see row counts without committing.

Usage:
    python manage.py migrate_legacy --tenant=personal
    python manage.py migrate_legacy --tenant=personal --dry-run
    python manage.py migrate_legacy --tenant=personal --phase=trades
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.events.models import Event
from apps.portfolio.models import Portfolio, PortfolioSnapshot
from apps.tenants.models import Tenant
from apps.trades.models import OptionsLeg, OptionsPosition, Trade

# ── Status mapping: legacy TradeJournal.status → trades.Trade.Status ─

LEGACY_STATUS_MAP = {
    "PENDING":   Trade.Status.PLAN,
    "PLANNED":   Trade.Status.PLAN,
    "APPROVED":  Trade.Status.APPROVED,
    "REJECTED":  Trade.Status.REJECTED,
    "EXECUTED":  Trade.Status.SENT,
    "FILLED":    Trade.Status.FILLED,
    "PARTIAL":   Trade.Status.PARTIAL,
    "PAPER":     Trade.Status.FILLED,    # paper fills are still fills
    "CANCELLED": Trade.Status.CANCELLED,
}

LEGACY_AUDIT_TYPE_MAP = {
    "PLANNER_REQ":  Event.Type.LLM_REQUEST,
    "PLANNER_RES":  Event.Type.LLM_RESPONSE,
    "PLANNER_ERR":  Event.Type.LLM_ERROR,
    "RISK_APPROVE": Event.Type.RISK_APPROVED,
    "RISK_REJECT":  Event.Type.RISK_REJECTED,
    "EXECUTION":    Event.Type.ORDER_SENT,
    "RECONCILE":    Event.Type.SYSTEM_AI_RESUMED,  # closest fit; "system reconciliation"
}


class Command(BaseCommand):
    help = "Lift legacy SQLite trading.* data into v2 Postgres tables."

    def add_arguments(self, parser):
        parser.add_argument("--tenant", required=True,
                            help="Tenant slug or name to attach migrated rows to")
        parser.add_argument("--dry-run", action="store_true",
                            help="Print counts without writing anything")
        parser.add_argument("--phase", default="all",
                            choices=["all", "trades", "audit", "options", "snapshots",
                                      "signals", "watchlist", "knowledge", "system"],
                            help="Restrict to a single phase (for retries)")

    def handle(self, *args, **opts):
        # Lazy-import legacy models so the command imports cleanly even
        # if the legacy app isn't installed in some environments.
        try:
            from trading.models import (  # noqa: F401  — local to lift
                AuditLog as LegacyAuditLog,
                PortfolioSnapshot as LegacyPortfolioSnapshot,
                SignalLog as LegacySignalLog,
                StraddlePosition as LegacyStraddle,
                StrategyDoc as LegacyStrategyDoc,
                SystemControl as LegacySystemControl,
                TradeJournal as LegacyTradeJournal,
                TraderNote as LegacyTraderNote,
                WatchlistEntry as LegacyWatchlist,
            )
        except ImportError as e:
            raise CommandError(
                "Could not import legacy trading.* models. Make sure "
                "INSTALLED_APPS includes 'trading' (dev.py does)."
            ) from e

        self.LegacyTradeJournal = LegacyTradeJournal
        self.LegacyAuditLog = LegacyAuditLog
        self.LegacyStraddle = LegacyStraddle
        self.LegacyPortfolioSnapshot = LegacyPortfolioSnapshot
        self.LegacySignalLog = LegacySignalLog
        self.LegacyWatchlist = LegacyWatchlist
        self.LegacyStrategyDoc = LegacyStrategyDoc
        self.LegacySystemControl = LegacySystemControl
        self.LegacyTraderNote = LegacyTraderNote

        tenant = self._resolve_tenant(opts["tenant"])
        portfolio = self._resolve_portfolio(tenant)
        self.dry = opts["dry_run"]
        phase = opts["phase"]

        self.stdout.write(
            f"\nLegacy data lift\n"
            f"  tenant    : {tenant.id} ({tenant.name})\n"
            f"  portfolio : {portfolio.id} ({portfolio.name})\n"
            f"  dry_run   : {self.dry}\n"
            f"  phase     : {phase}\n"
        )

        if phase in ("all", "trades"):
            self._lift_trades(tenant, portfolio)
        if phase in ("all", "audit"):
            self._lift_audit(tenant)
        if phase in ("all", "options"):
            self._lift_straddles(tenant, portfolio)
        if phase in ("all", "snapshots"):
            self._lift_snapshots(tenant, portfolio)
        if phase in ("all", "signals"):
            self._lift_signals(tenant)
        if phase in ("all", "watchlist"):
            self._lift_watchlist(tenant)
        if phase in ("all", "knowledge"):
            self._lift_knowledge(tenant)
        if phase in ("all", "system"):
            self._lift_system(tenant)

        self.stdout.write(self.style.SUCCESS("\nDone."))

    # ── Tenant + Portfolio resolution ──────────────────────────────────

    def _resolve_tenant(self, slug_or_name: str) -> Tenant:
        t = Tenant.objects.filter(slug=slug_or_name).first()
        if t is None:
            t = Tenant.objects.filter(name=slug_or_name).first()
        if t is None:
            raise CommandError(f"Tenant {slug_or_name!r} not found.")
        return t

    def _resolve_portfolio(self, tenant: Tenant) -> Portfolio:
        p = Portfolio.objects.filter(tenant=tenant).order_by("created_at").first()
        if p is None:
            from django.conf import settings
            p = Portfolio.objects.create(
                tenant=tenant, name="Default",
                capital=Decimal(str(settings.ALPHADESK.get("DEFAULT_CAPITAL", 500_000))),
                mode="paper",
            )
            self.stdout.write(f"  · created Portfolio {p.id} for tenant")
        return p

    # ── Phase 2: TradeJournal → Trade ──────────────────────────────────

    def _lift_trades(self, tenant: Tenant, portfolio: Portfolio):
        existing_legacy_ids = set(
            Trade.objects.filter(tenant=tenant, legacy_trade_journal_id__isnull=False)
            .values_list("legacy_trade_journal_id", flat=True)
        )
        qs = self.LegacyTradeJournal.objects.using("legacy").exclude(
            id__in=existing_legacy_ids
        )
        total = qs.count()
        self.stdout.write(f"\n· Trades: {total} new legacy rows to migrate")
        if self.dry or total == 0:
            return

        created = 0
        with transaction.atomic():
            for tj in qs.iterator(chunk_size=200):
                Trade.objects.create(
                    tenant=tenant,
                    portfolio=portfolio,
                    symbol=tj.symbol,
                    exchange="NSE",  # legacy is equity-only at this layer
                    side=tj.side,
                    product="INTRADAY",
                    entry_price=Decimal(str(tj.entry_price)),
                    stop_loss=Decimal(str(tj.stop_loss)),
                    target=Decimal(str(tj.target)),
                    quantity=tj.quantity,
                    lot_size=1,
                    confidence=Decimal(str(tj.confidence or 0)),
                    reasoning=tj.reasoning or "",
                    risk_approved=tj.risk_approved,
                    risk_reason=(tj.risk_reason or "")[:255],
                    status=LEGACY_STATUS_MAP.get(tj.status, Trade.Status.PLAN),
                    fill_price=Decimal(str(tj.fill_price)) if tj.fill_price else None,
                    fill_quantity=tj.fill_quantity,
                    realized_pnl=Decimal(str(tj.pnl)) if tj.pnl is not None else None,
                    pnl_percent=Decimal(str(tj.pnl_percent)) if tj.pnl_percent is not None else None,
                    origin=Trade.Origin.WORKFLOW if tj.reasoning else Trade.Origin.MANUAL,
                    trade_date=tj.trade_date,
                    legacy_trade_journal_id=tj.id,
                )
                created += 1
        self.stdout.write(f"  ✓ created {created} Trade rows")

    # ── Phase 3: AuditLog → Event ──────────────────────────────────────

    def _lift_audit(self, tenant: Tenant):
        # Build trade-journal-id → trade-uuid lookup
        tj_to_trade: dict[int, UUID] = dict(
            Trade.objects.filter(tenant=tenant, legacy_trade_journal_id__isnull=False)
            .values_list("legacy_trade_journal_id", "id")
        )

        qs = self.LegacyAuditLog.objects.using("legacy").all()
        total = qs.count()
        self.stdout.write(f"\n· AuditLog: {total} legacy events")
        if self.dry or total == 0:
            return

        created = 0
        with transaction.atomic():
            for al in qs.iterator(chunk_size=500):
                trade_uuid = tj_to_trade.get(al.trade_journal_id) if al.trade_journal_id else None
                Event.objects.create(
                    tenant=tenant,
                    ts=al.created_at,
                    type=LEGACY_AUDIT_TYPE_MAP.get(al.event_type, Event.Type.SYSTEM_AI_RESUMED),
                    severity=(Event.Severity.ERROR if al.event_type == "PLANNER_ERR"
                              else Event.Severity.WARN if al.event_type == "RISK_REJECT"
                              else Event.Severity.INFO),
                    actor_kind=(Event.ActorKind.WORKFLOW if al.event_type.startswith("PLANNER")
                                 else Event.ActorKind.RISK_ENGINE if al.event_type.startswith("RISK")
                                 else Event.ActorKind.SYSTEM),
                    trade_id=trade_uuid,
                    payload={
                        "prompt": al.prompt or "",
                        "response": al.response or "",
                        "model": al.model_name or "",
                        "tokens": al.tokens_used,
                        "latency_ms": al.latency_ms,
                        "risk": al.risk_details or {},
                        "execution": al.execution_details or {},
                    },
                    text=f"{al.event_type} {al.symbol}".strip(),
                )
                created += 1
        self.stdout.write(f"  ✓ created {created} Event rows")

    # ── Phase 4: StraddlePosition → OptionsPosition + Legs + Events ───

    def _lift_straddles(self, tenant: Tenant, portfolio: Portfolio):
        existing = set(
            OptionsPosition.objects.filter(tenant=tenant, legacy_straddle_id__isnull=False)
            .values_list("legacy_straddle_id", flat=True)
        )
        qs = self.LegacyStraddle.objects.using("legacy").exclude(id__in=existing)
        total = qs.count()
        self.stdout.write(f"\n· StraddlePositions: {total} new legacy rows")
        if self.dry or total == 0:
            return

        created_pos = created_legs = created_events = 0
        with transaction.atomic():
            for sp in qs.iterator(chunk_size=100):
                pos = OptionsPosition.objects.create(
                    tenant=tenant,
                    portfolio=portfolio,
                    position_type=OptionsPosition.PositionType.SHORT_STRADDLE,
                    underlying=sp.underlying,
                    expiry=sp.expiry,
                    lot_size=sp.lot_size,
                    lots=sp.lots,
                    status=self._map_straddle_status(sp.status),
                    net_delta=Decimal(str(sp.net_delta)),
                    current_pnl_inr=Decimal(str(sp.current_pnl_inr)),
                    realized_pnl=Decimal(str(sp.realized_pnl)),
                    opened_at=sp.opened_at,
                    closed_at=sp.closed_at,
                    trade_date=sp.trade_date,
                    legacy_straddle_id=sp.id,
                )
                created_pos += 1

                OptionsLeg.objects.create(
                    position=pos,
                    leg_role=OptionsLeg.LegRole.SHORT_CE,
                    symbol=sp.ce_symbol, token=sp.ce_token,
                    qty=sp.lot_size * sp.lots,
                    open_price=Decimal(str(sp.ce_sell_price)),
                    current_price=Decimal(str(sp.ce_current_price)),
                )
                OptionsLeg.objects.create(
                    position=pos,
                    leg_role=OptionsLeg.LegRole.SHORT_PE,
                    symbol=sp.pe_symbol, token=sp.pe_token,
                    qty=sp.lot_size * sp.lots,
                    open_price=Decimal(str(sp.pe_sell_price)),
                    current_price=Decimal(str(sp.pe_current_price)),
                )
                created_legs += 2

                # management_log[] → Event rows, linked via options_position FK
                for entry in (sp.management_log or []):
                    Event.objects.create(
                        tenant=tenant,
                        ts=self._parse_ts(entry.get("time")) or sp.last_updated,
                        type=Event.Type.STRADDLE_LEG_ROLLED,
                        actor_kind=Event.ActorKind.WORKFLOW,
                        options_position=pos,
                        payload={
                            "legacy_straddle_id": sp.id,
                            "action": entry.get("action"),
                            "nifty": entry.get("nifty"),
                            "pnl_inr": entry.get("pnl_inr"),
                            "note": entry.get("note"),
                            "executed": entry.get("executed"),
                        },
                        text=f"{entry.get('action', 'action')} — {entry.get('note', '')}",
                    )
                    created_events += 1

        self.stdout.write(
            f"  ✓ created {created_pos} OptionsPosition rows, "
            f"{created_legs} legs, {created_events} management events"
        )

    def _map_straddle_status(self, legacy_status: str) -> str:
        return {
            "ACTIVE": OptionsPosition.Status.ACTIVE,
            "PARTIAL": OptionsPosition.Status.PARTIAL,
            "HEDGED": OptionsPosition.Status.HEDGED,
            "CLOSED": OptionsPosition.Status.CLOSED,
        }.get(legacy_status, OptionsPosition.Status.ACTIVE)

    def _parse_ts(self, value: Any):
        if not value:
            return None
        if isinstance(value, str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return value

    # ── Phase 6: SignalLog → strategies.Signal ───────────────────────

    def _lift_signals(self, tenant: Tenant):
        from apps.strategies.models import Signal

        existing = set(
            Signal.objects.filter(tenant=tenant, legacy_signal_log_id__isnull=False)
            .values_list("legacy_signal_log_id", flat=True)
        )
        qs = self.LegacySignalLog.objects.using("legacy").exclude(id__in=existing)
        total = qs.count()
        self.stdout.write(f"\n· SignalLog: {total} new legacy rows")
        if self.dry or total == 0:
            return

        # Build legacy-trade-journal-id → new-trade lookup so signals
        # whose `trade_journal_id` is set can be relinked.
        tj_to_trade = dict(
            Trade.objects.filter(tenant=tenant, legacy_trade_journal_id__isnull=False)
            .values_list("legacy_trade_journal_id", "id")
        )

        created = 0
        with transaction.atomic():
            for sl in qs.iterator(chunk_size=500):
                Signal.objects.create(
                    tenant=tenant,
                    symbol=sl.symbol,
                    signal_date=sl.signal_date,
                    signal_time=sl.signal_time,
                    source=sl.source,
                    strategy=sl.strategy,
                    side=sl.side,
                    entry_price=sl.entry_price,
                    stoploss=sl.stoploss,
                    target=sl.target,
                    confidence=sl.confidence,
                    risk_reward=sl.risk_reward,
                    reasons=sl.reasons or [],
                    indicators=sl.indicators or {},
                    outcome=sl.outcome,
                    outcome_reason=sl.outcome_reason or "",
                    trade_id=tj_to_trade.get(sl.trade_journal_id) if sl.trade_journal_id else None,
                    eod_price=sl.eod_price,
                    max_favorable_move=sl.max_favorable_move,
                    max_adverse_move=sl.max_adverse_move,
                    legacy_signal_log_id=sl.id,
                )
                created += 1
        self.stdout.write(f"  ✓ created {created} Signal rows")

    # ── Phase 7: WatchlistEntry → strategies.WatchlistEntry ──────────

    def _lift_watchlist(self, tenant: Tenant):
        from apps.strategies.models import WatchlistEntry as V2Watchlist

        existing = set(
            V2Watchlist.objects.filter(tenant=tenant, legacy_watchlist_id__isnull=False)
            .values_list("legacy_watchlist_id", flat=True)
        )
        qs = self.LegacyWatchlist.objects.using("legacy").exclude(id__in=existing)
        total = qs.count()
        self.stdout.write(f"\n· WatchlistEntry: {total} new legacy rows")
        if self.dry or total == 0:
            return

        created = 0
        with transaction.atomic():
            for w in qs.iterator(chunk_size=500):
                V2Watchlist.objects.update_or_create(
                    tenant=tenant, symbol=w.symbol, scan_date=w.scan_date,
                    defaults=dict(
                        score=w.score,
                        bias=w.bias,
                        setups=w.setups or [],
                        prev_high=w.prev_high,
                        prev_low=w.prev_low,
                        prev_close=w.prev_close,
                        prev_atr=w.prev_atr,
                        orb_high=w.orb_high,
                        orb_low=w.orb_low,
                        vwap=w.vwap,
                        outcome=w.outcome,
                        triggered_setup=w.triggered_setup or "",
                        reason=w.reason or "",
                        legacy_watchlist_id=w.id,
                    ),
                )
                created += 1
        self.stdout.write(f"  ✓ upserted {created} WatchlistEntry rows")

    # ── Phase 8: StrategyDoc → rag.KnowledgeDoc ──────────────────────

    def _lift_knowledge(self, tenant: Tenant):
        from apps.rag.models import KnowledgeDoc

        existing = set(
            KnowledgeDoc.objects.filter(tenant=tenant, legacy_strategy_doc_id__isnull=False)
            .values_list("legacy_strategy_doc_id", flat=True)
        )
        qs = self.LegacyStrategyDoc.objects.using("legacy").exclude(id__in=existing)
        total = qs.count()
        self.stdout.write(f"\n· StrategyDoc: {total} new legacy rows")
        if self.dry or total == 0:
            return

        created = 0
        with transaction.atomic():
            for sd in qs.iterator(chunk_size=200):
                KnowledgeDoc.objects.create(
                    tenant=tenant,
                    title=sd.title,
                    content=sd.content,
                    category=sd.category,
                    is_active=sd.is_active,
                    legacy_strategy_doc_id=sd.id,
                )
                created += 1
        self.stdout.write(f"  ✓ created {created} KnowledgeDoc rows")

    # ── Phase 9: SystemControl + TraderNote ──────────────────────────

    def _lift_system(self, tenant: Tenant):
        from apps.system.models import SystemControl, TraderNote

        sc_qs = self.LegacySystemControl.objects.using("legacy").all()
        sc_total = sc_qs.count()
        tn_qs = self.LegacyTraderNote.objects.using("legacy").all()
        tn_total = tn_qs.count()
        self.stdout.write(
            f"\n· SystemControl: {sc_total} rows · TraderNote: {tn_total} rows"
        )
        if self.dry:
            return

        sc_created = tn_created = 0
        with transaction.atomic():
            for sc in sc_qs.iterator(chunk_size=200):
                SystemControl.objects.update_or_create(
                    tenant=tenant, key=sc.key,
                    defaults={"value": sc.value or {}},
                )
                sc_created += 1
            for tn in tn_qs.iterator(chunk_size=200):
                TraderNote.objects.update_or_create(
                    tenant=tenant, symbol=tn.symbol,
                    defaults={"note": tn.note or ""},
                )
                tn_created += 1
        self.stdout.write(
            f"  ✓ upserted {sc_created} SystemControl + {tn_created} TraderNote rows"
        )

    # ── Phase 5: PortfolioSnapshot ────────────────────────────────────

    def _lift_snapshots(self, tenant: Tenant, portfolio: Portfolio):
        # Skip if any snapshot already exists for this portfolio + date
        existing_dates = set(
            PortfolioSnapshot.objects.filter(
                tenant=tenant, portfolio=portfolio
            ).dates("captured_at", "day")
        )
        qs = self.LegacyPortfolioSnapshot.objects.using("legacy").all()
        total = qs.count()
        self.stdout.write(f"\n· PortfolioSnapshots: {total} legacy rows")
        if self.dry or total == 0:
            return

        created = 0
        with transaction.atomic():
            for ps in qs.iterator(chunk_size=200):
                if ps.snapshot_date in existing_dates:
                    continue
                PortfolioSnapshot.objects.create(
                    tenant=tenant, portfolio=portfolio,
                    equity=Decimal(str(ps.capital + ps.total_pnl)),
                    day_pnl=Decimal(str(ps.daily_pnl)),
                    unrealized_pnl=Decimal("0"),
                    open_positions=ps.open_positions,
                )
                created += 1
        self.stdout.write(f"  ✓ created {created} PortfolioSnapshot rows")
