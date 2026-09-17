"""Tests for the auto-kind watchlist framework.

Five resolver paths, each over existing DB tables:
  - SIGNAL_RANK      → top-N by Signal count
  - SOURCE_HOT       → top-N filtered by Signal.source
  - RECENT_ACTIVE    → any Signal in last N hours
  - TRADED_RECENTLY  → from Trade.status ∈ realised set
  - SHORTLIST_TODAY  → from WatchlistEntry where scan_date = today

Plus contract behaviour:
  - POST create with auto kind seeds symbols inline
  - POST /refresh/ re-resolves on demand; MANUAL kinds 400
  - POST /add-symbols/, /remove-symbols/ are blocked on auto kinds
  - SOURCE_HOT without a valid config.source 400s on create
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from django.utils import timezone

from apps.notifications.models import Watchlist
from apps.notifications.services.watchlist_resolvers import (
    refresh_watchlist, resolve_symbols,
)
from apps.notifications.tasks.watchlists import refresh_auto_watchlists
from apps.strategies.models import Signal, WatchlistEntry
from apps.trading.models import Trade


pytestmark = pytest.mark.django_db


# ── Seed helpers ────────────────────────────────────────────────────────

def _signal(tenant, *, symbol, source, side="BUY", hours_ago=1, strategy="vwap"):
    ts = timezone.now() - timedelta(hours=hours_ago)
    return Signal.objects.create(
        tenant=tenant, symbol=symbol,
        signal_date=ts.date(), signal_time=ts,
        source=source, strategy=strategy, side=side,
        entry_price=100.0, stoploss=0.0, target=0.0,
    )


# ── Resolvers ───────────────────────────────────────────────────────────

class TestResolveSignalRank:
    def test_returns_top_n_by_count_within_window(self, owner):
        t = owner.memberships.first().tenant
        for _ in range(5): _signal(t, symbol="RELIANCE", source="TRADINGVIEW")
        for _ in range(3): _signal(t, symbol="TCS", source="SCREENER")
        _signal(t, symbol="ITC", source="TRADINGVIEW", hours_ago=24 * 30)   # outside 7d default

        wl = Watchlist(
            tenant=t, owner=owner, name="rank",
            kind=Watchlist.Kind.SIGNAL_RANK,
            config={"window_days": 7, "top_n": 5},
        )
        wl.save()
        wl.refresh_from_db()

        assert resolve_symbols(wl) == ["RELIANCE", "TCS"]

    def test_top_n_caps_result(self, owner):
        t = owner.memberships.first().tenant
        for sym in ("A", "B", "C", "D", "E"):
            _signal(t, symbol=sym, source="TRADINGVIEW")

        wl = Watchlist.objects.create(
            tenant=t, owner=owner, name="cap",
            kind=Watchlist.Kind.SIGNAL_RANK,
            config={"top_n": 3},
        )
        assert len(resolve_symbols(wl)) == 3


class TestResolveSourceHot:
    def test_filters_to_one_source(self, owner):
        t = owner.memberships.first().tenant
        for _ in range(5): _signal(t, symbol="RELIANCE", source="TRADINGVIEW")
        for _ in range(10): _signal(t, symbol="TCS", source="SCREENER")

        wl = Watchlist.objects.create(
            tenant=t, owner=owner, name="tv hot",
            kind=Watchlist.Kind.SOURCE_HOT,
            config={"source": "TRADINGVIEW", "top_n": 5},
        )
        # SCREENER has more total but the filter pins it to TRADINGVIEW only.
        assert resolve_symbols(wl) == ["RELIANCE"]

    def test_unknown_source_falls_back_to_rank(self, owner):
        t = owner.memberships.first().tenant
        for _ in range(2): _signal(t, symbol="RELIANCE", source="TRADINGVIEW")

        wl = Watchlist.objects.create(
            tenant=t, owner=owner, name="bad src",
            kind=Watchlist.Kind.SOURCE_HOT,
            config={"source": "BOGUS"},
        )
        # Resolver-level fallback — at the API level we'd 400 instead.
        assert resolve_symbols(wl) == ["RELIANCE"]


class TestResolveRecentActive:
    def test_includes_only_within_window(self, owner):
        t = owner.memberships.first().tenant
        _signal(t, symbol="RELIANCE", source="TRADINGVIEW", hours_ago=2)
        _signal(t, symbol="TCS",      source="TRADINGVIEW", hours_ago=48)    # outside 24h

        wl = Watchlist.objects.create(
            tenant=t, owner=owner, name="active",
            kind=Watchlist.Kind.RECENT_ACTIVE,
            config={"window_hours": 24},
        )
        assert resolve_symbols(wl) == ["RELIANCE"]


class TestResolveTradedRecently:
    def test_includes_realised_states_only(self, owner, paper_portfolio):
        t = owner.memberships.first().tenant
        for sym, status in [
            ("FILLED_SYM",   Trade.Status.FILLED),
            ("PARTIAL_SYM",  Trade.Status.PARTIAL),
            ("CLOSED_SYM",   Trade.Status.CLOSED),
            ("PLAN_SYM",     Trade.Status.PLAN),       # excluded
            ("REJECTED_SYM", Trade.Status.REJECTED),   # excluded
        ]:
            Trade.objects.create(
                tenant=t, portfolio=paper_portfolio,
                symbol=sym, exchange="NSE", side="BUY", product="INTRADAY",
                entry_price=Decimal("100"), stop_loss=Decimal("95"),
                target=Decimal("110"), quantity=1,
                status=status, trade_date=timezone.now().date(),
            )

        wl = Watchlist.objects.create(
            tenant=t, owner=owner, name="recent",
            kind=Watchlist.Kind.TRADED_RECENTLY,
            config={"window_days": 30},
        )
        out = set(resolve_symbols(wl))
        assert out == {"FILLED_SYM", "PARTIAL_SYM", "CLOSED_SYM"}


class TestResolveShortlistToday:
    def test_filters_by_outcome_and_date(self, owner):
        t = owner.memberships.first().tenant
        today = timezone.now().date()
        for sym, outcome, score in [
            ("HOT",    WatchlistEntry.Outcome.WATCHING,  90),
            ("WARM",   WatchlistEntry.Outcome.TRIGGERED, 80),
            ("DONE",   WatchlistEntry.Outcome.TRADED,    70),
            ("PASS",   WatchlistEntry.Outcome.SKIPPED,   60),  # excluded by default
            ("MISS",   WatchlistEntry.Outcome.NO_SIGNAL, 50),  # excluded
        ]:
            WatchlistEntry.objects.create(
                tenant=t, symbol=sym, scan_date=today,
                score=score, outcome=outcome,
            )

        wl = Watchlist.objects.create(
            tenant=t, owner=owner, name="today",
            kind=Watchlist.Kind.SHORTLIST_TODAY,
        )
        out = resolve_symbols(wl)
        # Sorted by score desc.
        assert out == ["HOT", "WARM", "DONE"]


# ── API contract ────────────────────────────────────────────────────────

class TestAutoKindCreate:
    def test_create_with_auto_kind_seeds_symbols(self, auth_client, owner):
        t = owner.memberships.first().tenant
        for _ in range(3): _signal(t, symbol="RELIANCE", source="TRADINGVIEW")

        resp = auth_client.post(
            "/api/v1/watchlists/",
            {
                "name": "Top TV",
                "kind": "SOURCE_HOT",
                "config": {"source": "TRADINGVIEW", "top_n": 10},
            },
            format="json",
        )
        assert resp.status_code == 201, resp.content
        data = resp.json()
        assert data["kind"] == "SOURCE_HOT"
        assert data["is_auto"] is True
        assert data["symbols"] == ["RELIANCE"]   # seeded inline
        assert data["symbols_refreshed_at"] is not None

    def test_source_hot_without_source_400s(self, auth_client):
        resp = auth_client.post(
            "/api/v1/watchlists/",
            {"name": "missing src", "kind": "SOURCE_HOT", "config": {}},
            format="json",
        )
        assert resp.status_code == 400
        assert "source" in str(resp.json()).lower()

    def test_create_with_manual_kind_keeps_typed_symbols(self, auth_client):
        resp = auth_client.post(
            "/api/v1/watchlists/",
            {"name": "Manual", "kind": "MANUAL", "symbols": ["reliance", "TCS"]},
            format="json",
        )
        assert resp.status_code == 201
        assert resp.json()["symbols"] == ["RELIANCE", "TCS"]
        assert resp.json()["is_auto"] is False


class TestRefreshAction:
    def test_refresh_re_resolves(self, auth_client, owner):
        t = owner.memberships.first().tenant
        wl = Watchlist.objects.create(
            tenant=t, owner=owner, name="x",
            kind=Watchlist.Kind.SIGNAL_RANK,
            config={"window_days": 7},
        )
        # No signals yet → first resolve = [].
        refresh_watchlist(wl)
        wl.refresh_from_db()
        assert wl.symbols == []

        _signal(t, symbol="TCS", source="SCREENER")
        resp = auth_client.post(
            f"/api/v1/watchlists/{wl.id}/refresh/",
        )
        assert resp.status_code == 200
        assert resp.json()["symbols"] == ["TCS"]

    def test_refresh_on_manual_400s(self, auth_client, owner):
        wl = Watchlist.objects.create(
            tenant=owner.memberships.first().tenant, owner=owner,
            name="manual", kind=Watchlist.Kind.MANUAL,
            symbols=["RELIANCE"],
        )
        resp = auth_client.post(
            f"/api/v1/watchlists/{wl.id}/refresh/",
        )
        assert resp.status_code == 400


class TestSymbolEditGuard:
    def test_add_symbols_blocked_on_auto_kind(self, auth_client, owner):
        wl = Watchlist.objects.create(
            tenant=owner.memberships.first().tenant, owner=owner,
            name="auto", kind=Watchlist.Kind.RECENT_ACTIVE,
        )
        resp = auth_client.post(
            f"/api/v1/watchlists/{wl.id}/add-symbols/",
            {"symbols": ["RELIANCE"]}, format="json",
        )
        assert resp.status_code == 400


# ── Periodic task ───────────────────────────────────────────────────────

class TestRefreshTask:
    def test_refresh_task_skips_manual_and_updates_auto(self, owner):
        t = owner.memberships.first().tenant
        manual = Watchlist.objects.create(
            tenant=t, owner=owner, name="m",
            kind=Watchlist.Kind.MANUAL,
            symbols=["UNTOUCHED"],
        )
        auto = Watchlist.objects.create(
            tenant=t, owner=owner, name="a",
            kind=Watchlist.Kind.SIGNAL_RANK,
        )
        _signal(t, symbol="RELIANCE", source="TRADINGVIEW")

        summary = refresh_auto_watchlists()
        manual.refresh_from_db()
        auto.refresh_from_db()

        assert summary["refreshed"] >= 1
        assert manual.symbols == ["UNTOUCHED"]
        assert auto.symbols == ["RELIANCE"]
        assert auto.symbols_refreshed_at is not None
