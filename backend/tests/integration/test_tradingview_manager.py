"""Tests for the TradingView Manager surface — watchlist CRUD + grouped
signals aggregation."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from django.utils import timezone

from apps.notifications.models import Watchlist
from apps.strategies.models import Signal


pytestmark = pytest.mark.django_db


# ── Watchlists ──────────────────────────────────────────────────────────

class TestWatchlistCRUD:
    def test_create_normalises_symbols(self, auth_client):
        resp = auth_client.post(
            "/api/v1/watchlists/",
            {
                "name": "NIFTY top picks",
                "symbols": ["reliance", "  TCS ", "RELIANCE", "hdfcbank"],
            },
            format="json",
        )
        assert resp.status_code == 201, resp.content
        data = resp.json()
        # Uppercased, trimmed, deduped, order preserved
        assert data["symbols"] == ["RELIANCE", "TCS", "HDFCBANK"]
        assert data["symbol_count"] == 3

    def test_name_unique_per_owner(self, auth_client):
        url = "/api/v1/watchlists/"
        r1 = auth_client.post(url, {"name": "dup", "symbols": []}, format="json")
        assert r1.status_code == 201
        r2 = auth_client.post(url, {"name": "dup", "symbols": []}, format="json")
        assert r2.status_code == 400

    def test_add_and_remove_symbols(self, auth_client, owner):
        wl = Watchlist.objects.create(
            tenant=owner.memberships.first().tenant,
            owner=owner,
            name="bag",
            symbols=["RELIANCE"],
        )
        # add
        resp = auth_client.post(
            f"/api/v1/watchlists/{wl.id}/add-symbols/",
            {"symbols": ["TCS", "infy"]},
            format="json",
        )
        assert resp.status_code == 200
        assert resp.json()["symbols"] == ["RELIANCE", "TCS", "INFY"]

        # remove
        resp = auth_client.post(
            f"/api/v1/watchlists/{wl.id}/remove-symbols/",
            {"symbols": ["reliance"]},
            format="json",
        )
        assert resp.status_code == 200
        assert resp.json()["symbols"] == ["TCS", "INFY"]

    def test_url_pattern_does_not_collide_with_link_detail(self, auth_client):
        """`tradingview/<uuid>/` must NOT match `tradingview/watchlists/`."""
        resp = auth_client.get("/api/v1/watchlists/")
        assert resp.status_code == 200
        # If routing collided, this would 404 with "no link with pk=watchlists".


# ── Grouped signals ────────────────────────────────────────────────────

class TestGroupedSignals:
    @pytest.fixture
    def seeded_signals(self, owner):
        tenant = owner.memberships.first().tenant
        now = timezone.now()
        seed = [
            ("RELIANCE", "BUY",  "vwap",   "TRADINGVIEW", now - timedelta(hours=1)),
            ("RELIANCE", "BUY",  "vwap",   "TRADINGVIEW", now - timedelta(hours=2)),
            ("RELIANCE", "SELL", "vwap",   "TRADINGVIEW", now - timedelta(hours=3)),
            ("TCS",      "BUY",  "breakout","TRADINGVIEW", now - timedelta(hours=4)),
            ("TCS",      "SELL", "breakout","SCREENER",    now - timedelta(hours=5)),
            ("HDFCBANK", "BUY",  "ok",      "OK_SCANNER",  now - timedelta(days=10)),  # OUT of 7d window
        ]
        for sym, side, strat, src, ts in seed:
            Signal.objects.create(
                tenant=tenant,
                symbol=sym,
                signal_date=ts.date(),
                signal_time=ts,
                source=src,
                strategy=strat,
                side=side,
                entry_price=100.0,
                stoploss=0.0,
                target=0.0,
            )
        return tenant

    def test_default_groups_by_symbol_and_excludes_old(
        self, auth_client, seeded_signals,
    ):
        resp = auth_client.get("/api/v1/notifications/tradingview/signals/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["by"] == "symbol"

        rows = {r["key"]: r for r in body["rows"]}
        # HDFCBANK was 10 days old — outside the 7-day default window.
        assert "HDFCBANK" not in rows

        assert rows["RELIANCE"]["count"] == 3
        assert rows["RELIANCE"]["buys"] == 2
        assert rows["RELIANCE"]["sells"] == 1
        # Sorted by count desc → RELIANCE before TCS.
        assert body["rows"][0]["key"] == "RELIANCE"

    def test_filter_by_source(self, auth_client, seeded_signals):
        resp = auth_client.get(
            "/api/v1/notifications/tradingview/signals/?source=TRADINGVIEW",
        )
        assert resp.status_code == 200
        keys = {r["key"] for r in resp.json()["rows"]}
        # TCS appeared once via TRADINGVIEW and once via SCREENER. With the
        # source filter, only the TV row counts.
        assert "RELIANCE" in keys
        assert "TCS" in keys
        # The HDFCBANK row was OK_SCANNER — excluded.
        assert "HDFCBANK" not in keys

    def test_group_by_strategy(self, auth_client, seeded_signals):
        resp = auth_client.get(
            "/api/v1/notifications/tradingview/signals/?by=strategy",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["by"] == "strategy"
        keys = {r["key"] for r in body["rows"]}
        assert "vwap" in keys
        assert "breakout" in keys

    def test_group_by_day_returns_iso_dates(self, auth_client, seeded_signals):
        resp = auth_client.get(
            "/api/v1/notifications/tradingview/signals/?by=day",
        )
        assert resp.status_code == 200
        for row in resp.json()["rows"]:
            # ISO date strings: 2026-05-19
            datetime.fromisoformat(row["key"])

    def test_filter_by_watchlist(self, auth_client, owner, seeded_signals):
        wl = Watchlist.objects.create(
            tenant=owner.memberships.first().tenant,
            owner=owner,
            name="just RELIANCE",
            symbols=["RELIANCE"],
        )
        resp = auth_client.get(
            f"/api/v1/notifications/tradingview/signals/?watchlist={wl.id}",
        )
        assert resp.status_code == 200
        keys = {r["key"] for r in resp.json()["rows"]}
        assert keys == {"RELIANCE"}

    def test_invalid_by_returns_400(self, auth_client):
        resp = auth_client.get("/api/v1/notifications/tradingview/signals/?by=bogus")
        assert resp.status_code == 400

    def test_empty_returns_empty_rows(self, auth_client):
        resp = auth_client.get("/api/v1/notifications/tradingview/signals/")
        assert resp.status_code == 200
        assert resp.json()["rows"] == []


# ── Grouped-signals drill-in ────────────────────────────────────────────

class TestGroupedSignalsDetail:
    """The detail endpoint backs the row-click drill-in sheet. The sheet
    passes the *same* by/key/days/source filters it's grouped under, so the
    raw rows it shows must match the bucket the operator clicked — across
    every group-by dimension, not just symbol."""

    DETAIL = "/api/v1/notifications/tradingview/signals/detail/"

    @pytest.fixture
    def seeded(self, owner):
        tenant = owner.memberships.first().tenant
        now = timezone.now()
        seed = [
            ("RELIANCE", "BUY",  "vwap",     "TRADINGVIEW", now - timedelta(hours=1)),
            ("RELIANCE", "SELL", "vwap",     "TRADINGVIEW", now - timedelta(hours=2)),
            ("TCS",      "BUY",  "breakout", "SCREENER",    now - timedelta(hours=3)),
        ]
        for sym, side, strat, src, ts in seed:
            Signal.objects.create(
                tenant=tenant, symbol=sym, signal_date=ts.date(), signal_time=ts,
                source=src, strategy=strat, side=side,
                entry_price=100.0, stoploss=0.0, target=0.0,
            )
        return tenant

    def test_by_symbol(self, auth_client, seeded):
        resp = auth_client.get(f"{self.DETAIL}?by=symbol&key=RELIANCE")
        assert resp.status_code == 200, resp.content
        body = resp.json()
        assert body["by"] == "symbol"
        assert body["key"] == "RELIANCE"
        assert {r["symbol"] for r in body["rows"]} == {"RELIANCE"}
        assert len(body["rows"]) == 2

    def test_by_strategy(self, auth_client, seeded):
        resp = auth_client.get(f"{self.DETAIL}?by=strategy&key=vwap")
        assert resp.status_code == 200
        rows = resp.json()["rows"]
        assert all(r["strategy"] == "vwap" for r in rows)
        assert len(rows) == 2

    def test_by_source(self, auth_client, seeded):
        resp = auth_client.get(f"{self.DETAIL}?by=source&key=SCREENER")
        assert resp.status_code == 200
        rows = resp.json()["rows"]
        assert {r["symbol"] for r in rows} == {"TCS"}

    def test_by_day(self, auth_client, seeded):
        key = timezone.now().date().isoformat()
        resp = auth_client.get(f"{self.DETAIL}?by=day&key={key}")
        assert resp.status_code == 200
        # All three seeded signals fired today.
        assert len(resp.json()["rows"]) == 3

    def test_source_filter_composes(self, auth_client, seeded):
        # Drill into the RELIANCE symbol bucket but with a source filter that
        # excludes it — should come back empty, mirroring the feed.
        resp = auth_client.get(f"{self.DETAIL}?by=symbol&key=RELIANCE&source=SCREENER")
        assert resp.status_code == 200
        assert resp.json()["rows"] == []

    def test_missing_key_returns_400(self, auth_client, seeded):
        resp = auth_client.get(f"{self.DETAIL}?by=symbol")
        assert resp.status_code == 400

    def test_invalid_by_returns_400(self, auth_client, seeded):
        resp = auth_client.get(f"{self.DETAIL}?by=bogus&key=RELIANCE")
        assert resp.status_code == 400

    def test_malformed_day_key_returns_400(self, auth_client, seeded):
        resp = auth_client.get(f"{self.DETAIL}?by=day&key=not-a-date")
        assert resp.status_code == 400
