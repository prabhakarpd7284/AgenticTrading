"""Zerodha Kite Connect adapter.

Kite uses a request-token → access-token OAuth flow. The access_token has
a daily TTL (until ~6 AM IST next day) — operators must re-login each
trading day. The adapter expects an already-exchanged access_token in
credentials; the /brokers/{id}/connect/ endpoint is responsible for
running the request-token exchange and persisting the result.

Credentials dict shape:
  {
    "api_key": "...",        # Kite app's API key
    "api_secret": "...",     # used for refresh + checksum
    "access_token": "...",   # daily token from login flow
  }
Meta dict carries non-secret bits like ``user_id`` (Zerodha client id).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from apps.market_data.adapters.base import (
    BrokerAdapterBase, BrokerHealth, Holding, Margin, Position,
)

logger = logging.getLogger(__name__)


class ZerodhaAdapter(BrokerAdapterBase):
    name = "zerodha"

    def __init__(self, credentials: dict[str, Any], meta: dict[str, Any] | None = None):
        super().__init__(credentials, meta)
        self._kite = None

    # ── Auth ──────────────────────────────────────────────────────────
    def authenticate(self) -> bool:
        """Verify the daily access_token by calling profile().

        The constructor doesn't actually contact Kite — only an API call
        does. We hit profile() because it's the cheapest authenticated
        endpoint, so an expired token surfaces here instead of being
        masked by an empty positions() response downstream.
        """
        try:
            from kiteconnect import KiteConnect  # type: ignore
        except ImportError:
            logger.error("zerodha.auth.sdk_missing: pip install kiteconnect")
            return False
        api_key = self.credentials.get("api_key")
        access_token = self.credentials.get("access_token")
        if not api_key or not access_token:
            logger.error("zerodha.auth.missing_creds keys=%s", list(self.credentials))
            return False
        try:
            self._kite = KiteConnect(api_key=api_key)
            self._kite.set_access_token(access_token)
            profile = self._kite.profile()
        except Exception as e:
            # Includes kiteconnect.exceptions.TokenException for expired tokens.
            self._kite = None
            logger.warning("zerodha.auth.token_check_failed: %s", e)
            return False
        return bool(isinstance(profile, dict) and profile.get("user_id"))

    # ── Health ────────────────────────────────────────────────────────
    def health_check(self) -> BrokerHealth:
        t0 = time.time()
        if self._kite is None and not self.authenticate():
            return BrokerHealth(ok=False, detail="auth failed")
        try:
            profile = self._kite.profile()  # type: ignore[union-attr]
            return BrokerHealth(
                ok=bool(profile.get("user_id")),
                detail=str(profile.get("user_name", "")),
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return BrokerHealth(ok=False, detail=str(e)[:200])

    # ── Read API ──────────────────────────────────────────────────────
    def fetch_positions(self) -> list[Position]:
        if self._kite is None and not self.authenticate():
            raise RuntimeError("Zerodha authentication failed — daily re-login required")
        r = self._kite.positions()  # type: ignore[union-attr]
        # Kite returns {"net": [...], "day": [...]} — `net` is the open book.
        rows = (r or {}).get("net") or []
        if not isinstance(rows, list):
            return []
        return [self._to_position(row) for row in rows if isinstance(row, dict)]

    def fetch_holdings(self) -> list[Holding]:
        if self._kite is None and not self.authenticate():
            raise RuntimeError("Zerodha authentication failed — daily re-login required")
        r = self._kite.holdings()  # type: ignore[union-attr]
        if not isinstance(r, list):
            return []
        return [self._to_holding(row) for row in r if isinstance(row, dict)]

    def fetch_margin(self) -> Margin:
        if self._kite is None and not self.authenticate():
            raise RuntimeError("Zerodha authentication failed — daily re-login required")
        r = self._kite.margins()  # type: ignore[union-attr]
        # Kite returns {"equity": {...}, "commodity": {...}} — sum equity bucket.
        equity = (r or {}).get("equity") or {}
        if not isinstance(equity, dict):
            return Margin(available_cash=0, used=0, total=0)
        avail = _f((equity.get("available") or {}).get("cash"))
        used = _f((equity.get("utilised") or {}).get("debits"))
        return Margin(available_cash=avail, used=used, total=avail + used, raw=r or {})

    # ── Normalisers ───────────────────────────────────────────────────
    @staticmethod
    def _to_position(row: dict) -> Position:
        return Position(
            symbol=row.get("tradingsymbol") or "",
            exchange=row.get("exchange") or "NSE",
            product=row.get("product") or "MIS",
            quantity=_i(row.get("quantity")),
            avg_price=_f(row.get("average_price")),
            last_price=_f(row.get("last_price")),
            pnl=_f(row.get("pnl")),
            mtm=_f(row.get("m2m") or row.get("unrealised")),
            instrument_token=str(row.get("instrument_token") or ""),
            raw=row,
        )

    @staticmethod
    def _to_holding(row: dict) -> Holding:
        return Holding(
            symbol=row.get("tradingsymbol") or "",
            exchange=row.get("exchange") or "NSE",
            quantity=_i(row.get("quantity")),
            avg_price=_f(row.get("average_price")),
            last_price=_f(row.get("last_price")),
            pnl=_f(row.get("pnl")),
            instrument_token=str(row.get("instrument_token") or ""),
            raw=row,
        )


def _f(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _i(v) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return 0
