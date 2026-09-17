"""Angel One SmartAPI adapter — multi-account capable.

Unlike the legacy ``trading.services.data_service.BrokerClient`` singleton
(which auths from process-level .env), this adapter takes per-account
credentials at construction time. That lets a single tenant link multiple
Angel One accounts.

Credentials dict shape:
  {
    "api_key": "...",
    "client_code": "...",          # SmartAPI username
    "password": "...",
    "totp_secret": "...",          # base32 secret for pyotp
  }

Rate limiting:
  Every SmartAPI call routes through ``BrokerClient.get_instance().throttle()``
  — the process-wide universal rate-limit manager. Multiple linked Angel
  accounts therefore share one 0.4s queue, so they can't race past Angel's
  per-API-key rate limit ("Access denied because of exceeding access rate").
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

from apps.market_data.adapters.base import (
    BrokerAdapterBase, BrokerHealth, Holding, Margin, Position,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BrokerRateLimited(Exception):
    """Raised by _throttled while the rate-limit breaker is open (no API hit)."""


def _throttled(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a SmartAPI call after taking a slot on the universal throttle.

    Defined as a free function (not a method) so importing the legacy
    BrokerClient is lazy — strategies / tests that never link an Angel
    account don't pay the import cost. Falls back to a no-op if the
    legacy module isn't available (e.g. in tests that mock the broker).

    Guarded by the Redis-coordinated rate-limit breaker living on
    BrokerClient (shared across ALL ForkPool workers + the legacy candle
    path). While the breaker is open the call fast-fails with
    BrokerRateLimited instead of touching the API, so an account-level
    cooldown can actually elapse. A rate-limit / connection-timeout error
    trips the breaker; any clean call resets it.
    """
    bc = None
    try:
        from trading.services.data_service import (  # type: ignore
            BrokerClient, _is_transient_broker_error,
        )
        bc = BrokerClient.get_instance()
    except Exception as e:  # pragma: no cover — defensive only
        logger.debug("angel.throttle.unavailable: %s", e)
        return fn(*args, **kwargs)

    # Half-open gate: deny while OPEN, allow a single probe once the cooldown
    # has elapsed. Only the probe call resets the breaker on success — a normal
    # CLOSED call doesn't touch Redis, and only one probe runs at a time (no
    # re-storm), so legitimate calls aren't starved for a fixed window.
    gate = bc.breaker_gate()
    if gate == "open":
        raise BrokerRateLimited("Angel rate-limit breaker open — backing off")
    bc.throttle()

    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        if _is_transient_broker_error(e):
            bc.trip_breaker()
        raise
    if gate == "probe":
        bc.reset_breaker()
    return result


class AngelOneAdapter(BrokerAdapterBase):
    name = "angel_one"

    def __init__(self, credentials: dict[str, Any], meta: dict[str, Any] | None = None):
        super().__init__(credentials, meta)
        self._api = None
        self._session_expires = 0.0

    # ── Auth ──────────────────────────────────────────────────────────
    def authenticate(self) -> bool:
        from SmartApi import SmartConnect  # type: ignore
        import pyotp

        api_key = self.credentials.get("api_key")
        client_code = self.credentials.get("client_code") or self.credentials.get("username")
        password = self.credentials.get("password")
        totp_secret = self.credentials.get("totp_secret")
        if not all([api_key, client_code, password, totp_secret]):
            logger.error("angel_one.auth.missing_creds keys=%s", list(self.credentials))
            return False
        try:
            self._api = SmartConnect(api_key=api_key)
            otp = pyotp.TOTP(totp_secret).now()
            data = _throttled(self._api.generateSession, client_code, password, otp)
            if not isinstance(data, dict) or not data.get("data"):
                _msg = data.get("message") if isinstance(data, dict) else None
                _code = data.get("errorcode") if isinstance(data, dict) else None
                logger.error("angel_one.auth.failed code=%s msg=%s", _code, _msg)
                return False
            self._session_expires = time.time() + 22 * 3600
            return True
        except BrokerRateLimited as e:
            # Breaker open — concise one-liner, no traceback spam per refresh.
            logger.warning("angel_one.auth.rate_limited: %s", e)
            return False
        except Exception as e:
            logger.exception("angel_one.auth.exception: %s", e)
            return False

    def _ensure(self) -> bool:
        if self._api is None or time.time() > self._session_expires:
            return self.authenticate()
        return True

    # ── Health ────────────────────────────────────────────────────────
    def health_check(self) -> BrokerHealth:
        t0 = time.time()
        if not self._ensure():
            return BrokerHealth(ok=False, detail="auth failed")
        try:
            _throttled(self._api.getProfile, self._api.refresh_token)  # type: ignore[attr-defined]
            return BrokerHealth(ok=True, latency_ms=(time.time() - t0) * 1000)
        except Exception as e:
            return BrokerHealth(ok=False, detail=str(e)[:200])

    # ── Read API ──────────────────────────────────────────────────────
    def fetch_positions(self) -> list[Position]:
        if not self._ensure():
            return []
        try:
            r = _throttled(self._api.position)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("angel_one.positions.failed: %s", e)
            return []
        rows = (r or {}).get("data") or []
        if not isinstance(rows, list):
            return []
        return [self._to_position(row) for row in rows if isinstance(row, dict)]

    def fetch_holdings(self) -> list[Holding]:
        if not self._ensure():
            return []
        try:
            r = _throttled(self._api.holding)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("angel_one.holdings.failed: %s", e)
            return []
        rows = (r or {}).get("data") or []
        if not isinstance(rows, list):
            return []
        return [self._to_holding(row) for row in rows if isinstance(row, dict)]

    def fetch_margin(self) -> Margin:
        if not self._ensure():
            return Margin(available_cash=0, used=0, total=0)
        try:
            r = _throttled(self._api.rmsLimit)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("angel_one.margin.failed: %s", e)
            return Margin(available_cash=0, used=0, total=0)
        d = (r or {}).get("data") or {}
        if not isinstance(d, dict):
            return Margin(available_cash=0, used=0, total=0)
        avail = _f(d.get("availablecash"))
        used = _f(d.get("utiliseddebits") or d.get("utilisedexposure"))
        return Margin(available_cash=avail, used=used, total=avail + used, raw=d)

    # ── Options chain — uses getMarketData(FULL) + getOptionGreek ─────
    def options_chain(
        self,
        underlying: str,
        expiry=None,
        strikes_window: int = 20,
    ):
        """Pull options chain from Angel via:

          1. Scrip master  — find option tokens for (underlying, expiry).
                              Cached by `_scrip_master_service`.
          2. getMarketData — FULL mode for ±N strikes around ATM.
                              Returns LTP + depth + OI for each leg.
          3. getOptionGreek — single call returns delta/gamma/theta/vega/IV
                              for ALL strikes of (underlying, expiry).
                              Cheap — only one extra request per chain fetch.

        Both calls are merged into the broker-neutral OptionsChainSnapshot.
        """
        from apps.market_data.adapters.base import (
            OptionQuote, OptionsChainRow, OptionsChainSnapshot,
        )
        from apps.market_data.services.scrip_master import (
            get_underlying_token, list_option_strikes, nearest_expiry,
        )
        from datetime import datetime, timezone

        if not self._ensure():
            return None

        u = underlying.upper()
        # 1) figure out expiry — if none provided, pick the nearest
        expiry = expiry or nearest_expiry(u)
        if not expiry:
            logger.warning("angel.chain.no_expiry underlying=%s", u)
            return None

        # 2) spot LTP via getMarketData LTP mode
        spot_token = get_underlying_token(u)
        if not spot_token:
            return None
        try:
            r = _throttled(
                self._api.getMarketData,  # type: ignore[union-attr]
                mode="LTP", exchangeTokens={"NSE": [str(spot_token)]},
            )
            fetched = ((r or {}).get("data") or {}).get("fetched") or []
            spot = _f(fetched[0].get("ltp")) if fetched else 0.0
        except Exception as e:
            logger.error("angel.chain.spot_failed: %s", e)
            return None
        if spot <= 0:
            return None

        # 3) ATM + strikes window from scrip master
        all_strikes = list_option_strikes(u, expiry)
        if not all_strikes:
            return None
        atm = min(all_strikes, key=lambda s: abs(s["strike"] - spot))["strike"]
        # take ± strikes_window steps around ATM (regardless of step size)
        strikes_sorted = sorted({s["strike"] for s in all_strikes})
        atm_idx = strikes_sorted.index(atm) if atm in strikes_sorted else 0
        lo = max(0, atm_idx - strikes_window)
        hi = min(len(strikes_sorted), atm_idx + strikes_window + 1)
        window = set(strikes_sorted[lo:hi])
        tokens_by_pair: dict[tuple[int, str], dict] = {}
        for s in all_strikes:
            if s["strike"] not in window:
                continue
            tokens_by_pair[(s["strike"], s["opt"])] = s

        # 4) FULL-mode quotes in chunks of 50 (Angel API limit)
        nfo_tokens = [str(s["token"]) for s in tokens_by_pair.values()
                       if s.get("exch_seg") in ("NFO", "BFO")]
        quotes_by_token: dict[str, dict] = {}
        for i in range(0, len(nfo_tokens), 50):
            chunk = nfo_tokens[i:i + 50]
            try:
                qr = _throttled(
                    self._api.getMarketData,  # type: ignore[union-attr]
                    mode="FULL", exchangeTokens={"NFO": chunk},
                )
                for row in ((qr or {}).get("data") or {}).get("fetched") or []:
                    quotes_by_token[str(row.get("symbolToken"))] = row
            except Exception as e:
                logger.warning("angel.chain.quote_chunk_failed i=%d: %s", i, e)

        # 5) Greeks — one call for all strikes of (underlying, expiry).
        # Angel wants expiry as DDMMMYYYY uppercase (e.g. "28APR2026") which
        # is the same format we already use; pass through unchanged.
        greeks_by_pair: dict[tuple[int, str], dict] = {}
        try:
            gr = _throttled(
                self._api.optionGreek,  # type: ignore[union-attr]
                {"name": u, "expirydate": expiry},
            )
            for row in (gr or {}).get("data") or []:
                k = _i(row.get("strikePrice"))
                opt = (row.get("optionType") or "").upper()
                if k > 0 and opt in ("CE", "PE"):
                    greeks_by_pair[(k, opt)] = row
        except Exception as e:
            logger.info("angel.chain.greeks_missing: %s", e)

        # 6) Build the snapshot
        rows: list[OptionsChainRow] = []
        for strike in sorted(window):
            row = OptionsChainRow(strike=strike)
            for opt in ("CE", "PE"):
                meta = tokens_by_pair.get((strike, opt))
                if not meta:
                    continue
                q = quotes_by_token.get(str(meta["token"])) or {}
                depth = q.get("depth") or {}
                bid_d = (depth.get("buy") or [{}])[0]
                ask_d = (depth.get("sell") or [{}])[0]
                g = greeks_by_pair.get((strike, opt)) or {}
                quote = OptionQuote(
                    token=str(meta["token"]),
                    symbol=str(meta.get("symbol", "")),
                    strike=strike, opt=opt,
                    ltp=_f(q.get("ltp")),
                    bid=_f(bid_d.get("price")),
                    ask=_f(ask_d.get("price")),
                    bid_qty=_i(bid_d.get("quantity")),
                    ask_qty=_i(ask_d.get("quantity")),
                    volume=_i(q.get("tradeVolume") or q.get("volTraded")),
                    oi=_i(q.get("opnInterest")),
                    oi_change=_i(q.get("netChangeOpnInterest")),
                    iv=_f(g.get("impliedVolatility")) / 100.0 if g else 0.0,
                    delta=_f(g.get("delta")),
                    gamma=_f(g.get("gamma")),
                    theta=_f(g.get("theta")),
                    vega=_f(g.get("vega")),
                    raw=q,
                )
                if opt == "CE":
                    row.ce = quote
                else:
                    row.pe = quote
            if row.ce or row.pe:
                rows.append(row)

        # 7) VIX (best effort) + PCR (computable from chain)
        vix = None
        try:
            vix_tok = get_underlying_token("INDIAVIX")
            if vix_tok:
                vr = _throttled(
                    self._api.getMarketData,  # type: ignore[union-attr]
                    mode="LTP", exchangeTokens={"NSE": [str(vix_tok)]},
                )
                vfetched = ((vr or {}).get("data") or {}).get("fetched") or []
                if vfetched:
                    vix = _f(vfetched[0].get("ltp"))
        except Exception:
            pass

        ce_oi = sum(r.ce.oi for r in rows if r.ce)
        pe_oi = sum(r.pe.oi for r in rows if r.pe)
        pcr_oi = (pe_oi / ce_oi) if ce_oi else None

        return OptionsChainSnapshot(
            underlying=u, spot=spot, expiry=expiry,
            fetched_at=datetime.now(timezone.utc),
            rows=rows, source="angel_one",
            vix=vix, pcr_oi=pcr_oi, atm_strike=atm,
        )

    # ── Normalisers ───────────────────────────────────────────────────
    @staticmethod
    def _to_position(row: dict) -> Position:
        return Position(
            symbol=row.get("tradingsymbol") or row.get("symbolname") or "",
            exchange=row.get("exchange") or "NSE",
            product=row.get("producttype") or "INTRADAY",
            quantity=_i(row.get("netqty")),
            avg_price=_f(row.get("avgnetprice") or row.get("netvalue")),
            last_price=_f(row.get("ltp")),
            pnl=_f(row.get("pnl") or row.get("realised")),
            mtm=_f(row.get("unrealised")),
            instrument_token=str(row.get("symboltoken") or ""),
            raw=row,
        )

    @staticmethod
    def _to_holding(row: dict) -> Holding:
        return Holding(
            symbol=row.get("tradingsymbol") or "",
            exchange=row.get("exchange") or "NSE",
            quantity=_i(row.get("quantity")),
            avg_price=_f(row.get("averageprice")),
            last_price=_f(row.get("ltp")),
            pnl=_f(row.get("profitandloss")),
            instrument_token=str(row.get("symboltoken") or ""),
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
