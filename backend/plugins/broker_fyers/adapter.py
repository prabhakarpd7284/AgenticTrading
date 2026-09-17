"""Fyers v3 adapter — multi-account capable.

Fyers uses an OAuth2 flow (login URL → auth_code → access_token). The
exchanged access_token is what we persist; refresh-tokens are not part of
the public v3 API at time of writing, so operators re-login daily.

Credentials dict shape:
  {
    "app_id": "...",          # e.g. "XYZA-100"
    "secret_key": "...",
    "access_token": "...",    # exchanged via /generate-authcode
  }
Meta carries non-secret bits like ``client_id`` (Fyers user id).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from apps.market_data.adapters.base import (
    BrokerAdapterBase, BrokerHealth, Holding, Margin, Position,
)

logger = logging.getLogger(__name__)


class FyersAdapter(BrokerAdapterBase):
    name = "fyers"

    def __init__(self, credentials: dict[str, Any], meta: dict[str, Any] | None = None):
        super().__init__(credentials, meta)
        self._api = None

    # ── Auth ──────────────────────────────────────────────────────────
    def authenticate(self) -> bool:
        """Verify the daily access_token by calling get_profile().

        Constructor doesn't talk to Fyers — only an API call does. We
        probe get_profile() because Fyers returns ``{"s":"error",...}``
        for expired tokens, and we'd rather surface that here than have
        every downstream fetch silently return ``[]``.
        """
        try:
            from fyers_apiv3 import fyersModel  # type: ignore
        except ImportError:
            logger.error("fyers.auth.sdk_missing: pip install fyers-apiv3")
            return False
        app_id = self.credentials.get("app_id")
        access_token = self.credentials.get("access_token")
        if not app_id or not access_token:
            logger.error("fyers.auth.missing_creds keys=%s", list(self.credentials))
            return False
        try:
            self._api = fyersModel.FyersModel(
                client_id=app_id, token=access_token, log_path="",
            )
            profile = self._api.get_profile()
        except Exception as e:
            self._api = None
            logger.warning("fyers.auth.token_check_failed: %s", e)
            return False
        if not (isinstance(profile, dict) and profile.get("s") == "ok"):
            msg = (profile or {}).get("message") if isinstance(profile, dict) else ""
            logger.warning("fyers.auth.token_rejected msg=%s", msg)
            self._api = None
            return False
        return True

    # ── Health ────────────────────────────────────────────────────────
    def health_check(self) -> BrokerHealth:
        t0 = time.time()
        if self._api is None and not self.authenticate():
            return BrokerHealth(ok=False, detail="auth failed")
        try:
            profile = self._api.get_profile()  # type: ignore[union-attr]
            ok = isinstance(profile, dict) and profile.get("s") == "ok"
            return BrokerHealth(
                ok=ok,
                detail=str((profile or {}).get("data", {}).get("name", ""))[:200],
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return BrokerHealth(ok=False, detail=str(e)[:200])

    # ── Read API ──────────────────────────────────────────────────────
    def fetch_positions(self) -> list[Position]:
        if self._api is None and not self.authenticate():
            raise RuntimeError("Fyers authentication failed — daily re-login required")
        r = self._api.positions()  # type: ignore[union-attr]
        _raise_if_error(r, "positions")
        rows = ((r or {}).get("netPositions") or [])
        if not isinstance(rows, list):
            return []
        return [self._to_position(row) for row in rows if isinstance(row, dict)]

    def fetch_holdings(self) -> list[Holding]:
        if self._api is None and not self.authenticate():
            raise RuntimeError("Fyers authentication failed — daily re-login required")
        r = self._api.holdings()  # type: ignore[union-attr]
        _raise_if_error(r, "holdings")
        rows = ((r or {}).get("holdings") or [])
        if not isinstance(rows, list):
            return []
        return [self._to_holding(row) for row in rows if isinstance(row, dict)]

    def fetch_margin(self) -> Margin:
        if self._api is None and not self.authenticate():
            raise RuntimeError("Fyers authentication failed — daily re-login required")
        r = self._api.funds()  # type: ignore[union-attr]
        _raise_if_error(r, "funds")
        # Fyers `funds` returns {"s":"ok","fund_limit":[{title,equityAmount,...}, ...]}
        items = (r or {}).get("fund_limit") or []
        avail = used = total = 0.0
        for it in items:
            if not isinstance(it, dict):
                continue
            title = (it.get("title") or "").lower()
            amt = _f(it.get("equityAmount"))
            if "available" in title or "clearbalance" in title:
                avail += amt
            elif "utilized" in title or "utilised" in title:
                used += amt
            elif "total" in title:
                total += amt
        if total == 0:
            total = avail + used
        return Margin(available_cash=avail, used=used, total=total, raw=r or {})

    # ── Historical seconds candles (scalp strategy) ──────────────────
    def history(self, symbol: str, resolution: str = "5S", range_from: str = "",
                range_to: str = "", cont_flag: int = 1) -> list[list]:
        """Seconds/minute historical candles ``[[epoch,o,h,l,c,v], …]``.

        Used by the scalp strategy for intra-candle pressure (Angel can't go
        below 1-minute). Delegates to :mod:`plugins.broker_fyers.history`.
        """
        if self._api is None and not self.authenticate():
            raise RuntimeError("Fyers authentication failed — daily re-login required")
        from .history import fetch_history
        return fetch_history(self._api, symbol, resolution, range_from, range_to, cont_flag)

    # ── Options chain — Fyers serves the whole chain in one call ──────
    def options_chain(
        self,
        underlying: str,
        expiry=None,
        strikes_window: int = 20,
    ):
        """Fyers's `optionchain` endpoint is the cleanest of the three brokers
        — one HTTP call returns the spot LTP, the full strike grid, OI, IV,
        and greeks for the requested expiry. We just normalise it into the
        broker-neutral OptionsChainSnapshot.

        Fyers expects a tradingsymbol like `NSE:NIFTY50-INDEX`, with
        `strikecount` controlling the window (each side). `timestamp` is the
        expiry epoch in seconds — empty string means the nearest expiry.
        """
        from apps.market_data.adapters.base import (
            OptionQuote, OptionsChainRow, OptionsChainSnapshot,
        )
        from datetime import datetime, timezone

        if self._api is None and not self.authenticate():
            raise RuntimeError("Fyers authentication failed — daily re-login required")

        symbol_map = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "FINNIFTY": "NSE:FINNIFTY-INDEX",
            "SENSEX": "BSE:SENSEX-INDEX",
        }
        u = underlying.upper()
        fy_symbol = symbol_map.get(u)
        if not fy_symbol:
            logger.warning("fyers.chain.unknown_underlying=%s", u)
            return None

        # Fyers wants an epoch timestamp for expiry; "" → nearest
        timestamp_arg = ""
        if expiry:
            try:
                exp_dt = datetime.strptime(expiry, "%d%b%Y")
                # Fyers epoch for expiry is in seconds (IST midnight)
                timestamp_arg = str(int(exp_dt.timestamp()))
            except ValueError:
                logger.warning("fyers.chain.expiry_parse_failed=%s", expiry)

        try:
            r = self._api.optionchain(data={  # type: ignore[union-attr]
                "symbol": fy_symbol,
                "strikecount": strikes_window,
                "timestamp": timestamp_arg,
            })
            _raise_if_error(r, "optionchain")
        except Exception as e:
            logger.error("fyers.chain.failed: %s", e)
            return None

        data = (r or {}).get("data") or {}
        # Fyers response shape:
        #   data.indiavixData.ltp
        #   data.optionsChain  → list of legs each with {strike_price, option_type,
        #       ltp, bid, ask, ltpchp, oi, oich, volume, prev_oi}
        #   data.callOi / data.putOi (aggregate)
        #   data.indexLtp
        spot = _f(data.get("indexLtp") or data.get("ltp"))
        vix = _f((data.get("indiavixData") or {}).get("ltp")) or None
        chain_rows: dict[int, OptionsChainRow] = {}

        for row in data.get("optionsChain") or []:
            if not isinstance(row, dict):
                continue
            strike = _i(row.get("strike_price"))
            opt = (row.get("option_type") or "").upper()
            if strike <= 0 or opt not in ("CE", "PE"):
                continue
            quote = OptionQuote(
                token=str(row.get("fyToken") or row.get("symbol", "")),
                symbol=str(row.get("symbol") or ""),
                strike=strike, opt=opt,
                ltp=_f(row.get("ltp")),
                bid=_f(row.get("bid")),
                ask=_f(row.get("ask")),
                volume=_i(row.get("volume")),
                oi=_i(row.get("oi")),
                oi_change=_i(row.get("oich") or row.get("oi_change")),
                iv=_f(row.get("iv") or row.get("implied_volatility")) / 100.0,
                delta=_f(row.get("delta")),
                gamma=_f(row.get("gamma")),
                theta=_f(row.get("theta")),
                vega=_f(row.get("vega")),
                raw=row,
            )
            cr = chain_rows.setdefault(strike, OptionsChainRow(strike=strike))
            if opt == "CE":
                cr.ce = quote
            else:
                cr.pe = quote

        rows = [chain_rows[k] for k in sorted(chain_rows)]
        atm = None
        if rows and spot > 0:
            atm = min((r.strike for r in rows), key=lambda k: abs(k - spot))

        # Aggregate PCR — Fyers gives callOi / putOi directly when present
        call_oi = _f(data.get("callOi"))
        put_oi = _f(data.get("putOi"))
        pcr_oi = (put_oi / call_oi) if call_oi else None

        return OptionsChainSnapshot(
            underlying=u, spot=spot, expiry=expiry or "",
            fetched_at=datetime.now(timezone.utc),
            rows=rows, source="fyers",
            vix=vix, pcr_oi=pcr_oi, atm_strike=atm,
        )

    # ── Normalisers ───────────────────────────────────────────────────
    @staticmethod
    def _to_position(row: dict) -> Position:
        return Position(
            symbol=row.get("symbol") or "",
            exchange=str(row.get("exchange") or "NSE"),
            product=row.get("productType") or "INTRADAY",
            quantity=_i(row.get("netQty")),
            avg_price=_f(row.get("netAvg")),
            last_price=_f(row.get("ltp")),
            pnl=_f(row.get("pl")),
            mtm=_f(row.get("unrealized_profit")),
            instrument_token=str(row.get("fyToken") or ""),
            raw=row,
        )

    @staticmethod
    def _to_holding(row: dict) -> Holding:
        return Holding(
            symbol=row.get("symbol") or "",
            exchange=str(row.get("exchange") or "NSE"),
            quantity=_i(row.get("quantity")),
            avg_price=_f(row.get("costPrice")),
            last_price=_f(row.get("ltp")),
            pnl=_f(row.get("pl")),
            instrument_token=str(row.get("fyToken") or ""),
            raw=row,
        )


def _raise_if_error(resp, endpoint: str) -> None:
    """Fyers wraps every response with ``s=ok|error``. Surface errors so
    the snapshot layer marks the link errored instead of silently
    storing an empty payload."""
    if not isinstance(resp, dict):
        raise RuntimeError(f"Fyers {endpoint}: unexpected response shape")
    if resp.get("s") == "error":
        msg = resp.get("message") or resp.get("code") or "unknown error"
        raise RuntimeError(f"Fyers {endpoint}: {msg}")


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
