"""
Data Service — fetches and enriches market data from Angel One SmartAPI.

Reuses your existing BrokerClient and add_new_high_low_indicators logic.
"""
import os
import time
import threading
import pandas as pd
import pyotp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from logzero import logger
from SmartApi import SmartConnect
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Symbol master
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Cross-process throttle plumbing
# ──────────────────────────────────────────────
# Lazily-resolved Redis client used by BrokerClient._cross_process_throttle.
# Cached per-process so we don't re-pay the connection setup on every
# SmartAPI call. ``None`` means we tried and Redis isn't reachable — the
# throttle then falls back to in-process only (correct in a single-worker
# dev setup, degraded but not broken in prod if Redis is down).
_THROTTLE_REDIS_CLIENT: Any = None
_THROTTLE_REDIS_PROBED: bool = False


def _get_redis_for_throttle():
    """Return a redis.Redis client for the universal throttle, or None.

    Uses REDIS_URL (same as Celery / Channels), so no separate config.
    Imported lazily so test environments without redis-py installed
    aren't forced to install it just to import this module.
    """
    global _THROTTLE_REDIS_CLIENT, _THROTTLE_REDIS_PROBED
    if _THROTTLE_REDIS_PROBED:
        return _THROTTLE_REDIS_CLIENT
    _THROTTLE_REDIS_PROBED = True
    try:
        import redis  # type: ignore
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(url, socket_timeout=2)
        # Cheap liveness probe so we fail-fast on a misconfig.
        client.ping()
        _THROTTLE_REDIS_CLIENT = client
    except Exception as e:
        logger.warning("smartapi.throttle.redis_unavailable: %s", e)
        _THROTTLE_REDIS_CLIENT = None
    return _THROTTLE_REDIS_CLIENT


# Error fingerprints that should OPEN the rate-limit breaker: the account-level
# rate-limit ban plus the connection give-ups it manifests as (Angel drops/
# refuses connections once the key is throttled). Tripping on these backs the
# whole fleet off instead of hammering an unreachable/banned endpoint.
_TRANSIENT_BROKER_MARKERS = (
    "exceeding access rate",
    "max retries exceeded",
    "timed out",
    "connection aborted",
    "connection reset",
    "connection refused",
)


def _is_transient_broker_error(err) -> bool:
    s = str(err).lower()
    return any(m in s for m in _TRANSIENT_BROKER_MARKERS)


def load_symbol_master(file_path: str) -> list:
    """
    Load Angel One OpenAPI scrip master JSON.
    Returns raw list of instrument dicts.
    Format: [{"token": "2142", "symbol": "MFSL-EQ", "name": "MFSL",
              "exch_seg": "NSE", ...}, ...]
    """
    import json
    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Legacy dict-of-dicts format — flatten
        return list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    return []


class TokenFetcher:
    """Map NSE ticker symbols to exchange tokens using Angel One scrip master."""

    def __init__(self, instruments: list):
        """
        Args:
            instruments: list of dicts from Angel One OpenAPI scrip master.
                         Each dict has keys: token, symbol, name, exch_seg, etc.
        """
        # Build lookup: "NSE:MFSL-EQ" → "2142"
        self._lookup = {}
        for inst in instruments:
            seg = inst.get("exch_seg", "")
            sym = inst.get("symbol", "")
            tok = inst.get("token", "")
            if seg and sym and tok:
                self._lookup[f"{seg}:{sym}"] = tok

        logger.info(f"TokenFetcher loaded {len(self._lookup)} instruments")

    def get_token(self, symbol: str, exchange: str = "NSE") -> Optional[str]:
        """
        Get token for a ticker.
        Accepts: 'MFSL' or 'NSE:MFSL-EQ'
        """
        # Normalize to 'NSE:SYMBOL-EQ' format
        if ":" not in symbol:
            sym_key = f"{exchange}:{symbol}-EQ"
        else:
            sym_key = symbol

        token = self._lookup.get(sym_key)
        if not token:
            logger.error(f"Token not found for {sym_key} (have {len(self._lookup)} instruments)")
        return token


# ──────────────────────────────────────────────
# Broker client
# ──────────────────────────────────────────────
class BrokerClient:
    """
    Centralized Angel One SmartAPI gateway.

    ALL broker API calls go through this class. It provides:
      - Single authenticated session (thread-safe login)
      - Global rate limiter (0.4s min gap between ANY API call)
      - TTL cache for ltpData (avoids redundant spot/LTP fetches)
      - Retry with backoff on rate-limit errors
      - Request counting for monitoring

    Use the module-level singleton: `from trading.services.data_service import broker`
    """

    # ── Class-level singleton ──
    _instance: Optional["BrokerClient"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "BrokerClient":
        """Get or create the process-wide singleton."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.api_key = os.getenv("SMARTAPI_KEY")
        self.username = os.getenv("SMARTAPI_USERNAME")
        self.password = os.getenv("SMARTAPI_PASSWORD")
        self.totp_secret = os.getenv("SMARTAPI_TOTP_SECRET")
        self.smart_api = SmartConnect(self.api_key)
        self._logged_in = False

        # Rate limiting: min 0.4s between any API call
        self._last_call_time = 0.0
        self._rate_lock = threading.Lock()
        self._min_interval = 0.4  # seconds

        # LTP cache: {cache_key: (timestamp, result)}
        self._ltp_cache: Dict[str, tuple] = {}
        self._ltp_cache_ttl = 5  # seconds

        # Stats
        self._call_count = 0
        self._cache_hits = 0

        # Login lock
        self._login_lock = threading.Lock()

        # Rate-limit circuit breaker (in-process fallback; the authoritative
        # state is Redis-shared so all worker processes + the v2 adapter +
        # this legacy client honour one cooldown). See breaker_* methods.
        self._breaker_open_until = 0.0   # time.monotonic() deadline
        self._breaker_trips = 0
        self._breaker_lock = threading.Lock()

    # Lua: atomically read the recorded last-call timestamp (ms), compute
    # how long we still need to wait, and overwrite the slot with our
    # claim. KEYS[1] = redis key, ARGV[1] = min gap in ms, ARGV[2] = now ms.
    # Returns the number of ms to sleep before the call is allowed (0 if
    # the slot is free immediately).
    _LUA_THROTTLE = """
        local last = tonumber(redis.call('GET', KEYS[1]) or '0')
        local gap = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local next_slot = last + gap
        local wait_ms
        if next_slot <= now then
            wait_ms = 0
            redis.call('SET', KEYS[1], now, 'PX', 60000)
        else
            wait_ms = next_slot - now
            redis.call('SET', KEYS[1], next_slot, 'PX', 60000)
        end
        return wait_ms
    """
    _REDIS_THROTTLE_KEY = "alphadesk:smartapi:throttle:last_call_ms"

    def _throttle(self):
        """Enforce min interval between SmartAPI calls — process-wide
        AND cross-process when Redis is available.

        The in-process lock keeps the local fast-path correct (Daphne /
        celery worker / management command all in one process). Redis
        adds cross-process coordination so multiple Celery workers, plus
        the Daphne process, plus the legacy CLI scripts all share the
        same queue. Falls back gracefully to local-only when Redis is
        unreachable.
        """
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.monotonic()
            self._call_count += 1
        self._cross_process_throttle()

    def _cross_process_throttle(self) -> None:
        """Sleep until our turn in the Redis-coordinated slot queue."""
        try:
            r = _get_redis_for_throttle()
        except Exception:
            return  # Redis unavailable — local throttle was the best we could do.
        if r is None:
            return
        # Per-minute SmartAPI call counter (best effort) for the broker-monitor
        # dashboard — counts every throttled call across ALL processes.
        try:
            ckey = f"{self._REDIS_CALLS_PREFIX}{int(time.time() // 60)}"
            r.incr(ckey)
            r.expire(ckey, 3700)
        except Exception:
            pass
        try:
            wait_ms = r.eval(
                self._LUA_THROTTLE, 1,
                self._REDIS_THROTTLE_KEY,
                int(self._min_interval * 1000),
                int(time.time() * 1000),
            )
            wait_ms = int(wait_ms or 0)
        except Exception:
            return
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)

    _REDIS_CALLS_PREFIX = "alphadesk:smartapi:calls:"

    def recent_call_rate(self, minutes: int = 30) -> list:
        """Per-minute SmartAPI call counts for the last `minutes` (oldest first).

        Returns [{"minute": <epoch_seconds>, "calls": int}, ...] — drives the
        broker-monitor dashboard's call-volume chart. Empty if Redis is down."""
        try:
            r = _get_redis_for_throttle()
        except Exception:
            r = None
        if r is None:
            return []
        now_min = int(time.time() // 60)
        keys = [f"{self._REDIS_CALLS_PREFIX}{now_min - i}" for i in range(minutes)]
        try:
            vals = r.mget(keys)
        except Exception:
            return []
        out = [
            {"minute": (now_min - i) * 60, "calls": int(v) if v else 0}
            for i, v in enumerate(vals)
        ]
        out.reverse()
        return out

    def breaker_status(self) -> dict:
        """Snapshot of the rate-limit breaker for the monitor dashboard."""
        remaining_ms = self.breaker_remaining_ms()
        trips = 0
        try:
            r = _get_redis_for_throttle()
            if r is not None:
                t = r.get(self._REDIS_BREAKER_TRIPS_KEY)
                trips = int(t) if t else 0
        except Exception:
            pass
        state = "open" if remaining_ms > 0 else ("recovering" if trips > 0 else "closed")
        return {
            "open": remaining_ms > 0,
            "state": state,
            "cooldown_remaining_s": remaining_ms // 1000,
            "trips": trips,
        }

    # Public alias — the v2 multi-tenant AngelOneAdapter uses this to
    # share the 0.4s gap with every other SmartAPI call in the process,
    # so multiple linked accounts can't race past Angel's rate limit.
    throttle = _throttle

    # ── Rate-limit circuit breaker (Redis-coordinated, HALF-OPEN) ─────
    # Angel One's "exceeding access rate" is an ACCOUNT-level cooldown that
    # denies EVERY call. We must back off — but a blunt "block everything for
    # the whole cooldown" STARVES legitimate calls (a backtest / position
    # refresh fails for 30-300s even after Angel has recovered). So this is a
    # proper HALF-OPEN breaker, shared across all processes via Redis:
    #
    #   CLOSED → call normally.
    #   OPEN   → (short cooldown) deny fast, no network hit.
    #   HALF-OPEN (cooldown elapsed) → let exactly ONE probe call through
    #            (Redis NX lock); everyone else still denied. Probe succeeds →
    #            CLOSED (full traffic resumes immediately). Probe fails →
    #            OPEN again with escalated backoff.
    #
    # Net: at most one in-flight probe at a time (never a re-storm), the base
    # cooldown is short so calls aren't starved, and traffic resumes the moment
    # Angel recovers instead of after a fixed window. In-process deadline is the
    # fallback when Redis is unreachable.
    _REDIS_BREAKER_KEY = "alphadesk:smartapi:breaker:open_until_ms"
    _REDIS_BREAKER_TRIPS_KEY = "alphadesk:smartapi:breaker:trips"
    _REDIS_BREAKER_PROBE_KEY = "alphadesk:smartapi:breaker:probe"
    _BREAKER_BASE_COOLDOWN = 5.0    # seconds — first trip (short → fast recovery)
    _BREAKER_MAX_COOLDOWN = 60.0    # cap — still probe ≥ every 60s during a hard ban
    _BREAKER_PROBE_TTL = 10         # seconds — auto-release a stuck probe lock

    def breaker_gate(self) -> str:
        """Half-open gate. Returns 'closed' | 'probe' | 'open'.

        'closed' → make the call normally.
        'probe'  → cooldown elapsed and THIS call won the single probe slot;
                   caller MUST reset_breaker() on success / trip_breaker() on a
                   rate-limit failure.
        'open'   → cooling down; do NOT touch Angel.
        Cross-process via Redis; falls back to in-process open/closed."""
        now_ms = int(time.time() * 1000)
        try:
            r = _get_redis_for_throttle()
        except Exception:
            r = None
        if r is None:
            return "open" if (self._breaker_open_until - time.monotonic()) > 0 else "closed"
        try:
            raw = r.get(self._REDIS_BREAKER_KEY)
            if not raw:
                return "closed"
            if now_ms < int(raw):
                return "open"
            # Cooldown elapsed → half-open. Exactly one caller wins the probe.
            got = r.set(self._REDIS_BREAKER_PROBE_KEY, "1", nx=True, ex=self._BREAKER_PROBE_TTL)
            return "probe" if got else "open"
        except Exception:
            return "closed"   # fail-open on a Redis blip — don't starve on infra

    def breaker_remaining_ms(self) -> int:
        """Milliseconds until the rate-limit breaker's cooldown elapses (0 = not
        open). Used for the operator-facing cooldown message, not the gate."""
        now_ms = int(time.time() * 1000)
        try:
            r = _get_redis_for_throttle()
        except Exception:
            r = None
        if r is not None:
            try:
                raw = r.get(self._REDIS_BREAKER_KEY)
                deadline = int(raw) if raw else 0
                return max(0, deadline - now_ms)
            except Exception:
                pass
        rem = self._breaker_open_until - time.monotonic()
        return int(rem * 1000) if rem > 0 else 0

    def trip_breaker(self) -> float:
        """Open/extend the breaker with exponential backoff (Redis-shared trip
        counter, deadline only ever EXTENDS). Clears the half-open probe lock so
        the next window can probe. Returns the cooldown in seconds."""
        with self._breaker_lock:
            self._breaker_trips += 1
            trips = self._breaker_trips
        cooldown = min(self._BREAKER_BASE_COOLDOWN * (2 ** (trips - 1)), self._BREAKER_MAX_COOLDOWN)
        now_ms = int(time.time() * 1000)
        try:
            r = _get_redis_for_throttle()
        except Exception:
            r = None
        if r is not None:
            try:
                shared_trips = int(r.incr(self._REDIS_BREAKER_TRIPS_KEY))
                r.pexpire(self._REDIS_BREAKER_TRIPS_KEY,
                          int(self._BREAKER_MAX_COOLDOWN * 1000) + 60000)
                cooldown = min(
                    self._BREAKER_BASE_COOLDOWN * (2 ** (shared_trips - 1)),
                    self._BREAKER_MAX_COOLDOWN,
                )
                deadline_ms = now_ms + int(cooldown * 1000)
                existing = r.get(self._REDIS_BREAKER_KEY)
                if not existing or int(existing) < deadline_ms:
                    r.set(self._REDIS_BREAKER_KEY, deadline_ms, px=int(cooldown * 1000) + 60000)
                r.delete(self._REDIS_BREAKER_PROBE_KEY)
            except Exception:
                pass
        self._breaker_open_until = time.monotonic() + cooldown
        logger.warning("smartapi.breaker_open cooldown=%.0fs (trips=%d)", cooldown, trips)
        return cooldown

    def reset_breaker(self) -> None:
        """Close the breaker — a probe (or clean call) succeeded. Clears the
        deadline, trip counter and probe lock so full traffic resumes."""
        with self._breaker_lock:
            self._breaker_open_until = 0.0
            self._breaker_trips = 0
        try:
            r = _get_redis_for_throttle()
            if r is not None:
                r.delete(
                    self._REDIS_BREAKER_KEY,
                    self._REDIS_BREAKER_TRIPS_KEY,
                    self._REDIS_BREAKER_PROBE_KEY,
                )
        except Exception:
            pass

    def login(self) -> bool:
        """Authenticate with Angel One (thread-safe, idempotent)."""
        with self._login_lock:
            if self._logged_in:
                return True
            try:
                totp = pyotp.TOTP(self.totp_secret).now()
                self.smart_api.generateSession(self.username, self.password, totp)
                self._logged_in = True
                logger.info("Broker login successful.")
                return True
            except Exception as e:
                logger.error(f"Broker login failed: {e}")
                self._logged_in = False
                return False

    def ensure_login(self):
        """Login if not already logged in."""
        if not self._logged_in:
            self.login()

    @property
    def is_logged_in(self) -> bool:
        return self._logged_in

    # ──────────────────────────────────────────────
    # LTP Data (with built-in cache)
    # ──────────────────────────────────────────────
    def ltp(self, exchange: str, symbol: str, token: str) -> dict:
        """
        Fetch LTP data with automatic caching.

        Returns parsed dict: {ltp, open, high, low, prev_close} or {} on error.
        Cached for 5 seconds — safe to call frequently.
        """
        cache_key = f"ltp:{exchange}:{token}"
        now = time.monotonic()

        # Check cache
        entry = self._ltp_cache.get(cache_key)
        if entry is not None:
            ts, result = entry
            if now - ts < self._ltp_cache_ttl:
                self._cache_hits += 1
                return result

        # Fetch fresh
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.ltpData(exchange, symbol, token)
            result = self._parse_ltp(r)
            self._ltp_cache[cache_key] = (time.monotonic(), result)
            return result
        except Exception as e:
            logger.error(f"ltpData failed ({exchange}:{symbol}): {e}")
            return {}

    @staticmethod
    def _parse_ltp(r) -> dict:
        """Parse ltpData response. Guards against string error responses."""
        if not isinstance(r, dict):
            return {}
        d = r.get("data")
        if not isinstance(d, dict):
            return {}
        return {
            "ltp":        float(d.get("ltp", 0)),
            "open":       float(d.get("open", 0)),
            "high":       float(d.get("high", 0)),
            "low":        float(d.get("low", 0)),
            "prev_close": float(d.get("close", 0)),
        }

    # ──────────────────────────────────────────────
    # Candle Data
    # ──────────────────────────────────────────────
    # Known index tokens → exchange segment. BSE indices used to fall back
    # to the default exchange="NSE" and silently return zero candles.
    _INDEX_EXCHANGE = {
        "99926000": "NSE",   # NIFTY 50
        "99926009": "NSE",   # NIFTY Bank
        "99926017": "NSE",   # India VIX
        "99926037": "NSE",   # NIFTY Next 50 (assorted)
        "99919000": "BSE",   # SENSEX
        "99919012": "BSE",   # BANKEX
    }

    def fetch_candles(
        self,
        symbol_token: str,
        start: str,
        end: str,
        interval: str = "FIVE_MINUTE",
        exchange: str | None = None,
    ) -> List:
        """Fetch OHLCV candle data with retry on rate-limit errors.

        Dates in '%Y-%m-%d %H:%M' format. If `exchange` is omitted, it's
        auto-derived from the token: known index tokens use their static
        segment (NSE/BSE); other tokens fall back to ticker_service for
        NSE/NFO/BSE/BFO/MCX classification; ultimate default is NSE.
        Pass exchange explicitly to override.
        """
        self.ensure_login()

        # Auto-derive exchange when caller hasn't specified one. Known
        # index tokens have a static map (NIFTY/BANKNIFTY=NSE, SENSEX=BSE).
        # For options + non-index tokens callers should pass exchange
        # explicitly (NFO for NIFTY/BANKNIFTY options, BFO for SENSEX),
        # otherwise we default to NSE — same legacy behaviour.
        if exchange is None:
            exchange = self._INDEX_EXCHANGE.get(str(symbol_token), "NSE")

        # Defensive clamp: Angel One returns AB1012 if `fromdate` or `todate`
        # is in the future relative to broker wall-clock. This happens on every
        # pre-market fetch where the caller passes "today 09:15 → 15:30".
        # Short-circuit if the requested window can't possibly have data yet,
        # so callers see [] + a clear log instead of a stack trace from the SDK.
        try:
            from trading.utils.time_utils import cap_end_time
            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M")
            if start > now_str:
                logger.warning(
                    f"fetch_candles: window {start}..{end} is entirely in the future "
                    f"(now={now_str}); skipping broker call. "
                    f"Use trading.utils.time_utils.last_trading_day() for a fallback date."
                )
                return []
            capped = cap_end_time(end, now=now)
            if capped != end:
                logger.info(f"fetch_candles: clamped end {end} -> {capped}")
                end = capped
        except Exception as e:
            logger.debug(f"fetch_candles: clamp skipped ({e})")

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": start,
            "todate": end,
        }
        # Honour the shared rate-limit breaker: during an account-level
        # cooldown every candle call is denied anyway, and retrying just
        # extends the ban. Honour the half-open gate: deny while OPEN, allow a
        # single probe when the cooldown has elapsed.
        gate = self.breaker_gate()
        if gate == "open":
            logger.debug("Candle fetch skipped — rate-limit breaker open")
            return []

        max_retries = 2
        for attempt in range(max_retries + 1):
            self._throttle()
            try:
                response = self.smart_api.getCandleData(params)
                if response is None:
                    if attempt < max_retries:
                        logger.warning(f"Rate limited on candles, retry {attempt+1}/{max_retries} in 2s...")
                        time.sleep(2)
                        continue
                    logger.warning("Candle fetch returned None after retries")
                    return []
                if gate == "probe":
                    self.reset_breaker()   # probe succeeded → resume full traffic
                return response.get("data", []) or []
            except Exception as e:
                # A rate-limit ban / connection give-up: trip the shared breaker
                # and stop — retrying with 2s sleeps only amplifies the storm
                # (this except branch, not the None branch, is where Angel's
                # "exceeding access rate" actually lands).
                if _is_transient_broker_error(e):
                    self.trip_breaker()
                    logger.warning(f"Candle fetch hit broker rate-limit/timeout; breaker tripped: {str(e)[:120]}")
                    return []
                logger.exception(f"Candle fetch failed: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return []
        return []

    # ──────────────────────────────────────────────
    # Portfolio / Orders
    # ──────────────────────────────────────────────
    def fetch_holdings(self) -> List[Dict]:
        """Fetch current portfolio holdings."""
        self.ensure_login()
        self._throttle()
        try:
            return self.smart_api.holding() or []
        except Exception as e:
            logger.error(f"Holdings fetch failed: {e}")
            return []

    def fetch_positions(self) -> Dict:
        """Fetch open positions."""
        self.ensure_login()
        self._throttle()
        try:
            return self.smart_api.position() or {}
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return {}

    def fetch_order_book(self) -> list:
        """Fetch today's order book."""
        self.ensure_login()
        self._throttle()
        try:
            book = self.smart_api.orderBook()
            if isinstance(book, dict) and book.get("data"):
                return book["data"]
            return []
        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Batch Market Data (THE key optimization)
    # ──────────────────────────────────────────────
    def market_data_batch(
        self,
        tokens: Dict[str, List[str]],
        mode: str = "OHLC",
    ) -> List[dict]:
        """
        Fetch market data for up to 50 instruments in ONE API call.

        This is 50x more efficient than calling ltpData() per stock.

        Args:
            tokens: {"NSE": ["2885", "1333"], "NFO": ["57710"]}
            mode: "LTP" (just price), "OHLC" (price + OHLC), "FULL" (everything)
                  FULL includes: volume, OI, 52wk hi/lo, circuit limits, bid/ask depth

        Returns:
            List of dicts with: exchange, tradingSymbol, symbolToken, ltp, open,
            high, low, close, percentChange, tradeVolume, opnInterest, etc.
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.getMarketData(mode, tokens)
            if isinstance(r, dict) and r.get("status"):
                data = r.get("data", {})
                return data.get("fetched", [])
            return []
        except Exception as e:
            logger.error(f"Batch market data failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Option Greeks (real IV, delta, gamma from exchange)
    # ──────────────────────────────────────────────
    def option_greeks(self, underlying: str, expiry: str) -> List[dict]:
        """
        Fetch real option greeks from Angel One (only works during market hours).

        Args:
            underlying: "NIFTY" or "BANKNIFTY"
            expiry: "17MAR2026"

        Returns:
            List of dicts with: strikePrice, CE_delta, CE_gamma, CE_theta,
            CE_vega, CE_impliedVolatility, PE_* equivalents, etc.
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.optionGreek({"name": underlying, "expirydate": expiry})
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"Option greeks fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # OI Data (historical open interest)
    # ──────────────────────────────────────────────
    def oi_data(self, exchange: str, token: str, from_date: str, to_date: str) -> List[dict]:
        """
        Fetch historical open interest data.

        Args:
            exchange: "NFO"
            token: instrument token
            from_date: "2026-03-10 09:15"
            to_date: "2026-03-17 15:30"
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.getOIData({
                "exchange": exchange,
                "symboltoken": token,
                "fromdate": from_date,
                "todate": to_date,
            })
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"OI data fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Margin / RMS Limits (real capital, not hardcoded)
    # ──────────────────────────────────────────────
    def margin_available(self) -> dict:
        """
        Fetch real-time margin/capital from broker.

        Returns:
            {net, availablecash, collateral, utilisedexposure, ...}
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.rmsLimit()
            if isinstance(r, dict) and isinstance(r.get("data"), dict):
                return r["data"]
            return {}
        except Exception as e:
            logger.error(f"RMS limit fetch failed: {e}")
            return {}

    # ──────────────────────────────────────────────
    # Trade Book (today's executed trades)
    # ──────────────────────────────────────────────
    def fetch_trade_book(self) -> list:
        """Fetch all executed trades for today."""
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.tradeBook()
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"Trade book fetch failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Search Scrip (live symbol search)
    # ──────────────────────────────────────────────
    def search_scrip(self, exchange: str, query: str) -> list:
        """
        Live symbol search by name.

        Args:
            exchange: "NSE", "NFO", "BSE"
            query: partial name like "RELIANCE"
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.searchScrip(exchange, query)
            if isinstance(r, dict) and r.get("data"):
                return r["data"] if isinstance(r["data"], list) else []
            return []
        except Exception as e:
            logger.error(f"Search scrip failed: {e}")
            return []

    # ──────────────────────────────────────────────
    # Estimate Charges (brokerage + STT + stamp duty)
    # ──────────────────────────────────────────────
    def estimate_charges(self, orders: list) -> dict:
        """
        Estimate brokerage and charges for a list of orders.

        Args:
            orders: list of order param dicts
        """
        self.ensure_login()
        self._throttle()
        try:
            r = self.smart_api.estimateCharges({"orders": orders})
            if isinstance(r, dict):
                return r.get("data", {})
            return {}
        except Exception as e:
            logger.error(f"Estimate charges failed: {e}")
            return {}

    # ──────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "logged_in": self._logged_in,
            "api_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "ltp_cache_size": len(self._ltp_cache),
        }


# ── Module-level singleton ──
broker = BrokerClient.get_instance()


# ──────────────────────────────────────────────
# Data enrichment (from your existing logic)
# ──────────────────────────────────────────────
def enrich_ohlcv(data: List) -> pd.DataFrame:
    """
    Enrich raw OHLCV candles with indicators.
    Input: list of [timestamp, open, high, low, close, volume]
    """
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame([dict(zip(cols, row)) for row in data])

    if df.empty:
        return df

    df["cumulative_high"] = df["high"].cummax()
    df["cumulative_low"] = df["low"].cummin()

    df["new_high"] = df["high"] == df["cumulative_high"]
    df["new_low"] = df["low"] == df["cumulative_low"]
    df.at[0, "new_high"] = False
    df.at[0, "new_low"] = False

    first_close = df["close"].iloc[0]
    first_open = df["open"].iloc[0]

    df["range"] = df["cumulative_high"] - df["cumulative_low"]
    df["range_percent"] = (df["range"] / first_close) * 100 if first_close else 0

    df["high_drawdown"] = 100 * (df["cumulative_high"] - df["close"]).abs() / first_open if first_open else 0
    df["low_drawdown"] = 100 * (df["cumulative_low"] - df["close"]).abs() / first_open if first_open else 0

    df["OH"] = df["open"] == df["high"]
    df["OL"] = df["open"] == df["low"]
    df["doji"] = (df["open"] - df["close"]).abs() < 0.10
    df["pivot"] = (df["low"] + df["high"]) / 2

    return df


# ──────────────────────────────────────────────
# High-level data service
# ──────────────────────────────────────────────
class DataService:
    """
    Top-level data service used by graph nodes.
    Uses the singleton BrokerClient for all API calls and
    ticker_service for symbol resolution.
    """

    def __init__(self):
        self._broker: Optional[BrokerClient] = None

    def _ensure_broker(self):
        if self._broker is None:
            self._broker = BrokerClient.get_instance()
        self._broker.ensure_login()

    def fetch_intraday(
        self,
        symbol: str,
        date: Optional[str] = None,
        interval: str = "FIVE_MINUTE",
    ) -> Dict[str, Any]:
        """
        Fetch enriched intraday data for a symbol.

        Session-phase aware: if ``date`` is today and the market hasn't opened
        yet (or is closed / weekend), the response pivots to ``last_trading_day``
        and the returned dict carries ``is_live=False`` + ``data_age_days`` so
        the planner knows the structure isn't from a live session.

        Args:
            symbol: NSE symbol, e.g. 'MFSL'
            date: Date string '%Y-%m-%d'. If None, auto-selects via
                ``time_utils.get_candle_date_range``.
            interval: FIVE_MINUTE, ONE_HOUR, etc.

        Returns:
            dict with keys: symbol, date, requested_date, is_live, data_age_days,
                            session_phase, candle_count, last_close, day_high,
                            day_low, range_pct, summary
        """
        from datetime import date as _date

        from trading.utils.time_utils import (
            can_fetch_candles, get_session_phase, last_trading_day,
        )

        self._ensure_broker()

        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol)
        if not token:
            return {"error": f"Token not found for {symbol}", "symbol": symbol}

        requested = date or _date.today().isoformat()
        phase = get_session_phase()
        is_today = requested == _date.today().isoformat()

        # If the caller asked for today but live candles aren't available,
        # pivot to the last completed session. Explicit past dates are
        # passed through untouched.
        if is_today and not can_fetch_candles():
            effective = last_trading_day().isoformat()
        else:
            effective = requested

        raw = self._broker.fetch_candles(
            token, f"{effective} 09:15", f"{effective} 15:30", interval,
        )
        if not raw:
            return {
                "error": "No candle data returned",
                "symbol": symbol,
                "requested_date": requested,
                "effective_date": effective,
                "session_phase": phase,
            }

        df = enrich_ohlcv(raw)
        days_old = (_date.today() - _date.fromisoformat(effective)).days
        is_live = is_today and effective == requested

        # Build summary dict for the graph state
        last_row = df.iloc[-1]
        return {
            "symbol": symbol,
            "date": effective,
            "requested_date": requested,
            "is_live": is_live,
            "data_age_days": days_old,
            "session_phase": phase,
            "candle_count": len(df),
            "open": float(df.iloc[0]["open"]),
            "last_close": float(last_row["close"]),
            "day_high": float(df["high"].max()),
            "day_low": float(df["low"].min()),
            "range_pct": float(last_row.get("range_percent", 0)),
            "new_highs": int(df["new_high"].sum()),
            "new_lows": int(df["new_low"].sum()),
            "doji_count": int(df["doji"].sum()),
            "pivot": float(last_row.get("pivot", 0)),
            "summary": self._build_text_summary(
                df, symbol, effective,
                is_live=is_live, session_phase=phase, days_old=days_old,
            ),
        }

    def _build_text_summary(
        self,
        df: pd.DataFrame,
        symbol: str,
        date: str,
        is_live: bool = True,
        session_phase: str = "REGULAR",
        days_old: int = 0,
    ) -> str:
        """Build a text summary of market data for LLM context.

        The header makes data freshness explicit so the planner can choose
        between live-structure reasoning and prior-session reasoning.
        """
        if df.empty:
            return f"No data available for {symbol} on {date}"

        last = df.iloc[-1]
        first = df.iloc[0]

        if is_live:
            header = f"Market Data for {symbol} on {date} (LIVE — {session_phase}):"
        else:
            age = "yesterday" if days_old == 1 else f"{days_old} days ago"
            header = (
                f"Market Data for {symbol} — last completed session {date} "
                f"({age}). Current phase: {session_phase}. "
                f"Live ticks not yet available; reason from this structure."
            )

        return (
            f"{header}\n"
            f"  Open: {first['open']:.2f} | Last Close: {last['close']:.2f}\n"
            f"  Day High: {df['high'].max():.2f} | Day Low: {df['low'].min():.2f}\n"
            f"  Range: {last.get('range', 0):.2f} ({last.get('range_percent', 0):.2f}%)\n"
            f"  New Highs: {df['new_high'].sum()} | New Lows: {df['new_low'].sum()}\n"
            f"  Candles: {len(df)} | Doji: {df['doji'].sum()}\n"
            f"  Pivot: {last.get('pivot', 0):.2f}\n"
            f"  Last 5 closes: {list(df['close'].tail(5).round(2))}\n"
            f"  Trend: {'Bullish' if last['close'] > first['open'] else 'Bearish'} "
            f"({((last['close'] - first['open']) / first['open'] * 100):.2f}%)"
        )

    def fetch_historical(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = "FIVE_MINUTE",
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical candles across multiple days from Angel One broker.

        Angel One limits each API call to a single day for intraday intervals,
        so we loop day-by-day (skipping weekends) and merge results.

        Args:
            symbol: NSE symbol, e.g. 'MFSL'
            from_date: Start date '%Y-%m-%d'
            to_date: End date '%Y-%m-%d'
            interval: FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY

        Returns:
            List of candle dicts: [{date, open, high, low, close, volume}, ...]
            For daily interval: one candle per day (OHLCV aggregated).
            For intraday intervals: individual candles per trading session.
        """
        self._ensure_broker()

        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol)
        if not token:
            logger.error(f"Token not found for {symbol}")
            return []

        start_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(to_date, "%Y-%m-%d").date()

        all_candles: List[Dict[str, Any]] = []

        logger.info(
            f"Fetching historical data: {symbol} | {from_date} → {to_date} | interval={interval}"
        )

        # Angel One getCandleData caps the span PER CALL by interval. Fetch the
        # whole lookback in as few calls as possible by chunking the range to
        # each interval's max span — instead of one call per day, which made a
        # single 120-day intraday scan ~83 calls/symbol and tripped the broker
        # rate limit. (NSE weekends/holidays are skipped server-side, so we
        # request the full window and let grouping yield only trading days.)
        _INTERVAL_MAX_DAYS = {
            "ONE_MINUTE": 30, "THREE_MINUTE": 60,
            "FIVE_MINUTE": 100, "TEN_MINUTE": 100,
            "FIFTEEN_MINUTE": 200, "THIRTY_MINUTE": 200,
            "ONE_HOUR": 400, "ONE_DAY": 2000,
        }
        cap_days = _INTERVAL_MAX_DAYS.get(interval, 100)

        raw_rows: List[list] = []
        win_start = start_dt
        while win_start <= end_dt:
            win_end = min(win_start + timedelta(days=cap_days - 1), end_dt)
            chunk = self._broker.fetch_candles(
                token,
                f"{win_start.strftime('%Y-%m-%d')} 09:15",
                f"{win_end.strftime('%Y-%m-%d')} 15:30",
                interval,
            )
            if chunk:
                raw_rows.extend(chunk)
            win_start = win_end + timedelta(days=1)

        # ONE_DAY rows are already daily granularity — map straight through.
        if interval == "ONE_DAY":
            for row in raw_rows:
                ts = row[0]
                all_candles.append({
                    "date": ts[:10] if isinstance(ts, str) else str(ts),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": int(row[5]),
                })
            logger.info(f"Historical fetch complete: {len(all_candles)} daily candles for {symbol}")
            return all_candles

        # Intraday intervals → aggregate each trading day into one daily OHLCV
        # row (identical contract to the old day-by-day loop). raw_rows are in
        # ascending time order, so per-day first-open / last-close stay correct.
        by_day: Dict[str, list] = {}
        for row in raw_rows:
            ts = row[0]
            day = ts[:10] if isinstance(ts, str) else str(ts)
            by_day.setdefault(day, []).append(row)

        for day_str, rows in by_day.items():
            highs = [r[2] for r in rows]
            lows = [r[3] for r in rows]
            volumes = [r[5] for r in rows]
            all_candles.append({
                "date": day_str,
                "open": float(rows[0][1]),
                "high": float(max(highs)),
                "low": float(min(lows)),
                "close": float(rows[-1][4]),
                "volume": int(sum(volumes)),
            })

        logger.info(f"Historical fetch complete: {len(all_candles)} trading days for {symbol}")
        return all_candles

    def fetch_intraday_candles(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = "FIVE_MINUTE",
    ) -> List[Dict[str, Any]]:
        """
        Fetch raw intraday candles (not aggregated) across multiple days.

        Uses per-day disk cache at /tmp/intraday_cache/ to avoid redundant
        broker API calls. Second run for same symbol/day/interval is instant.

        Args:
            symbol: NSE symbol (e.g. 'RELIANCE')
            from_date: Start date '%Y-%m-%d'
            to_date: End date '%Y-%m-%d'
            interval: THREE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, etc.

        Returns:
            List of candle dicts: [{timestamp, open, high, low, close, volume}, ...]
        """
        import json as _json
        from pathlib import Path

        self._ensure_broker()

        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol)
        if not token:
            logger.error(f"Token not found for {symbol}")
            return []

        cache_dir = Path("/tmp/intraday_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        interval_short = {
            "ONE_MINUTE": "1m", "THREE_MINUTE": "3m", "FIVE_MINUTE": "5m",
            "TEN_MINUTE": "10m", "FIFTEEN_MINUTE": "15m", "THIRTY_MINUTE": "30m",
            "ONE_HOUR": "1h",
        }.get(interval, interval)

        start_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
        current = start_dt
        all_candles: List[Dict[str, Any]] = []
        cached_days = 0
        fetched_days = 0

        while current <= end_dt:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            day_str = current.strftime("%Y-%m-%d")
            cache_file = cache_dir / f"{symbol}_{interval_short}_{day_str}.json"

            # Try disk cache first
            if cache_file.exists():
                try:
                    day_candles = _json.loads(cache_file.read_text())
                    all_candles.extend(day_candles)
                    cached_days += 1
                    current += timedelta(days=1)
                    continue
                except (_json.JSONDecodeError, IOError):
                    pass

            # Fetch from broker
            raw = self._broker.fetch_candles(
                token, f"{day_str} 09:15", f"{day_str} 15:30", interval
            )

            day_candles = []
            if raw:
                for row in raw:
                    ts, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
                    day_candles.append({
                        "timestamp": ts if isinstance(ts, str) else str(ts),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": int(v),
                    })
                # Cache to disk (only complete trading days, not today)
                from datetime import date as _date
                if current < _date.today():
                    try:
                        cache_file.write_text(_json.dumps(day_candles))
                    except IOError:
                        pass
                fetched_days += 1

            all_candles.extend(day_candles)
            current += timedelta(days=1)

        logger.info(
            f"Intraday fetch: {symbol} {interval_short} {from_date}→{to_date}: "
            f"{len(all_candles)} candles ({cached_days} cached, {fetched_days} fetched)"
        )
        return all_candles

    def fetch_multi_timeframe(
        self,
        symbol: str,
        date_str: str,
        intervals: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch candles at multiple timeframes for the same symbol/date.
        Used by scanning agents for multi-TF structure confirmation.

        Args:
            symbol: NSE ticker e.g. 'RELIANCE'
            date_str: Date string '%Y-%m-%d'
            intervals: List of intervals. Default: ['FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR']

        Returns:
            {"5m": [candles], "15m": [candles], "1h": [candles]}
        """
        if intervals is None:
            intervals = ["FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR"]

        self._ensure_broker()
        from trading.services.ticker_service import ticker_service
        token = ticker_service.get_token(symbol)
        if not token:
            return {"error": f"Token not found for {symbol}"}

        # Cap end time for today
        from datetime import datetime as _dt
        now = _dt.now()
        if date_str == now.strftime("%Y-%m-%d"):
            if now.hour < 9 or (now.hour == 9 and now.minute < 16):
                return {"error": "Market not open yet"}
            end = f"{date_str} {min(now, now.replace(hour=15, minute=30)).strftime('%H:%M')}"
        else:
            end = f"{date_str} 15:30"

        interval_labels = {
            "ONE_MINUTE": "1m", "THREE_MINUTE": "3m", "FIVE_MINUTE": "5m",
            "TEN_MINUTE": "10m", "FIFTEEN_MINUTE": "15m", "THIRTY_MINUTE": "30m",
            "ONE_HOUR": "1h", "ONE_DAY": "1d",
        }

        result = {}
        for interval in intervals:
            label = interval_labels.get(interval, interval)
            raw = self._broker.fetch_candles(token, f"{date_str} 09:15", end, interval)
            result[label] = raw or []

        return result

    def fetch_batch_ltp(self, symbols: List[str]) -> List[dict]:
        """
        Fetch LTP for multiple symbols in ONE API call (up to 50).
        Returns list of dicts with: symbol, ltp, open, high, low, prev_close, volume, percentChange.
        """
        self._ensure_broker()
        from trading.services.ticker_service import ticker_service

        tokens = []
        token_map = {}
        for sym in symbols[:50]:  # API limit is 50
            tok = ticker_service.get_token(sym)
            if tok:
                tokens.append(tok)
                token_map[tok] = sym

        if not tokens:
            return []

        fetched = self._broker.market_data_batch({"NSE": tokens}, mode="FULL")

        results = []
        for item in fetched:
            tok = str(item.get("symbolToken", ""))
            sym = token_map.get(tok)
            if not sym:
                continue
            results.append({
                "symbol": sym,
                "ltp": float(item.get("ltp", 0)),
                "open": float(item.get("open", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "prev_close": float(item.get("close", 0)),
                "volume": int(item.get("tradeVolume", 0)),
                "oi": int(item.get("opnInterest", 0)),
                "pct_change": float(item.get("percentChange", 0)),
                "low_52w": float(item.get("52WeekLow", 0)),
                "high_52w": float(item.get("52WeekHigh", 0)),
                "upper_circuit": float(item.get("upperCircuit", 0)),
                "lower_circuit": float(item.get("lowerCircuit", 0)),
            })

        return results

    def fetch_holdings(self) -> List[Dict]:
        """Fetch current broker holdings."""
        self._ensure_broker()
        return self._broker.fetch_holdings()

    def fetch_positions(self) -> Dict:
        """Fetch open positions."""
        self._ensure_broker()
        return self._broker.fetch_positions()
