"""
Tick Stream — live data via Angel One SmartWebSocketV2 + REST polling fallback.

WebSocket (V2):
  - URL: wss://smartapisocket.angelone.in/smart-stream
  - Mode 2 (QUOTE): LTP + OHLC + volume per tick — builds candles
  - Binary protocol with automatic parsing
  - Up to 50 tokens per subscribe call, multiple connections for >50
  - Auto-reconnect on disconnect

Polling fallback:
  - market_data_batch() every N seconds, 50 per batch
"""
import json
import os
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from logzero import logger


class TickStream:
    """
    Live tick data from Angel One via WebSocket V2.
    Falls back to REST polling if websocket fails.
    """

    def __init__(
        self,
        symbols: List[str],
        on_tick: Callable[[str, float, int], None],
        poll_interval: float = 5.0,
    ):
        self.symbols = symbols
        self.on_tick = on_tick
        self.poll_interval = poll_interval

        self._running = False
        self._mode = "idle"
        self._threads: list[threading.Thread] = []

        self._token_to_symbol: Dict[str, str] = {}
        self._symbol_to_token: Dict[str, str] = {}

        # Stats
        self._tick_count = 0
        self._ws_errors = 0
        self._last_tick_time = 0.0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start websocket first, polling fallback."""
        if self._running:
            return
        self._running = True
        self._resolve_tokens()

        if self._try_websocket_v2():
            return
        logger.warning("WebSocket V2 unavailable, falling back to REST polling")
        self._start_polling()

    def stop(self):
        self._running = False
        self._mode = "idle"
        logger.info(f"Tick stream stopped (ticks: {self._tick_count})")

    def _resolve_tokens(self):
        from trading.services.ticker_service import ticker_service
        for sym in self.symbols:
            token = ticker_service.get_token(sym)
            if token:
                self._token_to_symbol[token] = sym
                self._symbol_to_token[sym] = token
        logger.info(f"Resolved {len(self._token_to_symbol)}/{len(self.symbols)} symbols")

    # ──────────────────────────────────────────────
    # WebSocket V2 (binary protocol, QUOTE mode)
    # ──────────────────────────────────────────────

    def _try_websocket_v2(self) -> bool:
        """Connect via SmartWebSocketV2. Returns True if started."""
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
            from trading.services.data_service import BrokerClient

            broker = BrokerClient.get_instance()
            broker.ensure_login()

            api = broker.smart_api
            auth_token = getattr(api, 'jwtToken', None) or getattr(api, 'access_token', None)
            feed_token = getattr(api, 'feed_token', None)
            api_key = os.getenv("SMARTAPI_KEY", "")
            client_code = broker.username

            if not all([auth_token, feed_token, api_key, client_code]):
                logger.warning(f"Missing WS V2 creds: auth={bool(auth_token)} feed={bool(feed_token)} key={bool(api_key)}")
                return False

            all_tokens = list(self._token_to_symbol.keys())
            if not all_tokens:
                return False

            # V2 supports subscribe in batches — one connection handles all
            # but subscribe call is limited to ~50 tokens per list entry
            BATCH = 50
            token_batches = [all_tokens[i:i + BATCH] for i in range(0, len(all_tokens), BATCH)]

            logger.info(f"WebSocket V2: {len(all_tokens)} symbols, {len(token_batches)} batch(es)")

            sws = SmartWebSocketV2(
                auth_token, api_key, client_code, feed_token,
                max_retry_attempt=5, retry_strategy=0, retry_delay=5,
            )

            def on_data(wsapp, data):
                try:
                    token = str(data.get("token", "")).strip()
                    # V2 prices are in paise — divide by 100
                    ltp_raw = data.get("last_traded_price", 0)
                    ltp = ltp_raw / 100.0 if ltp_raw > 1000 else float(ltp_raw)

                    volume = int(data.get("volume_trade_for_the_day", 0))

                    symbol = self._token_to_symbol.get(token)
                    if symbol and ltp > 0 and self._running:
                        self._tick_count += 1
                        self._last_tick_time = time.monotonic()
                        self.on_tick(symbol, ltp, volume)
                except Exception as e:
                    logger.debug(f"WS V2 parse: {e}")

            def on_open(wsapp):
                logger.info("WebSocket V2 connected — subscribing in QUOTE mode")
                for batch in token_batches:
                    token_list = [{"exchangeType": 1, "tokens": batch}]
                    sws.subscribe("screener", 2, token_list)  # mode 2 = QUOTE
                    logger.info(f"  Subscribed batch of {len(batch)} tokens")

            def on_error(wsapp, error):
                self._ws_errors += 1
                logger.error(f"WS V2 error: {str(error)[:200]}")

            def on_close(wsapp):
                logger.warning("WS V2 closed")
                if self._running:
                    logger.info("WS V2 will auto-reconnect")

            def on_control_message(wsapp, msg):
                pass  # heartbeats handled internally

            sws.on_data = on_data
            sws.on_open = on_open
            sws.on_error = on_error
            sws.on_close = on_close
            sws.on_control_message = on_control_message

            # Suppress V2's noisy internal logging
            sws.on_message = lambda ws, msg: None

            t = threading.Thread(target=self._ws_v2_run, args=(sws,), daemon=True)
            t.start()
            self._threads.append(t)
            self._mode = "websocket"
            return True

        except ImportError as e:
            logger.warning(f"SmartWebSocketV2 import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"WS V2 setup failed: {e}")
            return False

    def _ws_v2_run(self, sws):
        """Run websocket V2 in thread. Blocks until disconnect."""
        try:
            sws.connect()
        except Exception as e:
            logger.error(f"WS V2 fatal: {e}")
            if self._running:
                logger.info("Falling back to polling")
                self._start_polling()

    # ──────────────────────────────────────────────
    # Polling fallback
    # ──────────────────────────────────────────────

    def _start_polling(self):
        if self._mode == "polling":
            return
        self._mode = "polling"
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        self._threads.append(t)
        logger.info(f"REST polling started ({self.poll_interval}s interval)")

    def _poll_loop(self):
        from trading.services.data_service import BrokerClient

        broker = BrokerClient.get_instance()
        broker.ensure_login()

        tokens = list(self._token_to_symbol.keys())
        if not tokens:
            logger.error("No tokens to poll")
            return

        BATCH_SIZE = 50
        token_batches = [tokens[i:i + BATCH_SIZE] for i in range(0, len(tokens), BATCH_SIZE)]
        logger.info(f"Polling {len(tokens)} symbols in {len(token_batches)} batch(es)")

        while self._running:
            try:
                from trading.utils.time_utils import is_market_open
                if not is_market_open():
                    time.sleep(30)
                    continue

                for batch in token_batches:
                    if not self._running:
                        break
                    try:
                        fetched = broker.market_data_batch({"NSE": batch}, mode="FULL")
                        for item in fetched:
                            token = str(item.get("symbolToken", ""))
                            symbol = self._token_to_symbol.get(token)
                            if symbol:
                                ltp = float(item.get("ltp", 0))
                                volume = int(item.get("tradeVolume", 0))
                                if ltp > 0:
                                    self._tick_count += 1
                                    self._last_tick_time = time.monotonic()
                                    self.on_tick(symbol, ltp, volume)
                    except Exception as e:
                        logger.error(f"Batch poll error: {e}")

            except Exception as e:
                logger.error(f"Poll loop error: {e}")

            time.sleep(self.poll_interval)

    def get_stats(self) -> dict:
        return {
            "mode": self._mode,
            "running": self._running,
            "symbols_resolved": len(self._token_to_symbol),
            "symbols_total": len(self.symbols),
            "ticks_received": self._tick_count,
            "ws_errors": self._ws_errors,
            "last_tick_age_s": round(time.monotonic() - self._last_tick_time, 1) if self._last_tick_time else None,
        }
