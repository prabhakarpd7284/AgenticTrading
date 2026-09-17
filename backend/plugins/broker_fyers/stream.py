"""Fyers v3 live tick socket — real-time LTP for the scalp strategy's live mode.

Wraps ``data_ws.FyersDataSocket`` (which runs its own background thread) and
normalises each message into the engine's tick shape. The caller bridges ticks
into asyncio with ``loop.call_soon_threadsafe`` (the socket thread must not touch
the event loop directly).
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class FyersTickStream:
    def __init__(self, app_id: str, access_token: str,
                 on_tick: Callable[[dict], None],
                 on_connect: Callable[[], None] | None = None):
        self._access_token = f"{app_id}:{access_token}"
        self._on_tick = on_tick
        self._on_connect = on_connect
        self._ws = None
        self._symbols: list[str] = []

    def start(self, symbols: list[str], data_type: str = "SymbolUpdate") -> None:
        from fyers_apiv3.FyersWebsocket import data_ws  # type: ignore

        self._symbols = symbols

        def _on_open() -> None:
            try:
                self._ws.subscribe(symbols=self._symbols, data_type=data_type)
            except Exception as e:  # noqa: BLE001
                logger.error("fyers.stream.subscribe_failed: %s", e)
            if self._on_connect:
                self._on_connect()

        def _on_message(msg) -> None:
            try:
                if not isinstance(msg, dict):
                    return
                ltp = msg.get("ltp")
                if ltp is None:
                    return
                self._on_tick({
                    "symbol": msg.get("symbol"),
                    "ltp": float(ltp),
                    "vol": float(msg.get("vol_traded_today") or 0),
                    "ts": float(msg.get("last_traded_time") or 0),
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("fyers.stream.bad_message: %s", e)

        def _on_error(msg) -> None:
            logger.warning("fyers.stream.error: %s", msg)

        self._ws = data_ws.FyersDataSocket(
            access_token=self._access_token, log_path="", litemode=False,
            write_to_file=False, reconnect=True,
            on_connect=_on_open, on_message=_on_message, on_error=_on_error,
        )
        self._ws.connect()

    def stop(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close_connection()
        except Exception as e:  # noqa: BLE001
            logger.debug("fyers.stream.close_failed: %s", e)
        self._ws = None
