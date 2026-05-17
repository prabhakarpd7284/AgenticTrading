"""
Telegram Alert Service — sends screener signals with proper controls.

Controls:
  - Rate limiting: max N alerts per minute, max M per hour
  - Quiet hours: no alerts outside market hours (configurable)
  - Dedup: same signal (symbol+strategy) not repeated within cooldown
  - Kill switch: disable/enable via SystemControl model
  - Digest mode: batch alerts into periodic summaries
  - Per-strategy mute: silence individual strategies

Setup:
  1. Create a Telegram bot via @BotFather → get BOT_TOKEN
  2. Get your chat_id (send /start to the bot, then check getUpdates)
  3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Usage:
    from plugins.strategy_screener.telegram import TelegramAlertService

    alerts = TelegramAlertService()
    alerts.send_signal(signal)         # send a screener signal
    alerts.send_status("Engine up")    # send a status message
    alerts.mute_strategy("VWAP Bounce Long")
    alerts.unmute_strategy("VWAP Bounce Long")
"""
import os
import time
import threading
import urllib.request
import urllib.parse
import json
from collections import deque
from datetime import datetime, time as dt_time
from typing import Optional

from logzero import logger

from plugins.strategy_screener.signals import Signal


class TelegramAlertService:
    """
    Telegram bot integration with rate limiting, dedup, and controls.

    All sends are non-blocking (async via thread pool).
    Failures are logged but never stop the screener.
    """

    def __init__(
        self,
        bot_token: str = None,
        chat_id: str = None,
        max_per_minute: int = 5,
        max_per_hour: int = 30,
        signal_cooldown: int = 300,  # seconds between same symbol+strategy
        quiet_start: dt_time = dt_time(15, 35),  # no alerts after market
        quiet_end: dt_time = dt_time(9, 10),      # no alerts before market
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.signal_cooldown = signal_cooldown
        self.quiet_start = quiet_start
        self.quiet_end = quiet_end

        # Engine reference for chart rendering (set via set_engine)
        self._engine = None

        # Rate limiting
        self._minute_sends: deque = deque()  # timestamps of last sends
        self._hour_sends: deque = deque()

        # Dedup: (symbol, strategy) → last_send_timestamp
        self._last_signal: dict[tuple[str, str], float] = {}

        # Per-strategy muting
        self._muted_strategies: set[str] = set()

        # Kill switch
        self._enabled = True

        # Stats
        self.total_sent = 0
        self.total_dropped = 0
        self.total_errors = 0

        # Digest buffer
        self._digest_buffer: list[Signal] = []
        self._digest_lock = threading.Lock()

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # ──────────────────────────────────────────────
    # Controls
    # ──────────────────────────────────────────────

    def enable(self):
        """Enable alerts."""
        self._enabled = True
        logger.info("Telegram alerts ENABLED")

    def disable(self):
        """Disable alerts (kill switch)."""
        self._enabled = False
        logger.info("Telegram alerts DISABLED")

    def mute_strategy(self, strategy_name: str):
        """Mute a specific strategy."""
        self._muted_strategies.add(strategy_name)
        logger.info(f"Telegram: muted strategy '{strategy_name}'")

    def unmute_strategy(self, strategy_name: str):
        """Unmute a specific strategy."""
        self._muted_strategies.discard(strategy_name)
        logger.info(f"Telegram: unmuted strategy '{strategy_name}'")

    def set_engine(self, engine):
        """Set engine reference for chart rendering in signals."""
        self._engine = engine

    def set_rate_limits(self, per_minute: int = None, per_hour: int = None):
        if per_minute is not None:
            self.max_per_minute = per_minute
        if per_hour is not None:
            self.max_per_hour = per_hour

    # ──────────────────────────────────────────────
    # Sending
    # ──────────────────────────────────────────────

    def send_signal(self, signal: Signal) -> bool:
        """
        Send a screener signal as chart + caption to Telegram.

        Falls back to text-only if chart rendering fails or no engine is set.
        Returns True if sent, False if dropped (rate limit / dedup / muted).
        """
        if not self._should_send(signal):
            self.total_dropped += 1
            return False

        sent = False

        # Try chart + caption
        if self._engine:
            try:
                store = self._engine.stores.get(signal.symbol)
                if store:
                    bars_5m = list(store.bars.get("5m", []))
                    chart_bytes = signal.render_chart_png(bars_5m)
                    if chart_bytes:
                        caption = signal.format_telegram()
                        sent = self.send_chart(chart_bytes, caption=caption, parse_mode="Markdown")
            except Exception as e:
                logger.debug(f"Chart render failed for {signal.symbol}: {e}")

        # Fallback: text-only
        if not sent:
            message = signal.format_telegram()
            sent = self._send_message(message, parse_mode="Markdown")

        if sent:
            self._record_send(signal)
            self.total_sent += 1

        return sent

    def send_status(self, text: str) -> bool:
        """Send a plain status message (not subject to signal rate limits)."""
        if not self._enabled or not self.is_configured:
            return False
        return self._send_message(f"📊 *Status*\n{text}", parse_mode="Markdown")

    def send_digest(self) -> bool:
        """Send buffered signals as a single digest message."""
        with self._digest_lock:
            if not self._digest_buffer:
                return False
            signals = self._digest_buffer.copy()
            self._digest_buffer.clear()

        buys = [s for s in signals if s.side == "BUY"]
        sells = [s for s in signals if s.side == "SELL"]
        now = datetime.now().strftime("%H:%M")

        lines = [
            f"📋 *Screener Digest* — {now}",
            f"`{len(buys)} buys` | `{len(sells)} sells` | {len(signals)} total",
            "",
        ]

        def _fmt(s):
            arrow = "🟢" if s.side == "BUY" else "🔴"
            conf = round(s.confidence * 100)
            return (
                f"{arrow} *{s.symbol}* `{s.entry:.0f}` → `{s.target:.0f}`"
                f"  R:R *{s.risk_reward:.1f}* | {conf}%"
                f"\n     _{s.strategy}_"
            )

        if buys:
            lines.append("*— BUY —*")
            for s in sorted(buys, key=lambda x: x.confidence, reverse=True):
                lines.append(_fmt(s))
            lines.append("")

        if sells:
            lines.append("*— SELL —*")
            for s in sorted(sells, key=lambda x: x.confidence, reverse=True):
                lines.append(_fmt(s))

        return self._send_message("\n".join(lines), parse_mode="Markdown")

    def buffer_for_digest(self, signal: Signal):
        """Add signal to digest buffer instead of sending immediately."""
        with self._digest_lock:
            self._digest_buffer.append(signal)

    # ──────────────────────────────────────────────
    # Checks
    # ──────────────────────────────────────────────

    def _should_send(self, signal: Signal) -> bool:
        """Check all controls before sending."""
        if not self._enabled:
            return False
        if not self.is_configured:
            return False

        # Kill switch from DB
        if self._is_killed():
            return False

        # Quiet hours
        now = datetime.now().time()
        if self.quiet_start <= now or now <= self.quiet_end:
            return False

        # Strategy muted
        if signal.strategy in self._muted_strategies:
            return False

        # Dedup
        key = (signal.symbol, signal.strategy)
        last = self._last_signal.get(key, 0)
        if time.monotonic() - last < self.signal_cooldown:
            return False

        # Rate limit — per minute
        now_mono = time.monotonic()
        while self._minute_sends and now_mono - self._minute_sends[0] > 60:
            self._minute_sends.popleft()
        if len(self._minute_sends) >= self.max_per_minute:
            return False

        # Rate limit — per hour
        while self._hour_sends and now_mono - self._hour_sends[0] > 3600:
            self._hour_sends.popleft()
        if len(self._hour_sends) >= self.max_per_hour:
            return False

        return True

    def _record_send(self, signal: Signal):
        """Record this send for rate limiting and dedup."""
        now = time.monotonic()
        self._minute_sends.append(now)
        self._hour_sends.append(now)
        self._last_signal[(signal.symbol, signal.strategy)] = now

    def _is_killed(self) -> bool:
        """Check SystemControl for kill switch (non-blocking).

        Tenant-aware SystemControl row stores ``value`` as JSON (formerly a
        plain string in the legacy `trading.SystemControl`). A row with
        either ``"disabled"`` or ``{"disabled": true}`` / ``{"enabled":
        false}`` halts alerts. Matches across all tenants — same semantics
        as `apps.system.services.flags.get_flag` when no tenant is
        supplied: a single trader pressing the kill switch silences the
        screener regardless of which tenant they're in.
        """
        try:
            from apps.system.models import SystemControl
            for ctrl in SystemControl.objects.filter(key="screener_alerts").only("value"):
                v = ctrl.value
                if v == "disabled":
                    return True
                if isinstance(v, dict):
                    if v.get("disabled") is True:
                        return True
                    if v.get("enabled") is False:
                        return True
        except Exception:
            pass
        return False

    # ──────────────────────────────────────────────
    # HTTP
    # ──────────────────────────────────────────────

    def _send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message via Telegram Bot API (non-blocking)."""
        thread = threading.Thread(
            target=self._do_send, args=(text, parse_mode), daemon=True
        )
        thread.start()
        return True

    def send_chart(self, image_bytes: bytes, caption: str = "", parse_mode: str = "Markdown") -> bool:
        """Send a chart image to Telegram (blocking)."""
        if not self.is_configured:
            return False
        try:
            import io
            boundary = "----TgChart"
            body = io.BytesIO()

            # chat_id field
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{self.chat_id}\r\n'.encode())

            # caption field
            if caption:
                body.write(f"--{boundary}\r\n".encode())
                body.write(f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n'.encode())
                body.write(f"--{boundary}\r\n".encode())
                body.write(f'Content-Disposition: form-data; name="parse_mode"\r\n\r\n{parse_mode}\r\n'.encode())

            # photo field
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n'.encode())
            body.write(f"Content-Type: image/png\r\n\r\n".encode())
            body.write(image_bytes)
            body.write(f"\r\n--{boundary}--\r\n".encode())

            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            req = urllib.request.Request(
                url, data=body.getvalue(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=15)
            return resp.status == 200
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Telegram photo send error: {e}")
            return False

    def _do_send(self, text: str, parse_mode: str):
        """Actual HTTP POST to Telegram (runs in thread)."""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            if resp.status != 200:
                self.total_errors += 1
                logger.error(f"Telegram send failed: HTTP {resp.status}")
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Telegram send error: {e}")

    # ──────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "configured": self.is_configured,
            "enabled": self._enabled,
            "total_sent": self.total_sent,
            "total_dropped": self.total_dropped,
            "total_errors": self.total_errors,
            "muted_strategies": list(self._muted_strategies),
            "digest_buffer_size": len(self._digest_buffer),
        }
