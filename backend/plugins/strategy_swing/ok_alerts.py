"""
Oliver Kell Telegram Alerts — phase transition notifications.

Reuses the same TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars
as the intraday screener alerts.
"""
import os
import threading
import urllib.request
import urllib.parse
import json
from typing import List

from logzero import logger

from plugins.strategy_swing.ok_cycles import (
    BULLISH_ACTIONABLE,
    BEARISH_ACTIONABLE,
    CyclePhase,
    CycleResult,
    TrendState,
)


# Phase → emoji mapping
PHASE_EMOJI = {
    CyclePhase.REVERSAL_EXTENSION: "🔄",
    CyclePhase.WEDGE_POP: "🚀",
    CyclePhase.EMA_CROSSBACK_BULL: "📈",
    CyclePhase.BASIN_BREAK_BULL: "💥",
    CyclePhase.EXHAUSTION_EXTENSION: "⚠️",
    CyclePhase.WEDGE_DROP: "📉",
    CyclePhase.EMA_CROSSBACK_BEAR: "🔻",
    CyclePhase.BASIN_BREAK_BEAR: "💣",
    CyclePhase.NONE: "•",
}

PHASE_NAME = {
    CyclePhase.REVERSAL_EXTENSION: "Reversal Extension",
    CyclePhase.WEDGE_POP: "Wedge Pop",
    CyclePhase.EMA_CROSSBACK_BULL: "EMA Crossback",
    CyclePhase.BASIN_BREAK_BULL: "Basin Break",
    CyclePhase.EXHAUSTION_EXTENSION: "Exhaustion Extension",
    CyclePhase.WEDGE_DROP: "Wedge Drop",
    CyclePhase.EMA_CROSSBACK_BEAR: "Bear EMA Crossback",
    CyclePhase.BASIN_BREAK_BEAR: "Bear Basin Break",
    CyclePhase.NONE: "No Phase",
}

TREND_EMOJI = {
    TrendState.BULLISH: "🟢",
    TrendState.BEARISH: "🔴",
    TrendState.NEUTRAL: "⚪",
}


def _format_alert(result: CycleResult) -> str:
    """Format a single CycleResult as a Telegram message."""
    emoji = PHASE_EMOJI.get(result.phase, "•")
    phase_name = PHASE_NAME.get(result.phase, result.phase.value)
    trend_d = TREND_EMOJI.get(result.trend_daily, "⚪")
    trend_w = TREND_EMOJI.get(result.trend_weekly, "⚪")

    aligned_str = "✅ Aligned" if result.aligned else "❌ Not aligned"

    lines = [
        f"{emoji} *{result.symbol}* — {phase_name} ({result.phase.value})",
        f"Action: *{result.action}* | Confidence: {result.confidence:.0%}",
        f"Trend: {trend_d} Daily {result.trend_daily.value} | {trend_w} Weekly {result.trend_weekly.value}",
        f"{aligned_str}",
        f"Close: ₹{result.last_close:,.2f}",
        f"EMA10: ₹{result.ema_fast:,.2f} | EMA20: ₹{result.ema_mid:,.2f} | EMA50: ₹{result.ema_slow:,.2f}",
    ]

    if result.volume_ratio > 0:
        lines.append(f"Volume: {result.volume_ratio:.1f}x avg")

    return "\n".join(lines)


def _format_digest(results: List[CycleResult]) -> str:
    """Format a daily digest of all actionable results."""
    buy_results = [r for r in results if r.phase in BULLISH_ACTIONABLE]
    short_results = [r for r in results if r.phase in BEARISH_ACTIONABLE]
    watch_results = [r for r in results if r.phase == CyclePhase.REVERSAL_EXTENSION]

    lines = ["📊 *Oliver Kell Daily Scan*\n"]

    if buy_results:
        lines.append("*🟢 BUY Setups:*")
        for r in buy_results:
            aligned = "✅" if r.aligned else ""
            lines.append(
                f"  {PHASE_EMOJI[r.phase]} {r.symbol} — {r.phase.value} "
                f"₹{r.last_close:,.0f} ({r.confidence:.0%}) {aligned}"
            )
        lines.append("")

    if short_results:
        lines.append("*🔴 SHORT/AVOID:*")
        for r in short_results:
            lines.append(
                f"  {PHASE_EMOJI[r.phase]} {r.symbol} — {r.phase.value} "
                f"₹{r.last_close:,.0f}"
            )
        lines.append("")

    if watch_results:
        lines.append("*🔄 WATCH (Potential Bottoms):*")
        for r in watch_results:
            lines.append(f"  {r.symbol} — ₹{r.last_close:,.0f}")
        lines.append("")

    # Summary
    total = len(results)
    phases = {}
    for r in results:
        phases[r.phase.value] = phases.get(r.phase.value, 0) + 1

    lines.append(f"_Scanned {total} stocks_")
    phase_str = " | ".join(f"{k}: {v}" for k, v in sorted(phases.items()) if k != "NONE")
    if phase_str:
        lines.append(f"_Phases: {phase_str}_")

    return "\n".join(lines)


class OKAlertService:
    """Sends Oliver Kell phase alerts via Telegram."""

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_alert(self, result: CycleResult) -> bool:
        """Send a single phase alert."""
        if not self.is_configured:
            logger.warning("Telegram not configured — skipping OK alert")
            return False
        message = _format_alert(result)
        return self._send(message)

    def send_digest(self, results: List[CycleResult]) -> bool:
        """Send a daily digest of all results."""
        if not self.is_configured:
            logger.warning("Telegram not configured — skipping OK digest")
            return False

        # Only include results with detected phases
        active = [r for r in results if r.phase != CyclePhase.NONE]
        if not active:
            logger.info("No active OK phases — skipping digest")
            return False

        message = _format_digest(active)
        return self._send(message)

    def _send(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message via Telegram Bot API (non-blocking)."""
        thread = threading.Thread(
            target=self._do_send, args=(text, parse_mode), daemon=True
        )
        thread.start()
        return True

    def _do_send(self, text: str, parse_mode: str):
        """Actual HTTP send (runs in background thread)."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram API returned {resp.status}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
