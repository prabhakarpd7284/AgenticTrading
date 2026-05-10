"""
Signal — the output of the screener when conditions are met.

Contains everything needed to act on a trade opportunity:
entry, SL, target, R:R, reasons, and indicator context.
"""
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Signal:
    """A trade opportunity detected by the screener."""
    timestamp: datetime
    symbol: str
    strategy: str
    side: str                         # "BUY" | "SELL"
    entry: float                      # entry price
    stoploss: float                   # SL price
    target: float                     # target price
    risk_reward: float                # target_dist / sl_dist
    risk_points: float                # abs(entry - SL)
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0           # 0-1

    # Context for dashboard / review
    indicators: dict = field(default_factory=dict)
    timeframes_checked: list[str] = field(default_factory=list)

    @property
    def target_points(self) -> float:
        return abs(self.target - self.entry)

    @property
    def risk_pct(self) -> float:
        """Risk as % of entry price."""
        return round(self.risk_points / self.entry * 100, 2) if self.entry > 0 else 0

    def format_cli(self) -> str:
        """Compact CLI output."""
        arrow = "▲" if self.side == "BUY" else "▼"
        return (
            f"{arrow} {self.symbol} | {self.strategy} | {self.side}\n"
            f"  Entry: {self.entry:.2f} | SL: {self.stoploss:.2f} | "
            f"Target: {self.target:.2f} | R:R {self.risk_reward:.1f}\n"
            f"  Risk: {self.risk_points:.1f} pts ({self.risk_pct}%) | "
            f"Confidence: {self.confidence:.0%}\n"
            f"  Reasons: {' + '.join(self.reasons)}"
        )

    def format_telegram(self) -> str:
        """Rich Telegram message format (MarkdownV2-safe, uses Markdown)."""
        is_buy = self.side == "BUY"
        arrow = "🟢" if is_buy else "🔴"
        side_word = "LONG" if is_buy else "SHORT"

        # Confidence bar: ████░░░░ 75%
        filled = round(self.confidence * 8)
        conf_bar = "█" * filled + "░" * (8 - filled)

        # R:R visual
        rr_stars = "⭐" * min(int(self.risk_reward), 5)

        # Strategy tag
        strat_tag = self.strategy.upper().replace(" ", "_")

        # Indicator context (compact)
        ind = self.indicators or {}
        ctx_parts = []
        if ind.get("vwap"):
            vw = ind["vwap"]
            pos = "above" if self.entry > vw else "below"
            ctx_parts.append(f"VWAP `{vw:.0f}` ({pos})")
        if ind.get("rsi_14"):
            ctx_parts.append(f"RSI `{ind['rsi_14']:.0f}`")
        if ind.get("atr_14"):
            ctx_parts.append(f"ATR `{ind['atr_14']:.1f}`")
        ctx_line = " • ".join(ctx_parts) if ctx_parts else ""

        # Price levels with visual direction
        if is_buy:
            levels = (
                f"🎯 Target  `{self.target:>10,.2f}`  (+{self.target_points:.1f})\n"
                f"➡️ Entry   `{self.entry:>10,.2f}`\n"
                f"🛑 SL      `{self.stoploss:>10,.2f}`  (-{self.risk_points:.1f})"
            )
        else:
            levels = (
                f"🛑 SL      `{self.stoploss:>10,.2f}`  (-{self.risk_points:.1f})\n"
                f"➡️ Entry   `{self.entry:>10,.2f}`\n"
                f"🎯 Target  `{self.target:>10,.2f}`  (+{self.target_points:.1f})"
            )

        # Time
        ts = self.timestamp.strftime("%H:%M") if self.timestamp else ""

        lines = [
            f"{arrow} *{self.symbol}* — {side_word}",
            f"`#{strat_tag}`  🕐 {ts}",
            "",
            levels,
            "",
            f"R:R *{self.risk_reward:.1f}x* {rr_stars}  |  Risk *{self.risk_pct}%*",
            f"Confidence `{conf_bar}` *{self.confidence:.0%}*",
        ]

        if ctx_line:
            lines.append(f"📊 {ctx_line}")

        return "\n".join(lines)

    def render_chart_png(self, candles_5m: list = None) -> Optional[bytes]:
        """
        Render a mini candlestick chart with entry/SL/target lines.
        Returns PNG bytes or None if no data.

        candles_5m: list of CandleBar or dicts with open/high/low/close/volume/timestamp
        """
        if not candles_5m or len(candles_5m) < 3:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 4), gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
            fig.patch.set_facecolor("#0e1117")
            ax1.set_facecolor("#0e1117")
            ax2.set_facecolor("#0e1117")

            n = len(candles_5m)
            for i, c in enumerate(candles_5m):
                o = c.open if hasattr(c, 'open') else c["open"]
                h = c.high if hasattr(c, 'high') else c["high"]
                lo = c.low if hasattr(c, 'low') else c["low"]
                cl = c.close if hasattr(c, 'close') else c["close"]
                vol = c.volume if hasattr(c, 'volume') else c.get("volume", 0)
                color = "#26a69a" if cl >= o else "#ef5350"
                ax1.plot([i, i], [lo, h], color=color, linewidth=0.7)
                ax1.bar(i, abs(cl - o) or 0.01, bottom=min(o, cl), width=0.6, color=color)
                ax2.bar(i, vol, color=color, alpha=0.5, width=0.6)

            # Entry / SL / Target lines
            ax1.axhline(y=self.entry, color="#ffffff", linewidth=1.2, linestyle="-", alpha=0.9)
            ax1.text(n + 0.3, self.entry, f"E {self.entry:.1f}", fontsize=7, color="#fff", va="center")

            ax1.axhline(y=self.stoploss, color="#ef5350", linewidth=1, linestyle="--", alpha=0.8)
            ax1.text(n + 0.3, self.stoploss, f"SL {self.stoploss:.1f}", fontsize=7, color="#ef5350", va="center")

            ax1.axhline(y=self.target, color="#26a69a", linewidth=1, linestyle="--", alpha=0.8)
            ax1.text(n + 0.3, self.target, f"T {self.target:.1f}", fontsize=7, color="#26a69a", va="center")

            # VWAP line
            vwap_val = (self.indicators or {}).get("vwap", 0)
            if vwap_val > 0:
                ax1.axhline(y=vwap_val, color="#ff9800", linewidth=0.8, linestyle=":", alpha=0.6)
                ax1.text(n + 0.3, vwap_val, f"V", fontsize=6, color="#ff9800", va="center")

            # Mark the signal bar
            signal_bar_idx = n - 1  # last bar is the trigger
            ax1.axvline(x=signal_bar_idx, color="#ffffff", linewidth=0.3, alpha=0.3)

            side_color = "#26a69a" if self.side == "BUY" else "#ef5350"
            ax1.set_title(
                f"{self.symbol} — {self.side} @ {self.timestamp.strftime('%H:%M') if self.timestamp else ''}",
                color=side_color, fontsize=10, fontweight="bold",
            )

            for ax in (ax1, ax2):
                ax.tick_params(colors="white", labelsize=6)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("#333")
                ax.spines["left"].set_color("#333")

            # X-axis labels
            timestamps = []
            for c in candles_5m:
                ts = c.timestamp if hasattr(c, 'timestamp') else c.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("+05:30", ""))
                timestamps.append(ts)

            if timestamps and timestamps[0]:
                step = max(1, n // 8)
                positions = list(range(0, n, step))
                labels = [timestamps[i].strftime("%H:%M") if timestamps[i] else "" for i in positions]
                ax2.set_xticks(positions)
                ax2.set_xticklabels(labels, rotation=0, ha="center")

            ax2.set_ylabel("Vol", color="white", fontsize=7)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf.read()
        except Exception:
            return None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": self.side,
            "entry": self.entry,
            "stoploss": self.stoploss,
            "target": self.target,
            "risk_reward": self.risk_reward,
            "risk_points": self.risk_points,
            "risk_pct": self.risk_pct,
            "reasons": self.reasons,
            "confidence": self.confidence,
            "indicators": self.indicators,
        }

    def persist(self, source: str = "SCREENER"):
        """
        Save this signal to SignalLog for post-hoc analysis.
        Non-blocking — log failures don't stop trading flow.
        """
        try:
            from trading.models import SignalLog
            SignalLog.objects.create(
                symbol=self.symbol,
                signal_date=self.timestamp.date(),
                signal_time=self.timestamp,
                source=source,
                strategy=self.strategy,
                side=self.side,
                entry_price=self.entry,
                stoploss=self.stoploss,
                target=self.target,
                confidence=self.confidence,
                risk_reward=self.risk_reward,
                reasons=self.reasons,
                indicators=self.indicators,
            )
        except Exception as e:
            from logzero import logger
            logger.warning(f"SignalLog persist failed (non-blocking): {e}")
