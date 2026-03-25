"""
Reusable Streamlit UI components — TradingView embed, risk gauges,
payoff diagram, P&L cards.
"""
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────
# TradingView Chart Embed
# ──────────────────────────────────────────────
def render_tradingview_chart(symbol: str, exchange: str = "NSE", height: int = 450):
    """
    Embed a TradingView lightweight chart for the given NSE symbol.
    Uses TradingView's embed widget (no API key needed).
    """
    tv_symbol = f"{exchange}:{symbol}"

    html = f"""
    <div id="tradingview_widget" style="height: {height}px;">
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:100%;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "15",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "allow_symbol_change": true,
        "calendar": false,
        "support_host": "https://www.tradingview.com",
        "studies": [
          "STD;RSI",
          "STD;SMA"
        ]
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    </div>
    """
    st.components.v1.html(html, height=height + 20)


def render_tradingview_mini(symbol: str, exchange: str = "NSE", height: int = 200):
    """Smaller TradingView widget for scanner view."""
    tv_symbol = f"{exchange}:{symbol}"

    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
      {{
        "symbol": "{tv_symbol}",
        "width": "100%",
        "height": "{height}",
        "locale": "en",
        "dateRange": "1D",
        "colorTheme": "dark",
        "isTransparent": true,
        "autosize": true
      }}
      </script>
    </div>
    """
    st.components.v1.html(html, height=height + 10)


# ──────────────────────────────────────────────
# Risk Gauge
# ──────────────────────────────────────────────
def render_risk_gauge(label: str, current: float, maximum: float, unit: str = "%"):
    """Render a progress-bar style risk gauge with color coding."""
    pct = (current / maximum * 100) if maximum > 0 else 0
    pct = min(pct, 100)

    if pct >= 80:
        color = "red"
    elif pct >= 50:
        color = "orange"
    else:
        color = "green"

    st.markdown(f"**{label}**")
    st.progress(pct / 100)
    st.caption(f"{current:.1f}{unit} / {maximum:.1f}{unit} ({pct:.0f}%)")


# ──────────────────────────────────────────────
# Payoff Diagram (Straddle)
# ──────────────────────────────────────────────
def render_payoff_diagram(
    strike: int,
    ce_sell: float,
    pe_sell: float,
    lot_size: int,
    lots: int,
    nifty_spot: float,
):
    """Generate and display a matplotlib short straddle payoff diagram."""
    total_lots = lot_size * lots
    premium_received = (ce_sell + pe_sell) * total_lots

    # Generate NIFTY range
    low = strike - 800
    high = strike + 800
    nifty_levels = np.linspace(low, high, 200)

    # P&L at each level
    pnl = []
    for level in nifty_levels:
        ce_value = max(0, level - strike)
        pe_value = max(0, strike - level)
        net = ((ce_sell - ce_value) + (pe_sell - pe_value)) * total_lots
        pnl.append(net)

    pnl = np.array(pnl)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    # Plot P&L line
    ax.fill_between(nifty_levels, pnl, 0, where=(pnl >= 0), color="#00C853", alpha=0.3)
    ax.fill_between(nifty_levels, pnl, 0, where=(pnl < 0), color="#FF1744", alpha=0.3)
    ax.plot(nifty_levels, pnl, color="white", linewidth=1.5)

    # Zero line
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")

    # Mark strike and current spot
    ax.axvline(x=strike, color="#FFC107", linewidth=1, linestyle="--", label=f"Strike {strike}")
    ax.axvline(x=nifty_spot, color="#2196F3", linewidth=1, linestyle="--", label=f"NIFTY {nifty_spot:.0f}")

    # Breakeven points
    upper_be = strike + ce_sell + pe_sell
    lower_be = strike - ce_sell - pe_sell
    ax.axvline(x=upper_be, color="#FF9800", linewidth=0.8, linestyle=":", label=f"Upper BE {upper_be:.0f}")
    ax.axvline(x=lower_be, color="#FF9800", linewidth=0.8, linestyle=":", label=f"Lower BE {lower_be:.0f}")

    ax.set_xlabel("NIFTY at Expiry", color="white", fontsize=10)
    ax.set_ylabel("P&L (INR)", color="white", fontsize=10)
    ax.set_title("Short Straddle Payoff at Expiry", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.legend(loc="lower left", fontsize=8, facecolor="#1E1E1E", edgecolor="gray", labelcolor="white")
    ax.grid(True, alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────
# Premium Decay Chart
# ──────────────────────────────────────────────
def render_premium_decay_chart(snapshots: list):
    """
    Plot combined premium over time (from session_state snapshots).
    snapshots: list of {"time": str, "premium": float}
    """
    if len(snapshots) < 2:
        st.info("Need at least 2 data points for premium decay chart. Auto-refresh will collect them.")
        return

    times = [s["time"] for s in snapshots]
    premiums = [s["premium"] for s in snapshots]

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.plot(times, premiums, color="#00BCD4", linewidth=2, marker="o", markersize=3)
    ax.fill_between(times, premiums, alpha=0.2, color="#00BCD4")

    ax.set_xlabel("Time", color="white", fontsize=9)
    ax.set_ylabel("Combined Premium (pts)", color="white", fontsize=9)
    ax.set_title("Premium Decay Tracker", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, alpha=0.15)

    # Show every nth label to avoid crowding
    n = max(1, len(times) // 10)
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % n != 0:
            label.set_visible(False)

    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────
# P&L Bar Chart (per symbol)
# ──────────────────────────────────────────────
def render_symbol_pnl_chart(symbols_data: dict):
    """Bar chart of P&L by symbol."""
    if not symbols_data:
        st.info("No trade data for P&L chart.")
        return

    syms = list(symbols_data.keys())
    pnls = [symbols_data[s]["total_pnl"] for s in syms]

    colors = ["#00C853" if p >= 0 else "#FF1744" for p in pnls]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    bars = ax.bar(syms, pnls, color=colors, alpha=0.85)

    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_ylabel("P&L (INR)", color="white", fontsize=10)
    ax.set_title("P&L by Symbol (last 30 days)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.15)

    for bar, val in zip(bars, pnls):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:+,.0f}", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=8, color="white",
        )

    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────
# Alert Banner
# ──────────────────────────────────────────────
def render_alerts(alerts: list):
    """Render alert banners with severity-based styling."""
    if not alerts:
        st.success("No active alerts")
        return

    for alert in alerts:
        severity = alert.get("severity", "info")
        msg = alert.get("message", "")
        action = alert.get("action", "")

        if severity == "critical":
            st.error(f"**{msg}** — {action}")
        elif severity == "warning":
            st.warning(f"**{msg}** — {action}")
        else:
            st.info(f"**{msg}** — {action}")


# ──────────────────────────────────────────────
# Custom Indicator Chart (candlestick + Camarilla + BB + VWAP)
# ──────────────────────────────────────────────
def render_indicator_chart(
    candles: list,
    symbol: str,
    pivots: dict = None,
    bb: dict = None,
    vwap_val: float = 0,
    ema9: float = 0,
    ema20: float = 0,
    entry: float = 0,
    sl: float = 0,
    target: float = 0,
    height: int = 8,
):
    """
    Custom candlestick chart with our indicators overlaid.

    candles: list of dicts with open, high, low, close, volume
    pivots: Camarilla pivots dict {S3, S4, R3, R4, P}
    """
    if not candles or len(candles) < 3:
        st.info("Not enough candle data for chart.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, height), gridspec_kw={"height_ratios": [3, 1]},
                                     sharex=True)
    fig.patch.set_facecolor("#0e1117")
    ax1.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")

    n = len(candles)
    x = list(range(n))

    # Candlesticks
    for i, c in enumerate(candles):
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        color = "#26a69a" if cl >= o else "#ef5350"
        # Wick
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
        # Body
        body_bottom = min(o, cl)
        body_height = abs(cl - o) or 0.01
        ax1.bar(i, body_height, bottom=body_bottom, width=0.6, color=color, edgecolor=color)

    # Camarilla pivots
    if pivots:
        for level, color, style in [
            ("S4", "#ef5350", "--"), ("S3", "#ef5350", "-"),
            ("P", "#ffffff", ":"),
            ("R3", "#26a69a", "-"), ("R4", "#26a69a", "--"),
        ]:
            val = pivots.get(level, 0)
            if val > 0:
                ax1.axhline(y=val, color=color, linestyle=style, linewidth=0.7, alpha=0.6)
                ax1.text(n + 0.5, val, f"{level} {val:.0f}", fontsize=7, color=color, va="center")

    # Bollinger Bands
    if bb and bb.get("upper", 0) > 0:
        closes = [c["close"] for c in candles]
        # Compute rolling BB for each point
        period = min(20, len(closes))
        import math
        uppers, lowers, mids = [], [], []
        for i in range(len(closes)):
            window = closes[max(0, i - period + 1):i + 1]
            mid = sum(window) / len(window)
            std = math.sqrt(sum((x - mid) ** 2 for x in window) / len(window))
            uppers.append(mid + 2 * std)
            lowers.append(mid - 2 * std)
            mids.append(mid)
        ax1.plot(x, uppers, color="#2196f3", linewidth=0.6, alpha=0.5)
        ax1.plot(x, lowers, color="#2196f3", linewidth=0.6, alpha=0.5)
        ax1.fill_between(x, uppers, lowers, alpha=0.05, color="#2196f3")

    # VWAP
    if vwap_val > 0:
        ax1.axhline(y=vwap_val, color="#ff9800", linestyle="-", linewidth=1, alpha=0.7)
        ax1.text(n + 0.5, vwap_val, f"VWAP {vwap_val:.0f}", fontsize=7, color="#ff9800", va="center")

    # EMAs
    closes = [c["close"] for c in candles]
    if len(closes) >= 9:
        from trading.utils.indicators import _ema
        ema9_line = _ema(closes, 9)
        offset = len(closes) - len(ema9_line)
        ax1.plot(range(offset, len(closes)), ema9_line, color="#ffeb3b", linewidth=0.8, alpha=0.7, label="EMA9")
    if len(closes) >= 20:
        ema20_line = _ema(closes, 20)
        offset = len(closes) - len(ema20_line)
        ax1.plot(range(offset, len(closes)), ema20_line, color="#9c27b0", linewidth=0.8, alpha=0.7, label="EMA20")

    # Entry / SL / Target lines
    if entry > 0:
        ax1.axhline(y=entry, color="#ffffff", linewidth=1.2, linestyle="-", alpha=0.9)
        ax1.text(n + 0.5, entry, f"ENTRY {entry:.1f}", fontsize=7, color="#ffffff", va="center", fontweight="bold")
    if sl > 0:
        ax1.axhline(y=sl, color="#ef5350", linewidth=1.2, linestyle="-", alpha=0.9)
        ax1.text(n + 0.5, sl, f"SL {sl:.1f}", fontsize=7, color="#ef5350", va="center", fontweight="bold")
    if target > 0:
        ax1.axhline(y=target, color="#26a69a", linewidth=1.2, linestyle="-", alpha=0.9)
        ax1.text(n + 0.5, target, f"TGT {target:.1f}", fontsize=7, color="#26a69a", va="center", fontweight="bold")

    ax1.set_title(f"{symbol} — 5min", color="white", fontsize=11)
    ax1.tick_params(colors="white", labelsize=7)
    ax1.spines["bottom"].set_color("#333")
    ax1.spines["left"].set_color("#333")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Volume bars
    volumes = [c.get("volume", 0) for c in candles]
    colors = ["#26a69a" if c["close"] >= c["open"] else "#ef5350" for c in candles]
    ax2.bar(x, volumes, color=colors, alpha=0.6, width=0.6)
    ax2.set_ylabel("Volume", color="white", fontsize=8)
    ax2.tick_params(colors="white", labelsize=7)
    ax2.spines["bottom"].set_color("#333")
    ax2.spines["left"].set_color("#333")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
