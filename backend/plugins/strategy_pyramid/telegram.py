"""Telegram report for Pyramid Strategy results."""
import json
import os
import threading
import urllib.request

from logzero import logger


def send_pyramid_report(data: dict) -> bool:
    """
    Send a formatted pyramid strategy report to Telegram.

    Args:
        data: dict from run_pyramid_with_chart_data()

    Returns:
        True if send was initiated (non-blocking), False if not configured.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning("Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False

    text = format_pyramid_telegram(data)
    thread = threading.Thread(
        target=_do_send, args=(bot_token, chat_id, text), daemon=True
    )
    thread.start()
    return True


def format_pyramid_telegram(data: dict) -> str:
    """Format pyramid result as Telegram HTML message."""
    kpis = data.get("kpis", {})
    entries = data.get("entries", [])
    exit_data = data.get("exit")
    config = data.get("config", {})
    symbol = data.get("symbol", "?")

    won = kpis.get("won", False)
    verdict = "WINNER" if won else "LOSS"
    emoji = "\U0001f4b0" if won else "\U0001f534"  # money bag or red circle

    pnl_pts = kpis.get("total_pnl_pts", 0)
    pnl_inr = kpis.get("total_pnl_inr", 0)
    peak_inr = kpis.get("peak_unrealized_inr", 0)

    lines = [
        f"{emoji} <b>Pyramid Strategy — {symbol}</b>",
        "",
    ]

    # Config line
    dry = " (DRY RUN)" if config.get("dry_run") else ""
    lines.append(
        f"<i>{config.get('underlying', '?')} {config.get('strike', '?')} "
        f"{config.get('type', '?')} exp {config.get('expiry', '?')} "
        f"| {config.get('date', '?')}{dry}</i>"
    )
    lines.append(
        f"Capital: \u20b9{config.get('capital', 0):,.0f} | "
        f"Risk: {config.get('risk_pct', 0)}% | "
        f"Profit risk: {config.get('profit_risk', 0):.0%}"
    )
    lines.append("")

    # KPIs
    sign = "+" if pnl_inr >= 0 else ""
    lines.append("<b>Results:</b>")
    lines.append(f"  P&L: <b>{sign}\u20b9{pnl_inr:,.0f}</b> ({sign}{pnl_pts:.1f} pts)")
    lines.append(f"  Peak unrealized: +\u20b9{peak_inr:,.0f}")
    lines.append(
        f"  Lots: {kpis.get('total_lots', 0)} (peak {kpis.get('peak_lots', 0)}) | "
        f"Pyramids: {kpis.get('pyramid_count', 0)}"
    )
    lines.append(
        f"  Avg entry: {kpis.get('avg_entry', 0):.2f} | "
        f"Exit: {kpis.get('exit_price', 0):.2f} ({kpis.get('exit_reason', '?')})"
    )
    lines.append("")

    # Entry flow
    lines.append("<b>Pyramid Flow:</b>")
    for i, e in enumerate(entries):
        ts = e.get("t", "")
        time_str = ts[11:16] if len(ts) > 16 else ts
        tag = "Entry" if i == 0 else f"Add #{i}"
        lines.append(
            f"  {tag}: {e.get('price', 0):.2f} "
            f"+{e.get('lots', 0)} lots "
            f"(\u03a3{e.get('cumulative_lots', 0)}) "
            f"SL {e.get('sl', 0):.2f} "
            f"@ {time_str}"
        )

    if exit_data:
        ts = exit_data.get("t", "")
        time_str = ts[11:16] if len(ts) > 16 else ts
        lines.append(
            f"  Exit: {exit_data.get('price', 0):.2f} "
            f"{exit_data.get('lots', 0)} lots "
            f"({exit_data.get('reason', '?')}) @ {time_str}"
        )

    lines.append("")
    lines.append(f"<b>Verdict: {verdict}</b>")

    return "\n".join(lines)


def _do_send(bot_token: str, chat_id: str, text: str):
    """Actual HTTP POST to Telegram (runs in background thread)."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.status == 200:
            logger.info("Pyramid report sent to Telegram")
        else:
            logger.error(f"Telegram send failed: HTTP {resp.status}")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
