"""Scalper Loss-Streak Cool-Down Enforcer.

Scans recent closed TradeJournal rows for a consecutive-losing-trades
streak. When the streak hits a configured cap (default 3), surfaces a
COOL_DOWN_ACTIVE flag so @RiskGuard / the planner can skip new entries
for the rest of the session.

  GET /portfolios/cool-down/  →  {
    streak: <int>,            # consecutive losses (now)
    longest_streak_today: int,
    threshold: 3,
    state: NORMAL | WARNING | COOL_DOWN,
    cooldown_until: ISO,      # null when state != COOL_DOWN
    recent_trades: [...],
    note: "..."
  }

When state = COOL_DOWN, the operator (or RiskGuard) is expected to skip
new entries until the cool-down expires (15:30 IST by default — the
session boundary). Toggling the streak resets when a winning trade fires.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = timezone(timedelta(hours=5, minutes=30))


_DEFAULT_THRESHOLD = 3
_SESSION_END = time(15, 30)


def build_cool_down(tenant=None, *, threshold: int = _DEFAULT_THRESHOLD) -> dict[str, Any]:
    from trading.models import TradeJournal

    today = date.today()
    trades = list(
        TradeJournal.objects
        .filter(trade_date=today, pnl__isnull=False)
        .order_by("created_at")
    )

    # Build the wins/losses sequence in chronological order
    sequence = []
    for t in trades:
        pnl = float(t.pnl or 0)
        if pnl == 0: continue   # scratches don't break the streak
        sequence.append({
            "trade_id": t.id, "symbol": t.symbol,
            "pnl_inr": round(pnl, 2),
            "outcome": "WIN" if pnl > 0 else "LOSS",
            "ts": t.created_at.isoformat() if t.created_at else None,
        })

    # Streak length = trailing run of LOSSes
    streak = 0
    for trade in reversed(sequence):
        if trade["outcome"] == "LOSS":
            streak += 1
        else:
            break

    longest = 0
    cur = 0
    for trade in sequence:
        if trade["outcome"] == "LOSS":
            cur += 1; longest = max(longest, cur)
        else:
            cur = 0

    if streak >= threshold:
        state = "COOL_DOWN"
        now_ist = datetime.now(tz=_IST)
        deadline = datetime.combine(today, _SESSION_END, tzinfo=_IST)
        cooldown_until = deadline.isoformat() if deadline > now_ist else None
    elif streak >= threshold - 1:
        state = "WARNING"
        cooldown_until = None
    else:
        state = "NORMAL"
        cooldown_until = None

    return {
        "streak": streak,
        "longest_streak_today": longest,
        "threshold": threshold,
        "state": state,
        "cooldown_until": cooldown_until,
        "recent_trades": sequence[-10:],
        "note": (
            "COOL_DOWN_ACTIVE = skip new entries until session close "
            "(15:30 IST). One winning trade resets the streak — but if "
            "you take it on tilt, the streak resumes from the next loss."
        ),
    }
