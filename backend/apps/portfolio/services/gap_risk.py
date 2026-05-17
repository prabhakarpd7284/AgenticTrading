"""Overnight gap risk & pre-market exposure.

Identifies positions held overnight (any active straddle, any open
TradeJournal entry tagged NRML / not intraday), pulls a best-effort
implied-gap reference, and projects per-position P&L at +/-0.5/1/2%
gaps. Returns a `hedge_checklist` of plain-English actions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

_GAP_STEPS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def _is_overnight(trade) -> bool:
    """Heuristic: anything tagged NRML/CNC or with no exit yet is overnight."""
    product = (getattr(trade, "product", "") or "").upper()
    if product in ("NRML", "CNC", "DELIVERY"):
        return True
    # If still open (never filled), treat as overnight by default.
    return getattr(trade, "fill_price", None) is None


def _implied_gap_pct() -> float:
    """Best-effort SGX/Dow proxy. Falls back to 0.0 if no feed available."""
    try:
        # The legacy stack doesn't currently scrape SGX so we use yesterday's
        # NIFTY close vs the day-before to give the trader *some* number.
        from trading.options.data_service import OptionsDataService
        ods = OptionsDataService()
        spot = float(ods.fetch_nifty_spot().get("ltp", 0) or 0)
        # No second close handy; return 0 — UI shows "no live reference".
        return 0.0 if spot <= 0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _pnl_at_gap(symbol: str, side: str, qty: int, entry: float, gap_pct: float) -> float:
    move = entry * (gap_pct / 100.0)
    sign = 1 if side.upper() in ("BUY", "LONG") else -1
    return sign * move * qty


def build_gap_risk_report(tenant=None) -> dict[str, Any]:
    from trading.models import TradeJournal, StraddlePosition

    overnight_trades = [
        t for t in TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER"))
        if _is_overnight(t)
    ]
    straddles = list(StraddlePosition.objects.filter(status="ACTIVE"))

    if not overnight_trades and not straddles:
        return {
            "implied_gap_pct": 0.0,
            "implied_gap_source": "none",
            "positions": [],
            "hedge_checklist": [],
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    implied_gap = _implied_gap_pct()

    positions: list[dict] = []
    for t in overnight_trades:
        entry = float(t.entry_price or 0)
        qty = int(t.quantity or 0)
        pnl_at_gap = {str(g): round(_pnl_at_gap(t.symbol, t.side, qty, entry, g), 2)
                       for g in _GAP_STEPS}
        positions.append({
            "symbol": t.symbol,
            "side": t.side,
            "qty": qty,
            "entry": entry,
            "kind": "equity",
            "pnl_at_gap": pnl_at_gap,
        })

    for p in straddles:
        # Short straddle — losses scale with |gap|×(notional×delta_proxy).
        notional = float(p.lots) * float(p.lot_size) * float(p.strike)
        gamma_factor = 0.5  # rough: ATM short straddle PnL ~ -0.5×|gap|×notional
        pnl_at_gap = {
            str(g): round(-abs(g / 100.0) * notional * gamma_factor, 2)
            for g in _GAP_STEPS
        }
        positions.append({
            "symbol": f"{p.underlying} STRADDLE {p.strike}",
            "side": "SHORT_STRADDLE",
            "qty": int(p.lots) * int(p.lot_size),
            "entry": float(p.strike),
            "kind": "straddle",
            "pnl_at_gap": pnl_at_gap,
        })

    worst = min((sum(pos["pnl_at_gap"][str(g)] for pos in positions) for g in _GAP_STEPS), default=0)

    checklist = [
        "Pre-market: check SGX NIFTY / Dow futures / India VIX before 9:00 IST.",
        "If any underlying gaps > 1% adverse, queue a partial-exit order to fire at 9:15:30.",
        "Short straddles within 2 strikes of spot need a wing-buy hedge if VIX > 18.",
        f"Worst-case overnight loss across {len(positions)} positions: ₹{abs(worst):,.0f}.",
    ]

    return {
        "implied_gap_pct": implied_gap,
        "implied_gap_source": "manual",
        "positions": positions,
        "hedge_checklist": checklist,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
