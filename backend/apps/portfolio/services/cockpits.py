"""Cockpit services — backend aggregations for the 10 trader-facing
views the AI Team identified.

Each function takes a tenant + optional params and returns a JSON-safe
dict. All read from legacy `trading.*` models via LegacyRouter so the
v2 frontend gets real numbers immediately — no v2 schema migration
required.

Where the underlying data layer doesn't yet capture something (e.g.
live Greeks per-position outside straddles), we return a structured
empty / sample-shaped response so the React side stays well-typed and
the AI Tester can verify the contract regardless of state.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from apps.common.margin_calc import (
    equity_margin, short_straddle_margin, summarize_buckets,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. Capital & Leverage Cockpit
# ─────────────────────────────────────────────────────────────────────────
def build_capital_cockpit(tenant=None) -> dict:
    """Deployed capital · free margin · notional + delta-adjusted notional ·
    leverage ratio across all open positions, split by bucket."""
    from trading.models import TradeJournal, StraddlePosition, PortfolioSnapshot

    try:
        snap = PortfolioSnapshot.objects.latest()
        total_capital = float(snap.capital)
    except PortfolioSnapshot.DoesNotExist:
        total_capital = 500_000.0

    open_trades = TradeJournal.objects.filter(status__in=("PENDING", "APPROVED", "EXECUTED", "PAPER"))
    active_straddles = StraddlePosition.objects.filter(status="ACTIVE")

    estimates = []
    for t in open_trades:
        m = equity_margin(t.side, int(t.quantity), float(t.entry_price), product="MIS")
        estimates.append(m)
    for p in active_straddles:
        m = short_straddle_margin(
            lots=int(p.lots), lot_size=int(p.lot_size), strike=float(p.strike),
            ce_premium=float(p.ce_sell_price), pe_premium=float(p.pe_sell_price),
            underlying_spot=float(p.strike),
        )
        estimates.append(m)

    buckets = summarize_buckets(estimates)
    notional = buckets["totals"]["notional"]
    margin = buckets["totals"]["total_margin"]
    # Delta-adjusted notional: equity is full notional; options use 0.5 ATM heuristic.
    delta_adj = 0.0
    for t in open_trades:
        delta_adj += float(t.entry_price) * t.quantity * (1.0 if t.side == "BUY" else -1.0)
    for p in active_straddles:
        # Short straddle ≈ flat delta at entry; weight at 0.5 of single-leg notional
        delta_adj += 0.5 * float(p.strike) * p.lot_size * p.lots

    return {
        "total_capital": round(total_capital, 2),
        "deployed_capital": round(notional, 2),
        "free_margin": round(max(0.0, total_capital - margin), 2),
        "notional_exposure": round(notional, 2),
        "delta_adjusted_exposure": round(abs(delta_adj), 2),
        "leverage_ratio": round((notional / margin) if margin else 1.0, 2),
        "margin_used": round(margin, 2),
        "premium_received": round(buckets["totals"].get("premium_received", 0), 2),
        "buckets": [
            {"key": k, **{kk: round(vv, 2) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}}
            for k, v in buckets["by_bucket"].items()
        ],
    }


# ─────────────────────────────────────────────────────────────────────────
# 2. Plan vs Actual Reconciliation
# ─────────────────────────────────────────────────────────────────────────
def build_plan_vs_actual(tenant=None, *, limit: int = 200, strategy: str | None = None) -> dict:
    """Compare AI plans (TradeJournal planned fields) vs realised
    fills, surfacing slippage_bps + sizing drift + SL-handling outcome."""
    from trading.models import TradeJournal

    qs = TradeJournal.objects.order_by("-created_at")[:limit]
    if strategy:
        qs = qs.filter(reasoning__icontains=strategy)

    rows = []
    slippage_samples = []
    for t in qs:
        planned_entry = float(t.entry_price)
        actual_entry = float(t.fill_price) if t.fill_price is not None else None
        slippage_bps = None
        if actual_entry and planned_entry:
            slippage_bps = round((actual_entry - planned_entry) / planned_entry * 10_000, 1)
            slippage_samples.append(abs(slippage_bps))
        # SL handling outcome: did the trade close at/near SL?
        sl_handled = (
            "filled_at_sl" if t.pnl is not None and t.pnl < 0 and t.status == "EXECUTED"
            else "filled_at_target" if t.pnl is not None and t.pnl > 0
            else "open" if t.status in ("PENDING", "APPROVED", "PAPER") and t.pnl is None
            else "rejected" if t.status == "REJECTED"
            else "unknown"
        )
        rows.append({
            "id": t.id, "trade_date": t.trade_date.isoformat() if t.trade_date else None,
            "symbol": t.symbol, "side": t.side, "status": t.status,
            "planned_entry": planned_entry, "actual_entry": actual_entry,
            "planned_qty": int(t.quantity),
            "actual_qty": int(t.fill_quantity) if t.fill_quantity else 0,
            "planned_sl": float(t.stop_loss), "planned_target": float(t.target),
            "realised_pnl": float(t.pnl) if t.pnl is not None else None,
            "slippage_bps": slippage_bps,
            "sl_handling": sl_handled,
            "confidence": float(t.confidence) if t.confidence else None,
            "reasoning_preview": (t.reasoning or "")[:160],
        })

    return {
        "count": len(rows),
        "avg_abs_slippage_bps": round(statistics.mean(slippage_samples), 1) if slippage_samples else None,
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. Options Greeks Heatmap
# ─────────────────────────────────────────────────────────────────────────
def build_greeks_heatmap(tenant=None) -> dict:
    """Aggregate Δ/Γ/Θ/V across open option positions, bucketed by
    (underlying, expiry). Gamma-risk flag fires when expiry is today."""
    from trading.models import StraddlePosition
    from trading.options.straddle.analyzer import _approx_delta

    today = date.today()
    grouped: dict[tuple, dict] = {}
    for p in StraddlePosition.objects.filter(status="ACTIVE"):
        key = (p.underlying, p.expiry.isoformat() if p.expiry else "?")
        cell = grouped.setdefault(key, {
            "underlying": p.underlying, "expiry": key[1],
            "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "positions": 0, "gamma_risk": False,
        })
        dte = max(0, (p.expiry - today).days) if p.expiry else 0
        ce_d = _approx_delta(float(p.strike), float(p.strike), "CE", dte)
        pe_d = _approx_delta(float(p.strike), float(p.strike), "PE", dte)
        # Short straddle: net delta ≈ -(ce_d + pe_d) × lots × lot_size
        notional_per_lot = p.lot_size * p.lots
        cell["delta"] += -(ce_d + pe_d) * notional_per_lot
        # Gamma is highest near ATM and explodes on expiry day — heuristic
        gamma_proxy = (1.0 / max(1, dte)) * 0.02
        cell["gamma"] += gamma_proxy * notional_per_lot
        # Theta capture proxy: combined sold premium / DTE
        theta_proxy = (float(p.ce_sell_price + p.pe_sell_price) / max(1, dte))
        cell["theta"] += theta_proxy * notional_per_lot
        # Vega proxy: ATM straddle is high-vega
        cell["vega"] += 0.15 * notional_per_lot
        cell["positions"] += 1
        if dte == 0:
            cell["gamma_risk"] = True

    rows = [
        {**c, "delta": round(c["delta"], 2), "gamma": round(c["gamma"], 4),
         "theta": round(c["theta"], 2), "vega": round(c["vega"], 2)}
        for c in grouped.values()
    ]
    return {
        "count": len(rows),
        "rows": sorted(rows, key=lambda r: (not r["gamma_risk"], r["expiry"])),
        "note": "Δ from Black-Scholes approximation; Γ/Θ/V are proxies "
                "(real per-position chain Greeks need an option-chain API).",
    }


# ─────────────────────────────────────────────────────────────────────────
# 4. Signal-to-Trade Funnel
# ─────────────────────────────────────────────────────────────────────────
def build_signal_funnel(tenant=None) -> dict:
    """Fired → risk_passed → executed → profitable, sliced by strategy
    inference (from reasoning text) + rejection_reason buckets."""
    from trading.models import TradeJournal

    qs = TradeJournal.objects.all()
    fired = qs.count()
    risk_passed = qs.filter(risk_approved=True).count()
    executed = qs.filter(status__in=("EXECUTED", "PAPER")).count()
    profitable = qs.filter(pnl__gt=0).count()

    # Per-strategy slice (best-effort — reasoning keyword)
    strategy_keys = {
        "directional": ["intraday", "5-min", "vwap", "structure"],
        "short_straddle": ["straddle", "premium decay", "theta"],
        "pyramid":    ["pyramid", "rsi", "ema5"],
        "vertical_spread": ["spread", "debit", "credit"],
    }
    by_strategy: dict[str, dict] = {}
    for s, keys in strategy_keys.items():
        sub = qs
        for k in keys:
            sub = sub | qs.filter(reasoning__icontains=k)
        sub = sub.distinct()
        by_strategy[s] = {
            "fired": sub.count(),
            "risk_passed": sub.filter(risk_approved=True).count(),
            "executed": sub.filter(status__in=("EXECUTED", "PAPER")).count(),
            "profitable": sub.filter(pnl__gt=0).count(),
        }

    # Rejection reason buckets
    rejected_qs = qs.filter(risk_approved=False).exclude(risk_reason="")
    reasons: dict[str, int] = defaultdict(int)
    for t in rejected_qs:
        key = (t.risk_reason or "unknown").split("(")[0].strip()[:80]
        reasons[key] += 1
    rejections = sorted(
        [{"reason": k, "count": v} for k, v in reasons.items()],
        key=lambda r: -r["count"],
    )[:10]

    return {
        "totals": {
            "fired": fired, "risk_passed": risk_passed,
            "executed": executed, "profitable": profitable,
            "capture_rate_pct": round((profitable / fired) * 100, 1) if fired else 0,
        },
        "by_strategy": by_strategy,
        "rejection_reasons": rejections,
    }


# ─────────────────────────────────────────────────────────────────────────
# 5. Daily Risk Budget & Drawdown Anatomy
# ─────────────────────────────────────────────────────────────────────────
def build_risk_budget(tenant=None, *, lookback_days: int = 30) -> dict:
    from trading.models import TradeJournal, PortfolioSnapshot

    MAX_RISK_PCT = 1.0
    MAX_DAILY_LOSS_PCT = 3.0
    try:
        snap = PortfolioSnapshot.objects.latest()
        capital = float(snap.capital)
        daily_loss = float(snap.daily_loss)
        daily_pnl = float(snap.daily_pnl)
    except PortfolioSnapshot.DoesNotExist:
        capital, daily_loss, daily_pnl = 500_000.0, 0.0, 0.0

    open_trades = TradeJournal.objects.filter(status__in=("PENDING", "APPROVED", "EXECUTED", "PAPER"))
    open_risk = sum(
        abs(float(t.entry_price) - float(t.stop_loss)) * t.quantity for t in open_trades
    )

    # 30-day drawdown waterfall — daily P&L from TradeJournal
    today = date.today()
    days = [today - timedelta(days=i) for i in range(lookback_days)][::-1]
    waterfall: list[dict] = []
    cumulative = 0.0
    peak = 0.0
    for d in days:
        day_pnl = sum(
            float(t.pnl or 0) for t in TradeJournal.objects.filter(trade_date=d)
        )
        cumulative += day_pnl
        peak = max(peak, cumulative)
        waterfall.append({
            "date": d.isoformat(), "pnl": round(day_pnl, 2),
            "cumulative": round(cumulative, 2),
            "drawdown": round(cumulative - peak, 2),
        })

    return {
        "capital": capital,
        "used_risk_pct": round((daily_loss / capital) * 100, 2) if capital else 0,
        "allowed_risk_pct": MAX_DAILY_LOSS_PCT,
        "max_risk_per_trade_pct": MAX_RISK_PCT,
        "open_risk_at_stop": round(open_risk, 2),
        "open_risk_pct_of_capital": round((open_risk / capital) * 100, 2) if capital else 0,
        "daily_pnl": daily_pnl,
        "daily_loss": daily_loss,
        "drawdown_waterfall": waterfall,
    }


# ─────────────────────────────────────────────────────────────────────────
# 6. Expiry-Day Tactical Cockpit
# ─────────────────────────────────────────────────────────────────────────
def build_expiry_cockpit(tenant=None, *, underlying: str = "NIFTY") -> dict:
    """3:15 PM countdown · ATM pin probability · close-list of expiring positions."""
    from trading.models import StraddlePosition

    today = date.today()
    now = datetime.now()
    cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
    countdown_seconds = max(0, int((cutoff - now).total_seconds()))

    expiring = StraddlePosition.objects.filter(
        status="ACTIVE", underlying=underlying, expiry=today,
    )
    close_list = [
        {
            "id": p.id, "leg": f"{p.underlying} {p.strike}",
            "ce_symbol": p.ce_symbol, "pe_symbol": p.pe_symbol,
            "lots": p.lots, "side": "BUY-TO-CLOSE",
            "reason": "Expiry today — mandatory close before 15:15 IST",
        }
        for p in expiring
    ]

    # ATM pin probability — placeholder; real impl needs open interest curve
    pin_strike = None
    if expiring.exists():
        pin_strike = expiring.first().strike

    return {
        "underlying": underlying,
        "is_expiry_day": expiring.exists(),
        "countdown_seconds": countdown_seconds,
        "pin_strike": pin_strike,
        "gamma_by_strike": [],  # needs option chain — left empty until wired
        "close_list": close_list,
        "active_count": expiring.count(),
    }


# ─────────────────────────────────────────────────────────────────────────
# 7. Broker vs Journal Reconciliation
# ─────────────────────────────────────────────────────────────────────────
def build_broker_reconciliation(tenant=None, *, on_date: date | None = None) -> dict:
    """EOD diff between Angel One ledger and TradeJournal — surfaces missed
    fills, ghost positions, charges + brokerage delta."""
    from trading.models import TradeJournal
    from trading.services.broker_service import BrokerService

    on_date = on_date or date.today()
    journal = list(TradeJournal.objects.filter(trade_date=on_date))
    journal_pnl = sum(float(t.pnl or 0) for t in journal)

    broker_positions: list[dict] = []
    broker_pnl = 0.0
    broker_err = None
    try:
        br = BrokerService()
        # In paper mode there's no live ledger — surface that as a note
        if br.mode == "paper":
            broker_err = "Paper mode — no live broker ledger to reconcile against."
        else:
            from trading.services.data_service import BrokerClient
            bc = BrokerClient.get_instance(); bc.ensure_login()
            broker_positions = bc.fetch_positions().get("net", []) or []
            broker_pnl = sum(float(p.get("pnl", 0)) for p in broker_positions)
    except Exception as e:  # noqa: BLE001
        broker_err = str(e)

    journal_symbols = {t.symbol for t in journal}
    broker_symbols = {p.get("tradingsymbol", "") for p in broker_positions}
    only_in_journal = sorted(journal_symbols - broker_symbols)
    only_in_broker = sorted(broker_symbols - journal_symbols)

    return {
        "on_date": on_date.isoformat(),
        "journal_count": len(journal),
        "broker_count": len(broker_positions),
        "journal_pnl": round(journal_pnl, 2),
        "broker_pnl": round(broker_pnl, 2),
        "pnl_delta": round(broker_pnl - journal_pnl, 2),
        "only_in_journal": only_in_journal[:30],
        "only_in_broker": only_in_broker[:30],
        "mismatched_count": len(only_in_journal) + len(only_in_broker),
        "broker_note": broker_err,
    }


# ─────────────────────────────────────────────────────────────────────────
# 8. Strategy Edge Decay (rolling expectancy)
# ─────────────────────────────────────────────────────────────────────────
def build_edge_decay(tenant=None, *, window: int = 20) -> dict:
    """Rolling N-trade expectancy, win-rate, avg R per strategy."""
    from trading.models import TradeJournal

    closed = TradeJournal.objects.filter(pnl__isnull=False).order_by("-trade_date")[:500]
    closed = list(reversed(list(closed)))  # chronological

    series_per_strategy: dict[str, list[dict]] = defaultdict(list)
    strategy_keys = {
        "directional": ["intraday", "vwap", "structure", "5-min"],
        "short_straddle": ["straddle"],
        "pyramid":    ["pyramid", "ema5"],
        "vertical_spread": ["spread"],
    }

    def classify(t) -> str:
        rsn = (t.reasoning or "").lower()
        for s, keys in strategy_keys.items():
            if any(k in rsn for k in keys):
                return s
        return "uncategorised"

    bucketed: dict[str, list] = defaultdict(list)
    for t in closed:
        bucketed[classify(t)].append(t)

    for s, ts in bucketed.items():
        for i in range(window, len(ts) + 1):
            slice_ = ts[i - window:i]
            wins = sum(1 for t in slice_ if (t.pnl or 0) > 0)
            losses = sum(1 for t in slice_ if (t.pnl or 0) < 0)
            avg_r = statistics.mean([
                (float(t.pnl or 0) / max(1.0, abs(float(t.entry_price) - float(t.stop_loss)) * t.quantity))
                for t in slice_
            ])
            expectancy = statistics.mean(float(t.pnl or 0) for t in slice_)
            series_per_strategy[s].append({
                "as_of": slice_[-1].trade_date.isoformat() if slice_[-1].trade_date else "",
                "expectancy": round(expectancy, 2),
                "win_rate": round((wins / max(1, wins + losses)) * 100, 1),
                "avg_r": round(avg_r, 2),
                "n": len(slice_),
            })

    return {
        "window": window,
        "series": {s: pts for s, pts in series_per_strategy.items()},
    }


# ─────────────────────────────────────────────────────────────────────────
# 9. Theta Decay & Premium-Burn Forecast Curve
# ─────────────────────────────────────────────────────────────────────────
def build_theta_forecast(tenant=None) -> dict:
    """Per-minute theta capture vs realised P&L through 3:15 PM IST for
    every open short-premium position. Realised pulled from current LTPs;
    projected uses linear theta decay between now and 15:15 IST."""
    from trading.models import StraddlePosition

    now = datetime.now()
    cutoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
    minutes_left = max(1, int((cutoff - now).total_seconds() / 60))
    today = date.today()

    series_per_position: list[dict] = []
    for p in StraddlePosition.objects.filter(status="ACTIVE"):
        dte = max(1, (p.expiry - today).days) if p.expiry else 1
        sold = float(p.ce_sell_price + p.pe_sell_price)
        current = float((p.ce_current_price or 0) + (p.pe_current_price or 0))
        realised_pts = sold - current
        # Per-minute theta budget — sold premium spread over remaining DTE * 375 min/session
        per_min_theta = sold / (dte * 375)
        points = []
        for i in range(0, minutes_left, 15):  # every 15 min
            t_off = now + timedelta(minutes=i)
            projected = realised_pts + per_min_theta * i
            points.append({
                "minute": t_off.strftime("%H:%M"),
                "projected_pts": round(projected, 2),
                "projected_inr": round(projected * p.lot_size * p.lots, 2),
                "realised_pts": round(realised_pts, 2),
            })
        series_per_position.append({
            "position_id": p.id,
            "leg": f"{p.underlying} {p.strike}",
            "dte": dte, "lot_size": p.lot_size, "lots": p.lots,
            "sold_premium": sold, "current_premium": current,
            "minutes_left": minutes_left,
            "breakeven_band": {"low": p.strike - sold, "high": p.strike + sold},
            "series": points,
        })

    return {"count": len(series_per_position), "positions": series_per_position}


# ─────────────────────────────────────────────────────────────────────────
# 10. Regime-Conditioned Strategy Allocation Heatmap
# ─────────────────────────────────────────────────────────────────────────
def build_regime_heatmap(tenant=None) -> dict:
    """Strategy × NIFTY regime expectancy matrix from TradeJournal +
    PulsePayload regime tag at trade time (best-effort)."""
    from trading.models import TradeJournal

    regimes = ["trending_up", "trending_down", "chop", "high_vix", "low_vix"]
    strategies = ["directional", "short_straddle", "pyramid", "vertical_spread"]

    # Without per-trade regime snapshots stored, infer regime from VIX phase
    # bands stored in reasoning (best-effort) and bucket symbols.
    cells = []
    for s in strategies:
        for r in regimes:
            # Crude proxy — match on reasoning keywords
            keys = {"trending_up": "uptrend", "trending_down": "downtrend",
                    "chop": "range", "high_vix": "vix spike", "low_vix": "calm"}
            qs = TradeJournal.objects.filter(reasoning__icontains=s).filter(
                reasoning__icontains=keys[r]
            )
            n = qs.count()
            avg_pnl = statistics.mean([float(t.pnl or 0) for t in qs if t.pnl is not None]) if n else 0
            cells.append({
                "strategy": s, "regime": r,
                "trade_count": n,
                "expectancy": round(avg_pnl, 2),
            })

    # Current regime — read from the pulse cache if available
    current_regime = "unknown"
    try:
        from django.core.cache import cache
        payload = cache.get("market_pulse:v1")
        if payload:
            reg = getattr(payload, "regime", None) or (payload.get("regime") if isinstance(payload, dict) else None)
            if reg:
                vol = reg.get("vol", "")
                trend = reg.get("trend", "")
                if vol == "extreme":   current_regime = "high_vix"
                elif vol == "low":     current_regime = "low_vix"
                elif trend == "up":    current_regime = "trending_up"
                elif trend == "down":  current_regime = "trending_down"
                else:                  current_regime = "chop"
    except Exception:  # noqa: BLE001
        pass

    return {
        "current_regime": current_regime,
        "strategies": strategies,
        "regimes": regimes,
        "cells": cells,
    }
