"""Per-trade chart + feedback endpoints powering the Monthly trade table's
"View chart" action and the 👍/👎 feedback control.

  GET  /api/v1/trades/<id>/chart/     — candles for the trade's window + the
                                        entry / stop / target / exit overlay.
  GET  /api/v1/trades/<id>/feedback/  — the latest trader feedback (if any).
  POST /api/v1/trades/<id>/feedback/  — record a 👍/👎 + optional note.

Intraday trades chart the session's 5-min candles; swing trades chart ~6 weeks
of daily candles around the exit. The overlay levels let the UI highlight the
trade visually (where it entered, where the stop/target sat, where it exited).
"""
from __future__ import annotations

from datetime import timedelta

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiParameter, extend_schema, inline_serializer,
)
from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.trading.models import Trade


def _fetch_candles(symbol: str, start: str, end: str, interval: str) -> list[dict]:
    """Broker candles for a window. Reuses the legacy DataService broker path."""
    from trading.services.data_service import DataService
    from trading.services.ticker_service import ticker_service

    ds = DataService()
    ds._ensure_broker()
    token = ticker_service.get_token(symbol) or ""
    if not token:
        return []
    raw = ds._broker.fetch_candles(token, start, end, interval) or []
    return [
        {"t": r[0], "o": float(r[1]), "h": float(r[2]),
         "l": float(r[3]), "c": float(r[4]), "v": int(r[5])}
        for r in raw
    ]


def _is_swing(trade: Trade) -> bool:
    return (trade.reasoning or "").startswith("[Swing")


def _nearest_candle_ts(candles, target_dt):
    """Timestamp of the candle closest to ``target_dt`` (a tz-aware datetime)."""
    from datetime import datetime as _dt

    if not candles or target_dt is None:
        return None
    best, best_diff = None, None
    for c in candles:
        try:
            ct = _dt.fromisoformat(str(c["t"]))
        except (ValueError, TypeError):
            continue
        diff = abs((ct - target_dt).total_seconds())
        if best_diff is None or diff < best_diff:
            best, best_diff = c["t"], diff
    return best


def _resolve_entry_exit(candles, trade):
    """Entry/exit candle timestamps — from the stored fill/close times when they
    fall inside the window (accurate), else reconstructed from the price action."""
    from datetime import datetime as _dt

    if candles:
        try:
            lo = _dt.fromisoformat(str(candles[0]["t"]))
            hi = _dt.fromisoformat(str(candles[-1]["t"]))

            def _in(dt):
                return dt is not None and lo <= dt <= hi

            e = _nearest_candle_ts(candles, trade.filled_at) if _in(trade.filled_at) else None
            x = _nearest_candle_ts(candles, trade.closed_at) if _in(trade.closed_at) else None
            if e and x:
                return e, x
        except (ValueError, TypeError):
            pass

    return _entry_exit_ts(
        candles, trade.side, float(trade.entry_price or 0),
        float(trade.stop_loss or 0) or None, float(trade.target or 0) or None,
        trade.close_reason or "",
    )


def _entry_exit_ts(candles, side, entry, stop, target, close_reason):
    """Reconstruct the entry + exit candle timestamps from the price action.

    We don't store intraday entry/exit times on derived trades, but the exit is
    fully determined by the same rule the simulation used (first SL/target touch,
    else last candle), and the entry is the first candle whose range brackets the
    entry price. This makes the chart markers line up with what actually happened.
    """
    if not candles:
        return None, None
    is_long = side == "BUY"

    # Entry: first candle where price reaches the entry level in the trade's
    # direction (a long fills as price rises into it; a short as it falls).
    entry_i = 0
    for i, c in enumerate(candles):
        if (is_long and c["h"] >= entry) or (not is_long and c["l"] <= entry):
            entry_i = i
            break

    exit_i = len(candles) - 1
    for i in range(entry_i + 1, len(candles)):
        c = candles[i]
        if close_reason == "SL_HIT" and stop is not None:
            if (is_long and c["l"] <= stop) or (not is_long and c["h"] >= stop):
                exit_i = i
                break
        elif close_reason == "TARGET_HIT" and target is not None:
            if (is_long and c["h"] >= target) or (not is_long and c["l"] <= target):
                exit_i = i
                break
    return candles[entry_i]["t"], candles[exit_i]["t"]


class TradeChartView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        parameters=[
            OpenApiParameter(
                "interval", OpenApiTypes.STR, OpenApiParameter.QUERY,
                description="Swing-trade candle interval: '1d' (default) or '1h'. "
                            "Ignored for intraday trades (always 5m).",
                enum=["1d", "1h"], required=False,
            ),
        ],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request, trade_id):
        trade = Trade.objects.filter(tenant=request.tenant, id=trade_id).first()
        if not trade:
            return Response({"detail": "Trade not found."}, status=404)

        import re

        d = trade.trade_date
        swing = _is_swing(trade)
        if swing:
            # Window the chart to the actual trade period (entry→exit) rather
            # than a fixed lookback, so the trade isn't squished into a corner.
            # Swing trades default to daily candles; the UI can request 1h for a
            # finer view (a narrower window keeps the candle count sane).
            entry_d = trade.filled_at.date() if trade.filled_at else d
            exit_d = trade.closed_at.date() if trade.closed_at else d
            hourly = (request.query_params.get("interval") or "1d").lower() == "1h"
            if hourly:
                interval, label = "ONE_HOUR", "1h"
                pad_before, pad_after = 3, 2
            else:
                interval, label = "ONE_DAY", "1d"
                pad_before, pad_after = 10, 5
            start = f"{(entry_d - timedelta(days=pad_before)).isoformat()} 09:15"
            end = f"{(exit_d + timedelta(days=pad_after)).isoformat()} 15:30"
        else:
            interval, label = "FIVE_MINUTE", "5m"
            start = f"{d.isoformat()} 09:15"
            end = f"{d.isoformat()} 15:30"

        candles = _fetch_candles(trade.symbol, start, end, interval)
        entry_ts, exit_ts = _resolve_entry_exit(candles, trade)
        strat = re.match(r"\[([^\]]+)\]", trade.reasoning or "")

        return Response({
            "trade_id": str(trade.id),
            "symbol": trade.symbol,
            "side": trade.side,
            "interval": label,
            "source": "swing" if swing else "intraday",
            "strategy": strat.group(1) if strat else "",
            "entry_price": float(trade.entry_price or 0),
            "stop": float(trade.stop_loss or 0) or None,
            "target": float(trade.target or 0) or None,
            "exit_price": float(trade.exit_price or 0) or None,
            "close_reason": trade.close_reason or "",
            "pnl": float(trade.realized_pnl or 0),
            "reasoning": trade.reasoning or "",
            "trade_date": d.isoformat(),
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "candles": candles,
        })


class TradeFeedbackView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request, trade_id):
        from apps.events.models import Event

        ev = (
            Event.objects.filter(
                tenant=request.tenant, type=Event.Type.TRADE_FEEDBACK, trade_id=trade_id,
            )
            .order_by("-ts").first()
        )
        if not ev:
            return Response({"vote": None, "note": ""})
        p = ev.payload or {}
        return Response({"vote": p.get("vote"), "note": p.get("note", ""), "ts": ev.ts.isoformat()})

    @extend_schema(
        request=inline_serializer(
            name="TradeFeedbackRequest",
            fields={
                "vote": serializers.ChoiceField(choices=["up", "down"]),
                "note": serializers.CharField(
                    required=False, allow_blank=True, max_length=1000,
                ),
            },
        ),
        responses=OpenApiTypes.OBJECT,
    )
    def post(self, request, trade_id):
        trade = Trade.objects.filter(tenant=request.tenant, id=trade_id).first()
        if not trade:
            return Response({"detail": "Trade not found."}, status=404)

        vote = str(request.data.get("vote", "")).lower()
        if vote not in ("up", "down"):
            return Response({"detail": "vote must be 'up' or 'down'."}, status=400)
        note = str(request.data.get("note", ""))[:1000]

        from apps.events.models import Event
        from apps.events.services.event_writer import emit

        emit(
            tenant=request.tenant,
            type=Event.Type.TRADE_FEEDBACK,
            text=f"Feedback {vote} on {trade.symbol} {trade.side} ({trade.trade_date})",
            trade_id=trade.id,
            payload={"vote": vote, "note": note, "symbol": trade.symbol,
                     "reasoning": trade.reasoning or ""},
        )
        return Response({"ok": True, "vote": vote, "note": note})
