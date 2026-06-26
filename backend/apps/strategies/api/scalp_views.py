"""Create a scalp simulation run.

Unlike ``AgentRunViewSet`` (which enqueues a Celery worker), the scalp sim is
*interactive*: this endpoint only creates the ``AgentRun`` row (config validated
against the scalp JSONSchema) and returns its id. The client then opens
``ws/scalp/<run_id>/`` to drive the controllable playback. No Celery is enqueued,
so the worker never races the WebSocket consumer.
"""
from __future__ import annotations

from jsonschema import Draft7Validator
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.agents_core.models import AgentRun
from apps.agents_core.registry import strategy_registry


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_scalp_run(request):
    config = dict(request.data or {})
    try:
        strat = strategy_registry.get("scalp")
    except (KeyError, LookupError):
        return Response({"detail": "scalp strategy not registered"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    errors = sorted(Draft7Validator(strat.schema().params or {}).iter_errors(config),
                    key=lambda e: e.path)
    if errors:
        return Response(
            {"config": [{"path": list(e.absolute_path), "message": e.message} for e in errors]},
            status=status.HTTP_400_BAD_REQUEST,
        )

    portfolio = _default_portfolio(request)
    if portfolio is None:
        return Response({"detail": "No portfolio for this tenant — create one first."},
                        status=status.HTTP_400_BAD_REQUEST)

    run = AgentRun.objects.create(
        tenant=request.tenant, triggered_by=request.user,
        strategy_name=strat.name, strategy_version=strat.version,
        portfolio=portfolio, config=config, status=AgentRun.Status.QUEUED,
    )
    return Response({"run_id": str(run.id)}, status=status.HTTP_201_CREATED)


def _default_portfolio(request):
    from apps.trading.models import Portfolio

    qs = Portfolio.objects.filter(tenant=request.tenant)
    return qs.filter(mode="paper").first() or qs.first()


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def scalp_defaults(request):
    """Resolved default session date + nearest expiry for an underlying, so the
    UI can pre-fill the Date/Expiry fields. ``date`` is last completed trading
    day (YYYY-MM-DD); ``expiry`` is DDMMMYY (nearest weekly/monthly)."""
    underlying = request.query_params.get("underlying", "NIFTY").upper()
    from datetime import date

    from plugins.strategy_scalp.data import resolve_session
    from plugins.broker_fyers.symbols import list_expiries

    try:
        day, exp = resolve_session(underlying, "", "")
        today = date.today()
        upcoming = [e for e in list_expiries(underlying) if e >= today][:12]
        strike, strike_step = _strike_default(request, underlying, exp)
        return Response({
            "date": day,
            "expiry": exp.strftime("%d%b%y").upper(),
            "expiries": [e.strftime("%d%b%y").upper() for e in upcoming],
            "strike": strike,
            "strike_step": strike_step,
        })
    except Exception as exc:  # noqa: BLE001 — never block the form on a resolution miss
        return Response({"date": "", "expiry": "", "expiries": [], "strike": None,
                         "strike_step": 50, "detail": str(exc)})


# Strike interval per index (fallback when the live chain is unavailable).
_STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100, "FINNIFTY": 50}


def _strike_default(request, underlying, exp):
    """(ATM strike, strike step) — best-effort from the live option chain so
    switching underlying lands on a valid ATM strike; falls back to a per-index
    step and leaves the strike unset (frontend keeps the current value)."""
    step = _STRIKE_STEP.get(underlying, 50)
    try:
        from apps.market_data.adapters.factory import build_adapter
        from plugins.strategy_scalp.data import _fyers_link

        adapter = build_adapter(_fyers_link(getattr(request, "tenant", None)))
        snap = adapter.options_chain(underlying, exp.strftime("%d%b%Y"), strikes_window=8)
        if snap and getattr(snap, "atm_strike", None):
            strikes = sorted({r.strike for r in (snap.rows or []) if r.strike})
            gaps = [b - a for a, b in zip(strikes, strikes[1:]) if b > a]
            return int(snap.atm_strike), (min(gaps) if gaps else step)
    except Exception:  # noqa: BLE001 — chain may be unavailable (market closed / no link)
        pass
    return None, step
