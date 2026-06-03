"""Options chain endpoint — exposes the broker-neutral OptionsChainSnapshot.

GET /api/v1/market-data/options-chain/?underlying=NIFTY&expiry=28MAY2026&strikes_window=10

Selection of which adapter quotes the chain:
  - User has an active BrokerLink ⇒ use that broker's options_chain().
  - No link, or link errored ⇒ fall back to the PaperBrokerAdapter so the
    UI is never empty. The response always includes `source` so the
    frontend can show "live" vs "paper" attribution.

The response shape mirrors OptionsChainSnapshot with `rows[].ce` and
`rows[].pe` as nullable OptionQuote dicts.
"""
from __future__ import annotations

import logging
from typing import Any

from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.market_data.adapters.factory import build_adapter
from apps.market_data.adapters.paper import PaperBrokerAdapter
from apps.market_data.models import BrokerLink

logger = logging.getLogger(__name__)


def _quote_to_dict(q) -> dict[str, Any]:
    """Serialise an OptionQuote (or None) for the API response."""
    if q is None:
        return None  # type: ignore[return-value]
    return {
        "token": q.token,
        "symbol": q.symbol,
        "strike": q.strike,
        "opt": q.opt,
        "ltp": q.ltp,
        "bid": q.bid,
        "ask": q.ask,
        "bid_qty": q.bid_qty,
        "ask_qty": q.ask_qty,
        "volume": q.volume,
        "oi": q.oi,
        "oi_change": q.oi_change,
        "iv": q.iv,
        "delta": q.delta,
        "gamma": q.gamma,
        "theta": q.theta,
        "vega": q.vega,
        "mid": q.mid,
        "spread_bps": q.spread_bps,
    }


def _snapshot_to_dict(s) -> dict[str, Any]:
    if s is None:
        return {}
    return {
        "underlying": s.underlying,
        "spot": s.spot,
        "expiry": s.expiry,
        "fetched_at": s.fetched_at.isoformat() if s.fetched_at else None,
        "source": s.source,
        "vix": s.vix,
        "pcr_oi": s.pcr_oi,
        "pcr_volume": s.pcr_volume,
        "atm_strike": s.atm_strike,
        "rows": [
            {
                "strike": r.strike,
                "ce": _quote_to_dict(r.ce),
                "pe": _quote_to_dict(r.pe),
            }
            for r in s.rows
        ],
    }


class OptionsChainView(APIView):
    """GET /api/v1/market-data/options-chain/

    Query params:
      underlying      NIFTY | BANKNIFTY | SENSEX  (required, default NIFTY)
      expiry          DDMMMYYYY uppercase, e.g. "28MAY2026" (optional)
      strikes_window  int, default 10 (gives 2N+1 strikes around ATM)
      source          paper | broker  (optional override — useful for the
                                       Options Desk while broker links
                                       are still being set up)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        underlying = (request.query_params.get("underlying") or "NIFTY").upper()
        expiry = request.query_params.get("expiry") or None
        try:
            strikes_window = int(request.query_params.get("strikes_window", 10))
        except (TypeError, ValueError):
            strikes_window = 10
        force_source = (request.query_params.get("source") or "").lower()

        snapshot = None
        attempted: list[dict[str, str]] = []   # [{name, ok, error?}]

        # 1) Try a live broker — prefer the default link, then most-recently-refreshed.
        if force_source != "paper":
            link = (
                BrokerLink.objects
                .filter(tenant=request.tenant, status="active")
                .order_by("-is_default", "-last_refreshed_at", "-created_at")
                .first()
            )
            if link is not None:
                adapter = build_adapter(link)
                if adapter is None:
                    attempted.append({
                        "name": link.broker_name, "ok": False,
                        "error": "adapter_unavailable",
                    })
                else:
                    try:
                        snapshot = adapter.options_chain(
                            underlying, expiry=expiry, strikes_window=strikes_window,
                        )
                        if snapshot is None:
                            attempted.append({
                                "name": link.broker_name, "ok": False,
                                "error": "broker_returned_empty",
                            })
                        else:
                            attempted.append({"name": link.broker_name, "ok": True})
                    except Exception as e:
                        logger.warning("options_chain.live_failed broker=%s err=%s",
                                        link.broker_name, e)
                        attempted.append({
                            "name": link.broker_name, "ok": False,
                            "error": str(e)[:200],
                        })

        # 2) Fall back to paper synthesis so the page is never empty.
        fallback_used = snapshot is None and force_source != "broker"
        if snapshot is None:
            snapshot = PaperBrokerAdapter().options_chain(
                underlying, expiry=expiry, strikes_window=strikes_window,
            )
            if snapshot is not None:
                attempted.append({"name": "paper", "ok": True})

        if not snapshot:
            return Response(
                {"error": "no_chain", "attempted_sources": attempted},
                status=503,
            )

        payload = _snapshot_to_dict(snapshot)
        payload["attempted_sources"] = attempted
        payload["is_fallback"] = fallback_used
        return Response(payload)
