"""Pine Script generator + endpoints.

The integration we care about: an operator exports a screener strategy as a
TradingView Pine v5 indicator; its ``alert()`` fires the SAME webhook JSON our
parser accepts; that flows back as a ``Signal``. So the tests assert two things
the feature lives or dies on:

  1. Every enabled strategy generates a script that can actually FIRE — no
     condition silently degraded to ``false``/``na`` (which would make the
     indicator inert on a chart).
  2. The alert JSON the script emits round-trips through the REAL parser
     (``apps.notifications.services.tradingview.parse_payload``) into the right
     symbol / action / price / strategy.
"""
from __future__ import annotations

import json
import re

import pytest

from apps.notifications.models import TradingViewLink
from apps.notifications.services.pinescript import (
    find_enabled_strategy, generate_pine, list_pine_strategies,
)
from apps.notifications.services.tradingview import parse_payload


pytestmark = pytest.mark.django_db

STRATEGIES_URL = "/api/v1/notifications/tradingview/pine-strategies/"
PINE_URL = "/api/v1/notifications/tradingview/pine/"


def _render_alert(code: str) -> str:
    """Simulate TradingView's runtime concatenation of the alert() message:
    syminfo.ticker → a symbol, str.tostring(close) → a price, SIDE/STRAT are
    baked constants. Returns the JSON string a real alert would POST."""
    msg = re.search(r"alert\('(.+?)', alert", code)
    assert msg, "generated script has no alert() call"
    side = re.search(r'SIDE  = "(.+?)"', code).group(1)
    strat = re.search(r'STRAT = "(.+?)"', code).group(1)
    return (
        msg.group(1)
        .replace("' + syminfo.ticker + '", "RELIANCE")
        .replace("' + SIDE + '", side)
        .replace("' + str.tostring(close) + '", "1455.5")
        .replace("' + STRAT + '", strat)
    )


# ── Generator unit tests ─────────────────────────────────────────────────

class TestGenerator:
    def test_every_enabled_strategy_can_fire(self):
        """No enabled strategy may degrade to an always-false fireSignal — that
        would ship an inert indicator. Guards against a new strategy adding a
        condition type the generator doesn't translate yet."""
        strategies = list_pine_strategies()
        assert strategies, "no enabled strategies to export"
        for s in strategies:
            code = generate_pine(find_enabled_strategy(s["key"]))
            assert code.startswith("//@version=5")
            assert "= false  // unsupported" not in code, (
                f"{s['key']} has an unsupported condition → can never fire"
            )
            # An INDICATOR_COMPARE whose indicator didn't resolve becomes
            # `indN = na` → the condition is dead. Catch that too.
            assert not re.search(r"^ind\d+ = na$", code, re.M), (
                f"{s['key']} has an unresolved indicator → dead condition"
            )
            assert "fireSignal = false" not in code

    def test_alert_json_round_trips_through_parser(self):
        """The whole point: generated alert JSON parses into a faithful Signal."""
        for s in list_pine_strategies():
            strat = find_enabled_strategy(s["key"])
            rendered = _render_alert(generate_pine(strat))
            # Valid JSON …
            json.loads(rendered)
            # … and the real parser extracts the right fields.
            parsed = parse_payload(rendered, "application/json")
            assert parsed.symbol == "RELIANCE"
            assert parsed.action == strat.side.upper()
            assert parsed.price == 1455.5
            assert parsed.strategy == strat.name

    def test_is_pine_v5_not_v4(self):
        code = generate_pine(find_enabled_strategy("ema-crossover-trend"))
        assert "//@version=5" in code
        # v4-isms that would fail to compile on a v5 indicator.
        assert "study(" not in code
        assert re.search(r"\bsecurity\(", code.replace("request.security(", "")) is None

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            generate_pine("does-not-exist")


# ── Endpoint tests ───────────────────────────────────────────────────────

class TestPineStrategiesEndpoint:
    def test_lists_enabled_strategies(self, auth_client):
        resp = auth_client.get(STRATEGIES_URL)
        assert resp.status_code == 200, resp.content
        rows = resp.json()
        keys = {r["key"] for r in rows}
        # Enabled set includes these; the disabled BB-fade / morning-range are out.
        assert {"breakout-long", "ema-crossover-trend", "vwap-bounce-long"} <= keys
        for r in rows:
            assert r["key"] and r["label"] and r["side"] in ("BUY", "SELL", "BOTH")

    def test_requires_auth(self, api_client):
        assert api_client.get(STRATEGIES_URL).status_code == 401


class TestPineEndpoint:
    def test_returns_code_for_known_strategy(self, auth_client):
        resp = auth_client.get(PINE_URL, {"strategy": "breakout-long"})
        assert resp.status_code == 200, resp.content
        body = resp.json()
        assert body["strategy"] == "breakout-long"
        code = body["code"]
        assert code.startswith("//@version=5")
        assert 'STRAT = "Breakout Long"' in code
        assert "alert(" in code and '"symbol":' in code

    def test_missing_strategy_param_is_400(self, auth_client):
        resp = auth_client.get(PINE_URL)
        assert resp.status_code == 400

    def test_unknown_strategy_is_400(self, auth_client):
        resp = auth_client.get(PINE_URL, {"strategy": "nope"})
        assert resp.status_code == 400

    def test_requires_auth(self, api_client):
        assert api_client.get(PINE_URL, {"strategy": "breakout-long"}).status_code == 401

    def test_link_embeds_webhook_url(self, auth_client, owner):
        link = TradingViewLink.objects.create(
            tenant=owner.memberships.first().tenant,
            owner=owner,
            display_name="pine link",
        )
        resp = auth_client.get(PINE_URL, {"strategy": "breakout-long", "link": str(link.id)})
        assert resp.status_code == 200
        assert link.webhook_secret in resp.json()["code"]

    def test_other_tenants_link_secret_not_embedded(self, auth_client, two_tenants):
        """Passing a link the caller doesn't own must NOT leak its secret."""
        other = TradingViewLink.objects.create(
            tenant=two_tenants.b,
            owner=two_tenants.owner_b,
            display_name="not yours",
        )
        resp = auth_client.get(PINE_URL, {"strategy": "breakout-long", "link": str(other.id)})
        assert resp.status_code == 200
        assert other.webhook_secret not in resp.json()["code"]
