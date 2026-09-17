"""
Horizon-aware swing enrichment (enrich_signals_v2).

Proves the behaviour v1 gets wrong: a swing whose target is hit on day 3 is
enriched over the multi-day window (max-move measured across days, outcome
resolved to a target hit) instead of being marked EXPIRED the same evening.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from unittest.mock import patch

import pytest
from django.utils import timezone


def _candle(d, o, h, l, c, v=1000):
    return {"date": d.strftime("%Y-%m-%d"), "open": o, "high": h, "low": l,
            "close": c, "volume": v}


def _mk_signal(tenant, symbol, strategy, side="BUY", entry=100.0, stop=96.0,
               target=110.0, days_ago=5):
    from apps.strategies.models import Signal
    sig_date = timezone.now().date() - timedelta(days=days_ago)
    return Signal.objects.create(
        tenant=tenant,
        symbol=symbol,
        signal_date=sig_date,
        signal_time=timezone.make_aware(datetime.combine(sig_date, time(15, 30))),
        source=Signal.Source.OK_SCANNER,
        strategy=strategy,
        side=side,
        entry_price=entry,
        stoploss=stop,
        target=target,
        confidence=0.8,
        risk_reward=2.5,
        reasons=["test"],
        indicators={"version": "v2", "tier": "medium", "time_stop_bars": 40},
        outcome=Signal.Outcome.PENDING,
    )


class _StubDS:
    """Stand-in for DataService — returns canned candles, no broker."""
    by_symbol: dict = {}

    def __init__(self, *a, **k):
        pass

    def fetch_historical(self, symbol, frm, to, interval="ONE_DAY"):
        return _StubDS.by_symbol.get(symbol, [])


@pytest.mark.django_db
def test_swing_target_hit_on_day3_resolves_over_window():
    from tests.factories import TenantFactory
    from apps.strategies.models import Signal

    tenant = TenantFactory()
    sig = _mk_signal(tenant, "WINSYM", "swing_v2_medium_BB")

    base = sig.signal_date
    # Day +1, +2 drift up; day +3 spikes through the 110 target; then drifts.
    _StubDS.by_symbol = {
        "WINSYM": [
            _candle(base + timedelta(days=1), 100, 103, 99, 102),
            _candle(base + timedelta(days=2), 102, 105, 101, 104),
            _candle(base + timedelta(days=3), 104, 111, 103, 109),   # target hit
            _candle(base + timedelta(days=4), 109, 112, 107, 110),
            _candle(base + timedelta(days=5), 110, 113, 108, 112),
        ],
    }

    # Call the command directly: `trading` commands aren't registered with
    # get_commands() under test settings (they run via call_command in prod
    # tasks). Invoking handle() exercises the full enrichment path regardless.
    from trading.management.commands.enrich_signals_v2 import Command
    with patch("trading.services.data_service.DataService", _StubDS):
        Command().handle(tenant=None, tier=None, all=False)

    sig.refresh_from_db()
    assert sig.max_favorable_move == pytest.approx(11.0, abs=0.01)   # 111 - 100
    assert sig.indicators["swing"]["result"] == "target"
    assert sig.indicators["swing"]["hold_bars"] == 3                 # hit on the 3rd bar
    assert sig.indicators["swing"]["realized_r"] == pytest.approx(2.5, abs=0.01)
    assert sig.outcome == Signal.Outcome.EXPIRED                     # resolved (not same-day-only)


@pytest.mark.django_db
def test_open_swing_stays_pending_inside_window():
    from tests.factories import TenantFactory
    from apps.strategies.models import Signal

    tenant = TenantFactory()
    sig = _mk_signal(tenant, "OPENSYM", "swing_v2_medium_WP")

    base = sig.signal_date
    # Neither target (110) nor stop (96) touched; window (~13d) still open.
    _StubDS.by_symbol = {
        "OPENSYM": [
            _candle(base + timedelta(days=1), 100, 104, 99, 103),
            _candle(base + timedelta(days=2), 103, 105, 101, 102),
            _candle(base + timedelta(days=3), 102, 106, 100, 105),
        ],
    }

    # Call the command directly: `trading` commands aren't registered with
    # get_commands() under test settings (they run via call_command in prod
    # tasks). Invoking handle() exercises the full enrichment path regardless.
    from trading.management.commands.enrich_signals_v2 import Command
    with patch("trading.services.data_service.DataService", _StubDS):
        Command().handle(tenant=None, tier=None, all=False)

    sig.refresh_from_db()
    assert sig.outcome == Signal.Outcome.PENDING                     # still live
    assert sig.indicators["swing"]["result"] == "open"
    assert sig.max_favorable_move == pytest.approx(6.0, abs=0.01)    # 106 - 100
    assert sig.max_adverse_move == pytest.approx(1.0, abs=0.01)      # 100 - 99 (day+1 low)
