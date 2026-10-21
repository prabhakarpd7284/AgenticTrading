"""Cooldown + sentiment filter in PlaceOrder (TRADING_RULES.md §1, §3).

Automated origins only; manual/UI orders are exempt. RiskEngine stays last.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from django.utils import timezone

from apps.agents_core.domain.contracts import RiskDecision
from apps.common.exceptions import RiskRejected
from apps.events.models import Event
from apps.news.models import NightlyScan, NightlyScanRow
from apps.trading.domain.entities import OrderDraft
from apps.trading.models import Order, Trade
from apps.trading.services.place_order import PlaceOrder

pytestmark = pytest.mark.django_db


class _Spy:
    def __init__(self):
        self.calls = []

    def validate(self, draft, **kwargs):
        self.calls.append(kwargs)
        return RiskDecision(approved=True)


def _draft(**over):
    base = dict(symbol="INFY", side="BUY", qty=5, order_type="LIMIT", product="INTRADAY",
                price=1500.0, sl=1490.0, tp=1520.0, origin="strategy")
    base.update(over)
    return OrderDraft(**base)


def _scan(label="neutral", event=False, for_date=None, allow_long=True, allow_short=True,
          status=NightlyScan.Status.SUCCESS):
    for_date = for_date or timezone.localdate()
    scan = NightlyScan.objects.create(for_date=for_date, session_date=for_date - timedelta(days=1),
                                      status=status, started_at=timezone.now())
    from apps.news.services.sentiment_filter import allowed_directions, allowed_label

    al, ash, reasons = allowed_directions(label=label, event_tomorrow=event,
                                          event_purpose="Financial Results")
    NightlyScanRow.objects.create(scan=scan, symbol="INFY", sentiment_label=label,
                                  event_tomorrow=event, allow_long=al, allow_short=ash,
                                  allowed=allowed_label(al, ash), reasons=reasons)
    return scan


def _loss_days(portfolio, n=3):
    from apps.market_data.services.market_calendar import is_trading_day

    d = timezone.localdate()
    days = []
    while len(days) < n:
        d -= timedelta(days=1)
        if is_trading_day(d):
            days.append(d)
    for day in days:
        Trade.objects.create(
            tenant=portfolio.tenant, portfolio=portfolio, symbol="TCS", side="BUY",
            entry_price=Decimal("100"), stop_loss=Decimal("95"), target=Decimal("110"),
            quantity=10, status=Trade.Status.CLOSED, exit_price=Decimal("95"),
            realized_pnl=Decimal("-50"), trade_date=day,
        )
    return days


def test_strong_negative_blocks_automated_buy(tenant, owner, paper_portfolio):
    _scan("strong_negative")
    with pytest.raises(RiskRejected, match="Sentiment filter"):
        PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, _draft())
    assert Order.objects.count() == 0
    ev = Event.objects.get(step_name="rules.sentiment_filter")
    assert ev.type == "risk.rejected" and ev.payload["rule"] == "sentiment"


def test_strong_negative_allows_automated_sell(tenant, owner, paper_portfolio):
    _scan("strong_negative")
    PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio,
                                    _draft(side="SELL", sl=1510.0, tp=1480.0))
    assert Order.objects.count() == 1


def test_strong_positive_blocks_automated_sell_not_buy(tenant, owner, paper_portfolio):
    _scan("strong_positive")
    with pytest.raises(RiskRejected):
        PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio,
                                        _draft(side="SELL", sl=1510.0, tp=1480.0))
    PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, _draft())
    assert Order.objects.count() == 1


def test_event_tomorrow_blocks_both_directions(tenant, owner, paper_portfolio):
    _scan("positive", event=True)
    for d in (_draft(), _draft(side="SELL", sl=1510.0, tp=1480.0)):
        with pytest.raises(RiskRejected, match="Financial Results"):
            PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, d)


def test_manual_orders_are_exempt(tenant, owner, paper_portfolio):
    _scan("strong_negative", event=True)
    _loss_days(paper_portfolio)
    PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, _draft(origin="ui"))
    assert Order.objects.count() == 1


def test_no_scan_does_not_block_but_tags(tenant, owner, paper_portfolio):
    res = PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, _draft())
    ev = Event.objects.get(step_name="rules.gate")
    assert ev.payload["order_id"] == res.id
    assert ev.payload["tags"]["sentiment"]["status"] == "no_scan"


def test_unscored_and_failed_scans_do_not_block(tenant, owner, paper_portfolio):
    _scan("unscored")
    PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, _draft())
    NightlyScan.objects.all().delete()
    _scan("strong_negative", status=NightlyScan.Status.FAILED)
    PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, _draft())
    assert Order.objects.count() == 2


def test_cooldown_blocks_automated_entry_and_records_shadow_trade(tenant, owner, paper_portfolio):
    _loss_days(paper_portfolio)
    with pytest.raises(RiskRejected, match="cooldown"):
        PlaceOrder(risk=_Spy()).execute(tenant, owner, paper_portfolio, _draft())
    assert Order.objects.count() == 0
    ev = Event.objects.get(step_name="rules.cooldown.shadow_trade")
    assert ev.payload["shadow"] is True and ev.payload["draft"]["symbol"] == "INFY"


def test_reduced_mode_passes_half_multiplier_to_risk_engine(tenant, owner, paper_portfolio,
                                                            monkeypatch):
    from apps.trading.services import trading_rules
    from apps.trading.services.cooldown import CooldownStatus

    monkeypatch.setattr(trading_rules, "evaluate_for_portfolio",
                        lambda p, d: CooldownStatus(as_of=d, status="reduced", risk_multiplier=0.5))
    spy = _Spy()
    PlaceOrder(risk=spy).execute(tenant, owner, paper_portfolio, _draft())
    assert spy.calls == [{"portfolio_id": paper_portfolio.id, "risk_multiplier": 0.5}]


def test_risk_engine_halves_risk_cap(paper_portfolio):
    from apps.trading.services.risk_engine import RiskEngine, TradeDraft

    # risk 4000 INR: inside 1% (5000) but outside 0.5% (2500) of 500k
    d = TradeDraft(symbol="INFY", side="BUY", entry_price=1000, stop_loss=990, target=1030,
                   quantity=400, confidence=0.9)
    eng = RiskEngine()
    assert eng.validate(d, portfolio_id=paper_portfolio.id).criteria["risk"]["max_allowed"] == 5000
    dec = eng.validate(d, portfolio_id=paper_portfolio.id, risk_multiplier=0.5)
    assert not dec.approved and "losing-streak" in dec.reason
