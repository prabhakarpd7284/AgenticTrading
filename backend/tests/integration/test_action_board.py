"""Action Board (UX redesign R3): build, dry-run risk, decisions, replay, API.

The Claude planner and market data are always mocked — tests never call the
CLI or Angel.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from django.utils import timezone

from apps.events.models import Event
from apps.news.models import NightlyScan, NightlyScanRow
from apps.system.models import RunReceipt
from apps.system.services.run_receipts import tracked_run
from apps.trading.models import Order, TodayAction, Trade
from apps.trading.services import action_board as ab
from apps.trading.services import action_decisions as decisions
from apps.trading.services.action_outcomes import evaluate_outcomes, simulate_plan, track_record
from tests.factories import MembershipFactory, PortfolioFactory, TenantFactory, UserFactory

pytestmark = pytest.mark.django_db

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 10, 21)            # a Wednesday, NSE trading day
LIVE_NOW = datetime(2026, 10, 21, 9, 36, tzinfo=IST)


# ─── helpers ────────────────────────────────────────────────────────────

def _row(scan, symbol, *, prev_c, allow_long=True, allow_short=True, event=False, data_date=None):
    allowed = "none" if event else ("both" if allow_long and allow_short else
                                    "long" if allow_long else "short" if allow_short else "none")
    return NightlyScanRow.objects.create(
        scan=scan, symbol=symbol, levels={"prev_hlc": {"h": prev_c * 1.01, "l": prev_c * 0.99, "c": prev_c}},
        allow_long=allow_long and not event, allow_short=allow_short and not event, allowed=allowed,
        event_tomorrow=event, event_purpose="Financial Results" if event else "", cpr_class="narrow",
        two_day_relationship="higher value", data_date=data_date or scan.session_date)


def _scan(day=DAY):
    scan = NightlyScan.objects.create(for_date=day, session_date=day - timedelta(days=1),
                                      status=NightlyScan.Status.SUCCESS, started_at=timezone.now())
    _row(scan, "INFY", prev_c=1480.0)                        # +1.35% at 1500
    _row(scan, "TCS", prev_c=3000.0, allow_short=False)      # +2.0%  (long only)
    _row(scan, "HDFCBANK", prev_c=1700.0)                    # -0.6%
    _row(scan, "ITC", prev_c=400.0)                          # +0.1%  (outside cap 3)
    _row(scan, "RELIANCE", prev_c=2900.0, event=True)        # excluded: results due
    return scan


LAST = {"INFY": 1500.0, "TCS": 3060.0, "HDFCBANK": 1690.0, "ITC": 400.4}


def _bars(day, last, n=21, until=None):
    t0 = datetime.combine(day, time(9, 15), tzinfo=IST)
    out = []
    for i in range(n):
        t = t0 + timedelta(minutes=i)
        if until is not None and t >= until:
            break
        out.append([t, last, last * 1.001, last * 0.999, last, 1000])
    return out


PLANS = {
    "INFY": {"symbol": "INFY", "side": "BUY", "entry_price": 1500.0, "stop_loss": 1490.0, "target": 1520.0,
             "quantity": 999, "reasoning": "Opening range break above PP with rising volume.", "confidence": 0.7},
    "TCS": {"symbol": "TCS", "side": "SELL", "entry_price": 3060.0, "stop_loss": 3080.0, "target": 3020.0,
            "quantity": 5, "reasoning": "Fade the gap.", "confidence": 0.7},
    "HDFCBANK": {"error": "Claude CLI timed out (120s)."},
    "ITC": {"symbol": "ITC", "side": "BUY", "entry_price": 400.0, "stop_loss": 398.0, "target": 404.0,
            "quantity": 1, "reasoning": "x", "confidence": 0.7},
}


@pytest.fixture
def mocked(monkeypatch):
    calls = []

    def planner(intent, summary, rag):
        sym = intent.split(" for ")[1].split(" ")[0]
        calls.append(sym)
        return dict(PLANS[sym])

    monkeypatch.setattr(ab, "call_planner", planner)
    monkeypatch.setattr(ab, "rag_context", lambda s: "")
    monkeypatch.setattr(ab, "load_bars", lambda sym, day, until: (_bars(day, LAST[sym], until=until), "db_1m"))
    return calls


def _build(**kw):
    with tracked_run("job", "action_board", {"date": DAY.isoformat()}, trigger="manual",
                     task_name="apps.trading.tasks.action_board.build_action_board") as run:
        out = ab.build_action_board(DAY, run=run, now=kw.pop("now", LIVE_NOW), telegram=False, **kw)
    return out, RunReceipt.objects.filter(name="action_board").order_by("-started_at").first()


# ─── build ──────────────────────────────────────────────────────────────

def test_build_ranks_caps_plans_sizes_and_dry_runs(tenant, owner, paper_portfolio, mocked):
    _scan()
    out, receipt = _build(max_planner_calls=3)

    # Ranking: |move| desc → TCS (2.0%), INFY (1.35%), HDFCBANK (0.59%); ITC beyond the cap.
    assert mocked == ["TCS", "INFY", "HDFCBANK"]
    assert out["candidates"] == 4 and out["planner_calls"] == 3 and out["planned"] == 1
    assert out["per_symbol"]["TCS"]["result"] == "direction_blocked"
    assert out["per_symbol"]["HDFCBANK"]["result"] == "failed"
    assert "timed out" in out["per_symbol"]["HDFCBANK"]["detail"]

    card = TodayAction.objects.get(symbol="INFY")
    assert card.tenant_id == tenant.id and card.side == "BUY" and card.strategy == "directional"
    # Deterministic sizing (1% risk → 500 sh, capped by 10% position → 33), never the LLM's 999.
    assert card.suggested_qty == 33
    assert card.risk_passed and card.status == TodayAction.Status.PROPOSED
    assert card.risk_check["mode"] == "dry_run" and card.risk_check["failed_criterion"] == ""
    assert card.risk_at_stake == Decimal("330.00")
    assert card.expires_at == datetime(2026, 10, 21, 15, 0, tzinfo=IST)
    assert len(card.reasons) == 3 and card.source_refs["nightly_scan_row_id"]
    assert card.source_refs["planner"]["llm_quantity_ignored"] == 999
    assert Order.objects.count() == 0          # a dry run never places anything
    assert not TodayAction.objects.filter(symbol="TCS").exists()

    assert receipt.status == "partial"         # one planner call failed, the rest worked
    names = {c["name"]: c["status"] for c in receipt.checks}
    assert names["scan_present"] == "pass" and names["data_fresh"] == "pass"
    assert names["planner"] == "warn"
    assert Event.objects.filter(step_name="action_board.card").count() == 1
    assert Event.objects.filter(step_name="action_board.scan_direction").count() == 1


def test_rebuild_is_idempotent_and_never_overwrites_a_decision(tenant, owner, paper_portfolio, mocked):
    _scan()
    _build(max_planner_calls=3)
    card = TodayAction.objects.get(symbol="INFY")
    TodayAction.objects.filter(pk=card.pk).update(status="skipped", decision_note="not today")
    _build(max_planner_calls=3)
    assert TodayAction.objects.filter(symbol="INFY").count() == 1
    card.refresh_from_db()
    assert card.status == "skipped" and card.decision_note == "not today"


def test_no_scan_fails_loudly_and_builds_nothing(tenant, owner, paper_portfolio, mocked):
    out, receipt = _build()
    assert receipt.status == "failed"
    assert any(c["name"] == "scan_present" and c["status"] == "fail" for c in receipt.checks)
    assert mocked == [] and TodayAction.objects.count() == 0


def test_stale_data_fails_check_and_skips_planner(tenant, owner, paper_portfolio, mocked, monkeypatch):
    _scan()
    # Bars stop at 09:20 — 16 minutes before the as-of time.
    monkeypatch.setattr(ab, "load_bars", lambda sym, day, until: (_bars(day, LAST[sym], n=6), "angel_1m"))
    monkeypatch.setitem(ab.DEFAULTS, "MIN_BARS", 3)
    from django.conf import settings
    monkeypatch.setattr(settings, "ACTION_BOARD", {**settings.ACTION_BOARD, "MIN_BARS": 3})
    out, receipt = _build()
    assert out["error"] == "stale data" and mocked == []
    assert receipt.status == "failed"
    assert any(c["name"] == "data_fresh" and c["status"] == "fail" for c in receipt.checks)


def test_stale_scan_levels_fail_before_any_planner_call(tenant, owner, paper_portfolio, mocked):
    scan = _scan()
    scan.rows.update(data_date=DAY - timedelta(days=2))     # levels predate the expected session
    out, receipt = _build()
    assert out["error"] == "stale scan levels" and mocked == []
    assert receipt.status == "failed"
    assert any(c["name"] == "levels_fresh" and c["status"] == "fail" for c in receipt.checks)


def test_planner_summary_says_session_so_far():
    bars = _bars(DAY, 1500.0)
    text = ab.market_summary("INFY", DAY, bars, LIVE_NOW)
    first = text.splitlines()[0]
    assert "session in progress" in first and "09:15 to 09:36" in first
    assert "completed session" not in text


def test_cooldown_records_shadow_cards(tenant, owner, paper_portfolio, mocked):
    from apps.market_data.services.market_calendar import is_trading_day

    _scan()
    d, n = DAY, 0
    while n < 3:
        d -= timedelta(days=1)
        if is_trading_day(d):
            Trade.objects.create(tenant=tenant, portfolio=paper_portfolio, symbol="TCS", side="BUY",
                                 entry_price=Decimal("100"), stop_loss=Decimal("95"), target=Decimal("110"),
                                 quantity=10, status=Trade.Status.CLOSED, exit_price=Decimal("95"),
                                 realized_pnl=Decimal("-50"), trade_date=d)
            n += 1
    _build(max_planner_calls=3)
    card = TodayAction.objects.get(symbol="INFY")
    assert card.shadow and card.status == "rejected"
    assert card.risk_check["failed_criterion"] == "cooldown"
    assert Event.objects.filter(step_name="rules.cooldown.shadow_trade").exists()


def test_skips_weekends_and_too_early(tenant, owner, paper_portfolio, mocked):
    out, receipt = _build(now=datetime(2026, 10, 21, 9, 5, tzinfo=IST))
    assert "too early" in out["skipped"] and receipt.status == "skipped"
    with tracked_run("job", "action_board", {}) as run:
        out = ab.build_action_board(date(2026, 10, 24), run=run, now=LIVE_NOW, telegram=False)
    assert "not an NSE trading day" in out["skipped"]


# ─── decisions ──────────────────────────────────────────────────────────

def _card(portfolio, **over):
    now = timezone.now()
    base = dict(
        tenant=portfolio.tenant, portfolio=portfolio, date=timezone.localdate(), kind="BUY", symbol="INFY",
        strategy="directional", strategy_version="1.0.0", side="BUY", entry_price=Decimal("1500"),
        entry_low=Decimal("1500"), entry_high=Decimal("1500"), stoploss=Decimal("1490"),
        target=Decimal("1520"), suggested_qty=33, capital_required=Decimal("49500"),
        risk_at_stake=Decimal("330"), risk_check={"approved": True}, risk_passed=True,
        confidence=Decimal("0.7"), reasons=["a", "b", "c"], planned_as_of=now - timedelta(minutes=5),
        expires_at=now + timedelta(hours=1), status="proposed")
    base.update(over)
    return TodayAction.objects.create(**base)


def test_approve_places_paper_order_through_place_order(tenant, owner, paper_portfolio, auth_client):
    card = _card(paper_portfolio)
    r = auth_client.post(f"/api/v1/actions/{card.id}/approve/", {"note": "go"}, format="json")
    assert r.status_code == 200, r.content
    body = r.json()
    assert body["status"] == "executed" and body["order"]["status"] == "queued"
    order = Order.objects.get()
    assert order.origin == "strategy" and order.order_type == "LIMIT" and order.qty == 33
    assert order.idempotency_key == f"action-{card.id}" and order.portfolio_id == paper_portfolio.id
    ev = Event.objects.get(step_name="action_board.approve")
    assert ev.type == "order.queued" and ev.actor_user_id == owner.id
    # Second approve → conflict, still one order.
    assert auth_client.post(f"/api/v1/actions/{card.id}/approve/").status_code == 409
    assert Order.objects.count() == 1


def test_approve_low_confidence_is_rejected_and_journaled(tenant, owner, paper_portfolio, auth_client):
    card = _card(paper_portfolio, confidence=Decimal("0.3"))
    r = auth_client.post(f"/api/v1/actions/{card.id}/approve/")
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    assert "Confidence" in r.json()["decision_note"]
    assert Order.objects.count() == 0
    assert Event.objects.get(step_name="action_board.approve").type == "risk.rejected"


def test_approve_refuses_live_portfolio(tenant, owner, paper_portfolio, auth_client):
    paper_portfolio.mode = "live"
    paper_portfolio.save()
    card = _card(paper_portfolio)
    r = auth_client.post(f"/api/v1/actions/{card.id}/approve/")
    assert r.status_code == 409 and "PAPER-only" in r.json()["detail"]
    assert Order.objects.count() == 0


def test_approve_respects_kill_switch(tenant, owner, paper_portfolio, auth_client):
    from apps.system.models import SystemControl

    SystemControl.objects.create(tenant=tenant, key="kill_switch", value=True)
    card = _card(paper_portfolio)
    r = auth_client.post(f"/api/v1/actions/{card.id}/approve/")
    assert r.json()["status"] == "rejected" and "kill switch" in r.json()["decision_note"]
    assert Order.objects.count() == 0


def test_skip_and_snooze(tenant, owner, paper_portfolio, auth_client):
    a = _card(paper_portfolio, symbol="INFY")
    b = _card(paper_portfolio, symbol="TCS")
    r = auth_client.post(f"/api/v1/actions/{a.id}/skip/", {"reason": "earnings week"}, format="json")
    assert r.status_code == 200 and r.json()["status"] == "skipped"
    assert r.json()["decision_note"] == "earnings week"
    r = auth_client.post(f"/api/v1/actions/{b.id}/snooze/", {"minutes": 15}, format="json")
    assert r.status_code == 200 and r.json()["status"] == "snoozed"
    # Snooze past expiry is refused.
    c = _card(paper_portfolio, symbol="ITC")
    assert auth_client.post(f"/api/v1/actions/{c.id}/snooze/", {"minutes": 600},
                            format="json").status_code == 409
    assert auth_client.post(f"/api/v1/actions/{c.id}/snooze/", {}, format="json").status_code == 400
    steps = set(Event.objects.values_list("step_name", flat=True))
    assert {"action_board.skip", "action_board.snooze"} <= steps


def test_lazy_expiry_and_unsnooze(tenant, owner, paper_portfolio, auth_client):
    now = timezone.now()
    old = _card(paper_portfolio, symbol="INFY", expires_at=now - timedelta(minutes=1))
    snz = _card(paper_portfolio, symbol="TCS", status="snoozed", snoozed_until=now - timedelta(minutes=1))
    r = auth_client.get("/api/v1/actions/today/")
    assert r.status_code == 200
    by = {c["symbol"]: c["status"] for c in r.json()["cards"]}
    assert by == {"INFY": "expired", "TCS": "proposed"}
    assert auth_client.post(f"/api/v1/actions/{old.id}/approve/").status_code == 409
    assert Event.objects.filter(step_name="action_board.expire").count() == 1
    assert snz.id


def test_tenant_isolation_and_roles(two_tenants, api_client):
    from rest_framework_simplejwt.tokens import RefreshToken

    from apps.accounts.api.jwt import _attach_tenant_claims

    t = two_tenants
    card_b = _card(t.portfolio_b)

    def client_for(user):
        from rest_framework.test import APIClient

        c = APIClient()
        tok = RefreshToken.for_user(user).access_token
        _attach_tenant_claims(tok, user)
        c.force_authenticate(user=user)
        c.credentials(HTTP_AUTHORIZATION=f"Bearer {tok}")
        return c

    a = client_for(t.owner_a)
    assert a.get("/api/v1/actions/today/").json()["cards"] == []
    assert a.post(f"/api/v1/actions/{card_b.id}/approve/").status_code == 404
    assert a.post(f"/api/v1/actions/{card_b.id}/skip/").status_code == 404
    card_b.refresh_from_db()
    assert card_b.status == "proposed"

    viewer = UserFactory()
    MembershipFactory(user=viewer, tenant=t.b, role="viewer")
    v = client_for(viewer)
    body = v.get("/api/v1/actions/today/").json()
    assert len(body["cards"]) == 1 and body["can_decide"] is False
    assert v.post(f"/api/v1/actions/{card_b.id}/skip/").status_code == 403
    assert client_for(t.owner_b).post(f"/api/v1/actions/{card_b.id}/skip/").status_code == 200


def test_today_endpoint_shape_and_board_status(tenant, owner, paper_portfolio, auth_client):
    r = auth_client.get("/api/v1/actions/today/")
    body = r.json()
    assert r.status_code == 200
    assert body["strategy"]["stage"] == "unproven — forward paper test"
    assert body["board"]["built"] is False and body["board"]["schedule_label"].endswith("IST")
    tr = body["track_record"]["directional"]
    assert tr["paper"]["n"] == 0 and tr["paper"]["label"] == "n=0 · unproven" and tr["proven"] is False


# ─── replay + track record ──────────────────────────────────────────────

def _t(h, m):
    return datetime(2026, 10, 21, h, m, tzinfo=IST)


def test_simulate_target_stop_noentry_eod():
    eod = _t(15, 14)
    bars = [[_t(9, 36), 1505, 1506, 1499, 1500, 1],   # fills at 1500
            [_t(9, 37), 1500, 1521, 1499, 1520, 1]]   # target
    out = simulate_plan(side="BUY", entry=1500, stop=1490, target=1520, qty=10, start=_t(9, 36), bars=bars, eod=eod)
    assert out["status"] == "target" and out["r"] == 2.0 and out["pnl"] == 200.0
    both = [[_t(9, 36), 1500, 1500, 1500, 1500, 1], [_t(9, 37), 1500, 1525, 1485, 1500, 1]]
    assert simulate_plan(side="BUY", entry=1500, stop=1490, target=1520, qty=10, start=_t(9, 36),
                         bars=both, eod=eod)["status"] == "stop"
    never = [[_t(9, 36), 1510, 1515, 1505, 1510, 1]]
    assert simulate_plan(side="BUY", entry=1500, stop=1490, target=1520, qty=10, start=_t(9, 36),
                         bars=never, eod=eod)["status"] == "no_entry"
    flat = [[_t(9, 36), 1500, 1501, 1500, 1500, 1], [_t(15, 13), 1504, 1505, 1503, 1504, 1],
            [_t(15, 20), 1600, 1600, 1600, 1600, 1]]
    out = simulate_plan(side="SELL", entry=1500, stop=1510, target=1480, qty=10, start=_t(9, 36),
                        bars=flat, eod=eod)
    assert out["status"] == "eod" and out["exit_price"] == 1504 and out["r"] == -0.4
    assert simulate_plan(side="BUY", entry=1, stop=0.5, target=2, qty=1, start=_t(9, 36), bars=[],
                         eod=eod)["status"] == "no_data"


def test_evaluate_outcomes_and_track_record(tenant, owner, paper_portfolio, monkeypatch):
    from apps.trading.services import action_outcomes

    bars = [[_t(9, 40), 1500, 1501, 1499, 1500, 1], [_t(10, 0), 1500, 1521, 1500, 1520, 1]]
    monkeypatch.setattr(action_outcomes, "db_bars", lambda sym, lo, hi: bars)
    skipped = _card(paper_portfolio, date=DAY, symbol="INFY", status="skipped",
                    planned_as_of=_t(9, 36), expires_at=_t(15, 0))
    _card(paper_portfolio, date=DAY, symbol="TCS", status="proposed", planned_as_of=_t(9, 36),
          expires_at=_t(15, 0))
    with tracked_run("job", "action_board_outcomes", {}) as run:
        out = evaluate_outcomes(DAY, run=run, now=_t(16, 10))
    assert out["expired"] == 1 and out["evaluated"] == 2 and out["outcomes"] == {"target": 2}
    skipped.refresh_from_db()
    assert skipped.outcome["r"] == 2.0

    # An approved card whose paper trade closed at the target.
    ex = _card(paper_portfolio, date=DAY, symbol="HDFCBANK", status="executed", planned_as_of=_t(9, 36),
               expires_at=_t(15, 0))
    order = Order.objects.create(tenant=tenant, portfolio=paper_portfolio, created_by=owner, symbol="HDFCBANK",
                                 side="BUY", qty=33, order_type="LIMIT", price=Decimal("1500"),
                                 status="filled", origin="strategy")
    TodayAction.objects.filter(pk=ex.pk).update(order=order)
    Trade.objects.create(tenant=tenant, portfolio=paper_portfolio, primary_order=order, symbol="HDFCBANK",
                         side="BUY", entry_price=Decimal("1500"), stop_loss=Decimal("1490"),
                         target=Decimal("1520"), quantity=33, status=Trade.Status.CLOSED,
                         exit_price=Decimal("1520"), realized_pnl=Decimal("660"), trade_date=DAY)
    tr = track_record(tenant)
    assert tr["paper"]["n"] == 1 and tr["paper"]["wins"] == 1 and tr["paper"]["avg_r"] == 2.0
    assert tr["missed"]["n"] == 2 and tr["missed"]["avg_r"] == 2.0
    assert tr["paper"]["label"] == "n=1 · unproven"


def test_track_record_is_tenant_scoped(two_tenants):
    _card(two_tenants.portfolio_b, status="skipped")
    assert track_record(two_tenants.a)["cards_total"] == 0
    assert track_record(two_tenants.b)["cards_total"] == 1


# ─── schedule + telegram ────────────────────────────────────────────────

def test_jobs_are_in_the_schedule_registry():
    from apps.system.services.run_schedule import RERUNNABLE, registry

    entries = {e.key: e for e in registry()}
    build = entries["action-board-build"]
    assert build.receipt_name == "action_board" and build.tracked
    assert [d.strftime("%H:%M") for d in build.due_times(DAY)] == ["09:35"]
    assert entries["action-board-outcomes"].receipt_name == "action_board_outcomes"
    assert build.due_times(date(2026, 10, 24)) == []          # Saturday
    assert "action_board" in RERUNNABLE


def test_telegram_summary_text(tenant, owner, paper_portfolio):
    card = _card(paper_portfolio)
    text = ab.format_board_telegram(DAY, {"candidates": 40, "planner_calls": 10, "planned": 1,
                                          "passed_risk": 1}, [card])
    assert "INFY" in text and "BUY 33 @ 1500.00" in text and "₹330 at risk" in text
    assert "unproven" in text and "http" in text
    assert "No actions today" in ab.format_board_telegram(DAY, {}, [])


def test_board_sends_telegram_when_live(tenant, owner, paper_portfolio, mocked, monkeypatch):
    _scan()
    sent = []
    monkeypatch.setattr(ab, "send_board_telegram", lambda day, res, cards: (sent.append(len(cards)) or (True, "")))
    with tracked_run("job", "action_board", {}) as run:
        out = ab.build_action_board(DAY, run=run, now=LIVE_NOW, telegram=True, max_planner_calls=3)
    assert sent == [1] and out["telegram_sent"] is True


def test_isolated_tenants_each_get_their_own_cards(mocked):
    t1, t2 = TenantFactory(), TenantFactory()
    for t in (t1, t2):
        u = UserFactory()
        MembershipFactory(user=u, tenant=t, role="owner")
        PortfolioFactory(tenant=t)
    _scan()
    _build(max_planner_calls=3)
    assert TodayAction.objects.filter(tenant=t1).count() == 1
    assert TodayAction.objects.filter(tenant=t2).count() == 1
    assert mocked == ["TCS", "INFY", "HDFCBANK"]               # planned once, carded per tenant
