"""Strategy Studio API (R4): CRUD + versions + templates + backtests, with
tenant isolation on every endpoint. Candles come from an in-memory fake
loader — tests never touch market data or a broker."""
from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from apps.strategies.models import Backtest, BacktestTrade, StrategyInstance, StrategyVersion
from apps.system.models import RunReceipt
from tests.studio_helpers import FakeLoader, simple_spec

pytestmark = pytest.mark.django_db

BASE = "/api/v1/strategies"


def _client(user):
    from rest_framework_simplejwt.tokens import RefreshToken

    from apps.accounts.api.jwt import _attach_tenant_claims
    c = APIClient()
    tok = RefreshToken.for_user(user).access_token
    _attach_tenant_claims(tok, user)
    c.force_authenticate(user=user)
    c.credentials(HTTP_AUTHORIZATION=f"Bearer {tok}")
    return c


@pytest.fixture
def fake_candles(monkeypatch):
    loader = FakeLoader()
    from apps.strategies.api import views
    from apps.strategies.services import backtest_service
    monkeypatch.setattr(backtest_service, "default_loader", lambda end: loader)
    monkeypatch.setattr(views, "_loader", lambda end: loader)
    return loader


@pytest.fixture
def pair(two_tenants):
    return two_tenants, _client(two_tenants.owner_a), _client(two_tenants.owner_b)


def _create_spec(c, name="EMA test", **over):
    r = c.post(f"{BASE}/instances/", {"name": name, "spec": simple_spec(**over)}, format="json")
    assert r.status_code == 201, r.json()
    return r.json()


# ── templates ───────────────────────────────────────────────────────────────

def test_templates_list_rule_and_plugin_templates_honestly(pair):
    _, ca, _ = pair
    rows = ca.get(f"{BASE}/templates/").json()
    by = {t["key"]: t for t in rows}
    assert by["rule.orb_breakout"]["backtest"] == "backtest" and by["rule.orb_breakout"]["spec"]
    assert by["plugin.directional"]["backtest"] == "forward_paper_only"
    assert by["plugin.pyramid"]["backtest"] == "unavailable" and "option" in by["plugin.pyramid"]["reason"]
    scr = by["plugin.intraday_screener"]
    assert scr["schema"]["properties"] and "BB Squeeze Breakout" in scr["broken"]
    assert "plugin.backtest" not in by                      # infrastructure, hidden


# ── CRUD + versions ─────────────────────────────────────────────────────────

def test_create_from_template_with_params_validates_schema(pair):
    _, ca, _ = pair
    bad = ca.post(f"{BASE}/instances/", {"name": "scr", "template_key": "plugin.intraday_screener",
                                          "params": {"max_risk_pct": 99}}, format="json")
    assert bad.status_code == 400 and bad.json()["errors"][0]["path"] == "max_risk_pct"
    ok = ca.post(f"{BASE}/instances/", {"name": "scr", "template_key": "plugin.intraday_screener",
                                         "params": {"max_risk_pct": 0.5}}, format="json")
    assert ok.status_code == 201
    d = ok.json()
    assert d["kind"] == "plugin" and d["definition"]["max_risk_pct"] == 0.5
    assert d["definition"]["universe"] == "all"              # schema default filled
    assert d["schema"]["properties"]["max_risk_pct"]
    assert d["stage"] == "draft" and d["next_step"]["code"] == "run_backtest"


def test_edit_creates_new_version_and_resets_stage(pair):
    t, ca, _ = pair
    inst = _create_spec(ca)
    assert inst["current_version"] == 1 and inst["explain"]
    StrategyInstance.objects.filter(id=inst["id"]).update(stage="backtested")
    spec = inst["definition"]
    same = ca.post(f"{BASE}/instances/{inst['id']}/versions/", {"spec": spec}, format="json")
    assert same.status_code == 200 and same.json()["created"] is False
    spec["exits"]["target"]["value"] = 3.0
    r = ca.post(f"{BASE}/instances/{inst['id']}/versions/", {"spec": spec, "note": "3R"}, format="json")
    assert r.status_code == 201 and r.json()["instance"]["current_version"] == 2
    assert r.json()["instance"]["stage"] == "draft"
    vs = ca.get(f"{BASE}/instances/{inst['id']}/versions/").json()
    assert [v["version"] for v in vs] == [2, 1] and vs[0]["note"] == "3R"
    assert vs[1]["definition"]["exits"]["target"]["value"] == 2.0          # v1 immutable
    bad = ca.post(f"{BASE}/instances/{inst['id']}/versions/",
                  {"spec": {**spec, "sizing": {"risk_pct": 9}}}, format="json")
    assert bad.status_code == 400 and bad.json()["errors"][0]["path"] == "sizing.risk_pct"


def test_patch_only_renames(pair):
    _, ca, _ = pair
    inst = _create_spec(ca)
    assert ca.patch(f"{BASE}/instances/{inst['id']}/", {"params": {}}, format="json").status_code == 400
    r = ca.patch(f"{BASE}/instances/{inst['id']}/", {"name": "Renamed"}, format="json")
    assert r.status_code == 200 and r.json()["name"] == "Renamed" and r.json()["current_version"] == 1


def test_duplicate_archive_and_delete_rules(pair):
    _, ca, _ = pair
    inst = _create_spec(ca)
    dup = ca.post(f"{BASE}/instances/{inst['id']}/duplicate/", {}, format="json")
    assert dup.status_code == 201 and dup.json()["name"] == "Copy of EMA test"
    assert dup.json()["definition"]["entry"] == inst["definition"]["entry"]
    a = ca.post(f"{BASE}/instances/{inst['id']}/archive/")
    assert a.json()["archived"] is True
    ids = [i["id"] for i in ca.get(f"{BASE}/instances/").json()]
    assert inst["id"] not in ids and dup.json()["id"] in ids
    assert inst["id"] in [i["id"] for i in ca.get(f"{BASE}/instances/?archived=1").json()]
    ca.post(f"{BASE}/instances/{inst['id']}/unarchive/")
    StrategyInstance.objects.filter(id=inst["id"]).update(stage="paper")
    blocked = ca.delete(f"{BASE}/instances/{inst['id']}/")
    assert blocked.status_code == 409 and "archive" in blocked.json()["detail"]
    assert ca.delete(f"{BASE}/instances/{dup.json()['id']}/").status_code == 204
    assert not StrategyInstance.objects.filter(id=dup.json()["id"]).exists()


# ── spec tools ──────────────────────────────────────────────────────────────

def test_spec_validate_and_preview(pair, fake_candles):
    _, ca, _ = pair
    v = ca.post(f"{BASE}/spec/validate/", {"spec": simple_spec()}, format="json").json()
    assert v["valid"] and v["spec_hash"] and v["explain"] and v["live_gate"]["passes"]
    bad = ca.post(f"{BASE}/spec/validate/", {"spec": {"timeframe": "7m"}}, format="json").json()
    assert not bad["valid"] and bad["errors"]
    p = ca.post(f"{BASE}/spec/preview-signals/", {"spec": simple_spec(), "symbol": "aaa",
                                                  "from_date": "2026-05-04", "to_date": "2026-05-08"},
                format="json")
    assert p.status_code == 200
    body = p.json()
    assert body["symbol"] == "AAA" and body["bars"] and body["signals"]
    assert all(s["fill_t"] > s["t"] for s in body["signals"])
    too_long = ca.post(f"{BASE}/spec/preview-signals/", {"spec": simple_spec(), "symbol": "AAA",
                                                         "from_date": "2026-01-01", "to_date": "2026-05-08"},
                       format="json")
    assert too_long.status_code == 400


# ── backtests ───────────────────────────────────────────────────────────────

def test_backtest_runs_end_to_end_with_receipt_and_stage(pair, fake_candles):
    _, ca, _ = pair
    inst = _create_spec(ca)
    r = ca.post(f"{BASE}/backtests/", {"instance": inst["id"], "from_date": "2026-05-04",
                                        "to_date": "2026-05-29", "symbols": ["AAA", "BBB"]}, format="json")
    assert r.status_code == 202, r.json()
    bt = ca.get(f"{BASE}/backtests/{r.json()['id']}/").json()
    assert bt["status"] == "done", bt["error"]
    assert bt["version"] == 1 and bt["metrics"]["trades"] > 0
    assert bt["provenance"]["synthetic"] is False and bt["provenance"]["definition"]["version"] == 1
    assert set(bt["split_metrics"]) >= {"is", "oos"} and bt["equity_curve"]
    trades = ca.get(f"{BASE}/backtests/{bt['id']}/trades/").json()
    assert len(trades) == bt["metrics"]["trades"] == BacktestTrade.objects.filter(backtest_id=bt["id"]).count()
    assert {t["segment"] for t in trades} <= {"is", "oos"}
    receipt = RunReceipt.objects.get(detail_kind="backtest", detail_id=bt["id"])
    assert receipt.status == "success"
    assert {c["name"]: c["status"] for c in receipt.checks}["real_metrics"] == "pass"
    assert receipt.outputs["trades"] == bt["metrics"]["trades"]
    detail = ca.get(f"{BASE}/instances/{inst['id']}/").json()
    assert detail["stage"] == "backtested" and detail["last_result"]["id"] == bt["id"]
    assert detail["next_step"]["code"] in ("too_few_trades", "paper_next")


def test_backtest_refuses_llm_plugin_and_bad_input(pair, fake_candles):
    _, ca, _ = pair
    d = ca.post(f"{BASE}/instances/", {"name": "llm", "template_key": "plugin.directional"}, format="json").json()
    assert d["next_step"]["code"] == "forward_paper_only"
    r = ca.post(f"{BASE}/backtests/", {"instance": d["id"], "from_date": "2026-05-04", "to_date": "2026-05-08",
                                        "symbols": ["AAA"]}, format="json")
    assert r.status_code == 400 and "LLM" in r.json()["errors"][0]["message"]
    inst = _create_spec(ca)
    r = ca.post(f"{BASE}/backtests/", {"instance": inst["id"], "from_date": "2026-05-08",
                                        "to_date": "2026-05-04"}, format="json")
    assert r.status_code == 400
    assert not Backtest.objects.filter(instance_id=inst["id"]).exists()


def test_failed_backtest_records_error(pair, fake_candles):
    _, ca, _ = pair
    fake_candles.missing = {"AAA", "BBB"}
    inst = _create_spec(ca)
    r = ca.post(f"{BASE}/backtests/", {"instance": inst["id"], "from_date": "2026-05-04",
                                        "to_date": "2026-05-08"}, format="json")
    bt = ca.get(f"{BASE}/backtests/{r.json()['id']}/").json()
    assert bt["status"] == "failed" and "no candle data" in bt["error"]
    assert RunReceipt.objects.get(detail_kind="backtest", detail_id=bt["id"]).status == "failed"


def test_cancel_queued_backtest(pair):
    t, ca, _ = pair
    inst = _create_spec(ca)
    obj = StrategyInstance.objects.get(id=inst["id"])
    bt = Backtest.objects.create(tenant=obj.tenant, instance=obj, version=obj.versions.first(),
                                 from_date="2026-05-04", to_date="2026-05-08")
    r = ca.post(f"{BASE}/backtests/{bt.id}/cancel/")
    assert r.status_code == 202 and r.json()["cancel_requested"] is True
    from apps.strategies.services.backtest_service import run_backtest_job
    assert run_backtest_job(str(bt.id), loader=FakeLoader()) == "cancelled"
    assert ca.post(f"{BASE}/backtests/{bt.id}/cancel/").status_code == 409


# ── tenant isolation on every endpoint ──────────────────────────────────────

def test_tenant_isolation_everywhere(pair, fake_candles):
    t, ca, cb = pair
    inst = _create_spec(ca)
    run = ca.post(f"{BASE}/backtests/", {"instance": inst["id"], "from_date": "2026-05-04",
                                          "to_date": "2026-05-15"}, format="json").json()
    iid, bid = inst["id"], run["id"]
    assert cb.get(f"{BASE}/instances/").json() == []
    assert cb.get(f"{BASE}/backtests/").json() == []
    for method, url, body in [
        ("get", f"{BASE}/instances/{iid}/", None),
        ("patch", f"{BASE}/instances/{iid}/", {"name": "x"}),
        ("delete", f"{BASE}/instances/{iid}/", None),
        ("get", f"{BASE}/instances/{iid}/versions/", None),
        ("post", f"{BASE}/instances/{iid}/versions/", {"spec": simple_spec()}),
        ("post", f"{BASE}/instances/{iid}/duplicate/", {}),
        ("post", f"{BASE}/instances/{iid}/archive/", None),
        ("post", f"{BASE}/instances/{iid}/unarchive/", None),
        ("get", f"{BASE}/backtests/{bid}/", None),
        ("get", f"{BASE}/backtests/{bid}/trades/", None),
        ("post", f"{BASE}/backtests/{bid}/cancel/", None),
        ("post", f"{BASE}/backtests/", {"instance": iid, "from_date": "2026-05-04", "to_date": "2026-05-08"}),
    ]:
        r = getattr(cb, method)(url, body, format="json") if body is not None else getattr(cb, method)(url)
        assert r.status_code == 404, (method, url, r.status_code)
    assert cb.get(f"{BASE}/backtests/?instance={iid}").json() == []
    # B's own rows don't leak into A either
    _create_spec(cb, name="B strategy")
    assert [i["name"] for i in ca.get(f"{BASE}/instances/").json()] == ["EMA test"]
    assert StrategyInstance.objects.get(id=iid).name == "EMA test"
    assert StrategyVersion.objects.filter(instance_id=iid).count() == 1


def test_anonymous_is_rejected():
    c = APIClient()
    for url in (f"{BASE}/instances/", f"{BASE}/templates/", f"{BASE}/backtests/", f"{BASE}/universe/"):
        assert c.get(url).status_code in (401, 403)


def test_universe_lists_candle_symbols(pair):
    from datetime import datetime, timedelta, timezone

    from apps.market_data.models import Candle, Symbol
    _, ca, _ = pair
    now = datetime.now(timezone.utc)
    for name in ("RELIANCE", "NIFTY"):
        s = Symbol.objects.create(exchange="NSE", tradingsymbol=name, token=name, name=name)
        Candle.objects.create(symbol=s, interval="1m", t=now - timedelta(days=1), o=1, h=1, l=1, c=1, v=1)
    from django.core.cache import cache
    cache.delete("strategies.universe.v1")
    u = ca.get(f"{BASE}/universe/").json()
    assert u["intraday"]["symbols"] == ["RELIANCE"] and u["intraday"]["indices"] == ["NIFTY"]
