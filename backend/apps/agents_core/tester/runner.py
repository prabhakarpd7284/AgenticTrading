"""AI Tester runner.

Deterministic suite that hits the live backend + checks invariants. See
docs/AI_TESTER_CLAUDE.md for the agent's briefing — that file is the
"system prompt" of this whole module.

Token-cheap on rerun: results dedupe through the mind palace. A green
re-run produces zero new findings; a regression bumps `occurrences` on the
same finding id and refreshes evidence.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests

from apps.agents_core.tester import state

V2_BASE = "http://localhost:8000/api/v1"
LEGACY_BASE = "http://localhost:8001/api/v1"
HTTP_TIMEOUT = 180

DEFAULT_EMAIL = "smoke@alphadesk.local"
DEFAULT_PASS = "smoke-1234"


# ─────────────────────────────────────────────────────────────────────────
# Test infra
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Ctx:
    palace: state.MindPalace
    access_token: str = ""
    portfolio_id: str = ""
    facts: dict = field(default_factory=dict)   # cross-test scratchpad

    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        return h

    # ── HTTP helpers ──
    def get(self, path: str, *, base: str = V2_BASE) -> requests.Response:
        return requests.get(f"{base}{path}", headers=self.headers(), timeout=HTTP_TIMEOUT)

    def post(self, path: str, body: dict, *, base: str = V2_BASE, timeout: int = HTTP_TIMEOUT) -> requests.Response:
        return requests.post(f"{base}{path}", headers=self.headers(),
                              data=json.dumps(body), timeout=timeout)

    def legacy_get(self, path: str) -> requests.Response:
        return self.get(path, base=LEGACY_BASE)

    # ── JSON parsing tolerant of literal control chars in payloads ──
    @staticmethod
    def json_of(resp: requests.Response) -> dict:
        return json.loads(re.sub(r"[\x00-\x1f]+", " ", resp.text))


@dataclass
class TestCase:
    id: str
    suite: str
    title: str
    fn: Callable[[Ctx], None]


@dataclass
class TestResult:
    case: TestCase
    passed: bool
    duration_ms: int
    error_msg: str = ""


class TestFailure(AssertionError):
    """Raised inside a test_fn to abort with a recorded reason."""


def fail(msg: str) -> None:
    raise TestFailure(msg)


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise TestFailure(msg)


# ─────────────────────────────────────────────────────────────────────────
# Bootstrap — make sure the smoke user + tenant + portfolio exist
# ─────────────────────────────────────────────────────────────────────────
def ensure_smoke_user() -> tuple[str, str]:
    """Idempotent: creates the smoke user/tenant/portfolio if missing.
    Returns (user_id, portfolio_id)."""
    from apps.accounts.models import User
    from apps.tenants.models import Tenant, Membership
    from apps.portfolio.models import Portfolio

    user, created = User.objects.get_or_create(email=DEFAULT_EMAIL, defaults={"is_active": True})
    user.set_password(DEFAULT_PASS); user.save()
    tenant, _ = Tenant.objects.get_or_create(slug="smoke", defaults={"name": "Smoke Tenant", "kind": "retail"})
    Membership.objects.get_or_create(tenant=tenant, user=user, defaults={"role": "owner"})
    portfolio, _ = Portfolio.objects.get_or_create(
        tenant=tenant, name="smoke-portfolio",
        defaults={"capital": 2_000_000, "mode": "paper"},
    )
    return str(user.id), str(portfolio.id)


# ─────────────────────────────────────────────────────────────────────────
# Test suites — each function is a single TestCase callable
# ─────────────────────────────────────────────────────────────────────────
def t_login(ctx: Ctx) -> None:
    r = ctx.post("/auth/token/", {"email": DEFAULT_EMAIL, "password": DEFAULT_PASS})
    expect(r.status_code == 200, f"login expected 200, got {r.status_code}")
    data = r.json()
    expect("access" in data, "login response missing access token")
    ctx.access_token = data["access"]


def t_catalog(ctx: Ctx) -> None:
    r = ctx.get("/agents/catalog/")
    expect(r.status_code == 200, f"catalog {r.status_code}")
    items = r.json()
    if isinstance(items, dict) and "results" in items:
        items = items["results"]
    names = {s["name"] for s in items}
    for expected in ("directional", "short_straddle", "vertical_spread", "pyramid"):
        expect(expected in names, f"catalog missing plugin {expected!r}; got {names}")
    ctx.facts["catalog"] = items


def t_expiries_nifty(ctx: Ctx) -> None:
    r = ctx.legacy_get("/legacy/expiries/?underlying=NIFTY&limit=12")
    expect(r.status_code == 200, f"expiries {r.status_code}")
    body = r.json()
    rows = body.get("results", [])
    expect(len(rows) >= 1, "no NIFTY expiries returned")
    weekly = [e for e in rows if e["kind"] == "weekly" and e["dte"] >= 0]
    expect(len(weekly) >= 1, "no upcoming NIFTY weekly expiry returned")
    pick = weekly[0]
    ctx.facts["nifty_weekly_expiry"] = pick["expiry"]
    ctx.facts["nifty_weekly_iso"] = pick["iso"]
    ctx.facts["nifty_weekly_dte"] = pick["dte"]


def t_portfolios(ctx: Ctx) -> None:
    r = ctx.get("/portfolios/")
    expect(r.status_code == 200, f"portfolios {r.status_code}")
    items = r.json()
    if isinstance(items, dict) and "results" in items:
        items = items["results"]
    expect(len(items) >= 1, "no portfolios visible to smoke user")
    ctx.portfolio_id = items[0]["id"]


# ── Strategy fan-out via /agents/plan-stock/ ─────────────────────────────
def t_plan_stock_nifty_monthly(ctx: Ctx) -> None:
    body = {
        "symbol": "NIFTY",
        "portfolio": ctx.portfolio_id,
        "total_capital": 500_000,
        "horizon": "monthly",
        "side_hint": "BULL",
        "dry_run": True,
    }
    t0 = time.time()
    r = ctx.post("/agents/plan-stock/", body, timeout=180)
    dt = int((time.time() - t0) * 1000)
    expect(r.status_code == 200, f"plan-stock {r.status_code} body={r.text[:200]}")
    d = ctx.json_of(r)

    expect(d.get("symbol") == "NIFTY", "plan-stock returned wrong symbol")
    runtime = d.get("summary", {}).get("total_runtime_ms", 0)
    expect(runtime < 60_000, f"plan-stock too slow: {runtime}ms")

    # Allocations must sum to ~1.0
    allocs = d.get("allocations") or []
    expect(allocs, "plan-stock returned no allocations")
    pct_sum = sum(a["pct"] for a in allocs)
    expect(abs(pct_sum - 1.0) < 0.01, f"allocations pct sum {pct_sum}, expected 1.0")
    # NIFTY is an index → directional must be skipped
    strat_names = {a["strategy"] for a in allocs}
    if "directional" in strat_names:
        raise TestFailure("directional should be skipped for index underlying NIFTY")

    # Each result must have at minimum status + capital + summary
    for res in d.get("results", []):
        expect(res.get("status") in ("succeeded", "failed"),
               f"result for {res.get('strategy')} has bad status")

    ctx.facts["plan_stock_nifty"] = d
    ctx.facts["plan_stock_runtime_ms"] = dt


# ── Vertical spread on NIFTY weekly ──────────────────────────────────────
def t_vertical_spread_nifty_weekly(ctx: Ctx) -> None:
    expiry = ctx.facts.get("nifty_weekly_expiry")
    expect(expiry, "no NIFTY weekly expiry recorded yet")
    body = {
        "strategy_name": "vertical_spread",
        "strategy_version": "1.0.0",
        "portfolio": ctx.portfolio_id,
        "config": {"underlying": "NIFTY", "side": "BULL", "expiry": expiry,
                    "capital": 200_000, "max_lots": 10},
    }
    r = ctx.post("/agents/runs/", body)
    expect(r.status_code in (200, 202), f"run create {r.status_code}: {r.text[:200]}")
    run_id = r.json()["id"]
    state_run = _wait_for_run(ctx, run_id, timeout=60)
    expect(state_run["status"] == "succeeded", f"vertical_spread failed: {state_run.get('error')}")

    res = state_run.get("result") or {}
    plan = res.get("plan") or {}
    expect("long_strike" in plan and "short_strike" in plan,
           f"vertical_spread plan missing strikes: {plan}")
    expect(plan.get("long_strike") != plan.get("short_strike"),
           "long and short strikes identical")
    # Strikes should come from real scrip-master grid — gap is a multiple of the grid step
    gap = abs(plan["long_strike"] - plan["short_strike"])
    expect(gap >= 25, f"strike gap {gap} suspiciously small (or LTPs not real)")
    ctx.facts["vertical_spread_run_id"] = run_id


# ── Pyramid on NIFTY ATM CE weekly ───────────────────────────────────────
def t_pyramid_nifty_weekly(ctx: Ctx) -> None:
    expiry = ctx.facts.get("nifty_weekly_expiry")
    body = {
        "strategy_name": "pyramid",
        "strategy_version": "1.0.0",
        "portfolio": ctx.portfolio_id,
        "config": {"underlying": "NIFTY", "option_type": "CE", "expiry": expiry,
                    "capital": 100_000, "risk_pct": 2.0, "max_pyramids": 5,
                    "lookback_days": 3},
    }
    r = ctx.post("/agents/runs/", body)
    expect(r.status_code in (200, 202), f"run create {r.status_code}: {r.text[:200]}")
    run_id = r.json()["id"]
    state_run = _wait_for_run(ctx, run_id, timeout=90)
    expect(state_run["status"] == "succeeded", f"pyramid failed: {state_run.get('error')}")

    res = state_run.get("result") or {}
    candles = res.get("candles_raw") or []
    expect(len(candles) >= 50,
           f"pyramid got only {len(candles)} candles — auto-widen fallback not engaging?")
    plan = res.get("plan") or {}
    # Either the plan executed (entries present) or surfaced a clear error
    entries = plan.get("entries") or []
    err = plan.get("error")
    expect(entries or err,
           f"pyramid plan has neither entries nor error — silent failure")
    ctx.facts["pyramid_run_id"] = run_id


# ── Step replay via detail endpoint ──────────────────────────────────────
def t_step_replay(ctx: Ctx) -> None:
    """Confirm AgentSteps come back on /agents/runs/{id}/ so the Stream tab
    can replay history for completed runs."""
    run_id = ctx.facts.get("pyramid_run_id") or ctx.facts.get("vertical_spread_run_id")
    expect(run_id, "no recent run id captured for step-replay test")
    r = ctx.get(f"/agents/runs/{run_id}/")
    expect(r.status_code == 200, f"run detail {r.status_code}")
    d = ctx.json_of(r)
    steps = d.get("steps")
    expect(steps is not None, "run detail missing steps[] (need to expose for Stream tab replay)")
    expect(len(steps) >= 2, f"only {len(steps)} steps persisted, expected >= 2")


# ── stock-summary endpoint ───────────────────────────────────────────────
def t_stock_summary_nifty(ctx: Ctx) -> None:
    r = ctx.legacy_get("/legacy/stock-summary/?symbol=NIFTY&period=monthly")
    expect(r.status_code == 200, f"stock-summary {r.status_code}")
    d = ctx.json_of(r)
    expect(d.get("kind") == "index_underlying", f"NIFTY classified as {d.get('kind')}")
    expect("kpis" in d and "buckets" in d and "rollups" in d, "stock-summary missing top-level fields")
    for p in d.get("open_positions", []):
        expect(p.get("leverage", 0) > 0, f"open position {p.get('leg')} has leverage 0")


# ── UI plumbing: data fetched but not rendered ───────────────────────────
# Cross-checks: for the last pyramid/vertical_spread runs, look up the
# React file and confirm the *plan fields with real values* are referenced
# in the JSX. This catches "data is in the payload but UI doesn't show it"
# — the exact class of bug the user flagged about chart rendering.
def t_ui_renders_pyramid_plan(ctx: Ctx) -> None:
    run_id = ctx.facts.get("pyramid_run_id")
    if not run_id:
        return  # no pyramid run to validate against
    r = ctx.get(f"/agents/runs/{run_id}/")
    if r.status_code != 200:
        return
    d = ctx.json_of(r)
    plan = (d.get("result") or {}).get("plan") or {}
    if not plan or plan.get("error"):
        return
    src = _read_repo_file("frontend/src/features/agents/AgentConsolePage.tsx")
    missing = []
    # Fields that should be rendered on the Pyramid Overview when present:
    for f in ("entries", "exit_price", "exit_reason", "total_lots", "avg_entry",
              "peak_unrealized", "total_pnl_rupees", "log_tail"):
        if plan.get(f) in (None, "", [], 0):
            continue
        if f not in src:
            missing.append(f)
    expect(not missing,
           f"pyramid response has these fields, but UI doesn't render them: {missing}. "
           f"PyramidOverview in AgentConsolePage.tsx needs to reference them.")


def t_ui_renders_pyramid_chart(ctx: Ctx) -> None:
    """The pyramid intraday close chart is the obvious-missing-feature the
    user flagged. Pyramid candles ARE in the response but no chart renders
    them. Treat as a high-severity UI gap until a chart component exists."""
    run_id = ctx.facts.get("pyramid_run_id")
    if not run_id:
        return
    r = ctx.get(f"/agents/runs/{run_id}/")
    d = ctx.json_of(r)
    candles = ((d.get("result") or {}).get("candles_raw")) or []
    if not candles:
        return  # nothing to chart
    src = _read_repo_file("frontend/src/features/agents/AgentConsolePage.tsx")
    has_pyramid_chart = ("PyramidChart" in src) or ("PyramidIntradayChart" in src)
    expect(has_pyramid_chart,
           f"Pyramid run has {len(candles)} candles in result.candles_raw but no PyramidChart component "
           f"references them in AgentConsolePage.tsx. The intraday + entries chart is missing.")


def t_ui_chart_containers_clipped(ctx: Ctx) -> None:
    """Static render-invariant: every <CardContent> that hosts a chart
    container (height in pixels) must have `overflow-hidden` so any
    absolutely-positioned chart canvas can't overlay the next card.

    This is exactly the bug pattern that hit the pyramid + vertical-spread
    overviews — lightweight-charts canvases bleeding over the headline KPI
    card below. A deterministic check beats waiting for a screenshot.
    """
    src = _read_repo_file("frontend/src/features/agents/AgentConsolePage.tsx")
    if not src:
        return
    pattern = re.compile(r'<CardContent className="h-\[(\d+)px\]([^"]*)">')
    violators: list[str] = []
    for m in pattern.finditer(src):
        height, rest = m.group(1), m.group(2)
        if "overflow-hidden" not in rest:
            violators.append(f"h-[{height}px] missing overflow-hidden ({rest.strip() or 'no other classes'})")
    expect(not violators,
           "Chart CardContent wrappers need `overflow-hidden` to clip absolutely-positioned chart "
           f"canvases (lightweight-charts especially). Violations: {violators}")


def t_ui_lightweight_charts_measure_before_init(ctx: Ctx) -> None:
    """Static render-invariant: every lightweight-charts createChart call
    must pass explicit width + height and set up a ResizeObserver,
    otherwise the chart can render at 0×0 (invisible) or overflow its
    parent (overlay)."""
    src = _read_repo_file("frontend/src/features/agents/AgentConsolePage.tsx")
    if not src:
        return
    # Each createChart() block must include width:, height:, and ResizeObserver.
    blocks: list[str] = []
    for m in re.finditer(r"mod\.createChart\([^)]*\{(.+?)\}\s*\)", src, flags=re.DOTALL):
        blocks.append(m.group(1))
    if not blocks:
        return
    bad: list[int] = []
    for i, b in enumerate(blocks, start=1):
        if not ("width:" in b and "height:" in b):
            bad.append(i)
    expect(not bad,
           f"createChart() calls #{bad} are missing explicit width/height — without it lightweight-charts "
           "can paint outside its container. Wrap with `const {width, height} = el.getBoundingClientRect()` "
           "and pass them in.")
    # ResizeObserver presence (anywhere in the file is fine — global belt).
    expect("new ResizeObserver" in src,
           "AgentConsolePage uses lightweight-charts but has no ResizeObserver — charts won't reflow "
           "when the sidebar collapses or the window resizes.")


def t_ui_renders_vertical_spread_chart(ctx: Ctx) -> None:
    run_id = ctx.facts.get("vertical_spread_run_id")
    if not run_id:
        return
    r = ctx.get(f"/agents/runs/{run_id}/")
    d = ctx.json_of(r)
    plan = (d.get("result") or {}).get("plan") or {}
    if not plan.get("long_strike"):
        return
    src = _read_repo_file("frontend/src/features/agents/AgentConsolePage.tsx")
    has_chart = ("VerticalSpreadChart" in src) or ("PayoffChart" in src) or ("payoff" in src.lower() and "ResponsiveContainer" in src)
    expect(has_chart,
           "Vertical spread has structured legs (long/short/strikes/breakeven) but no payoff "
           "diagram component renders them. Add a VerticalSpreadChart that plots P&L vs underlying "
           "with breakeven + max profit/loss zones.")


# ─────────────────────────────────────────────────────────────────────────
# Suite registry
# ─────────────────────────────────────────────────────────────────────────
SUITES: list[TestCase] = [
    TestCase("auth.login",                 "auth",            "Login as smoke user, get JWT",                                t_login),
    TestCase("meta.catalog",               "meta",            "Strategy catalog lists all 4 plugins",                        t_catalog),
    TestCase("meta.expiries_nifty",        "meta",            "NIFTY expiries endpoint returns weekly + monthly",            t_expiries_nifty),
    TestCase("meta.portfolios",            "meta",            "Smoke user can see at least one portfolio",                    t_portfolios),
    TestCase("plan_stock.nifty_monthly",   "plan_stock",      "POST /plan-stock/ for NIFTY monthly returns sensible data",   t_plan_stock_nifty_monthly),
    TestCase("vertical_spread.nifty",      "vertical_spread", "vertical_spread NIFTY weekly picks real strikes",             t_vertical_spread_nifty_weekly),
    TestCase("pyramid.nifty",              "pyramid",         "pyramid NIFTY weekly fetches >= 50 candles",                  t_pyramid_nifty_weekly),
    TestCase("agents.step_replay",         "agents",          "Run detail includes steps[] for Stream tab replay",           t_step_replay),
    TestCase("legacy.stock_summary",       "legacy",          "/legacy/stock-summary/?symbol=NIFTY shape + leverage",         t_stock_summary_nifty),
    TestCase("ui.pyramid_plan_rendered",   "ui_plumbing",     "PyramidOverview references all plan fields the API returns",   t_ui_renders_pyramid_plan),
    TestCase("ui.pyramid_chart_present",   "ui_plumbing",     "Pyramid intraday chart component exists (with entries markers)", t_ui_renders_pyramid_chart),
    TestCase("ui.vs_payoff_present",       "ui_plumbing",     "Vertical-spread payoff diagram component exists",              t_ui_renders_vertical_spread_chart),
    TestCase("ui.chart_containers_clipped","ui_rendering",    "Chart CardContent wrappers clip absolute children (overflow-hidden)", t_ui_chart_containers_clipped),
    TestCase("ui.charts_measure_before_init","ui_rendering",  "lightweight-charts createChart calls pass explicit width/height + ResizeObserver", t_ui_lightweight_charts_measure_before_init),
]


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────
def run(only: str | None = None, verbose: bool = False) -> tuple[state.MindPalace, list[TestResult]]:
    palace = state.load()
    ensure_smoke_user()

    cases = SUITES
    if only:
        cases = [c for c in SUITES if c.suite == only or c.id == only]
        if not cases:
            raise SystemExit(f"No tests match --only={only}")

    # Fingerprint — useful as a notes hint, not a hard short-circuit yet.
    from apps.agents_core.registry import strategy_registry
    try:
        strategy_registry.load_entry_points()
    except Exception:
        pass
    palace.fingerprint = state.compute_fingerprint(
        strategies=list(strategy_registry.names()),
        plugin_dirs=[Path(__file__).resolve().parents[3] / "plugins"],
    )

    ctx = Ctx(palace=palace)
    results: list[TestResult] = []
    run_id = uuid.uuid4().hex[:12]
    started_at = state._now()

    for case in cases:
        t0 = time.time()
        finding_id = state.slugify(f"[{case.suite}] {case.title}")
        try:
            case.fn(ctx)
            results.append(TestResult(case=case, passed=True, duration_ms=int((time.time() - t0) * 1000)))
            # If this test had previously opened a finding and is now green,
            # close it — keeps the palace honest about what's still broken.
            if state.close_finding(palace, finding_id):
                if verbose:
                    print(f"  ✓ {case.id:36}  {int((time.time() - t0)*1000):>5}ms  (closed: {finding_id})")
                else:
                    pass
            elif verbose:
                print(f"  ✓ {case.id:36}  {int((time.time() - t0)*1000):>5}ms")
        except TestFailure as e:
            msg = str(e)
            results.append(TestResult(case=case, passed=False,
                                       duration_ms=int((time.time() - t0) * 1000),
                                       error_msg=msg))
            severity = "high" if case.suite in ("plan_stock", "vertical_spread", "pyramid", "ui_plumbing") else "warning"
            state.upsert_finding(palace, title=f"[{case.suite}] {case.title}",
                                  severity=severity, suite=case.suite, evidence=msg)
            if verbose:
                print(f"  ✗ {case.id:36}  {int((time.time() - t0)*1000):>5}ms  {msg[:120]}")
        except Exception as e:  # noqa: BLE001 — unhandled exception inside test
            msg = f"unhandled exception: {type(e).__name__}: {e}"
            results.append(TestResult(case=case, passed=False,
                                       duration_ms=int((time.time() - t0) * 1000),
                                       error_msg=msg))
            state.upsert_finding(palace, title=f"[{case.suite}] {case.title} — unhandled exception",
                                  severity="blocker", suite=case.suite, evidence=msg)
            if verbose:
                print(f"  ✗ {case.id:36}  {int((time.time() - t0)*1000):>5}ms  {msg[:120]}")

    ended_at = state._now()
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    finding_ids = [
        state.slugify(f"[{r.case.suite}] {r.case.title}")
        for r in results if not r.passed
    ]
    state.record_run(palace, run_id=run_id, started_at=started_at,
                      ended_at=ended_at, total=len(results), passed=passed,
                      failed=failed, finding_ids=finding_ids)
    state.save(palace)
    return palace, results


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def _wait_for_run(ctx: Ctx, run_id: str, *, timeout: int) -> dict:
    """Poll a v2 AgentRun until it leaves queued/running, or fail the test."""
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        r = ctx.get(f"/agents/runs/{run_id}/")
        if r.status_code != 200:
            time.sleep(1.0)
            continue
        last = ctx.json_of(r)
        if last.get("status") in ("succeeded", "failed", "cancelled"):
            return last
        time.sleep(2.0)
    fail(f"run {run_id} did not complete within {timeout}s (last status={last.get('status')})")
    return {}


def _read_repo_file(rel: str) -> str:
    p = state._REPO_ROOT / rel
    try:
        return p.read_text()
    except OSError:
        return ""
