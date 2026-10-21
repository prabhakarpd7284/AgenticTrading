"""Strategy CRUD rules (UX plan §6, Studio plan §2.1/§7).

* Create — from a template (rule spec or plugin + params) or a hand-written
  spec; version 1 is written at once.
* Edit — the definition is immutable per version: a change writes version
  n+1 (and moves the stage back to draft: the new version is untested).
  Name/description edits don't version.
* Duplicate — a new strategy whose v1 is the source's current definition.
* Archive / unarchive — hides it; always allowed.
* Delete — refused while the strategy is in paper/live or has trades booked
  from it (archive instead); otherwise removes it with its backtests.
* ``next_step`` — the "what stage am I in and what is next" line.
"""
from __future__ import annotations

import copy

from django.db import transaction
from django.utils import timezone

from apps.strategies.models import Backtest, StrategyInstance, StrategyVersion
from apps.strategies.services import spec as spec_mod
from apps.strategies.services import templates as tpl


class DefinitionError(ValueError):
    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("; ".join(f"{e['path']}: {e['message']}" if e.get("path") else e["message"]
                                   for e in errors))


class DeleteBlocked(Exception):
    pass


def _normalise(kind: str, strategy_name: str, definition: dict) -> tuple[dict, str]:
    """Validate and return (definition, hash)."""
    if kind == "spec":
        res = spec_mod.validate_spec(definition)
        if not res.ok:
            raise DefinitionError(res.errors)
        return res.spec, spec_mod.spec_hash(res.spec)
    errs = tpl.validate_params(strategy_name, definition or {})
    if errs:
        raise DefinitionError(errs)
    return dict(definition or {}), spec_mod.params_hash(strategy_name, definition or {})


@transaction.atomic
def create_strategy(*, tenant, user, name: str, description: str = "", template_key: str = "",
                    spec: dict | None = None, strategy_name: str = "", params: dict | None = None,
                    note: str = "created") -> StrategyInstance:
    if template_key:
        t = tpl.get_template(template_key)
        if t is None:
            raise DefinitionError([{"path": "template_key", "message": f"unknown template {template_key!r}"}])
        kind, strategy_name = t["kind"], t["strategy_name"]
        if kind == "spec":
            definition = copy.deepcopy(spec if spec is not None else t["spec"])
        else:
            definition = {**(t["defaults"] or {}), **(params or {})}
        plugin_version = t.get("version") or "1.0.0"
    elif spec is not None:
        kind, strategy_name, definition, plugin_version = "spec", "spec", spec, str(spec_mod.SCHEMA_VERSION)
    elif strategy_name:
        sch = tpl.plugin_schema(strategy_name)
        if sch is None or strategy_name in tpl.HIDDEN_PLUGINS:
            raise DefinitionError([{"path": "strategy_name", "message": f"unknown strategy {strategy_name!r}"}])
        kind, definition, plugin_version = "plugin", dict(params or {}), sch["version"]
    else:
        raise DefinitionError([{"path": "", "message": "give a template_key, a spec, or a strategy_name"}])
    name = (name or "").strip()
    if not name:
        raise DefinitionError([{"path": "name", "message": "name is required"}])
    if kind == "spec":
        definition = copy.deepcopy(definition)
        definition.setdefault("meta", {})
        definition["meta"]["name"] = name
        if description:
            definition["meta"]["description"] = description
    definition, h = _normalise(kind, strategy_name, definition)
    inst = StrategyInstance.objects.create(
        tenant=tenant, strategy_name=strategy_name,
        strategy_version=str(spec_mod.SCHEMA_VERSION) if kind == "spec" else plugin_version,
        name=name[:120], description=description or "", kind=kind, template_key=template_key or "",
        params=definition, stage=StrategyInstance.Stage.DRAFT, current_version=1)
    StrategyVersion.objects.create(tenant=tenant, instance=inst, version=1,
                                   spec=definition if kind == "spec" else {},
                                   params=definition if kind == "plugin" else {},
                                   spec_hash=h, created_by=user if getattr(user, "pk", None) else None,
                                   note=note[:240])
    _emit(inst, "created", user)
    return inst


@transaction.atomic
def new_version(inst: StrategyInstance, *, user, definition: dict, note: str = "") -> StrategyVersion:
    if inst.stage in (StrategyInstance.Stage.PAPER, StrategyInstance.Stage.LIVE):
        raise DefinitionError([{"path": "", "message": f"stop {inst.stage} trading before editing the definition"}])
    inst = StrategyInstance.objects.select_for_update().get(pk=inst.pk)
    if inst.kind == "spec":
        definition = copy.deepcopy(definition)
        definition.setdefault("meta", {})
        definition["meta"]["name"] = definition["meta"].get("name") or inst.name
    definition, h = _normalise(inst.kind, inst.strategy_name, definition)
    cur = inst.versions.filter(version=inst.current_version).first()
    if cur is not None and cur.spec_hash == h:
        return cur          # nothing changed — no new version
    v = inst.current_version + 1
    ver = StrategyVersion.objects.create(
        tenant=inst.tenant, instance=inst, version=v,
        spec=definition if inst.kind == "spec" else {}, params=definition if inst.kind == "plugin" else {},
        spec_hash=h, created_by=user if getattr(user, "pk", None) else None, note=(note or "edited")[:240])
    inst.current_version = v
    inst.params = definition
    inst.stage = StrategyInstance.Stage.DRAFT
    inst.save(update_fields=["current_version", "params", "stage", "updated_at"])
    _emit(inst, f"version {v}", user)
    return ver


def duplicate(inst: StrategyInstance, *, user, name: str = "") -> StrategyInstance:
    cur = current_version(inst)
    definition = copy.deepcopy((cur.spec if inst.kind == "spec" else cur.params) if cur else inst.params)
    new_name = (name or f"Copy of {inst.name}")[:120]
    if inst.kind == "spec":
        return create_strategy(tenant=inst.tenant, user=user, name=new_name, description=inst.description,
                               spec=definition, note=f"duplicated from {inst.name} v{inst.current_version}")
    return create_strategy(tenant=inst.tenant, user=user, name=new_name, description=inst.description,
                           strategy_name=inst.strategy_name, params=definition,
                           note=f"duplicated from {inst.name} v{inst.current_version}")


def set_archived(inst: StrategyInstance, archived: bool, user=None) -> StrategyInstance:
    inst.archived_at = timezone.now() if archived else None
    inst.save(update_fields=["archived_at", "updated_at"])
    _emit(inst, "archived" if archived else "unarchived", user)
    return inst


def trades_booked(inst: StrategyInstance) -> int:
    """Paper/live trades attributed to this strategy (through the agent run
    that produced them). None exist until the Paper stage ships (P6)."""
    try:
        from apps.trading.models import Trade
        return Trade.objects.filter(tenant=inst.tenant,
                                    strategy_run__config__strategy_instance_id=str(inst.id)).count()
    except Exception:  # noqa: BLE001 — JSON lookups unsupported → be safe, count 0
        return 0


def delete_blocker(inst: StrategyInstance) -> str | None:
    if inst.stage in (StrategyInstance.Stage.PAPER, StrategyInstance.Stage.LIVE):
        return f"it is in the {inst.stage} stage — archive it instead"
    n = trades_booked(inst)
    if n:
        return f"it has {n} paper/live trade(s) — archive it instead to keep the record"
    return None


def delete(inst: StrategyInstance, user=None) -> None:
    why = delete_blocker(inst)
    if why:
        raise DeleteBlocked(f"Can't delete {inst.name}: {why}.")
    tenant, name = inst.tenant, inst.name
    inst.delete()
    try:
        from apps.events.services.event_writer import emit
        emit(tenant=tenant, type="workflow.step.completed", step_name="strategy_studio.deleted",
             text=f"Strategy deleted: {name}", actor_kind="user" if user else "workflow",
             actor_user=user if getattr(user, "pk", None) else None, broadcast=False)
    except Exception:  # noqa: BLE001
        pass


def current_version(inst: StrategyInstance) -> StrategyVersion | None:
    return inst.versions.filter(version=inst.current_version).first()


def last_backtest(inst: StrategyInstance, *, done_only: bool = False) -> Backtest | None:
    qs = inst.backtests.all()
    if done_only:
        qs = qs.filter(status=Backtest.Status.DONE)
    return qs.order_by("-created_at").first()


def next_step(inst: StrategyInstance) -> dict:
    b = tpl.backtestability(inst.kind, inst.strategy_name)
    if inst.archived_at:
        return {"code": "archived", "title": "Archived",
                "detail": "Unarchive it to backtest or edit it again.", "action": "unarchive"}
    if b["backtest"] == "forward_paper_only":
        return {"code": "forward_paper_only", "title": "Forward paper test only",
                "detail": b["reason"] + " The Paper stage of the Studio isn't built yet (plan P6).",
                "action": None}
    if b["backtest"] == "unavailable":
        return {"code": "not_backtestable", "title": "Can't be backtested here", "detail": b["reason"],
                "action": None}
    bt = last_backtest(inst, done_only=True)
    cur_bt = bt if (bt and bt.version and bt.version.version == inst.current_version) else None
    running = inst.backtests.filter(status__in=[Backtest.Status.QUEUED, Backtest.Status.RUNNING]).exists()
    if running:
        return {"code": "backtest_running", "title": "Backtest running",
                "detail": "Results appear on the Test tab when it finishes.", "action": None}
    if cur_bt is None:
        detail = f"Run a backtest of v{inst.current_version} on real candles."
        if bt is not None:
            detail = (f"The last backtest tested an older version (v{bt.version.version if bt.version else '?'}); "
                      f"run v{inst.current_version}.")
        return {"code": "run_backtest", "title": "Next: backtest it", "detail": detail, "action": "backtest"}
    m = cur_bt.metrics or {}
    if m.get("too_few_trades"):
        return {"code": "too_few_trades", "title": "Next: get more trades",
                "detail": f"The last backtest had only {m.get('trades', 0)} trade(s) (< {m.get('min_trades', 30)}). "
                          "Widen the universe or the period before trusting any metric.",
                "action": "backtest"}
    gate = (cur_bt.data_provenance or {}).get("live_gate") or {}
    gate_txt = f" Note: {gate['message']}" if gate.get("passes") is False else ""
    return {"code": "paper_next", "title": "Next: paper trade it",
            "detail": "Backtested on real data. Check the out-of-sample column, then paper-trade it — the "
                      "Paper stage of the Studio isn't built yet (plan P6)." + gate_txt,
            "action": None}


def _emit(inst, what: str, user) -> None:
    try:
        from apps.events.services.event_writer import emit
        emit(tenant=inst.tenant, type="workflow.step.completed", step_name="strategy_studio.strategy",
             text=f"Strategy {what}: {inst.name}", actor_kind="user" if user else "workflow",
             actor_user=user if getattr(user, "pk", None) else None,
             payload={"instance_id": str(inst.id), "version": inst.current_version, "what": what},
             broadcast=False)
    except Exception:  # noqa: BLE001
        pass
