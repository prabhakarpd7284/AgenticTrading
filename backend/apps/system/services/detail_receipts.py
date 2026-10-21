"""Receipts for runs that have their own detail table (UX redesign plan §8.1).

``AgentRun`` (agents_core) and ``Backtest`` (strategies) keep their tables;
each row gets one ``RunReceipt`` that mirrors its status and points back at it
(``detail_kind`` / ``detail_id``). The receipt is kept in sync from
``post_save`` signals (wired in ``apps.system.apps``) and from the bulk
``.update()`` paths that bypass signals (agent-run expiry). A run that ends
``failed`` becomes a failed receipt and alerts once — before this, a failed
agent run (the 2026-10-18 ``scalp`` run) surfaced nowhere.

Never raises: a receipt write must not break the run it describes.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

AGENT_STATUS = {"queued": "queued", "running": "running", "succeeded": "success",
                "failed": "failed", "cancelled": "skipped"}
BACKTEST_STATUS = {"queued": "queued", "running": "running", "done": "success", "failed": "failed"}

# Agent runs live under a 30-min expiry sweep; backtests get the sweeper's 6 h.
AGENT_MAX_SECONDS = 45 * 60
BACKTEST_MAX_SECONDS = 6 * 3600


def _upsert(*, detail_kind: str, detail_id: str, tenant_id, kind: str, name: str, status: str,
            started_at, finished_at, error: str, inputs: dict, outputs: dict, checks: list,
            links: list, max_seconds: int, trigger: str = "api", alert: bool = True):
    from apps.system.models import RunReceipt
    from apps.system.services.run_receipts import _jsonable

    receipt = RunReceipt.objects.filter(detail_kind=detail_kind, detail_id=detail_id).first()
    was = receipt.status if receipt else None
    if links:
        outputs = {**outputs, "links": links}
    fields = {
        "tenant_id": tenant_id, "kind": kind, "name": name[:64], "status": status,
        "trigger": trigger, "inputs": {k: _jsonable(v) for k, v in inputs.items()},
        "outputs": {k: _jsonable(v) for k, v in outputs.items()}, "checks": checks,
        "error": (error or "")[:4000], "finished_at": finished_at, "max_seconds": max_seconds,
    }
    if started_at is not None:
        fields["started_at"] = started_at
    if finished_at is not None and started_at is not None:
        fields["duration_seconds"] = round((finished_at - started_at).total_seconds(), 2)
    if receipt is None:
        receipt = RunReceipt.objects.create(detail_kind=detail_kind, detail_id=detail_id, **fields)
    else:
        for k, v in fields.items():
            setattr(receipt, k, v)
        receipt.save()
    if alert and status == "failed" and was != "failed":
        from apps.system.services.run_alerts import alert_receipt

        alert_receipt(receipt)
    return receipt


def sync_agent_run(run, *, alert: bool = True):
    try:
        status = AGENT_STATUS.get(run.status, "running")
        terminal = status in ("success", "failed", "skipped")
        error = run.error or ""
        if status == "failed" and error == "expired":
            error = ("expired: still queued/running 30 min after it was created — the worker never "
                     "picked it up or died mid-run (housekeeping.expire_runs)")
        checks = []
        if status == "failed":
            checks.append({"name": "agent_completed", "status": "fail", "message": error[:300] or "failed"})
        elif status == "success":
            checks.append({"name": "agent_completed", "status": "pass", "message": "graph finished"})
        outputs = {"strategy": run.strategy_name, "agent_status": run.status}
        if status == "skipped":
            outputs["cancelled"] = True
        result = run.result if isinstance(run.result, dict) else {}
        for k in ("decision", "action", "trades", "signals"):
            if k in result and isinstance(result[k], (str, int, float)):
                outputs[k] = result[k]
        return _upsert(
            detail_kind="agent_run", detail_id=str(run.id), tenant_id=run.tenant_id, kind="agent",
            name=f"agent.{run.strategy_name}", status=status,
            started_at=run.started_at or run.created_at,
            finished_at=(run.completed_at or run.updated_at) if terminal else None,
            error=error if status == "failed" else "",
            inputs={"strategy": run.strategy_name, "version": run.strategy_version,
                    "portfolio_id": str(run.portfolio_id), "config": run.config or {}},
            outputs=outputs, checks=checks, links=[{"label": "Agent run", "href": f"/agents/{run.id}"}],
            max_seconds=AGENT_MAX_SECONDS, trigger="manual", alert=alert,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("detail_receipt.agent_failed", run_id=str(getattr(run, "id", "?")), error=str(e))
        return None


def sync_backtest(bt, *, alert: bool = True):
    try:
        status = BACKTEST_STATUS.get(bt.status, "running")
        terminal = status in ("success", "failed")
        metrics = bt.metrics or {}
        checks = []
        if status == "success":
            numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
            if not metrics or not any(numeric):
                # run_backtest is still a placeholder that writes zeros (plan §1).
                checks.append({"name": "real_metrics", "status": "warn",
                               "message": "every metric is 0 — the generic backtest task is a placeholder"})
            else:
                checks.append({"name": "real_metrics", "status": "pass", "message": "metrics computed"})
        if status == "failed":
            checks.append({"name": "backtest_completed", "status": "fail",
                           "message": (bt.error or "failed")[:300]})
        instance = getattr(bt, "instance", None)
        name = getattr(instance, "strategy_name", "") or "backtest"
        return _upsert(
            detail_kind="backtest", detail_id=str(bt.id), tenant_id=bt.tenant_id, kind="backtest",
            name=f"backtest.{name}", status=status, started_at=bt.created_at,
            finished_at=bt.updated_at if terminal else None,
            error=bt.error if status == "failed" else "",
            inputs={"instance_id": str(bt.instance_id), "from": bt.from_date.isoformat(),
                    "to": bt.to_date.isoformat()},
            outputs={**{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                     "equity_points": len(bt.equity_curve or [])},
            checks=checks, links=[{"label": "Strategies", "href": "/strategies"}],
            max_seconds=BACKTEST_MAX_SECONDS, trigger="api", alert=alert,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("detail_receipt.backtest_failed", backtest_id=str(getattr(bt, "id", "?")), error=str(e))
        return None


def connect_signals() -> None:
    from django.db.models.signals import post_save

    from apps.agents_core.models import AgentRun
    from apps.strategies.models import Backtest

    def _agent_saved(sender, instance, **kwargs):
        if kwargs.get("raw"):
            return
        sync_agent_run(instance)

    def _backtest_saved(sender, instance, **kwargs):
        if kwargs.get("raw"):
            return
        sync_backtest(instance)

    post_save.connect(_agent_saved, sender=AgentRun, weak=False, dispatch_uid="run_receipt_agent_run")
    post_save.connect(_backtest_saved, sender=Backtest, weak=False, dispatch_uid="run_receipt_backtest")


def ensure_detail_receipts(limit: int = 500) -> int:
    """Create receipts for AgentRun / Backtest rows that have none (rows from
    before receipts existed, or saved through a path that skipped signals).
    Historic failures are marked alerted instead of pushing a Telegram burst.
    Called by the schedule watcher; cheap when everything is mirrored."""
    from django.utils import timezone

    from apps.agents_core.models import AgentRun
    from apps.strategies.models import Backtest
    from apps.system.models import RunReceipt

    created = 0
    for detail_kind, model, sync in (("agent_run", AgentRun, sync_agent_run),
                                     ("backtest", Backtest, sync_backtest)):
        have = set(RunReceipt.objects.filter(detail_kind=detail_kind).values_list("detail_id", flat=True))
        for row in model.objects.order_by("-created_at")[:limit]:
            if str(row.id) in have:
                continue
            receipt = sync(row, alert=False)
            if receipt is not None:
                created += 1
                if receipt.status == "failed" and receipt.alerted_at is None:
                    RunReceipt.objects.filter(pk=receipt.pk).update(alerted_at=timezone.now())
    return created
