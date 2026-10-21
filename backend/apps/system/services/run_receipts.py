"""``tracked_run`` — write a RunReceipt around anything that executes.

Usage as a context manager::

    with tracked_run("job", "pull_amfi_nav", inputs={...}) as run:
        ...
        run.output(navs_created=87)
        run.check(run_checks.rows_written("NAV rows", 87))

or as a decorator under ``@shared_task``::

    @shared_task(name="apps.x.tasks.y")
    @tracked_task("job", "y", task_name="apps.x.tasks.y")
    def y(force=False): ...

Code deeper in the call stack can reach the active handle with
``current_run()`` (a no-op handle when nothing is tracked), so pipeline
helpers can attach outputs/links without threading an argument through.

**Receipts never break the wrapped job.** Every receipt write is wrapped
in try/except and logged; when the receipt row cannot be created the
handle still collects data, it just persists nothing. The job's own
exceptions propagate unchanged (the receipt records them as ``failed``).
"""
from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import structlog

from apps.system.services import run_checks

log = structlog.get_logger(__name__)

_current: contextvars.ContextVar[RunHandle | None] = contextvars.ContextVar(
    "alphadesk_current_run", default=None)


class RunHandle:
    """Mutable collector for one run; persisted on exit."""

    def __init__(self, *, kind: str, name: str, tenant=None, primary_metric: str | None = None):
        self.kind = kind
        self.name = name
        self.tenant = tenant
        self.receipt = None
        self.outputs: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.checks: list[dict] = []
        self.provenance: dict[str, Any] = {}
        self.links: list[dict] = []
        self.primary_metric = primary_metric
        self.units_ok = 0
        self.units_failed = 0
        self.swallowed = 0
        self.swallowed_examples: list[str] = []
        self.skipped_reason: str | None = None
        self.partial_reason: str | None = None
        self.detail_kind = ""
        self.detail_id = ""

    # ── what the job reports ───────────────────────────────────────────
    def output(self, **kw: Any) -> RunHandle:
        self.outputs.update(kw)
        return self

    def warn(self, message: str) -> None:
        self.warnings.append(str(message)[:500])

    def check(self, check: dict | None) -> None:
        if check:
            self.checks.append(check)

    def link(self, label: str, href: str) -> None:
        self.links.append({"label": label, "href": href})

    def set_detail(self, kind: str, detail_id: Any) -> None:
        self.detail_kind, self.detail_id = kind, str(detail_id)

    def set_provenance(self, **kw: Any) -> None:
        self.provenance.update(kw)

    def unit_ok(self, n: int = 1) -> None:
        self.units_ok += n

    def unit_failed(self, n: int = 1, error: str | None = None) -> None:
        self.units_failed += n
        if error:
            self.swallowed_error(error, count=0)

    def swallowed_error(self, error: str, count: int = 1) -> None:
        self.swallowed += count
        if len(self.swallowed_examples) < 5:
            self.swallowed_examples.append(str(error)[:300])

    def skip(self, reason: str) -> None:
        self.skipped_reason = str(reason)[:300]

    def mark_partial(self, reason: str) -> None:
        self.partial_reason = str(reason)[:300]


class _NullHandle(RunHandle):
    """Returned by ``current_run()`` outside a tracked run — swallows calls."""

    def __init__(self):
        super().__init__(kind="", name="")


def current_run() -> RunHandle:
    return _current.get() or _NullHandle()


def _celery_task_id() -> str:
    try:
        from celery import current_task

        if current_task and current_task.request and current_task.request.id:
            return str(current_task.request.id)
    except Exception:  # noqa: BLE001
        pass
    return ""


def _jsonable(value: Any) -> Any:
    import json

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _open(handle: RunHandle, *, inputs: dict, trigger: str, task_name: str, max_seconds: int | None):
    try:
        from apps.system.models import RunReceipt

        handle.receipt = RunReceipt.objects.create(
            tenant=handle.tenant,
            kind=handle.kind,
            name=handle.name,
            task_name=task_name or "",
            trigger=trigger,
            celery_task_id=_celery_task_id(),
            status=RunReceipt.Status.RUNNING,
            inputs={k: _jsonable(v) for k, v in (inputs or {}).items()},
            **({"max_seconds": int(max_seconds)} if max_seconds else {}),
        )
    except Exception as e:  # noqa: BLE001 — receipts never break the job
        log.warning("run_receipt.open_failed", name=handle.name, error=str(e))
        handle.receipt = None


def _history(name: str, exclude_id) -> list[dict]:
    from apps.system.models import RunReceipt

    qs = (RunReceipt.objects
          .filter(name=name, status__in=[RunReceipt.Status.SUCCESS, RunReceipt.Status.PARTIAL])
          .exclude(id=exclude_id)
          .order_by("-started_at")
          .values("outputs", "duration_seconds")[: run_checks.HISTORY_WINDOW])
    return list(qs)


def _finalize(handle: RunHandle, exc: BaseException | None) -> None:
    receipt = handle.receipt
    if receipt is None:
        return
    try:
        from django.utils import timezone

        from apps.system.models import RunReceipt

        now = timezone.now()
        receipt.finished_at = now
        receipt.duration_seconds = round((now - receipt.started_at).total_seconds(), 2)

        checks = list(handle.checks)
        if exc is None and handle.skipped_reason is None:
            history = _history(handle.name, receipt.id)
            checks.append(run_checks.duration_anomaly(
                receipt.duration_seconds, [h["duration_seconds"] for h in history]))
            if handle.primary_metric:
                checks.append(run_checks.zero_output_anomaly(
                    handle.primary_metric, handle.outputs.get(handle.primary_metric),
                    [(h["outputs"] or {}).get(handle.primary_metric) for h in history]))
        if handle.swallowed or handle.units_failed:
            checks.append(run_checks.swallowed_errors(
                handle.swallowed + handle.units_failed, handle.swallowed_examples))

        failed_checks = [c for c in checks if c.get("status") == run_checks.FAIL]
        if exc is not None:
            status = RunReceipt.Status.FAILED
            receipt.error = f"{type(exc).__name__}: {exc}"[:4000]
        elif handle.skipped_reason is not None:
            status = RunReceipt.Status.SKIPPED
        elif failed_checks:
            status = RunReceipt.Status.FAILED
            receipt.error = "; ".join(f"{c['name']}: {c['message']}" for c in failed_checks)[:4000]
        elif handle.units_failed and not handle.units_ok:
            status = RunReceipt.Status.FAILED
            receipt.error = f"all {handle.units_failed} unit(s) failed"
        elif handle.units_failed or handle.partial_reason:
            status = RunReceipt.Status.PARTIAL
        else:
            status = RunReceipt.Status.SUCCESS

        outputs = {k: _jsonable(v) for k, v in handle.outputs.items()}
        if handle.skipped_reason is not None:
            outputs["skipped"] = handle.skipped_reason
        if handle.units_ok or handle.units_failed:
            outputs.setdefault("units_ok", handle.units_ok)
            outputs.setdefault("units_failed", handle.units_failed)
        if handle.links:
            outputs["links"] = handle.links
        warnings = list(handle.warnings)
        if handle.partial_reason:
            warnings.insert(0, handle.partial_reason)

        receipt.status = status
        receipt.outputs = outputs
        receipt.warnings = warnings
        receipt.checks = checks
        receipt.data_provenance = {k: _jsonable(v) for k, v in handle.provenance.items()}
        receipt.detail_kind = handle.detail_kind
        receipt.detail_id = handle.detail_id
        receipt.save()
    except Exception as e:  # noqa: BLE001
        log.warning("run_receipt.finalize_failed", name=handle.name, error=str(e))
        return

    try:
        if receipt.status == RunReceipt.Status.FAILED:
            from apps.system.services.run_alerts import alert_receipt

            alert_receipt(receipt)
    except Exception as e:  # noqa: BLE001
        log.warning("run_receipt.alert_failed", name=handle.name, error=str(e))


@contextmanager
def tracked_run(kind: str, name: str, inputs: dict | None = None, *, tenant=None,
                trigger: str = "beat", task_name: str = "", primary_metric: str | None = None,
                max_seconds: int | None = None):
    handle = RunHandle(kind=kind, name=name, tenant=tenant, primary_metric=primary_metric)
    _open(handle, inputs=inputs or {}, trigger=trigger, task_name=task_name, max_seconds=max_seconds)
    token = _current.set(handle)
    exc: BaseException | None = None
    try:
        yield handle
    except BaseException as e:  # noqa: BLE001 — record, then re-raise unchanged
        exc = e
        raise
    finally:
        _current.reset(token)
        _finalize(handle, exc)


def infer_trigger(kwargs: dict) -> str:
    explicit = kwargs.get("trigger")
    if explicit in ("beat", "manual", "api"):
        return explicit
    return "manual" if kwargs.get("force") else "beat"


def tracked_task(kind: str, name: str, *, task_name: str = "", primary_metric: str | None = None,
                 max_seconds: int | None = None,
                 post: Callable[[RunHandle, Any], None] | None = None):
    """Decorator form. Put it *under* ``@shared_task``.

    A dict result with a ``skipped`` key marks the receipt ``skipped``
    (e.g. "not a trading day"). ``post(handle, result)`` runs after the
    body to derive outputs/checks from the result; errors in it are logged,
    never raised.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            inputs = {k: v for k, v in kwargs.items() if k != "trigger"}
            with tracked_run(kind, name, inputs, trigger=infer_trigger(kwargs),
                             task_name=task_name, primary_metric=primary_metric,
                             max_seconds=max_seconds) as run:
                result = fn(*args, **kwargs)
                try:
                    if isinstance(result, dict) and result.get("skipped"):
                        run.skip(str(result["skipped"]))
                    elif post is not None:
                        post(run, result)
                except Exception as e:  # noqa: BLE001
                    log.warning("run_receipt.post_failed", name=name, error=str(e))
                return result
        return wrapper

    return deco
