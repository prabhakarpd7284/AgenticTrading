"""Tenant-aware system flag helpers.

Replacement for the Streamlit-era `dashboard_utils.data_layer.is_ai_paused`
and friends. Reads SystemControl rows directly so CLI scripts (which
typically have no tenant context) can still check the kill switch.
"""
from __future__ import annotations

from typing import Any

from apps.system.models import SystemControl


def get_flag(key: str, tenant_id: Any = None, default: Any = None) -> Any:
    """Look up a SystemControl flag.

    If ``tenant_id`` is given, return that tenant's value (or ``default``).
    If omitted (operator CLI use-case), return the most-truthy value across
    all tenants — that way a single trader pressing pause halts the
    legacy global commands the same way it always did.
    """
    qs = SystemControl.objects.filter(key=key)
    if tenant_id is not None:
        row = qs.filter(tenant_id=tenant_id).first()
        return row.value if row else default
    for row in qs.only("value"):
        if _truthy(row.value):
            return row.value
    return default


def is_ai_paused(tenant_id: Any = None) -> bool:
    """True if AI trading is paused for the given tenant (or any tenant)."""
    return bool(_truthy(get_flag("ai_trading_paused", tenant_id, default=False)))


def is_kill_switch_on(tenant_id: Any = None) -> bool:
    return bool(_truthy(get_flag("kill_switch", tenant_id, default=False)))


def _truthy(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("paused") or value.get("enabled") or value.get("value"))
    return bool(value)
