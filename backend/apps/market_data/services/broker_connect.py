"""Create-or-update a BrokerLink after probing the broker with the credentials.

Single write path shared by ``POST /api/v1/brokers/{name}/connect/`` and the
``zerodha_login`` management command, so credential validation, encryption
and the BrokerLink upsert live in exactly one place. Bad credentials raise
``BrokerConnectError`` and never reach the DB.
"""
from __future__ import annotations

from typing import Any

from django.utils import timezone

from apps.market_data.adapters.factory import get_adapter_class
from apps.market_data.models import BrokerLink
from apps.market_data.services.crypto import encrypt_credentials


class BrokerConnectError(Exception):
    """Unknown broker, empty credentials, or the broker rejected them."""


def connect_broker(
    *,
    tenant,
    owner,
    broker_name: str,
    credentials: dict[str, Any],
    meta: dict[str, Any] | None = None,
    display_name: str = "",
    take_snapshot: bool = True,
) -> BrokerLink:
    """Probe ``authenticate()`` with ``credentials``; on success upsert the
    tenant's BrokerLink (keyed on tenant/broker/owner/display_name) as ACTIVE
    with Fernet-encrypted credentials, then best-effort take a snapshot."""
    cls = get_adapter_class(broker_name)
    if cls is None:
        raise BrokerConnectError(f"Unknown broker: {broker_name}")
    meta = meta or {}
    if not isinstance(credentials, dict) or not credentials:
        raise BrokerConnectError("`credentials` must be a non-empty object")

    try:
        ok = cls(credentials=credentials, meta=meta).authenticate()
    except Exception as e:
        raise BrokerConnectError(f"Auth probe raised: {e}") from e
    if not ok:
        raise BrokerConnectError("Broker rejected the credentials")

    link, _ = BrokerLink.objects.update_or_create(
        tenant=tenant,
        broker_name=broker_name,
        owner=owner,
        display_name=display_name,
        defaults={
            "credential_blob": encrypt_credentials(credentials),
            "credential_meta": meta,
            "status": BrokerLink.Status.ACTIVE,
            "last_error": "",
            "last_refreshed_at": timezone.now(),
        },
    )
    if take_snapshot:
        # Imported lazily: broker_views imports this module.
        from apps.market_data.api.broker_views import _take_snapshot
        try:
            _take_snapshot(link)
        except Exception:
            pass
    return link
