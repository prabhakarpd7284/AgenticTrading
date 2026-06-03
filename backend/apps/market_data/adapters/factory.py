"""Resolve a BrokerLink into a ready-to-use adapter instance.

Combines:
  * adapter registry (entry-points + the internal paper adapter)
  * credential resolution (Fernet blob for local dev, ARN for prod)

Callers pass a BrokerLink row in and get back an authenticated adapter.
"""
from __future__ import annotations

import logging
from typing import Any

from apps.market_data.adapters.base import BrokerAdapterBase
from apps.market_data.services.crypto import decrypt_credentials

logger = logging.getLogger(__name__)


# Internal explicit registry (used by tests). Falls through to entry-points.
_REGISTRY: dict[str, type[BrokerAdapterBase]] = {}
_EP_CACHE: dict[str, type[BrokerAdapterBase]] | None = None


def register_adapter(name: str, cls: type[BrokerAdapterBase]) -> None:
    _REGISTRY[name] = cls
    logger.debug("broker.adapter.registered name=%s class=%s", name, cls.__name__)


def _entry_point_classes() -> dict[str, type]:
    """Walk the alphadesk.brokers entry-points and return their CLASSES.

    The shared PluginRegistry.load_entry_points() instantiates with no
    arguments which is incompatible with adapters that require credentials
    at construction. We need the class itself here so build_adapter() can
    pass the credentials in.
    """
    global _EP_CACHE
    if _EP_CACHE is not None:
        return _EP_CACHE
    import importlib.metadata as md
    out: dict[str, type] = {}
    try:
        eps = md.entry_points(group="alphadesk.brokers")
    except TypeError:  # py<3.10
        eps = md.entry_points().get("alphadesk.brokers", [])  # type: ignore[assignment]
    for ep in eps:
        try:
            out[ep.name] = ep.load()
        except Exception as e:
            logger.warning("broker.adapter.ep_load_failed name=%s err=%s", ep.name, e)
    _EP_CACHE = out
    return out


def get_adapter_class(name: str) -> type[BrokerAdapterBase] | None:
    if name in _REGISTRY:
        return _REGISTRY[name]
    return _entry_point_classes().get(name)


def list_adapters() -> list[str]:
    return sorted(set(_REGISTRY) | set(_entry_point_classes()))


def load_credentials(link) -> dict[str, Any]:
    """Decrypt local blob if present, else fetch from AWS Secrets Manager."""
    if link.credential_blob:
        return decrypt_credentials(link.credential_blob)
    if link.credential_arn:
        try:
            return _fetch_from_secrets_manager(link.credential_arn)
        except Exception as e:
            logger.error("broker.creds.aws_fetch_failed link=%s err=%s", link.id, e)
            return {}
    return {}


def _fetch_from_secrets_manager(arn: str) -> dict[str, Any]:
    """Pull a JSON secret from AWS Secrets Manager. Only used in prod."""
    import json
    import boto3  # type: ignore
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=arn)
    return json.loads(response["SecretString"])


def build_adapter(link) -> BrokerAdapterBase | None:
    """Build an authenticated adapter for a BrokerLink. Returns None if
    the broker plugin isn't registered or credentials are missing."""
    cls = get_adapter_class(link.broker_name)
    if cls is None:
        logger.warning("broker.adapter.missing name=%s link=%s", link.broker_name, link.id)
        return None
    creds = load_credentials(link)
    if not creds:
        logger.warning("broker.creds.missing link=%s name=%s", link.id, link.broker_name)
        return None
    return cls(credentials=creds, meta=link.credential_meta or {})
