"""Ops console safety — tiered access + command/arg hardening (RCE fix)."""

from __future__ import annotations

import pytest

from apps.system.consumers import _bad_arg
from apps.system.services.ops_access import (
    PRODUCT_COMMANDS,
    TIER_ADMIN,
    TIER_PRODUCT,
    can_run,
    is_ops_admin,
    ops_tier,
)
from apps.system.services.ops_runner import _HIDDEN, validate_command


@pytest.mark.parametrize("cmd", ["shell", "dbshell", "loaddata", "dumpdata", "flush"])
def test_dangerous_commands_are_hidden(cmd):
    assert cmd in _HIDDEN
    with pytest.raises(ValueError, match="unknown or hidden"):
        validate_command(cmd)


@pytest.mark.parametrize(
    ("args", "bad"),
    [
        (["--settings=pwn"], "--settings=pwn"),
        (["-c", "import os"], "-c"),
        (["--pythonpath=/tmp"], "--pythonpath=/tmp"),
        (["--command=x"], "--command=x"),
        (["--strike", "23800", "--type", "CE"], None),
    ],
)
def test_bad_arg_blocks_injection(args, bad):
    assert _bad_arg(args) == bad


class _NonSuper:
    is_anonymous = False
    is_superuser = False


class _Super:
    is_anonymous = False
    is_superuser = True


def test_is_ops_admin_requires_superuser():
    assert is_ops_admin(None) is False
    assert is_ops_admin(_NonSuper()) is False
    assert is_ops_admin(_Super()) is True


def test_admin_tier_runs_anything_product_tier_does_not():
    assert can_run(TIER_ADMIN, "migrate") is True
    assert can_run(TIER_ADMIN, "run_trading_agent") is True

    assert can_run(TIER_PRODUCT, "run_trading_agent") is True
    assert can_run(TIER_PRODUCT, "enrich_signals") is True
    # The RCE-shaped ones stay admin-only even for a tenant owner.
    for cmd in ("migrate", "createsuperuser", "backfill_personal_tenants", "stockedge_login"):
        assert can_run(TIER_PRODUCT, cmd) is False
        assert cmd not in PRODUCT_COMMANDS

    # No tier → nothing runs.
    assert can_run(None, "run_trading_agent") is False


def test_console_disabled_kills_every_tier(settings):
    settings.OPS_CONSOLE_ENABLED = False
    assert ops_tier(_Super(), None) is None


def test_superuser_gets_admin_tier_without_a_tenant(settings):
    settings.OPS_CONSOLE_ENABLED = True
    assert ops_tier(_Super(), None) == TIER_ADMIN
    # A non-superuser with no tenant/membership gets nothing.
    assert ops_tier(_NonSuper(), None) is None


@pytest.mark.django_db
def test_tenant_owner_gets_product_tier(settings, django_user_model):
    from apps.tenants.models import Membership, Tenant

    settings.OPS_CONSOLE_ENABLED = True
    tenant = Tenant.objects.create(name="acme")
    user = django_user_model.objects.create_user(email="owner@acme.test", password="x")
    Membership.objects.create(user=user, tenant=tenant, role="owner", is_active=True)
    assert ops_tier(user, tenant) == TIER_PRODUCT

    trader = django_user_model.objects.create_user(email="t@acme.test", password="x")
    Membership.objects.create(user=trader, tenant=tenant, role="trader", is_active=True)
    assert ops_tier(trader, tenant) is None
