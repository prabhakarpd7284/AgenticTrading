"""Personal-tenant bootstrap.

Every authenticated user must belong to at least one tenant — our entire
authorisation model rests on `request.tenant` being non-null. The simplest,
least surprising way to guarantee that is: whenever we mint a JWT for a user
that has no active membership, lazily create a single-member "personal" tenant
and attach an owner membership.

This keeps onboarding one click away (retail users don't want to "create a
workspace" as a separate step) while still letting advisor / prop-desk tenants
be created via the explicit POST /api/v1/tenants/ endpoint.
"""
from __future__ import annotations

from django.db import transaction
from django.utils.text import slugify

from apps.accounts.models import User
from apps.tenants.models import Membership, Tenant


def ensure_tenant(user: User) -> Tenant:
    """Return the user's default tenant, creating one if they have none.

    Idempotent and safe under concurrent calls thanks to ``get_or_create`` +
    the ``Membership.unique_together`` constraint.
    """
    existing = (
        Membership.objects
        .filter(user=user, is_active=True)
        .select_related("tenant")
        .order_by("invited_at")
        .first()
    )
    if existing:
        return existing.tenant

    with transaction.atomic():
        base_slug = slugify(user.email.split("@")[0]) or f"user-{user.id.hex[:8]}"
        slug = base_slug
        i = 1
        while Tenant.objects.filter(slug=slug).exists():
            i += 1
            slug = f"{base_slug}-{i}"

        display = user.full_name or user.email.split("@")[0]
        tenant = Tenant.objects.create(
            name=f"{display}'s workspace",
            kind=Tenant.Kind.RETAIL,
            slug=slug,
        )
        Membership.objects.create(
            user=user,
            tenant=tenant,
            role=Membership.Role.OWNER,
            is_active=True,
        )
        return tenant
