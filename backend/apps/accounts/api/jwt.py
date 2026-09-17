"""Custom JWT issuance.

The default ``TokenObtainPairSerializer`` ships tokens with only ``user_id``.
Our ``TenantMiddleware`` resolves the current tenant from a ``tenant_id``
claim, and without it every downstream DRF view hits the ``TenantScoped``
permission and returns 403 — even for a fully authenticated user.

To keep the UX invariant "sign up → dashboard loads", we:

1. Lazily bootstrap a personal tenant for any user that has no membership,
2. Embed ``tenant_id`` + ``role`` in every access/refresh token,
3. Expose the resolved tenant in the login response body so the frontend
   can skip the extra ``/me`` round-trip.

If a user has memberships on multiple tenants they can still switch — the
refresh endpoint accepts a ``tenant_id`` override (see ``TenantRefreshView``),
falling back to the currently-embedded claim.
"""
from __future__ import annotations

from rest_framework_simplejwt.serializers import (
    TokenObtainPairSerializer,
    TokenRefreshSerializer,
)
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from apps.accounts.services.tenant_bootstrap import ensure_tenant
from apps.tenants.models import Membership


def _attach_tenant_claims(token, user, *, tenant_id: str | None = None):
    """Populate ``tenant_id`` / ``role`` claims on ``token`` in place."""
    tenant = None
    if tenant_id:
        membership = (
            Membership.objects
            .filter(user=user, tenant_id=tenant_id, is_active=True)
            .select_related("tenant")
            .first()
        )
        if membership:
            tenant = membership.tenant
            token["role"] = membership.role
    if tenant is None:
        tenant = ensure_tenant(user)
        role = (
            Membership.objects
            .filter(user=user, tenant=tenant, is_active=True)
            .values_list("role", flat=True)
            .first()
        )
        if role:
            token["role"] = role
    token["tenant_id"] = str(tenant.id)
    return tenant


class TenantTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        tenant = _attach_tenant_claims(token, user)
        # Cache tenant on the token so validate() can echo it without
        # minting a second token.
        token._tenant_echo = {  # type: ignore[attr-defined]
            "id": str(tenant.id), "name": tenant.name, "slug": tenant.slug,
            "role": token.get("role"),
        }
        return token

    def validate(self, attrs):
        # Drive get_token() so its side-effect populates the access token
        # with the tenant claim.  We then surface the cached echo dict.
        data = super().validate(attrs)
        # SimpleJWT instantiates a fresh refresh token in `super().validate()`
        # and discards the per-call cache, so re-derive the echo from the user.
        from apps.tenants.models import Membership
        membership = (
            Membership.objects
            .filter(user=self.user, is_active=True)
            .select_related("tenant")
            .order_by("invited_at")
            .first()
        )
        if membership:
            data["tenant"] = {
                "id": str(membership.tenant_id),
                "name": membership.tenant.name,
                "slug": membership.tenant.slug,
                "role": membership.role,
            }
        return data


class TenantTokenRefreshSerializer(TokenRefreshSerializer):
    """Refresh endpoint that re-issues tokens with the current tenant claim."""

    def validate(self, attrs):
        data = super().validate(attrs)
        # Re-hydrate the refresh token to access custom claims
        refresh = RefreshToken(attrs["refresh"])
        user_id = refresh.get("user_id")
        tenant_id = self.initial_data.get("tenant_id") or refresh.get("tenant_id")
        if user_id:
            from apps.accounts.models import User
            user = User.objects.filter(id=user_id).first()
            if user:
                # Re-issue access with fresh tenant claim
                access = refresh.access_token
                _attach_tenant_claims(access, user, tenant_id=tenant_id)
                data["access"] = str(access)
        return data


class TenantTokenObtainPairView(TokenObtainPairView):
    serializer_class = TenantTokenObtainPairSerializer


class TenantTokenRefreshView(TokenRefreshView):
    serializer_class = TenantTokenRefreshSerializer
