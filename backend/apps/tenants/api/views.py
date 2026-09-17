from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiParameter, extend_schema, extend_schema_view,
)
from rest_framework import serializers, viewsets
from rest_framework.exceptions import PermissionDenied

from apps.common.pagination import CursorPagination
from apps.tenants.models import Membership, Tenant

# Tenant + Membership both carry UUID primary keys. Pin the detail-route {id}
# param to UUID so the generated schema doesn't default it to "string" (the
# cause of the "could not derive type of path parameter" warning). Annotation
# only — routing + lookup_field are untouched.
_UUID_PK = [OpenApiParameter("id", OpenApiTypes.UUID, OpenApiParameter.PATH)]
_uuid_detail_schema = extend_schema_view(
    retrieve=extend_schema(parameters=_UUID_PK),
    update=extend_schema(parameters=_UUID_PK),
    partial_update=extend_schema(parameters=_UUID_PK),
    destroy=extend_schema(parameters=_UUID_PK),
)


class MembershipPagination(CursorPagination):
    """Membership doesn't have ``created_at`` — it tracks ``invited_at`` instead.

    Without this override the default pagination's ``-created_at`` ordering
    raises ``FieldError`` when listing memberships.
    """

    ordering = "-invited_at"


class TenantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tenant
        fields = ["id", "name", "kind", "slug", "white_label_domain", "created_at"]
        read_only_fields = ["id", "created_at"]


class MembershipSerializer(serializers.ModelSerializer):
    class Meta:
        model = Membership
        fields = ["id", "user", "tenant", "role", "is_active", "invited_at"]
        # `tenant` is NEVER client-settable — the view forces it to the caller's
        # own tenant on create. Otherwise a user could POST a membership into
        # any tenant (with role=owner) and escalate cross-tenant.
        read_only_fields = ["id", "tenant", "invited_at"]


@_uuid_detail_schema
class TenantViewSet(viewsets.ModelViewSet):
    serializer_class = TenantSerializer

    def get_queryset(self):
        return Tenant.objects.filter(memberships__user=self.request.user).distinct()

    # Reads are scoped to the caller's tenants; mutating or deleting a tenant
    # requires OWNER. Without this any member (even a viewer) could PUT/PATCH or
    # DELETE their own tenant.
    def _assert_owner(self, tenant):
        is_owner = Membership.objects.filter(
            tenant=tenant, user=self.request.user, is_active=True,
            role=Membership.Role.OWNER,
        ).exists()
        if not is_owner:
            raise PermissionDenied("Only a tenant owner can modify or delete the tenant.")

    def perform_update(self, serializer):
        self._assert_owner(serializer.instance)
        serializer.save()

    def perform_destroy(self, instance):
        self._assert_owner(instance)
        instance.delete()


@_uuid_detail_schema
class MembershipViewSet(viewsets.ModelViewSet):
    serializer_class = MembershipSerializer
    pagination_class = MembershipPagination

    def get_queryset(self):
        # Tenant may be unset (e.g. anonymous request that slipped past auth);
        # return an empty queryset rather than raising.
        tenant = getattr(self.request, "tenant", None)
        if tenant is None:
            return Membership.objects.none()
        return Membership.objects.filter(tenant=tenant).order_by("-invited_at")

    # ── write authorization ──────────────────────────────────────────────
    # Reads are tenant-scoped via get_queryset; writes additionally require the
    # caller to be an owner/admin of their OWN tenant. This blocks both the
    # cross-tenant escalation (create) and in-tenant self-escalation (viewer
    # PATCHing their own role upward).
    def _assert_admin(self):
        tenant = getattr(self.request, "tenant", None)
        is_admin = tenant is not None and Membership.objects.filter(
            tenant=tenant, user=self.request.user, is_active=True,
            role__in=[Membership.Role.OWNER, Membership.Role.ADMIN],
        ).exists()
        if not is_admin:
            raise PermissionDenied("Only a tenant owner/admin can manage memberships.")

    def perform_create(self, serializer):
        self._assert_admin()
        serializer.save(tenant=self.request.tenant)

    def perform_update(self, serializer):
        self._assert_admin()
        serializer.save()

    def perform_destroy(self, instance):
        self._assert_admin()
        instance.delete()
