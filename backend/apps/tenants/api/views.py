from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiParameter, extend_schema, extend_schema_view,
)
from rest_framework import serializers, viewsets

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


@_uuid_detail_schema
class TenantViewSet(viewsets.ModelViewSet):
    serializer_class = TenantSerializer

    def get_queryset(self):
        return Tenant.objects.filter(memberships__user=self.request.user).distinct()


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
