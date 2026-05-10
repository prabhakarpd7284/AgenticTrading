from rest_framework import serializers, viewsets

from apps.common.pagination import CursorPagination
from apps.tenants.models import Membership, Tenant


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


class TenantViewSet(viewsets.ModelViewSet):
    serializer_class = TenantSerializer

    def get_queryset(self):
        return Tenant.objects.filter(memberships__user=self.request.user).distinct()


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
