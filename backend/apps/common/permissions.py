from rest_framework import permissions

from apps.system.services.ops_access import is_tenant_owner


class OwnerOnly(permissions.IsAuthenticated):
    """Allow only members with role=owner on the active tenant.

    Lived in apps.system.api.views until the ops console grew its own
    two-tier policy (apps.system.services.ops_access); the pipeline views
    still want plain owner semantics, so it moved here.
    """

    def has_permission(self, request, view) -> bool:
        if not super().has_permission(request, view):
            return False
        return is_tenant_owner(request.user, getattr(request, "tenant", None))


class TenantScoped(permissions.BasePermission):
    """Object-level check: object.tenant_id must equal request.tenant.id.

    Paired with ViewSets that override `get_queryset` to filter by tenant at the
    queryset layer. This class is a belt-and-braces guard for `/<pk>/` routes.
    """

    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        return getattr(request, "tenant", None) is not None

    def has_object_permission(self, request, view, obj):
        tenant_id = getattr(getattr(request, "tenant", None), "id", None)
        return tenant_id is not None and getattr(obj, "tenant_id", None) == tenant_id


class HasRole(permissions.BasePermission):
    """Checks view.required_roles against the user's membership role."""

    def has_permission(self, request, view):
        required = getattr(view, "required_roles", None)
        if not required:
            return True
        membership = getattr(request, "membership", None)
        return membership is not None and membership.role in required
