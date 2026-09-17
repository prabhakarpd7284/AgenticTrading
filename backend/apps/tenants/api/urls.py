"""URL wiring for the tenants app.

A subtle DRF router gotcha: ``router.register("", TenantViewSet)`` produces
the detail pattern ``^(?P<pk>[^/.]+)/$`` at the root of this namespace, which
happily eats sibling routes like ``/memberships/`` (capturing ``pk="memberships"``).

We defuse it with two routers mounted under explicit path prefixes so the
tenant-detail catchall can never shadow siblings.
"""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.tenants.api.views import TenantViewSet, MembershipViewSet

tenant_router = DefaultRouter()
tenant_router.register("", TenantViewSet, basename="tenant")

membership_router = DefaultRouter()
membership_router.register("", MembershipViewSet, basename="membership")

urlpatterns = [
    # More specific first — memberships never get swallowed by tenant detail.
    path("memberships/", include(membership_router.urls)),
    path("", include(tenant_router.urls)),
]
