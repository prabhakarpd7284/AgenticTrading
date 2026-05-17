"""REST surface for the dev-ops console.

Two read-only endpoints:
  GET /api/v1/ops/commands/           — list every visible management command
  GET /api/v1/ops/commands/<n>/help/  — full --help text for one command

The actual run lives over WebSocket (apps.system.consumers.OpsConsumer)
because the operator needs streaming stdout, not a polled API.

Both endpoints are owner-gated: only members with role=owner on the
active tenant can list/inspect ops. This is a dev console — not a
trader-facing feature.
"""
from __future__ import annotations

from django.conf import settings
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.system.services.ops_runner import (
    discover_commands,
    get_command_help,
    validate_command,
)


class OwnerOnly(IsAuthenticated):
    """Allow only members with role=owner on the active tenant."""

    def has_permission(self, request, view) -> bool:
        if not super().has_permission(request, view):
            return False
        tenant = getattr(request, "tenant", None)
        if tenant is None:
            return False
        return request.user.memberships.filter(
            tenant=tenant, role="owner", is_active=True,
        ).exists()


class OpsCommandListView(APIView):
    permission_classes = [OwnerOnly]

    def get(self, request):
        return Response({
            "commands": discover_commands(),
            "trading_mode": getattr(settings, "TRADING_MODE", "paper"),
        })


class OpsCommandHelpView(APIView):
    permission_classes = [OwnerOnly]

    def get(self, request, name: str):
        try:
            validate_command(name)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=404)
        return Response({"name": name, "help": get_command_help(name)})
