"""REST surface for the dev-ops console.

Two read-only endpoints:
  GET /api/v1/ops/commands/           — every command *this* caller may run
  GET /api/v1/ops/commands/<n>/help/  — full --help text for one command

The actual run lives over WebSocket (apps.system.consumers.OpsConsumer)
because the operator needs streaming stdout, not a polled API.

Both endpoints share the tier policy in apps.system.services.ops_access
with the consumer, so the list can never advertise a command the socket
would refuse: platform admins see everything visible, tenant owners see
the product commands the UI embeds via <OpButton>. When the console is
disabled (the default whenever DEBUG is off) both return 403.
"""
from __future__ import annotations

from django.conf import settings
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.system.services.ops_access import (
    TIER_ADMIN,
    can_run,
    ops_tier,
)
from apps.system.services.ops_runner import (
    discover_commands,
    get_command_help,
    validate_command,
)


class OpsAccess(IsAuthenticated):
    """Authenticated + a non-null ops tier (platform admin or tenant owner)."""

    def has_permission(self, request, view) -> bool:
        if not super().has_permission(request, view):
            return False
        tier = ops_tier(request.user, getattr(request, "tenant", None))
        # Stash it — the view needs the tier anyway, and it costs a query.
        request.ops_tier = tier  # type: ignore[attr-defined]
        return tier is not None


class OpsCommandListView(APIView):
    permission_classes = [OpsAccess]

    def get(self, request):
        tier = getattr(request, "ops_tier", None)
        commands = [c for c in discover_commands() if can_run(tier, c["name"])]
        return Response({
            "commands": commands,
            "trading_mode": getattr(settings, "TRADING_MODE", "paper"),
            "tier": tier,
            "is_admin": tier == TIER_ADMIN,
        })


class OpsCommandHelpView(APIView):
    permission_classes = [OpsAccess]

    def get(self, request, name: str):
        try:
            validate_command(name)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=404)
        if not can_run(getattr(request, "ops_tier", None), name):
            return Response(
                {"detail": f"{name!r} is platform-admin only."}, status=403,
            )
        return Response({"name": name, "help": get_command_help(name)})
