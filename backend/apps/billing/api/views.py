from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.billing.models import Plan, Subscription


class PlansView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response(list(Plan.objects.values("code", "name", "monthly_inr", "entitlements")))


class SubscriptionView(APIView):
    def get(self, request):
        sub = Subscription.objects.filter(tenant=request.tenant).first()
        if not sub:
            return Response(None)
        return Response({
            "plan": sub.plan.code,
            "status": sub.status,
            "current_period_end": sub.current_period_end,
        })


class WebhookView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, provider: str):
        # TODO: verify signature via provider-specific verifier
        # TODO: dispatch to billing.services.<provider>
        return Response({"received": True})
