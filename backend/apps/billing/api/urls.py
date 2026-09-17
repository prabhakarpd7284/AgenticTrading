from django.urls import path
from apps.billing.api.views import PlansView, SubscriptionView, WebhookView

urlpatterns = [
    path("plans/", PlansView.as_view(), name="plans"),
    path("subscription/", SubscriptionView.as_view(), name="subscription"),
    path("webhook/<str:provider>/", WebhookView.as_view(), name="billing-webhook"),
]
