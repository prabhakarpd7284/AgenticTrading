"""Runs API (UX redesign R1) — mounted at /api/v1/runs/ via config/urls.py."""
from django.urls import path

from apps.system.api.runs_views import (
    ReceiptDetailView,
    ReceiptListView,
    RerunView,
    ScheduleView,
    StatusView,
)

urlpatterns = [
    path("receipts/", ReceiptListView.as_view(), name="run-receipts"),
    path("receipts/<uuid:receipt_id>/", ReceiptDetailView.as_view(), name="run-receipt-detail"),
    path("schedule/", ScheduleView.as_view(), name="run-schedule"),
    path("status/", StatusView.as_view(), name="run-status"),
    path("rerun/<str:name>/", RerunView.as_view(), name="run-rerun"),
]
